import datetime
from typing import TYPE_CHECKING, Optional
from fastapi import  HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.router import Deployment


if TYPE_CHECKING:
    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache

    InternalUsageCache = _InternalUsageCache
else:
    InternalUsageCache = object




DEFAULT_REQUEST_BUDGET = 5
DEFAULT_REFILL_RATE = 10  # request per second
WORKLOAD_WINDOW_MINUTES = 5 #WORKLOAD IN PAST X MINUTES


class _PROXY_MaxAvailableCapacityLimiter(CustomLogger):
    """
    Limits user token consumption based on available system capacity.

    Tokens are refilled over time at a rate determined by current workload.
    Lower workload = faster refill, higher workload = slower refill.
    """

    def __init__(self, internal_usage_cache: InternalUsageCache):
        self.cache  = internal_usage_cache.dual_cache
        self._prev_load: float = 0.0

    # ==================== Hooks ====================

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> None:
        model = data["model"]
        api_key = user_api_key_dict.api_key
        cache_key = f"{api_key}:{model}"


        try:
            user_requests_left = await self._get_user_budget(model, cache_key)
        except HTTPException:
            raise

        except Exception as e:

            verbose_proxy_logger.error(f"Error in max available capacity rate limiter: {e}, allowing request")
            return None  # request allowed

        if user_requests_left <= 0:
            raise HTTPException(status_code=429, detail={"error": "Model capacity reached for {model}. Priority: {priority}, ..."})

        await self.cache.async_set_cache(f"{cache_key}:requests_left", -1)

        return None

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ) -> None:
        verbose_proxy_logger.debug("MaxAvailableCapacityLimiter: post call success")
        verbose_proxy_logger.debug(f"data: {data}")
        verbose_proxy_logger.debug(f"response: {response}")

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ) -> None:

        model = response_obj["model"]
        total_tokens = response_obj.get("usage").get("total_tokens", 0)

        await self.cache.async_increment_cache(f"{model}:tokens_used", total_tokens)
        return None




    # ==================== Budget Management ====================

    async def _get_user_budget(self, model: str, cache_key: str) -> int:
        """Get existing budget from cache or create new one."""
        requests_left = await self.cache.async_get_cache(f"{cache_key}:requests_left")

        return await self._create_user_budget(cache_key) if requests_left is None \
            else  await self._refill_user_budget(cache_key, model)

    async def _create_user_budget(self, cache_key: str) -> int:

        await self.cache.async_set_cache(f"{cache_key}:requests_left", DEFAULT_REQUEST_BUDGET)
        await self.cache.async_set_cache(f"{cache_key}:last_refill", datetime.datetime.now(datetime.timezone.utc).isoformat())

        return DEFAULT_REQUEST_BUDGET

    async def _refill_user_budget(self, model: str, cache_key: str) -> int:
        """Refill user budget based on elapsed time and current workload."""
        now = datetime.datetime.now(datetime.timezone.utc)
        timestamp = datetime.datetime.fromisoformat(await self.cache.async_get_cache(f"{cache_key}:timestamp"))
        elapsed_seconds = (now - timestamp).total_seconds()

        refill_rate = await self._calculate_refill_rate(model)
        requests_to_add = int(elapsed_seconds * refill_rate)
        new_requests = await self.cache.async_increment_cache(f"{cache_key}:requests_left", requests_to_add)
        await self.cache.async_set_cache(f"{cache_key}:requests_left", now.isoformat())

        return new_requests if new_requests is not None else 0

    # ==================== Refill Rate Calculation ====================

    async def _calculate_refill_rate(self, model: str, base_rate: float = 0.1) -> float: # TODO check saturation in dynamic_rate_limiter_v3.py
        """
        Calculate REQUEST refill rate based on system workload.

        Args:
            model: model name
            base_rate: Base refill rate in REQUESTS per second (default 0.1 = 6 req/min)

        Returns:
            Effective refill rate in requests per second
        """
        workload = await self._get_model_workload(model)

        if workload < 0.5:
            # Green zone: 20% bonus (0.12 req/s = 7.2 req/min)
            return base_rate * 1.2

        if workload < 0.8:
            # Yellow zone: linear decrease 100% -> 40%
            # 0.5 -> 0.1 req/s, 0.8 -> 0.04 req/s
            factor = 1.0 - (workload - 0.5) * 2  # 2 = 1/(0.8-0.5)
            return base_rate * max(factor, 0.4)

        # Red zone: exponential decrease 40% -> 2%
        # 0.8 -> 0.04 req/s, 0.9 -> 0.01 req/s, 1.0 -> 0.002 req/s
        factor = 0.4 * ((1.0 - workload) / 0.2) ** 2
        return base_rate * max(factor, 0.02)

    # ==================== Workload Calculation ====================

    async def _get_model_workload(self, model: str) -> float:
        """Calculate current workload for a model."""
        tokens_used = await self._fetch_tokens_used_in_window(model)
        tpm_limit = self._get_model_tpm_limit(model) * 5 #window is 5 minutes
        workload = self._calculate_load(tokens_used, tpm_limit)

        await self.cache.async_set_cache(f"{model}:workload", workload)
        return workload

    async def _fetch_tokens_used_in_window(self, model: str) -> int:
        tokens_used_interval = await self.cache.async_get_cache(f"{model}:tokens")
        cached_timestamp = await self.cache.async_get_cache(f"{model}:timestamp")
        if tokens_used_interval is None:
            return await self.load_used_tokens(model) # load tokens used from db


        now = datetime.datetime.now(datetime.timezone.utc)
        age_minutes = (now - cached_timestamp).total_seconds() / 60


        return await self.load_used_tokens(model) if age_minutes > WORKLOAD_WINDOW_MINUTES else tokens_used_interval




    async def load_used_tokens(self, model: str) -> int:
        """Fetch total tokens used by model in the configured time window."""
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            return 0

        sql_query = f"""
            SELECT COALESCE(SUM(total_tokens), 0) as total
            FROM "LiteLLM_SpendLogs"
            WHERE "endTime" >= NOW() - INTERVAL '{WORKLOAD_WINDOW_MINUTES} minutes'
            AND model = $1
        """
        db_response = await prisma_client.db.query_raw(sql_query, model)

        if db_response is None or len(db_response) == 0:
            return 0

        return int(db_response[0]["total"])

    def _get_model_tpm_limit(self, model: str) -> int:
        """Get TPM (tokens per minute) limit for a model deployment."""
        deployment = self._get_deployment(model)

        if deployment is None:
            return 0

        if deployment.litellm_params is None:
            return 0

        return deployment.litellm_params.tpm or 0

    def _get_deployment(self, model: str) -> Optional[Deployment]: # TODO CHECK IF NOT BETTER TO CACHE IT
        """Get deployment configuration for a model."""
        from litellm.proxy.proxy_server import llm_router

        if llm_router is None:
            return None

        return llm_router.get_deployment_by_model_group_name(model_group_name=model)

    # ==================== Load Calculation ====================

    def _calculate_load(self, tokens_used: int, max_tokens: int) -> float:
        """Calculate simple load ratio."""
        if max_tokens == 0:
            return 0.0
        return min(tokens_used / max_tokens, 1.0)
