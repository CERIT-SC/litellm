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
DEFAULT_REFILL_RATE = 1  # request per second
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

        workload = await self._get_model_workload(model)
        data = await self._get_or_create_user_budget(api_key, model, workload)

        if data["requests_left"] <= 0:
            raise HTTPException(status_code=429, detail={"error": "Model capacity reached for {model}. Priority: {priority}, ..."})

        requests_left = data.get("requests_left") or 0
        updated_data = {
            "model": data.get("model"),
            "requests_left": requests_left - 1,
            "timestamp": data.get("timestamp"),
        }

        cache_key = f"{api_key}:{model}"
        await self.cache.async_set_cache(cache_key, updated_data)

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
        verbose_proxy_logger.debug("Inside log success event")

        model = response_obj["model"]
        user_api_key_dict: UserAPIKeyAuth = kwargs.get("litellm_params", {}).get("metadata", {}).get("user_api_key_auth", {})
        api_key = user_api_key_dict.api_key
        cache_key = f"{api_key}:{model}"
        user_data = await self.cache.async_get_cache(cache_key)
        verbose_proxy_logger.debug(f"user data after change: {user_data}")
        await self.handle_succss_event(api_key, model)






    # ==================== Budget Management ====================

    async def _get_or_create_user_budget(
        self,
        api_key: Optional[str],
        model: str,
        workload: float,
    ) -> dict:
        """Get existing budget from cache or create new one."""
        if api_key is None:
            return {"requests_left": -1}

        cache_key = f"{api_key}:{model}"
        cached_data = await self.cache.async_get_cache(cache_key) # dict keys model_name, requests_left, timestamp

        if cached_data is None:
            return await self._create_user_budget(self.cache, cache_key, model)

        return await self._refill_user_budget(self.cache, cache_key, cached_data, workload)

    async def _create_user_budget(
        self,
        cache: DualCache,
        cache_key: str,
        model: str,
    ) -> dict:
        """Initialize a new user budget entry in cache."""
        budget_data = {
            "model": model,
            "requests_left": DEFAULT_REQUEST_BUDGET,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
        }
        await cache.async_set_cache(cache_key, budget_data)
        return budget_data

    async def _refill_user_budget(
        self,
        cache: DualCache,
        cache_key: str,
        cached_data: dict,
        workload: float,
    ) -> dict:
        """Refill user budget based on elapsed time and current workload."""
        now = datetime.datetime.now(datetime.timezone.utc)
        timestamp = cached_data.get("timestamp")

        if timestamp is None:
            return cached_data

        elapsed_seconds = (now - timestamp).total_seconds()
        refill_rate = self._calculate_refill_rate(workload)
        requests_to_add = int(elapsed_seconds * refill_rate) * 0 #TODO FOR TESTING ONLY

        current_requests = cached_data.get("requests_left", 0)
        updated_data = {
            "model": cached_data.get("model"),
            "requests_left": current_requests + requests_to_add,
            "timestamp": now,
        }

        await cache.async_set_cache(cache_key, updated_data)
        return updated_data

    # ==================== Refill Rate Calculation ====================

    def _calculate_refill_rate(self, workload: float, base_rate: float = 0.1) -> float: # TODO check saturation in dynamic_rate_limiter_v3.py
        """
        Calculate REQUEST refill rate based on system workload.

        Args:
            workload: System load ratio (0.0 to 1.0)
            base_rate: Base refill rate in REQUESTS per second (default 0.1 = 6 req/min)

        Returns:
            Effective refill rate in requests per second
        """
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
        tpm_limit = self._get_model_tpm_limit(model)
        return self._calculate_load(tokens_used, tpm_limit)

    async def _fetch_tokens_used_in_window(self, model: str) -> int:
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

    def _get_deployment(self, model: str) -> Optional[Deployment]:
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
