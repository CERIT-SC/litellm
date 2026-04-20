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


from typing import TypedDict

class CacheDataUser(TypedDict):
    model_name: str
    requests_left: int
    last_refill: str  # ISO format string for JSON serialization

class CacheDataModel(TypedDict):
    workload: float
    tokens_used: int
    timestamp: str

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

        verbose_proxy_logger.debug(f"MaxAvailableCapacityLimiter: pre call hook {datetime.datetime.now(datetime.timezone.utc).isoformat(sep=' ')}")

        try:
            workload = await self._get_model_workload(model)
            user_data = await self._get_user_budget(api_key, model, workload)
        except HTTPException:
            raise
        except Exception as e:

            verbose_proxy_logger.error(f"Error in max available capacity rate limiter: {e}, allowing request")
            return None  # request allowed



        if user_data["requests_left"] <= 0:
            raise HTTPException(status_code=429, detail={"error": "Model capacity reached for {model}. Priority: {priority}, ..."})

        updated_data: CacheDataUser = {
            "model_name": user_data["model_name"],
            "requests_left": user_data["requests_left"] - 1,
            "last_refill": user_data["last_refill"],
        }

        cache_key = f"{api_key}:{model}"
        await self.cache.async_set_cache(cache_key, updated_data)
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
        verbose_proxy_logger.debug("Inside log success event")

        model = response_obj["model"]
        user_api_key_dict: UserAPIKeyAuth = kwargs.get("litellm_params", {}).get("metadata", {}).get("user_api_key_auth", {})
        api_key = user_api_key_dict.api_key
        cache_key = f"{api_key}:{model}"

        total_tokens = response_obj.get("usage").get("total_tokens", 0)

        model_data: CacheDataModel = await self.cache.async_get_cache(model)

        await self.cache.async_set_cache(model, {
            "workload": model_data["workload"],
            "tokens_used": model_data["tokens_used"] + total_tokens,
            "timestamp": model_data["timestamp"],
        })





    # ==================== Budget Management ====================

    async def _get_user_budget(
        self,
        api_key: Optional[str],
        model: str,
        workload: float,
    ) -> CacheDataUser:
        """Get existing budget from cache or create new one."""
        if api_key is None:
            raise HTTPException(status_code=429, detail={"error": "API key not provided"})

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
    ) -> CacheDataUser:
        """Initialize a new user budget entry in cache."""
        budget_data: CacheDataUser = {
            "model_name": model,
            "requests_left": DEFAULT_REQUEST_BUDGET,
            "last_refill": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        await cache.async_set_cache(cache_key, budget_data)
        return budget_data

    async def _refill_user_budget(
        self,
        cache: DualCache,
        cache_key: str,
        cached_data: CacheDataUser,
        workload: float,
    ) -> CacheDataUser:
        """Refill user budget based on elapsed time and current workload."""
        now = datetime.datetime.now(datetime.timezone.utc)
        timestamp = datetime.datetime.fromisoformat(cached_data["last_refill"])
        elapsed_seconds = (now - timestamp).total_seconds()

        refill_rate = self._calculate_refill_rate(workload)
        requests_to_add = int(elapsed_seconds * refill_rate)

        current_requests = cached_data["requests_left"]

        updated_data: CacheDataUser = {
            "model_name": cached_data["model_name"],
            "requests_left": current_requests + requests_to_add,
            "last_refill": now.isoformat(),
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
        tpm_limit = self._get_model_tpm_limit(model) * 5 #window is 5 minutes
        workload = self._calculate_load(tokens_used, tpm_limit)
        data: CacheDataModel = {
            "workload": workload,
            "tokens_used": tokens_used,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        await self.cache.async_set_cache(model, data)

        return workload

    async def _fetch_tokens_used_in_window(self, model: str) -> int:
        cache_data: CacheDataModel = await self.cache.async_get_cache(model)
        verbose_proxy_logger.debug(f"cache data workload {cache_data}")

        if cache_data is None or cache_data["workload"] is None:
            return await self.load_used_tokens(model)

        cached_timestamp = datetime.datetime.fromisoformat(cache_data["timestamp"])
        now = datetime.datetime.now(datetime.timezone.utc)
        age_minutes = (now - cached_timestamp).total_seconds() / 60


        return await self.load_used_tokens(model) if age_minutes > WORKLOAD_WINDOW_MINUTES else cache_data["tokens_used"]




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
