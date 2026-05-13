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


MAX_REQUEST_BUDGET = 5
BASE_REFILL_RATE = 0.1  # requests per second (6 req/min)
WORKLOAD_WINDOW_MINUTES = 5
WORKLOAD_REFRESH_SECONDS = 30

REFILL_AND_DECREMENT_LUA = """
local requests_key = KEYS[1]
local timestamp_key = KEYS[2]
local refill_rate = tonumber(ARGV[1])
local max_budget = tonumber(ARGV[2])

local requests_left = redis.call('GET', requests_key)
local stored_timestamp = redis.call('GET', timestamp_key)

local time_result = redis.call('TIME')
local current_time = tonumber(time_result[1])

if requests_left == false then
    -- First time: initialize at max_budget, then try to consume 1
    redis.call('SET', requests_key, max_budget - 1)
    redis.call('SET', timestamp_key, current_time)
    return 1
end

requests_left = tonumber(requests_left)

if stored_timestamp == false then
    redis.call('SET', timestamp_key, current_time)
else
    local elapsed_seconds = current_time - tonumber(stored_timestamp)
    local requests_to_add = elapsed_seconds * refill_rate
    requests_left = math.min(requests_left + requests_to_add, max_budget)
    redis.call('SET', timestamp_key, current_time)
end

-- Check and decrement
if requests_left >= 1 then
    redis.call('SET', requests_key, requests_left - 1)
    return 1
end

redis.call('SET', requests_key, requests_left)
return 0
"""


class _PROXY_MaxAvailableCapacityLimiter(CustomLogger):
    """
    Limits user token consumption based on available system capacity.

    Tokens are refilled over time at a rate determined by current workload.
    Lower workload = faster refill, higher workload = slower refill.
    """

    def __init__(self, internal_usage_cache: InternalUsageCache):
        self.cache  = internal_usage_cache.dual_cache
        self._prev_load: float = 0.0
        if self.cache.redis_cache is None:
            raise Exception("Redis cache is required for MaxAvailableCapacityLimiter")

        self._refill_and_decrement_script = self.cache.redis_cache.async_register_script(REFILL_AND_DECREMENT_LUA)

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
            granted = await self._try_consume_budget(model, cache_key)
        except Exception as e:
            verbose_proxy_logger.error(f"Error in max available capacity rate limiter: {e}, allowing request")
            raise HTTPException(status_code=500, detail={"error": "Internal error in MaxAvailableCapacityLimiter"})

        if not granted:
            raise HTTPException(status_code=429, detail={"error": f"Model capacity reached for {model}."})


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

    # ==================== Budget Management ====================

    async def _try_consume_budget(self, model: str, cache_key: str) -> bool:
        """Refill user budget and atomically consume one request using a Lua script in Redis.

        Returns True if the request was granted, False if denied.
        """
        refill_rate = await self._calculate_refill_rate(model, base_rate=BASE_REFILL_RATE)

        redis_cache = self.cache.redis_cache
        if redis_cache is None:
            raise Exception("Redis cache is not configured for MaxAvailableCapacityLimiter")

        requests_key = f"{cache_key}:requests_left"
        timestamp_key = f"{cache_key}:timestamp"

        result = await self._refill_and_decrement_script(
            keys=[requests_key, timestamp_key],
            args=[refill_rate, MAX_REQUEST_BUDGET],
        )

        return int(result) == 1

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
        tpm_limit = self._get_model_tpm_limit(model) * WORKLOAD_WINDOW_MINUTES
        workload = self._calculate_load(tokens_used, tpm_limit)

        await self.cache.async_set_cache(f"{model}:workload", workload) # TODO: never used
        return workload

    async def _fetch_tokens_used_in_window(self, model: str) -> int:
        cached_timestamp = await self.cache.async_get_cache(f"{model}:timestamp")
        tokens_used_last_update = None
        if cached_timestamp is not None:
            tokens_used_last_update = float(cached_timestamp)
        
        now = datetime.datetime.now().timestamp()
        if tokens_used_last_update is not None and (now - tokens_used_last_update) < WORKLOAD_REFRESH_SECONDS:
            cached_tokens = await self.cache.async_get_cache(f"{model}:tokens")
            if cached_tokens is not None:
                return int(cached_tokens)

        used_tokens = await self._load_used_tokens(model)

        await self.cache.async_set_cache(f"{model}:tokens", used_tokens)
        await self.cache.async_set_cache(f"{model}:timestamp", now)

        return used_tokens

    async def _load_used_tokens(self, model: str) -> int:
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

        return int(db_response[0].get("total", 0))

    def _get_model_tpm_limit(self, model: str) -> int:
        """Get TPM (tokens per minute) limit for a model deployment."""
        deployment = self._get_deployment(model)

        if deployment is None or deployment.model_info is None:
            raise Exception(f"Deployment or model info not found for model: {model}")

        max_tpm = int(deployment.model_info["max_tpm"])

        return max_tpm

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
