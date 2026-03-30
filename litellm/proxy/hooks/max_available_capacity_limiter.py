import datetime
from litellm import Router
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.router import Deployment
from typing import Optional


class _PROXY_MaxAvailableCapacityLimiter(CustomLogger):
    # Class variables or attributes
    def __init__(self):
        self.llm_router: Optional[Router] = None
        self.prev_load = 0;

    def update_variables(self, llm_router: Router):
        """Update the router reference. Called during proxy initialization."""
        self.llm_router = llm_router

    async def async_pre_call_hook(
            self,
            user_api_key_dict: UserAPIKeyAuth,
            cache: DualCache,
            data: dict,
            call_type: str,
    ):

        verbose_proxy_logger.debug("test build number: 7")

        workload = await self.get_model_workload(data["model"])
        user_used_toknes = await self.get_user_used_tokens(cache, user_api_key_dict.api_key, data["model"])

    def get_refill_rate(self, base_rate, load):
        if load < 0.5:
            # Zelená zóna: Bonus za nízku záťaž!
            return base_rate * 1.2
        elif load < 0.8:
            # Žltá zóna: Lineárny pokles
            # 0.5 -> 100%, 0.8 -> 40%
            factor = 1.0 - (load - 0.5) * 2  # 2 = 1/(0.8-0.5)
            return base_rate * max(factor, 0.4)
        else:
            # Červená zóna: Drastický pokles
            # 0.8 -> 40%, 0.9 -> 10%, 1.0 -> 2%
            factor = 0.4 * ((1 - load) / 0.2) ** 2
            return base_rate * max(factor, 0.02)

    async def get_user_used_tokens(self, cache: DualCache, user_api_key: Optional[str], model: str):
        key = f"{user_api_key}:{model}"
        if user_api_key is None:
            return -1

        data = await cache.async_get_cache(key)

        if data is None:
            return await self.set_user_model_cache(cache, key, model)
        return data

    async def set_user_model_cache(self, cache: DualCache, key: str, model: str):
        data = {
            "model": model,
            "tokens_left": 100,
            "time_stamp": datetime.datetime.now(datetime.timezone.utc)

        }
        await cache.async_set_cache(key, data)
        return data


    def get_deployment_by_model_name(self, model_name: str) -> Optional[Deployment]:
        """
        Get deployment for a given model name.

        Args:
            model_name: The model group name (e.g., "gpt-4", "deepseek-v3.2")

        Returns:
            Deployment object containing litellm_params with tpm/rpm limits,
            or None if not found.
        """

        if self.llm_router is None:

            return None

        return self.llm_router.get_deployment_by_model_group_name(
            model_group_name=model_name
        )

    def calculate_load(self, curr_tokens_in_use: int, max_tokens: int) -> float:
        return 0 if max_tokens == 0 else min(curr_tokens_in_use / max_tokens, 1.0)

    def calculate_load_ema(self, curr_tokens_in_use: int, max_tokens: int) -> float:

        if max_tokens == 0:
            instant_load = 0.0
        else:
            instant_load = min(curr_tokens_in_use / max_tokens, 1.0)


        alpha = 0.25 #weight of instal load

        if not hasattr(self, 'prev_load') or self.prev_load is None:
            self.prev_load = instant_load

        smooth_load = alpha * instant_load + (1 - alpha) * self.prev_load
        self.prev_load = smooth_load

        return round(smooth_load, 3)

    async def get_generated_toknes_by_model_and_time(self, model: str):
        from litellm.proxy.proxy_server import prisma_client
        if prisma_client is None:
            return -1
        
        sql_querry = """SELECT SUM(total_tokens) FROM "LiteLLM_SpendLogs" sl WHERE sl."endTime" >= NOW() - INTERVAL '5 minutes' AND model = $1;"""
        db_response = await prisma_client.db.query_raw(sql_querry, model)
        if db_response is None:
            return -1

        return db_response[0]["sum"]

    async def get_model_workload(self, model) -> float:
        tokens_used = await self.get_generated_toknes_by_model_and_time(model)
        verbose_proxy_logger.debug(f"tokens used: {tokens_used}")
        deployment = self.get_deployment_by_model_name(model)
        tpm_limit = 0

        if deployment is not None and deployment.litellm_params.tpm is not None:
            tpm_limit = deployment.litellm_params.tpm

        return self.calculate_load(100 , tpm_limit)

    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: UserAPIKeyAuth, response
    ):
        verbose_proxy_logger.debug("Inside Max Available Capacity Limiter Post-Call-Success Hook")
        verbose_proxy_logger.debug("data:", data)
        verbose_proxy_logger.debug("response:", response)

