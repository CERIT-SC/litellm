from fastapi import HTTPException

from litellm import Router, verbose_logger
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
        verbose_proxy_logger.debug("update variables call")
        self.llm_router = llm_router


    def get_deployment_by_model_name(self, model_name: str) -> Optional[Deployment]:
        """
        Get deployment for a given model name.

        Args:
            model_name: The model group name (e.g., "gpt-4", "deepseek-v3.2")

        Returns:
            Deployment object containing litellm_params with tpm/rpm limits,
            or None if not found.
        """
        verbose_proxy_logger.warning("INSIDE DEPLOYMENT BY MODEL NAME")
        if self.llm_router is None:
            verbose_proxy_logger.warning(
                "llm_router is not initialized. Call update_variables() first."
            )
            return None

        return self.llm_router.get_deployment_by_model_group_name(
            model_group_name=model_name
        )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ):
        verbose_proxy_logger.debug("Inside Max Available Capacity Limiter Pre-Call Hook")
        # verbose_proxy_logger.debug(f"data: {data}" )
        # verbose_proxy_logger.debug(f"call_type: {data}")
        verbose_proxy_logger.debug(f"router: {self.llm_router}")
        verbose_proxy_logger.debug("before deployment call")
        deployment = self.get_deployment_by_model_name("deepseek-v3.2")
        verbose_proxy_logger.debug(f"deployment: {deployment}")

    def calculate_load(self, curr_tokens_in_use: int, max_tokens: int) -> float:
        if max_tokens == 0:
            return 0.0
        return min(curr_tokens_in_use / max_tokens, 1.0)

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
            return 0
        
        sql_querry = """SELECT SUM(total_tokens) FROM "LiteLLM_SpendLogs" sl WHERE sl."endTime" >= NOW() - INTERVAL '5 minutes' AND model = $1;"""
        db_response = await prisma_client.db.query_raw(sql_querry, model)
        if db_response is None:
            return []

        return db_response

    async def model_work_load(self, model="deepseek-v3.2"):
        from litellm.proxy.proxy_server import prisma_client
        
        
        tokens_used = await self.get_generated_toknes_by_model_and_time(model)
        deployment = self.get_deployment_by_model_name(model)

        if deployment is not None:
            tpm_limit = deployment.litellm_params.tpm
            rpm_limit = deployment.litellm_params.rpm
            verbose_proxy_logger.debug(
                f"Model {model}: TPM limit={tpm_limit}, RPM limit={rpm_limit}"
            )
        else:
            verbose_proxy_logger.warning(f"No deployment found for model: {model}")
            return -1

        self.calculate_load(tokens_used, tpm_limit)
        

    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: UserAPIKeyAuth, response
    ):
        verbose_proxy_logger.debug("Inside Max Available Capacity Limiter Post-Call-Success Hook")
        verbose_proxy_logger.debug("data:", data)
        verbose_proxy_logger.debug("response:", response)

