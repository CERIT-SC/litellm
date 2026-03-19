from fastapi import HTTPException

from litellm import verbose_logger
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth


class _PROXY_MaxAvailableCapacityLimiter(CustomLogger):
    # Class variables or attributes
    def __init__(self):
        pass

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ):
        verbose_proxy_logger.debug("Inside Max Available Capacity Limiter Pre-Call Hook")
        verbose_proxy_logger.debug("data:", data)
        verbose_proxy_logger.debug("call_type:", data)

    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: UserAPIKeyAuth, response
    ):
        verbose_proxy_logger.debug("Inside Max Available Capacity Limiter Post-Call-Success Hook")
        verbose_proxy_logger.debug("data:", data)
        verbose_proxy_logger.debug("response:", response)