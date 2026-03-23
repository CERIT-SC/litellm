from fastapi import HTTPException

from litellm import verbose_logger
from typing import  Optional
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
        verbose_proxy_logger.debug(f"data: {data}" )
        verbose_proxy_logger.debug(f"call_type: {data}")
        
        await self.db_call()
        


    async def db_call(self):
        from litellm.proxy.proxy_server import prisma_client
        db_resp = await self.get_generated_toknes_by_model_and_time("deepseek-v3.2", prisma_client)
        verbose_proxy_logger.debug(f"db_resp: {db_resp}")


    async def get_generated_toknes_by_model_and_time(self, model: str, prisma_client):
        sql_querry = """SELECT SUM(total_tokens) FROM "LiteLLM_SpendLogs" sl WHERE sl."endTime" >= NOW() - INTERVAL '5 minutes' AND model = $1;"""
        db_response = await prisma_client.db.query_raw(sql_querry, model)
        if db_response is None:
            return []

        return db_response


    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: UserAPIKeyAuth, response
    ):
        verbose_proxy_logger.debug("Inside Max Available Capacity Limiter Post-Call-Success Hook")
        verbose_proxy_logger.debug("data:", data)
        verbose_proxy_logger.debug("response:", response)