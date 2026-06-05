"""
Redis-based request queue limiter for LiteLLM proxy.

This module provides a queue-based rate limiting mechanism that:
- Allows up to MAX_TOTAL_REQUESTS per API key - GLOBAL
- Processes maximum MAX_CONCURRENT_REQUESTS concurrently per API key - GLOBAL
- Queues remaining requests in Redis using FIFO ordering
- Releases queued requests when running requests complete
- Waits internally for queued requests to be processed

The queue is GLOBAL per API key, not per-model.

Key principle: The queue list stores ONLY pending request IDs. The running key
stores a list of currently running request IDs. No markers are stored in the queue.
"""

import asyncio
import os
from typing import TYPE_CHECKING, Any, Optional, Union

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache

    Span = Union[_Span, Any]
    InternalUsageCache = _InternalUsageCache
else:
    Span = Any
    InternalUsageCache = Any
    DualCache = Any
    UserAPIKeyAuth = Any


# Default values for queue configuration
MAX_CONCURRENT_REQUESTS = 2  # Maximum requests processed concurrently per API key
QUEUE_POLL_INTERVAL = 2  # Seconds between queue checks
QUEUE_MAX_WAIT_TIME = 600  # Maximum time (seconds) to wait in queue before

# Lua script for atomically trying to acquire a slot
# Returns: "RUNNING" (-1) if request can run immediately, queue position (int) if queued, "FULL" (-2) if queue is full
TRY_ACQUIRE_SLOT_LUA = """
local running_key = KEYS[1]
local queue_key = KEYS[2]
local max_concurrent = tonumber(ARGV[1])
local max_total = tonumber(ARGV[2])
local request_id = ARGV[3]
local queue_key_ttl = tonumber(ARGV[4])

-- Get current running count
local running_count = redis.call('LLEN', running_key)

-- Get current total count (running + queued)
local queue_length = redis.call('LLEN', queue_key)
local total_count = running_count + queue_length

-- Check if we can run immediately
if running_count < max_concurrent then
    -- Add request ID to running list and set TTL. Queue stores only pending requests.
    redis.call('RPUSH', running_key, request_id)
    redis.call('EXPIRE', running_key, queue_key_ttl)
    return -1
end

-- Check if we can queue
if total_count < max_total then
    -- Add to queue and return queue position, set TTL on queue
    redis.call('RPUSH', queue_key, request_id)
    redis.call('EXPIRE', queue_key, queue_key_ttl)
    return tostring(queue_length + 1)
end

-- Queue is full
return -2
"""


# This removes the request ID from the running list. Queue promotion is handled by polling in _wait_for_slot().
# ARGV[1] = queue_key_ttl (TTL in seconds)
# ARGV[2] = request_id
RELEASE_SLOT_LUA = """
local running_key = KEYS[1]
local queue_key = KEYS[2]
local queue_key_ttl = tonumber(ARGV[1])
local request_id = ARGV[2]

-- Remove the request ID from the running list and refresh TTL
redis.call('LREM', running_key, 1, request_id)
redis.call('EXPIRE', running_key, queue_key_ttl)

-- Queue promotion is handled by CHECK_AND_PROMOTE_LUA during polling in _wait_for_slot().
-- We do not manipulate the queue here to avoid race conditions.
return 'OK'
"""

# Lua script for checking if a request is at the front of the queue and can run
CHECK_AND_PROMOTE_LUA = """
local running_key = KEYS[1]
local queue_key = KEYS[2]
local request_id = ARGV[1]
local max_concurrent = tonumber(ARGV[2])
local queue_key_ttl = tonumber(ARGV[3])

-- Get current running count
local running_count = redis.call('LLEN', running_key)

-- Check if we can run now
if running_count >= max_concurrent then
    return 0
end

-- Check if this request is at the front of the queue
local first_item = redis.call('LINDEX', queue_key, 0)
if first_item == request_id then
    -- Remove from queue and add request ID to running list, refresh TTL on both keys
    redis.call('LPOP', queue_key)
    redis.call('RPUSH', running_key, request_id)
    redis.call('EXPIRE', running_key, queue_key_ttl)
    redis.call('EXPIRE', queue_key, queue_key_ttl)
    return 1
end

return 0
"""

# Lua script for cleaning up a queued request (removes from queue without affecting running count)
CLEANUP_QUEUED_LUA = """
local queue_key = KEYS[1]
local request_id = ARGV[1]
local queue_key_ttl = tonumber(ARGV[2])

-- Remove the request from queue and refresh TTL (queue only stores pending request IDs)
local removed = redis.call('LREM', queue_key, 1, request_id)
redis.call('EXPIRE', queue_key, queue_key_ttl)

return tostring(removed)
"""


class _PROXY_RequestQueueLimiter(CustomLogger):
    """
    Redis-based request queue limiter for LiteLLM proxy.
    
    This limiter manages request queuing using Redis, allowing:
    - Up to MAX_CONCURRENT_REQUESTS to run concurrently per API key
    - Up to MAX_TOTAL_REQUESTS total (running + queued) per API key
    - FIFO queue for pending requests
    - Internal waiting for queued requests
    
    The queue is GLOBAL per API key, not per-model.
    """

    def __init__(self, internal_usage_cache: InternalUsageCache):
        """
        Initialize the request queue limiter.
        
        Args:
            internal_usage_cache: The internal usage cache instance with Redis access
        """
        super().__init__()
        self.internal_usage_cache = internal_usage_cache
        
        # Register Lua scripts with Redis
        self._try_acquire_script = None
        self._release_slot_script = None
        self._check_and_promote_script = None
        self._cleanup_queued_script = None
        
        if self.internal_usage_cache.dual_cache.redis_cache is not None:
            try:
                self._try_acquire_script = (
                    self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                        TRY_ACQUIRE_SLOT_LUA
                    )
                )
                self._release_slot_script = (
                    self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                        RELEASE_SLOT_LUA
                    )
                )
                self._check_and_promote_script = (
                    self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                        CHECK_AND_PROMOTE_LUA
                    )
                )
                self._cleanup_queued_script = (
                    self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                        CLEANUP_QUEUED_LUA
                    )
                )
                verbose_proxy_logger.debug(
                    "RequestQueueLimiter: Lua scripts registered successfully"
                )
            except Exception as e:
                verbose_proxy_logger.warning(
                    f"RequestQueueLimiter: Failed to register Lua scripts: {str(e)}"
                )
        
        # Configuration constants
        self.max_concurrent_requests = int(
            os.getenv("LITELLM_QUEUE_MAX_CONCURRENT_REQUESTS", MAX_CONCURRENT_REQUESTS)
        )

        # Queue wait configuration
        self.queue_poll_interval = int(os.getenv("LITELLM_QUEUE_POLL_INTERVAL", QUEUE_POLL_INTERVAL))
        self.max_queue_wait_time = int(os.getenv("LITELLM_QUEUE_MAX_WAIT_TIME", QUEUE_MAX_WAIT_TIME))

    def _get_queue_keys(
        self, api_key: str
    ) -> tuple[str, str]:
        """
        Get the Redis keys for the running list and queue for an API key.
        
        Args:
            api_key: The API key to get keys for
            
        Returns:
            Tuple of (running_key, queue_key)
        """
        # Use hash tag to ensure keys are in same slot for Redis cluster
        running_key = f"{{{api_key}}}:queue:running"
        queue_key = f"{{{api_key}}}:queue:pending"
        return running_key, queue_key

    def _get_request_id(self, data: dict) -> str | None:
        """Extract the request ID from the data dictionary if available."""
        request_id = data.get("litellm_call_id", None)
        
        return str(request_id) or None

    async def _wait_for_slot(
        self,
        api_key: str,
        request_id: str,
    ) -> None:
        """
        Wait for a slot to become available in the queue.
        
        This method polls Redis to check if the request has been promoted
        from the queue to running state.
        
        Args:
            api_key: The API key for this request
            request_id: The unique request ID
            
        Raises:
            HTTPException: 429 if the request times out waiting in queue
        """
        running_key, queue_key = self._get_queue_keys(api_key)

        event_loop = asyncio.get_event_loop()
        start_time = event_loop.time()
        
        while True:
            # Check elapsed time
            elapsed = event_loop.time() - start_time
            if elapsed > self.max_queue_wait_time:
                verbose_proxy_logger.warning(
                    f"RequestQueueLimiter: Request {request_id[:8]} timed out waiting in queue after {elapsed:.1f}s"
                )
                # Clean up this request from the queue
                await self._cleanup_queued_request(api_key, request_id)
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Request timed out waiting in queue. Waited {elapsed:.1f}s. "
                        f"Maximum wait time is {self.max_queue_wait_time}s."
                    ),
                    headers={
                        "retry-after": "10",
                        "x-rate-limit-queue-status": "timeout",
                    },
                )
            
            # Check if we can run now
            if self._check_and_promote_script is not None:
                try:
                    result = await self._check_and_promote_script(
                        keys=[running_key, queue_key],
                        args=[request_id, self.max_concurrent_requests, self.max_queue_wait_time],
                    )
                    
                    if result == 1:
                        verbose_proxy_logger.debug(
                            f"RequestQueueLimiter: Request {request_id[:8]} promoted to running after waiting"
                        )
                        return  # Slot acquired, exit wait loop
                    
                    # Still waiting, update position info
                    if result == 0:
                        # Get current queue position
                        if self.internal_usage_cache.dual_cache.redis_cache is not None:
                            queue_items = self.internal_usage_cache.dual_cache.redis_cache.redis_client.lrange(queue_key, 0, -1)
                            current_position = None
                            for idx, item in enumerate(queue_items):
                                if item == request_id:
                                    current_position = idx + 1
                                    break
                            if current_position is not None:
                                verbose_proxy_logger.debug(
                                    f"RequestQueueLimiter: Request {request_id[:8]} at queue position {current_position}"
                                )
                except Exception as e:
                    verbose_proxy_logger.debug(
                        f"RequestQueueLimiter: Check/promote script failed: {str(e)}"
                    )

            await asyncio.sleep(self.queue_poll_interval)

    async def _cleanup_queued_request(
        self,
        api_key: str,
        request_id: str,
    ) -> None:
        """
        Clean up a queued request that didn't complete (e.g., client disconnected, timeout).
        
        This method removes a request from the queue. The queue only stores pending
        request IDs, so no running count adjustment is needed.
        
        Args:
            api_key: The API key associated with the request
            request_id: The unique request ID to clean up
        """
        try:
            _, queue_key = self._get_queue_keys(api_key)
            
            verbose_proxy_logger.debug(
                f"RequestQueueLimiter: Cleaning up queued request {request_id[:8]} for API key {api_key[:8]}"
            )
            
            if self._cleanup_queued_script is not None:
                try:
                    await self._cleanup_queued_script(
                        keys=[queue_key],
                        args=[request_id, self.max_queue_wait_time],
                    )
                    verbose_proxy_logger.debug(
                        f"RequestQueueLimiter: Cleanup completed for request {request_id[:8]}"
                    )
                except Exception as e:
                    verbose_proxy_logger.warning(
                        f"RequestQueueLimiter: Failed to cleanup queued request: {str(e)}"
                    )
                    
        except Exception as e:
            verbose_proxy_logger.exception(
                f"RequestQueueLimiter: Error in cleanup: {str(e)}"
            )

    def do_rate_limit_check(self, user_api_key_dict: UserAPIKeyAuth) -> bool:
        """
        Synchronous rate limit check to determine if the request should be processed.
        
        Args:
            user_api_key_dict: The user API key authentication dictionary
        
        Returns:
            True if the request should be processed, False if it should be rejected immediately
        """
        max_parallel_requests = user_api_key_dict.max_parallel_requests
        if max_parallel_requests is None:
            return False  # No limit for this user, allow request to proceed
        
        return True

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Pre-call hook to try to acquire a slot in the queue.
        
        This hook is called before the LLM API call is made. It attempts to:
        1. Acquire a running slot if available (running < MAX_CONCURRENT_REQUESTS)
        2. Queue the request if running slots are full but total < MAX_TOTAL_REQUESTS
        3. Wait for a slot to become available (internal waiting)
        4. Raise HTTPException 429 if the queue is full or wait times out
        
        Args:
            user_api_key_dict: User API key authentication dictionary
            cache: Dual cache instance
            data: Request data dictionary
            call_type: Type of call being made
            
        Returns:
            - None: Allow request to proceed
            - Raises HTTPException 429: If queue is full or timeout
            
        Raises:
            HTTPException: 429 if the request queue is full or times out
        """
        if not self.do_rate_limit_check(user_api_key_dict):
            return None
        
        max_parallel_requests = user_api_key_dict.max_parallel_requests
        try:
            # Get the API key from the user_api_key_dict
            api_key = getattr(user_api_key_dict, "api_key", None)
            if not api_key:
                verbose_proxy_logger.debug(
                    "RequestQueueLimiter: No API key found, skipping queue check"
                )
                return None
            
            request_id = self._get_request_id(data)
            if not request_id:
                verbose_proxy_logger.debug(
                    "RequestQueueLimiter: No request ID found, skipping queue check"
                )
                return None
            
            # Get Redis keys for this API key
            running_key, queue_key = self._get_queue_keys(api_key)
            
            verbose_proxy_logger.debug(
                f"RequestQueueLimiter: Trying to acquire slot for API key {api_key[:8]}..."
            )
            
            # Try to acquire a slot using Lua script
            if self._try_acquire_script is None:
                verbose_proxy_logger.debug(
                    "RequestQueueLimiter: Lua scripts not registered, allowing request"
                )
                return None
            

            try:
                result = await self._try_acquire_script(
                    keys=[running_key, queue_key],
                    args=[
                        self.max_concurrent_requests,
                        max_parallel_requests,
                        request_id,
                        self.max_queue_wait_time,
                    ],
                )
                
                verbose_proxy_logger.debug(
                    f"RequestQueueLimiter: Acquire result for {api_key[:8]}: {result}"
                )
                
                if result == -2:
                    # Queue is full, reject the request
                    verbose_proxy_logger.warning(
                        f"RequestQueueLimiter: Queue full for API key {api_key[:8]}, rejecting request"
                    )
                    raise HTTPException(
                        status_code=429,
                        detail=(
                            f"Request queue is full. Maximum {max_parallel_requests} "
                            f"requests allowed (running + queued). Please try again later."
                        ),
                        headers={
                            "retry-after": "30",
                            "x-rate-limit-queue-status": "full",
                            "x-rate-limit-max-concurrent": str(self.max_concurrent_requests),
                            "x-rate-limit-max-total": str(max_parallel_requests),
                        },
                    )
                
                elif result == -1:
                    # Request can run immediately - request ID was added to running list by Lua script
                    verbose_proxy_logger.debug(
                        f"RequestQueueLimiter: Request {request_id[:8]} allowed to run immediately for API key {api_key[:8]}"
                    )
                    return None
                else:
                    # Request is queued, result is the queue position
                    queue_position = int(result)
                    verbose_proxy_logger.info(
                        f"RequestQueueLimiter: Request {request_id[:8]} queued at position {queue_position} for API key {api_key[:8]}. Waiting for slot..."
                    )
                    
                    # Wait for a slot to become available (internal waiting)
                    await self._wait_for_slot(api_key, request_id)
                    
                    verbose_proxy_logger.debug(
                        f"RequestQueueLimiter: Request {request_id[:8]} acquired slot after waiting"
                    )
                    # Request ID was added to running list when queued request was promoted via CHECK_AND_PROMOTE_LUA
                    return None
                    
            except HTTPException:
                # Re-raise HTTP exceptions
                raise
            except Exception as e:
                verbose_proxy_logger.warning(
                    f"RequestQueueLimiter: Lua script execution failed: {str(e)}, falling back to allowing request"
                )
                # If Lua script fails, allow the request to proceed
                return None
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            verbose_proxy_logger.exception(
                f"RequestQueueLimiter: Error in pre_call_hook: {str(e)}"
            )
            # On error, allow the request to proceed
            return None

    async def async_log_success_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """
        Post-call success hook to release a slot when a request completes successfully.
        
        This hook removes the request ID from the running list. Queue promotion is handled by
        the CHECK_AND_PROMOTE_LUA script during polling in _wait_for_slot().
        
        Args:
            kwargs: Request kwargs from the logging system
            response_obj: The response object from the LLM API
            start_time: Request start time
            end_time: Request end time
        """
        verbose_proxy_logger.debug(
            "RequestQueueLimiter: In async_log_success_event"
        )

        try:
            standard_logging_object = kwargs.get("standard_logging_object") or {}
            standard_logging_metadata = standard_logging_object.get("metadata") or {}
            api_key = standard_logging_metadata.get("user_api_key_hash")
            
            if not api_key:
                verbose_proxy_logger.debug(
                    "RequestQueueLimiter: No API key found in success event, skipping slot release"
                )
                return
            
            request_id = self._get_request_id(standard_logging_object)
            if request_id is None:
                verbose_proxy_logger.debug(
                    "RequestQueueLimiter: No request ID found, skipping slot release"
                )
                return
            
            # Get Redis keys for this API key
            running_key, queue_key = self._get_queue_keys(api_key)
            
            verbose_proxy_logger.debug(
                f"RequestQueueLimiter: Releasing slot for API key {api_key[:8]} after successful request"
            )
            
            # Release the slot using Lua script
            if self._release_slot_script is not None:
                try:
                    result = await self._release_slot_script(
                        keys=[running_key, queue_key],
                        args=[self.max_queue_wait_time, request_id],
                    )
                    verbose_proxy_logger.debug(
                        f"RequestQueueLimiter: Slot released for API key {api_key[:8]}, next queued: {result}"
                    )
                except Exception as e:
                    verbose_proxy_logger.warning(
                        f"RequestQueueLimiter: Failed to release slot: {str(e)}"
                    )
            else:
                verbose_proxy_logger.debug(
                    "RequestQueueLimiter: Release script not registered"
                )
                
        except Exception as e:
            verbose_proxy_logger.exception(
                f"RequestQueueLimiter: Error in async_log_success_event: {str(e)}"
            )

    async def async_log_failure_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """
        Log failure event to release a slot when a request fails.
        
        This hook removes the request ID from the running list. Queue promotion is handled by
        the CHECK_AND_PROMOTE_LUA script during polling in _wait_for_slot().
        
        Args:
            kwargs: Request kwargs from the logging system
            response_obj: The response object (may be None for failures)
            start_time: Request start time
            end_time: Request end time
        """

        verbose_proxy_logger.debug(
            "RequestQueueLimiter: In async_log_failure_event"
        )

        try:
            standard_logging_object = kwargs.get("standard_logging_object") or {}
            standard_logging_metadata = standard_logging_object.get("metadata") or {}
            api_key = standard_logging_metadata.get("user_api_key_hash")
            
            if not api_key:
                verbose_proxy_logger.debug(
                    "RequestQueueLimiter: No API key found in failure event, skipping slot release"
                )
                return
            
            request_id = self._get_request_id(standard_logging_object)
            if request_id is None:
                verbose_proxy_logger.debug(
                    "RequestQueueLimiter: No request ID found, skipping slot release"
                )
                return
            
            # Get Redis keys for this API key
            running_key, queue_key = self._get_queue_keys(api_key)
            
            verbose_proxy_logger.debug(
                f"RequestQueueLimiter: Releasing slot for API key {api_key[:8]} after failed request"
            )
            
            # Release the slot using Lua script
            if self._release_slot_script is not None:
                try:
                    result = await self._release_slot_script(
                        keys=[running_key, queue_key],
                        args=[self.max_queue_wait_time, request_id],
                    )
                    verbose_proxy_logger.debug(
                        f"RequestQueueLimiter: Slot released for API key {api_key[:8]} after failure, next queued: {result}"
                    )
                except Exception as e:
                    verbose_proxy_logger.warning(
                        f"RequestQueueLimiter: Failed to release slot after failure: {str(e)}"
                    )
            else:
                verbose_proxy_logger.debug(
                    "RequestQueueLimiter: Release script not registered"
                )
                
        except Exception as e:
            verbose_proxy_logger.exception(
                f"RequestQueueLimiter: Error in async_log_failure_event: {str(e)}"
            )
