"""
Rate Limiter for API Clients

Implements token bucket algorithm to prevent hitting API rate limits.
"""

import time
import threading
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for API calls
    
    Algorithm:
    - Tokens are added to bucket at constant rate
    - Each API call consumes one token
    - If bucket is empty, wait until token is available
    - Prevents exceeding rate limits
    """
    
    def __init__(self, 
                 rate: float = 10.0, 
                 capacity: Optional[int] = None,
                 burst: Optional[int] = None):
        """
        Initialize token bucket rate limiter
        
        Args:
            rate: Number of tokens per second (requests per second)
            capacity: Maximum tokens in bucket (None = same as rate)
            burst: Maximum burst size (None = same as capacity)
        """
        self.rate = float(rate)  # Tokens per second
        self.capacity = capacity if capacity is not None else int(rate)
        self.burst = burst if burst is not None else self.capacity
        
        # Current token count
        self.tokens = float(self.capacity)
        
        # Last update time
        self.last_update = time.time()
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.total_requests = 0
        self.total_wait_time = 0.0
        self.max_wait_time = 0.0
    
    def _add_tokens(self) -> None:
        """Add tokens to bucket based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_update
        
        # Add tokens based on rate
        tokens_to_add = elapsed * self.rate
        
        # Cap at capacity
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_update = now
    
    def acquire(self, tokens: int = 1, wait: bool = True) -> bool:
        """
        Acquire tokens for API call
        
        Args:
            tokens: Number of tokens to acquire (default: 1)
            wait: If True, wait until tokens available. If False, return False if not available
            
        Returns:
            True if tokens acquired, False if not available and wait=False
        """
        with self.lock:
            # Update tokens based on elapsed time
            self._add_tokens()
            
            # Check if enough tokens available
            if self.tokens >= tokens:
                # Consume tokens
                self.tokens -= tokens
                self.total_requests += tokens
                return True
            
            # Not enough tokens
            if not wait:
                return False
            
            # Calculate wait time
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.rate
            
            # Add tokens while waiting
            self.tokens = tokens
            self.last_update = time.time() + wait_time
            
            # Update statistics
            self.total_wait_time += wait_time
            self.max_wait_time = max(self.max_wait_time, wait_time)
            
            # Wait
            if wait_time > 0.001:  # Only log if significant wait
                logger.debug(f"Rate limit: waiting {wait_time:.2f}s for {tokens} tokens")
            time.sleep(wait_time)
            
            # Consume tokens
            self.tokens -= tokens
            self.total_requests += tokens
            
            return True
    
    def get_stats(self) -> dict:
        """
        Get rate limiter statistics
        
        Returns:
            Dictionary with statistics
        """
        with self.lock:
            return {
                "total_requests": self.total_requests,
                "total_wait_time": self.total_wait_time,
                "max_wait_time": self.max_wait_time,
                "current_tokens": self.tokens,
                "rate": self.rate,
                "capacity": self.capacity
            }
    
    def reset_stats(self) -> None:
        """Reset statistics counters"""
        with self.lock:
            self.total_requests = 0
            self.total_wait_time = 0.0
            self.max_wait_time = 0.0


class AsyncTokenBucketRateLimiter:
    """
    Async token bucket rate limiter for async API calls

    Same algorithm as TokenBucketRateLimiter but async-compatible.
    """

    def __init__(self,
                 rate: float = 10.0,
                 capacity: Optional[int] = None,
                 burst: Optional[int] = None):
        """
        Initialize async token bucket rate limiter

        Args:
            rate: Number of tokens per second (requests per second)
            capacity: Maximum tokens in bucket (None = same as rate)
            burst: Maximum burst size (None = same as capacity)
        """
        self.rate = float(rate)  # Tokens per second
        self.capacity = capacity if capacity is not None else int(rate)
        self.burst = burst if burst is not None else self.capacity

        # Current token count
        self.tokens = float(self.capacity)

        # Last update time
        self.last_update = time.time()

        # Lock for thread safety (asyncio.Lock for async compatibility)
        self.lock = asyncio.Lock()

        # Statistics
        self.total_requests = 0
        self.total_wait_time = 0.0
        self.max_wait_time = 0.0

    async def _add_tokens(self) -> None:
        """Add tokens to bucket based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_update

        # Add tokens based on rate
        tokens_to_add = elapsed * self.rate

        # Cap at capacity
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_update = now

    async def acquire(self, tokens: int = 1, wait: bool = True) -> bool:
        """
        Acquire tokens for API call (async)

        Args:
            tokens: Number of tokens to acquire (default: 1)
            wait: If True, wait until tokens available. If False, return False if not available

        Returns:
            True if tokens acquired, False if not available and wait=False
        """
        async with self.lock:
            # Update tokens based on elapsed time
            await self._add_tokens()

            # Check if enough tokens available
            if self.tokens >= tokens:
                # Consume tokens
                self.tokens -= tokens
                self.total_requests += tokens
                return True

            # Not enough tokens
            if not wait:
                return False

            # Calculate wait time
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.rate

            # Record wait time statistics
            self.total_wait_time += wait_time
            self.max_wait_time = max(self.max_wait_time, wait_time)

            # Wait for tokens to be available
            await asyncio.sleep(wait_time)

            # Add tokens while waiting
            self.tokens = tokens
            self.last_update = time.time() + wait_time
            self.total_requests += tokens

            return True

    def get_stats(self) -> dict:
        """Get rate limiter statistics"""
        return {
            "rate": self.rate,
            "capacity": self.capacity,
            "burst": self.burst,
            "current_tokens": self.tokens,
            "total_requests": self.total_requests,
            "total_wait_time": self.total_wait_time,
            "max_wait_time": self.max_wait_time,
            "avg_wait_time": self.total_wait_time / max(1, self.total_requests),
        }


# Exchange-specific rate limiters
# These are conservative defaults - adjust based on actual exchange limits

HYPERLIQUID_RATE_LIMITS = {
    "info": TokenBucketRateLimiter(rate=10.0, capacity=20, burst=30),  # 10 req/s, burst to 30
    "exchange": TokenBucketRateLimiter(rate=5.0, capacity=10, burst=15),  # 5 req/s, burst to 15
}

BITUNIX_RATE_LIMITS = {
    "default": TokenBucketRateLimiter(rate=10.0, capacity=20, burst=30),  # 10 req/s
}

