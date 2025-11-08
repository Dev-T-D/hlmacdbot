"""
High-Performance Cache Manager for Trading Bot

Supports Redis with automatic fallback to in-memory LRU cache.
Optimized for concurrent access and real-time trading data.

Features:
- Redis backend with TTL support
- In-memory LRU cache fallback
- Cache warming for startup performance
- Automatic cache invalidation strategies
- Concurrent-safe operations
- Performance metrics and hit rate tracking

"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from collections import OrderedDict

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: Any
    timestamp: float
    ttl: int
    access_count: int = 0
    last_access: float = 0

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check if cache entry is expired."""
        if current_time is None:
            current_time = time.time()
        return current_time - self.timestamp > self.ttl

    def access(self, current_time: Optional[float] = None) -> None:
        """Record access to cache entry."""
        if current_time is None:
            current_time = time.time()
        self.access_count += 1
        self.last_access = current_time


class LRUCache:
    """Thread-safe LRU cache implementation."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                if not entry.is_expired():
                    entry.access()
                    self.cache.move_to_end(key)  # Mark as recently used
                    self._hits += 1
                    return entry.data
                else:
                    # Remove expired entry
                    del self.cache[key]

            self._misses += 1
            return None

    async def set(self, key: str, value: Any, ttl: int) -> None:
        """Set value in cache."""
        async with self._lock:
            current_time = time.time()

            if key in self.cache:
                # Update existing entry
                self.cache[key].data = value
                self.cache[key].timestamp = current_time
                self.cache[key].ttl = ttl
                self.cache[key].access(current_time)
                self.cache.move_to_end(key)
            else:
                # Add new entry
                if len(self.cache) >= self.max_size:
                    # Remove least recently used item
                    oldest_key, _ = self.cache.popitem(last=False)
                    logger.debug(f"Cache eviction: removed {oldest_key}")

                entry = CacheEntry(
                    data=value,
                    timestamp=current_time,
                    ttl=ttl,
                    access_count=1,
                    last_access=current_time
                )
                self.cache[key] = entry

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self.cache.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests) * 100 if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
        }


class CacheManager:
    """High-performance cache manager with Redis and LRU fallback."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        max_memory_cache_size: int = 1000,
        default_ttl: int = 300,  # 5 minutes
        enable_metrics: bool = True
    ):
        """
        Initialize cache manager.

        Args:
            redis_url: Redis connection URL (e.g., "redis://localhost:6379/0")
            max_memory_cache_size: Maximum entries in memory cache
            default_ttl: Default TTL in seconds
            enable_metrics: Enable performance metrics
        """
        self.redis_url = redis_url or "redis://localhost:6379/0"
        self.default_ttl = default_ttl
        self.enable_metrics = enable_metrics

        # Redis client
        self.redis_client: Optional[redis.Redis] = None
        self.redis_available = False

        # Memory fallback cache
        self.memory_cache = LRUCache(max_memory_cache_size)

        # Performance metrics
        self._operation_times: Dict[str, list] = {}
        self._lock = asyncio.Lock()

        logger.info(f"Cache manager initialized with Redis URL: {self.redis_url}")

    async def connect(self) -> None:
        """Connect to Redis if available."""
        if REDIS_AVAILABLE and self.redis_url:
            try:
                self.redis_client = redis.Redis.from_url(self.redis_url)
                # Test connection
                await self.redis_client.ping()
                self.redis_available = True
                logger.info("✅ Redis cache connected successfully")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using memory cache fallback.")
                self.redis_available = False
        else:
            logger.info("Redis not available. Using memory cache only.")

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_available = False

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        start_time = time.time() if self.enable_metrics else 0

        try:
            # Try Redis first
            if self.redis_available and self.redis_client:
                try:
                    data = await self.redis_client.get(key)
                    if data:
                        # Deserialize JSON data
                        value = json.loads(data.decode('utf-8'))
                        if self.enable_metrics:
                            self._record_operation_time("redis_get", time.time() - start_time)
                        logger.debug(f"Cache hit (Redis): {key}")
                        return value
                except Exception as e:
                    logger.warning(f"Redis get error for key {key}: {e}")

            # Fallback to memory cache
            value = await self.memory_cache.get(key)
            if value is not None:
                if self.enable_metrics:
                    self._record_operation_time("memory_get", time.time() - start_time)
                logger.debug(f"Cache hit (Memory): {key}")
                return value

            # Cache miss
            if self.enable_metrics:
                self._record_operation_time("cache_miss", time.time() - start_time)
            logger.debug(f"Cache miss: {key}")
            return None

        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        if ttl is None:
            ttl = self.default_ttl

        start_time = time.time() if self.enable_metrics else 0

        try:
            # Serialize data to JSON
            serialized_data = json.dumps(value, default=str)

            # Set in Redis
            if self.redis_available and self.redis_client:
                try:
                    await self.redis_client.setex(key, ttl, serialized_data)
                    if self.enable_metrics:
                        self._record_operation_time("redis_set", time.time() - start_time)
                except Exception as e:
                    logger.warning(f"Redis set error for key {key}: {e}")

            # Always set in memory cache as well
            await self.memory_cache.set(key, value, ttl)

            if self.enable_metrics:
                self._record_operation_time("memory_set", time.time() - start_time)

            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")

        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        start_time = time.time() if self.enable_metrics else 0
        deleted = False

        try:
            # Delete from Redis
            if self.redis_available and self.redis_client:
                try:
                    redis_deleted = await self.redis_client.delete(key)
                    if redis_deleted:
                        deleted = True
                except Exception as e:
                    logger.warning(f"Redis delete error for key {key}: {e}")

            # Delete from memory cache
            memory_deleted = await self.memory_cache.delete(key)
            if memory_deleted:
                deleted = True

            if deleted and self.enable_metrics:
                self._record_operation_time("cache_delete", time.time() - start_time)

            return deleted

        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        try:
            # Clear Redis
            if self.redis_available and self.redis_client:
                try:
                    await self.redis_client.flushdb()
                except Exception as e:
                    logger.warning(f"Redis clear error: {e}")

            # Clear memory cache
            await self.memory_cache.clear()

            logger.info("Cache cleared")

        except Exception as e:
            logger.error(f"Cache clear error: {e}")

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        # Try Redis first
        if self.redis_available and self.redis_client:
            try:
                return await self.redis_client.exists(key) > 0
            except Exception:
                pass

        # Fallback to memory cache check
        # Note: This is an approximation since memory cache doesn't have exists method
        return await self.memory_cache.get(key) is not None

    async def warm_cache(self, warmup_data: Dict[str, Any]) -> None:
        """Warm cache with initial data."""
        logger.info(f"Warming cache with {len(warmup_data)} entries...")

        tasks = []
        for key, (value, ttl) in warmup_data.items():
            tasks.append(self.set(key, value, ttl))

        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Cache warming completed")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        memory_stats = self.memory_cache.get_stats()

        stats = {
            "redis_available": self.redis_available,
            "memory_cache": memory_stats,
            "metrics_enabled": self.enable_metrics,
        }

        if self.enable_metrics:
            stats["operation_times"] = {}
            for operation, times in self._operation_times.items():
                if times:
                    stats["operation_times"][operation] = {
                        "count": len(times),
                        "avg_time": sum(times) / len(times),
                        "min_time": min(times),
                        "max_time": max(times),
                    }

        return stats

    def _record_operation_time(self, operation: str, duration: float) -> None:
        """Record operation timing for metrics."""
        if operation not in self._operation_times:
            self._operation_times[operation] = []

        # Keep only last 1000 measurements to prevent memory bloat
        if len(self._operation_times[operation]) >= 1000:
            self._operation_times[operation] = self._operation_times[operation][-999:]

        self._operation_times[operation].append(duration)

    # Cache key generation helpers
    @staticmethod
    def make_market_data_key(symbol: str, timeframe: str, limit: int) -> str:
        """Generate cache key for market data."""
        return f"market_data:{symbol}:{timeframe}:{limit}"

    @staticmethod
    def make_account_data_key(user_address: str) -> str:
        """Generate cache key for account data."""
        return f"account_data:{user_address}"

    @staticmethod
    def make_position_key(user_address: str, symbol: str) -> str:
        """Generate cache key for position data."""
        return f"position:{user_address}:{symbol}"

    @staticmethod
    def make_asset_metadata_key() -> str:
        """Generate cache key for asset metadata."""
        return "asset_metadata"

    @staticmethod
    def make_clearinghouse_key(user_address: str) -> str:
        """Generate cache key for clearinghouse state."""
        return f"clearinghouse:{user_address}"

    @staticmethod
    def make_indicator_key(symbol: str, timeframe: str, indicator_type: str) -> str:
        """Generate cache key for indicator data."""
        return f"indicator:{symbol}:{timeframe}:{indicator_type}"


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


async def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
        await _cache_manager.connect()
    return _cache_manager


async def initialize_cache_manager(
    redis_url: Optional[str] = None,
    max_memory_cache_size: int = 1000,
    default_ttl: int = 300
) -> CacheManager:
    """Initialize global cache manager."""
    global _cache_manager
    _cache_manager = CacheManager(
        redis_url=redis_url,
        max_memory_cache_size=max_memory_cache_size,
        default_ttl=default_ttl
    )
    await _cache_manager.connect()
    return _cache_manager


async def shutdown_cache_manager() -> None:
    """Shutdown global cache manager."""
    global _cache_manager
    if _cache_manager:
        await _cache_manager.disconnect()
        _cache_manager = None
