"""
Redis Cache Client

Redis client for caching and session management.

Copyright 2024 XQELM Research Team
Licensed under the Apache License, Version 2.0
"""

import json
import pickle
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager

import redis.asyncio as redis
from redis.asyncio import ConnectionPool
from loguru import logger

from ..utils.config import RedisConfig


class RedisClient:
    """
    Asynchronous Redis client for caching and session management.
    """
    
    def __init__(self, config: RedisConfig):
        """
        Initialize Redis client.
        
        Args:
            config: Redis configuration
        """
        self.config = config
        self.pool = None
        self.client = None
        
        # Cache key prefixes
        self.prefixes = {
            "query_cache": "xqelm:query:",
            "model_cache": "xqelm:model:",
            "session": "xqelm:session:",
            "rate_limit": "xqelm:ratelimit:",
            "user_data": "xqelm:user:",
            "legal_data": "xqelm:legal:",
            "quantum_state": "xqelm:quantum:",
            "embeddings": "xqelm:embeddings:",
            "analytics": "xqelm:analytics:"
        }
        
        logger.info(f"Initializing Redis client for {config.host}:{config.port}")
    
    async def connect(self) -> None:
        """Establish connection to Redis."""
        try:
            # Create connection pool
            self.pool = ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db,
                max_connections=self.config.max_connections,
                retry_on_timeout=True,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=False  # We'll handle encoding ourselves
            )
            
            # Create Redis client
            self.client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            await self.client.ping()
            
            logger.info("Successfully connected to Redis")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close connection to Redis."""
        if self.client:
            await self.client.close()
        if self.pool:
            await self.pool.disconnect()
        logger.info("Disconnected from Redis")
    
    def _make_key(self, prefix: str, key: str) -> str:
        """
        Create a prefixed cache key.
        
        Args:
            prefix: Key prefix
            key: Base key
            
        Returns:
            Prefixed key
        """
        return f"{self.prefixes.get(prefix, prefix)}:{key}"
    
    def _serialize_value(self, value: Any) -> bytes:
        """
        Serialize value for storage.
        
        Args:
            value: Value to serialize
            
        Returns:
            Serialized bytes
        """
        if isinstance(value, (str, int, float, bool)):
            return json.dumps(value).encode('utf-8')
        else:
            # Use pickle for complex objects
            return pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes) -> Any:
        """
        Deserialize value from storage.
        
        Args:
            data: Serialized data
            
        Returns:
            Deserialized value
        """
        try:
            # Try JSON first
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle
            return pickle.loads(data)
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        prefix: str = "query_cache"
    ) -> bool:
        """
        Set a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            prefix: Key prefix
            
        Returns:
            True if successful
        """
        try:
            cache_key = self._make_key(prefix, key)
            serialized_value = self._serialize_value(value)
            
            if ttl:
                result = await self.client.setex(cache_key, ttl, serialized_value)
            else:
                result = await self.client.set(cache_key, serialized_value)
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    async def get(
        self,
        key: str,
        prefix: str = "query_cache"
    ) -> Optional[Any]:
        """
        Get a value from cache.
        
        Args:
            key: Cache key
            prefix: Key prefix
            
        Returns:
            Cached value or None
        """
        try:
            cache_key = self._make_key(prefix, key)
            data = await self.client.get(cache_key)
            
            if data is None:
                return None
            
            return self._deserialize_value(data)
            
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None
    
    async def delete(
        self,
        key: str,
        prefix: str = "query_cache"
    ) -> bool:
        """
        Delete a value from cache.
        
        Args:
            key: Cache key
            prefix: Key prefix
            
        Returns:
            True if deleted
        """
        try:
            cache_key = self._make_key(prefix, key)
            result = await self.client.delete(cache_key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    async def exists(
        self,
        key: str,
        prefix: str = "query_cache"
    ) -> bool:
        """
        Check if a key exists in cache.
        
        Args:
            key: Cache key
            prefix: Key prefix
            
        Returns:
            True if exists
        """
        try:
            cache_key = self._make_key(prefix, key)
            result = await self.client.exists(cache_key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False
    
    async def expire(
        self,
        key: str,
        ttl: int,
        prefix: str = "query_cache"
    ) -> bool:
        """
        Set expiration time for a key.
        
        Args:
            key: Cache key
            ttl: Time to live in seconds
            prefix: Key prefix
            
        Returns:
            True if successful
        """
        try:
            cache_key = self._make_key(prefix, key)
            result = await self.client.expire(cache_key, ttl)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error setting expiration for key {key}: {e}")
            return False
    
    async def ttl(
        self,
        key: str,
        prefix: str = "query_cache"
    ) -> int:
        """
        Get time to live for a key.
        
        Args:
            key: Cache key
            prefix: Key prefix
            
        Returns:
            TTL in seconds (-1 if no expiration, -2 if key doesn't exist)
        """
        try:
            cache_key = self._make_key(prefix, key)
            return await self.client.ttl(cache_key)
            
        except Exception as e:
            logger.error(f"Error getting TTL for key {key}: {e}")
            return -2
    
    async def increment(
        self,
        key: str,
        amount: int = 1,
        prefix: str = "analytics"
    ) -> int:
        """
        Increment a counter.
        
        Args:
            key: Counter key
            amount: Increment amount
            prefix: Key prefix
            
        Returns:
            New counter value
        """
        try:
            cache_key = self._make_key(prefix, key)
            return await self.client.incrby(cache_key, amount)
            
        except Exception as e:
            logger.error(f"Error incrementing counter {key}: {e}")
            return 0
    
    async def decrement(
        self,
        key: str,
        amount: int = 1,
        prefix: str = "analytics"
    ) -> int:
        """
        Decrement a counter.
        
        Args:
            key: Counter key
            amount: Decrement amount
            prefix: Key prefix
            
        Returns:
            New counter value
        """
        try:
            cache_key = self._make_key(prefix, key)
            return await self.client.decrby(cache_key, amount)
            
        except Exception as e:
            logger.error(f"Error decrementing counter {key}: {e}")
            return 0
    
    async def set_hash(
        self,
        key: str,
        field: str,
        value: Any,
        prefix: str = "user_data"
    ) -> bool:
        """
        Set a field in a hash.
        
        Args:
            key: Hash key
            field: Field name
            value: Field value
            prefix: Key prefix
            
        Returns:
            True if successful
        """
        try:
            cache_key = self._make_key(prefix, key)
            serialized_value = self._serialize_value(value)
            result = await self.client.hset(cache_key, field, serialized_value)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error setting hash field {key}.{field}: {e}")
            return False
    
    async def get_hash(
        self,
        key: str,
        field: str,
        prefix: str = "user_data"
    ) -> Optional[Any]:
        """
        Get a field from a hash.
        
        Args:
            key: Hash key
            field: Field name
            prefix: Key prefix
            
        Returns:
            Field value or None
        """
        try:
            cache_key = self._make_key(prefix, key)
            data = await self.client.hget(cache_key, field)
            
            if data is None:
                return None
            
            return self._deserialize_value(data)
            
        except Exception as e:
            logger.error(f"Error getting hash field {key}.{field}: {e}")
            return None
    
    async def get_all_hash(
        self,
        key: str,
        prefix: str = "user_data"
    ) -> Dict[str, Any]:
        """
        Get all fields from a hash.
        
        Args:
            key: Hash key
            prefix: Key prefix
            
        Returns:
            Dictionary of field-value pairs
        """
        try:
            cache_key = self._make_key(prefix, key)
            data = await self.client.hgetall(cache_key)
            
            result = {}
            for field, value in data.items():
                field_str = field.decode('utf-8') if isinstance(field, bytes) else field
                result[field_str] = self._deserialize_value(value)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting all hash fields for {key}: {e}")
            return {}
    
    async def delete_hash_field(
        self,
        key: str,
        field: str,
        prefix: str = "user_data"
    ) -> bool:
        """
        Delete a field from a hash.
        
        Args:
            key: Hash key
            field: Field name
            prefix: Key prefix
            
        Returns:
            True if deleted
        """
        try:
            cache_key = self._make_key(prefix, key)
            result = await self.client.hdel(cache_key, field)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error deleting hash field {key}.{field}: {e}")
            return False
    
    async def add_to_set(
        self,
        key: str,
        value: Any,
        prefix: str = "legal_data"
    ) -> bool:
        """
        Add a value to a set.
        
        Args:
            key: Set key
            value: Value to add
            prefix: Key prefix
            
        Returns:
            True if added (False if already exists)
        """
        try:
            cache_key = self._make_key(prefix, key)
            serialized_value = self._serialize_value(value)
            result = await self.client.sadd(cache_key, serialized_value)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error adding to set {key}: {e}")
            return False
    
    async def remove_from_set(
        self,
        key: str,
        value: Any,
        prefix: str = "legal_data"
    ) -> bool:
        """
        Remove a value from a set.
        
        Args:
            key: Set key
            value: Value to remove
            prefix: Key prefix
            
        Returns:
            True if removed
        """
        try:
            cache_key = self._make_key(prefix, key)
            serialized_value = self._serialize_value(value)
            result = await self.client.srem(cache_key, serialized_value)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error removing from set {key}: {e}")
            return False
    
    async def get_set_members(
        self,
        key: str,
        prefix: str = "legal_data"
    ) -> List[Any]:
        """
        Get all members of a set.
        
        Args:
            key: Set key
            prefix: Key prefix
            
        Returns:
            List of set members
        """
        try:
            cache_key = self._make_key(prefix, key)
            data = await self.client.smembers(cache_key)
            
            return [self._deserialize_value(item) for item in data]
            
        except Exception as e:
            logger.error(f"Error getting set members for {key}: {e}")
            return []
    
    async def is_set_member(
        self,
        key: str,
        value: Any,
        prefix: str = "legal_data"
    ) -> bool:
        """
        Check if a value is a member of a set.
        
        Args:
            key: Set key
            value: Value to check
            prefix: Key prefix
            
        Returns:
            True if member
        """
        try:
            cache_key = self._make_key(prefix, key)
            serialized_value = self._serialize_value(value)
            result = await self.client.sismember(cache_key, serialized_value)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error checking set membership for {key}: {e}")
            return False
    
    async def push_to_list(
        self,
        key: str,
        value: Any,
        left: bool = True,
        prefix: str = "legal_data"
    ) -> int:
        """
        Push a value to a list.
        
        Args:
            key: List key
            value: Value to push
            left: Push to left (True) or right (False)
            prefix: Key prefix
            
        Returns:
            New list length
        """
        try:
            cache_key = self._make_key(prefix, key)
            serialized_value = self._serialize_value(value)
            
            if left:
                result = await self.client.lpush(cache_key, serialized_value)
            else:
                result = await self.client.rpush(cache_key, serialized_value)
            
            return result
            
        except Exception as e:
            logger.error(f"Error pushing to list {key}: {e}")
            return 0
    
    async def pop_from_list(
        self,
        key: str,
        left: bool = True,
        prefix: str = "legal_data"
    ) -> Optional[Any]:
        """
        Pop a value from a list.
        
        Args:
            key: List key
            left: Pop from left (True) or right (False)
            prefix: Key prefix
            
        Returns:
            Popped value or None
        """
        try:
            cache_key = self._make_key(prefix, key)
            
            if left:
                data = await self.client.lpop(cache_key)
            else:
                data = await self.client.rpop(cache_key)
            
            if data is None:
                return None
            
            return self._deserialize_value(data)
            
        except Exception as e:
            logger.error(f"Error popping from list {key}: {e}")
            return None
    
    async def get_list_range(
        self,
        key: str,
        start: int = 0,
        end: int = -1,
        prefix: str = "legal_data"
    ) -> List[Any]:
        """
        Get a range of values from a list.
        
        Args:
            key: List key
            start: Start index
            end: End index (-1 for end of list)
            prefix: Key prefix
            
        Returns:
            List of values
        """
        try:
            cache_key = self._make_key(prefix, key)
            data = await self.client.lrange(cache_key, start, end)
            
            return [self._deserialize_value(item) for item in data]
            
        except Exception as e:
            logger.error(f"Error getting list range for {key}: {e}")
            return []
    
    async def get_list_length(
        self,
        key: str,
        prefix: str = "legal_data"
    ) -> int:
        """
        Get the length of a list.
        
        Args:
            key: List key
            prefix: Key prefix
            
        Returns:
            List length
        """
        try:
            cache_key = self._make_key(prefix, key)
            return await self.client.llen(cache_key)
            
        except Exception as e:
            logger.error(f"Error getting list length for {key}: {e}")
            return 0
    
    async def cache_query_result(
        self,
        query_hash: str,
        result: Any,
        ttl: int = 3600
    ) -> bool:
        """
        Cache a query result.
        
        Args:
            query_hash: Hash of the query
            result: Query result
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully
        """
        return await self.set(query_hash, result, ttl, "query_cache")
    
    async def get_cached_query_result(
        self,
        query_hash: str
    ) -> Optional[Any]:
        """
        Get a cached query result.
        
        Args:
            query_hash: Hash of the query
            
        Returns:
            Cached result or None
        """
        return await self.get(query_hash, "query_cache")
    
    async def cache_model_state(
        self,
        model_id: str,
        state: Any,
        ttl: int = 86400
    ) -> bool:
        """
        Cache a model state.
        
        Args:
            model_id: Model identifier
            state: Model state
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully
        """
        return await self.set(model_id, state, ttl, "model_cache")
    
    async def get_cached_model_state(
        self,
        model_id: str
    ) -> Optional[Any]:
        """
        Get a cached model state.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Cached state or None
        """
        return await self.get(model_id, "model_cache")
    
    async def store_session(
        self,
        session_id: str,
        session_data: Dict[str, Any],
        ttl: int = 3600
    ) -> bool:
        """
        Store session data.
        
        Args:
            session_id: Session identifier
            session_data: Session data
            ttl: Time to live in seconds
            
        Returns:
            True if stored successfully
        """
        return await self.set(session_id, session_data, ttl, "session")
    
    async def get_session(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get session data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None
        """
        return await self.get(session_id, "session")
    
    async def delete_session(
        self,
        session_id: str
    ) -> bool:
        """
        Delete session data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted
        """
        return await self.delete(session_id, "session")
    
    async def track_rate_limit(
        self,
        client_id: str,
        window_seconds: int = 60,
        max_requests: int = 100
    ) -> Tuple[bool, int, int]:
        """
        Track rate limiting for a client.
        
        Args:
            client_id: Client identifier
            window_seconds: Time window in seconds
            max_requests: Maximum requests in window
            
        Returns:
            Tuple of (allowed, current_count, remaining_count)
        """
        try:
            current_time = int(datetime.now().timestamp())
            window_start = current_time - window_seconds
            
            # Use sorted set to track requests with timestamps
            cache_key = self._make_key("rate_limit", client_id)
            
            # Remove old entries
            await self.client.zremrangebyscore(cache_key, 0, window_start)
            
            # Count current requests
            current_count = await self.client.zcard(cache_key)
            
            if current_count < max_requests:
                # Add current request
                await self.client.zadd(cache_key, {str(current_time): current_time})
                await self.client.expire(cache_key, window_seconds)
                
                remaining = max_requests - current_count - 1
                return True, current_count + 1, remaining
            else:
                remaining = 0
                return False, current_count, remaining
                
        except Exception as e:
            logger.error(f"Error tracking rate limit for {client_id}: {e}")
            return True, 0, max_requests  # Allow on error
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics
        """
        try:
            info = await self.client.info()
            
            # Get key counts by prefix
            key_counts = {}
            for prefix_name, prefix_value in self.prefixes.items():
                pattern = f"{prefix_value}*"
                keys = await self.client.keys(pattern)
                key_counts[prefix_name] = len(keys)
            
            return {
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
                "key_counts": key_counts,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting cache statistics: {e}")
            return {}
    
    async def flush_cache(
        self,
        prefix: Optional[str] = None
    ) -> bool:
        """
        Flush cache data.
        
        Args:
            prefix: Optional prefix to flush (flushes all if None)
            
        Returns:
            True if successful
        """
        try:
            if prefix:
                # Flush specific prefix
                pattern = f"{self.prefixes.get(prefix, prefix)}*"
                keys = await self.client.keys(pattern)
                if keys:
                    await self.client.delete(*keys)
            else:
                # Flush all
                await self.client.flushdb()
            
            return True
            
        except Exception as e:
            logger.error(f"Error flushing cache: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Redis connection.
        
        Returns:
            Health check results
        """
        try:
            # Test basic operations
            test_key = "health_check_test"
            test_value = "test_value"
            
            # Set and get test
            await self.client.set(test_key, test_value, ex=10)
            retrieved_value = await self.client.get(test_key)
            await self.client.delete(test_key)
            
            if retrieved_value and retrieved_value.decode('utf-8') == test_value:
                return {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "database": "redis"
                }
            else:
                return {
                    "status": "unhealthy",
                    "timestamp": datetime.now().isoformat(),
                    "database": "redis",
                    "error": "Set/get test failed"
                }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "database": "redis",
                "error": str(e)
            }