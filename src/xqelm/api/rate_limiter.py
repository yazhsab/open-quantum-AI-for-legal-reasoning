"""
Rate Limiter Module

Rate limiting functionality for the XQELM API to prevent abuse
and ensure fair usage of resources.

Copyright 2024 XQELM Research Team
Licensed under the Apache License, Version 2.0
"""

import time
import asyncio
from typing import Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib

from loguru import logger


@dataclass
class RateLimitInfo:
    """Rate limit information."""
    requests_made: int
    requests_remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None


class TokenBucket:
    """
    Token bucket algorithm implementation for rate limiting.
    """
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        async with self._lock:
            now = time.time()
            
            # Refill tokens based on elapsed time
            elapsed = now - self.last_refill
            tokens_to_add = elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def get_tokens(self) -> float:
        """Get current number of tokens."""
        return self.tokens
    
    def time_until_tokens(self, tokens: int) -> float:
        """
        Calculate time until specified tokens are available.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Time in seconds until tokens are available
        """
        if self.tokens >= tokens:
            return 0.0
        
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


class SlidingWindowCounter:
    """
    Sliding window counter for rate limiting.
    """
    
    def __init__(self, window_size: int, max_requests: int):
        """
        Initialize sliding window counter.
        
        Args:
            window_size: Window size in seconds
            max_requests: Maximum requests in window
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = deque()
        self._lock = asyncio.Lock()
    
    async def is_allowed(self) -> bool:
        """
        Check if request is allowed.
        
        Returns:
            True if request is allowed, False otherwise
        """
        async with self._lock:
            now = time.time()
            
            # Remove old requests outside the window
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            
            # Check if we're under the limit
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    def get_request_count(self) -> int:
        """Get current request count in window."""
        now = time.time()
        
        # Count requests in current window
        count = 0
        for request_time in self.requests:
            if request_time > now - self.window_size:
                count += 1
        
        return count
    
    def time_until_reset(self) -> float:
        """
        Calculate time until window resets.
        
        Returns:
            Time in seconds until oldest request expires
        """
        if not self.requests:
            return 0.0
        
        now = time.time()
        oldest_request = self.requests[0]
        return max(0.0, self.window_size - (now - oldest_request))


class RateLimiter:
    """
    Comprehensive rate limiter with multiple algorithms and strategies.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000,
        window_seconds: int = 60,
        burst_capacity: int = 10,
        algorithm: str = "sliding_window"
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Requests per minute limit
            requests_per_hour: Requests per hour limit
            requests_per_day: Requests per day limit
            window_seconds: Window size for sliding window algorithm
            burst_capacity: Burst capacity for token bucket
            algorithm: Rate limiting algorithm ('sliding_window', 'token_bucket', 'fixed_window')
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day
        self.window_seconds = window_seconds
        self.burst_capacity = burst_capacity
        self.algorithm = algorithm
        
        # Storage for different clients
        self.client_limiters: Dict[str, Any] = {}
        self.client_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_requests": 0,
            "blocked_requests": 0,
            "first_request": None,
            "last_request": None
        })
        
        # Global statistics
        self.global_stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "unique_clients": 0,
            "start_time": datetime.now()
        }
        
        logger.info(f"Rate limiter initialized with {algorithm} algorithm")
    
    def _get_client_key(self, identifier: str) -> str:
        """
        Generate client key for rate limiting.
        
        Args:
            identifier: Client identifier (IP, user ID, API key, etc.)
            
        Returns:
            Hashed client key
        """
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]
    
    def _get_or_create_limiter(self, client_key: str) -> Any:
        """
        Get or create rate limiter for client.
        
        Args:
            client_key: Client key
            
        Returns:
            Rate limiter instance
        """
        if client_key not in self.client_limiters:
            if self.algorithm == "token_bucket":
                # Token bucket with refill rate based on requests per minute
                refill_rate = self.requests_per_minute / 60.0  # tokens per second
                self.client_limiters[client_key] = TokenBucket(
                    capacity=self.burst_capacity,
                    refill_rate=refill_rate
                )
            elif self.algorithm == "sliding_window":
                self.client_limiters[client_key] = SlidingWindowCounter(
                    window_size=self.window_seconds,
                    max_requests=self.requests_per_minute
                )
            else:  # fixed_window
                self.client_limiters[client_key] = {
                    "requests": 0,
                    "window_start": time.time()
                }
            
            self.global_stats["unique_clients"] += 1
        
        return self.client_limiters[client_key]
    
    async def is_allowed(
        self,
        identifier: str,
        weight: int = 1,
        check_multiple_limits: bool = True
    ) -> bool:
        """
        Check if request is allowed for client.
        
        Args:
            identifier: Client identifier
            weight: Request weight (for weighted rate limiting)
            check_multiple_limits: Check multiple time window limits
            
        Returns:
            True if request is allowed, False otherwise
        """
        client_key = self._get_client_key(identifier)
        
        # Update statistics
        self.global_stats["total_requests"] += 1
        client_stats = self.client_stats[client_key]
        client_stats["total_requests"] += 1
        
        now = datetime.now()
        if client_stats["first_request"] is None:
            client_stats["first_request"] = now
        client_stats["last_request"] = now
        
        # Get rate limiter for client
        limiter = self._get_or_create_limiter(client_key)
        
        # Check primary rate limit
        allowed = await self._check_primary_limit(limiter, weight)
        
        # Check additional limits if enabled
        if allowed and check_multiple_limits:
            allowed = await self._check_additional_limits(client_key, weight)
        
        # Update blocked statistics
        if not allowed:
            self.global_stats["blocked_requests"] += 1
            client_stats["blocked_requests"] += 1
            
            logger.warning(f"Rate limit exceeded for client: {client_key}")
        
        return allowed
    
    async def _check_primary_limit(self, limiter: Any, weight: int) -> bool:
        """
        Check primary rate limit based on algorithm.
        
        Args:
            limiter: Rate limiter instance
            weight: Request weight
            
        Returns:
            True if allowed, False otherwise
        """
        if self.algorithm == "token_bucket":
            return await limiter.consume(weight)
        elif self.algorithm == "sliding_window":
            # For weighted requests, check multiple times
            for _ in range(weight):
                if not await limiter.is_allowed():
                    return False
            return True
        else:  # fixed_window
            now = time.time()
            
            # Reset window if needed
            if now - limiter["window_start"] >= self.window_seconds:
                limiter["requests"] = 0
                limiter["window_start"] = now
            
            # Check limit
            if limiter["requests"] + weight <= self.requests_per_minute:
                limiter["requests"] += weight
                return True
            
            return False
    
    async def _check_additional_limits(self, client_key: str, weight: int) -> bool:
        """
        Check additional time-based limits (hourly, daily).
        
        Args:
            client_key: Client key
            weight: Request weight
            
        Returns:
            True if all limits pass, False otherwise
        """
        now = time.time()
        
        # Initialize additional limiters if needed
        if f"{client_key}_hourly" not in self.client_limiters:
            self.client_limiters[f"{client_key}_hourly"] = SlidingWindowCounter(
                window_size=3600,  # 1 hour
                max_requests=self.requests_per_hour
            )
        
        if f"{client_key}_daily" not in self.client_limiters:
            self.client_limiters[f"{client_key}_daily"] = SlidingWindowCounter(
                window_size=86400,  # 1 day
                max_requests=self.requests_per_day
            )
        
        # Check hourly limit
        hourly_limiter = self.client_limiters[f"{client_key}_hourly"]
        for _ in range(weight):
            if not await hourly_limiter.is_allowed():
                return False
        
        # Check daily limit
        daily_limiter = self.client_limiters[f"{client_key}_daily"]
        for _ in range(weight):
            if not await daily_limiter.is_allowed():
                return False
        
        return True
    
    def get_rate_limit_info(self, identifier: str) -> RateLimitInfo:
        """
        Get rate limit information for client.
        
        Args:
            identifier: Client identifier
            
        Returns:
            Rate limit information
        """
        client_key = self._get_client_key(identifier)
        
        if client_key not in self.client_limiters:
            # No requests made yet
            return RateLimitInfo(
                requests_made=0,
                requests_remaining=self.requests_per_minute,
                reset_time=datetime.now() + timedelta(seconds=self.window_seconds)
            )
        
        limiter = self.client_limiters[client_key]
        
        if self.algorithm == "sliding_window":
            requests_made = limiter.get_request_count()
            requests_remaining = max(0, self.requests_per_minute - requests_made)
            reset_time = datetime.now() + timedelta(seconds=limiter.time_until_reset())
        elif self.algorithm == "token_bucket":
            tokens = limiter.get_tokens()
            requests_made = self.burst_capacity - int(tokens)
            requests_remaining = int(tokens)
            reset_time = datetime.now() + timedelta(
                seconds=limiter.time_until_tokens(self.burst_capacity)
            )
        else:  # fixed_window
            requests_made = limiter["requests"]
            requests_remaining = max(0, self.requests_per_minute - requests_made)
            window_elapsed = time.time() - limiter["window_start"]
            reset_time = datetime.now() + timedelta(
                seconds=self.window_seconds - window_elapsed
            )
        
        retry_after = None
        if requests_remaining == 0:
            retry_after = int((reset_time - datetime.now()).total_seconds())
        
        return RateLimitInfo(
            requests_made=requests_made,
            requests_remaining=requests_remaining,
            reset_time=reset_time,
            retry_after=retry_after
        )
    
    def get_client_statistics(self, identifier: str) -> Dict[str, Any]:
        """
        Get statistics for specific client.
        
        Args:
            identifier: Client identifier
            
        Returns:
            Client statistics
        """
        client_key = self._get_client_key(identifier)
        stats = self.client_stats[client_key].copy()
        
        # Add rate limit info
        rate_limit_info = self.get_rate_limit_info(identifier)
        stats["rate_limit"] = {
            "requests_made": rate_limit_info.requests_made,
            "requests_remaining": rate_limit_info.requests_remaining,
            "reset_time": rate_limit_info.reset_time.isoformat(),
            "retry_after": rate_limit_info.retry_after
        }
        
        return stats
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """
        Get global rate limiter statistics.
        
        Returns:
            Global statistics
        """
        stats = self.global_stats.copy()
        
        # Calculate rates
        uptime = (datetime.now() - stats["start_time"]).total_seconds()
        if uptime > 0:
            stats["requests_per_second"] = stats["total_requests"] / uptime
            stats["block_rate"] = stats["blocked_requests"] / stats["total_requests"] if stats["total_requests"] > 0 else 0.0
        else:
            stats["requests_per_second"] = 0.0
            stats["block_rate"] = 0.0
        
        stats["uptime_seconds"] = uptime
        stats["start_time"] = stats["start_time"].isoformat()
        
        return stats
    
    def reset_client_limits(self, identifier: str) -> bool:
        """
        Reset rate limits for specific client.
        
        Args:
            identifier: Client identifier
            
        Returns:
            True if reset successful, False if client not found
        """
        client_key = self._get_client_key(identifier)
        
        if client_key not in self.client_limiters:
            return False
        
        # Remove client limiters
        keys_to_remove = [
            client_key,
            f"{client_key}_hourly",
            f"{client_key}_daily"
        ]
        
        for key in keys_to_remove:
            if key in self.client_limiters:
                del self.client_limiters[key]
        
        # Reset statistics
        if client_key in self.client_stats:
            del self.client_stats[client_key]
        
        logger.info(f"Reset rate limits for client: {client_key}")
        return True
    
    def cleanup_expired_clients(self, max_age_hours: int = 24) -> int:
        """
        Clean up expired client data.
        
        Args:
            max_age_hours: Maximum age in hours for client data
            
        Returns:
            Number of clients cleaned up
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        clients_to_remove = []
        
        for client_key, stats in self.client_stats.items():
            if stats["last_request"] and stats["last_request"] < cutoff_time:
                clients_to_remove.append(client_key)
        
        # Remove expired clients
        for client_key in clients_to_remove:
            # Remove limiters
            keys_to_remove = [
                client_key,
                f"{client_key}_hourly",
                f"{client_key}_daily"
            ]
            
            for key in keys_to_remove:
                if key in self.client_limiters:
                    del self.client_limiters[key]
            
            # Remove statistics
            if client_key in self.client_stats:
                del self.client_stats[client_key]
        
        if clients_to_remove:
            logger.info(f"Cleaned up {len(clients_to_remove)} expired clients")
        
        return len(clients_to_remove)
    
    def update_limits(
        self,
        requests_per_minute: Optional[int] = None,
        requests_per_hour: Optional[int] = None,
        requests_per_day: Optional[int] = None
    ) -> None:
        """
        Update rate limits dynamically.
        
        Args:
            requests_per_minute: New requests per minute limit
            requests_per_hour: New requests per hour limit
            requests_per_day: New requests per day limit
        """
        if requests_per_minute is not None:
            self.requests_per_minute = requests_per_minute
        
        if requests_per_hour is not None:
            self.requests_per_hour = requests_per_hour
        
        if requests_per_day is not None:
            self.requests_per_day = requests_per_day
        
        # Clear existing limiters to apply new limits
        self.client_limiters.clear()
        
        logger.info(f"Updated rate limits: {self.requests_per_minute}/min, {self.requests_per_hour}/hour, {self.requests_per_day}/day")


class AdaptiveRateLimiter(RateLimiter):
    """
    Adaptive rate limiter that adjusts limits based on system load and client behavior.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize adaptive rate limiter."""
        super().__init__(*args, **kwargs)
        
        self.base_requests_per_minute = self.requests_per_minute
        self.load_factor = 1.0
        self.client_trust_scores: Dict[str, float] = defaultdict(lambda: 1.0)
        
        logger.info("Adaptive rate limiter initialized")
    
    def update_system_load(self, cpu_usage: float, memory_usage: float) -> None:
        """
        Update system load factor.
        
        Args:
            cpu_usage: CPU usage percentage (0.0 to 1.0)
            memory_usage: Memory usage percentage (0.0 to 1.0)
        """
        # Calculate load factor based on system metrics
        avg_usage = (cpu_usage + memory_usage) / 2.0
        
        if avg_usage > 0.9:
            self.load_factor = 0.5  # Reduce limits by 50%
        elif avg_usage > 0.7:
            self.load_factor = 0.75  # Reduce limits by 25%
        elif avg_usage < 0.3:
            self.load_factor = 1.5  # Increase limits by 50%
        else:
            self.load_factor = 1.0  # Normal limits
        
        # Update current limits
        adjusted_limit = int(self.base_requests_per_minute * self.load_factor)
        self.update_limits(requests_per_minute=adjusted_limit)
        
        logger.debug(f"Updated load factor: {self.load_factor}, new limit: {adjusted_limit}/min")
    
    def update_client_trust_score(self, identifier: str, score_delta: float) -> None:
        """
        Update client trust score.
        
        Args:
            identifier: Client identifier
            score_delta: Change in trust score (-1.0 to 1.0)
        """
        client_key = self._get_client_key(identifier)
        current_score = self.client_trust_scores[client_key]
        new_score = max(0.1, min(2.0, current_score + score_delta))
        self.client_trust_scores[client_key] = new_score
        
        logger.debug(f"Updated trust score for {client_key}: {new_score}")
    
    async def is_allowed(self, identifier: str, weight: int = 1, **kwargs) -> bool:
        """
        Check if request is allowed with adaptive limits.
        
        Args:
            identifier: Client identifier
            weight: Request weight
            **kwargs: Additional arguments
            
        Returns:
            True if request is allowed, False otherwise
        """
        client_key = self._get_client_key(identifier)
        trust_score = self.client_trust_scores[client_key]
        
        # Adjust weight based on trust score
        adjusted_weight = max(1, int(weight / trust_score))
        
        return await super().is_allowed(identifier, adjusted_weight, **kwargs)