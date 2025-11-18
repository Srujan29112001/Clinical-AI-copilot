"""
Rate Limiting Middleware for API Protection
Prevents abuse and ensures fair usage
"""

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RateLimitExceeded(HTTPException):
    """Rate limit exceeded exception"""

    def __init__(self, retry_after: int):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)}
        )


class RateLimiter:
    """
    Token bucket rate limiter

    Features:
    - Per-user rate limiting
    - Per-IP rate limiting
    - Configurable limits
    - Automatic cleanup of old entries
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000,
    ):
        """
        Initialize rate limiter

        Args:
            requests_per_minute: Max requests per minute
            requests_per_hour: Max requests per hour
            requests_per_day: Max requests per day
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day

        # Storage: {identifier: {window: [timestamps]}}
        self.requests: Dict[str, Dict[str, list]] = defaultdict(
            lambda: {"minute": [], "hour": [], "day": []}
        )

        # Last cleanup time
        self.last_cleanup = datetime.now()

    def is_allowed(self, identifier: str) -> tuple[bool, Optional[int]]:
        """
        Check if request is allowed

        Args:
            identifier: User ID or IP address

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        now = datetime.now()

        # Cleanup old entries periodically
        if (now - self.last_cleanup).seconds > 300:  # Every 5 minutes
            self._cleanup()

        # Get request history
        history = self.requests[identifier]

        # Check limits for each window
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)

        # Clean up old timestamps
        history["minute"] = [ts for ts in history["minute"] if ts > minute_ago]
        history["hour"] = [ts for ts in history["hour"] if ts > hour_ago]
        history["day"] = [ts for ts in history["day"] if ts > day_ago]

        # Check minute limit
        if len(history["minute"]) >= self.requests_per_minute:
            oldest = min(history["minute"])
            retry_after = int((oldest + timedelta(minutes=1) - now).total_seconds())
            return False, max(retry_after, 1)

        # Check hour limit
        if len(history["hour"]) >= self.requests_per_hour:
            oldest = min(history["hour"])
            retry_after = int((oldest + timedelta(hours=1) - now).total_seconds())
            return False, max(retry_after, 1)

        # Check day limit
        if len(history["day"]) >= self.requests_per_day:
            oldest = min(history["day"])
            retry_after = int((oldest + timedelta(days=1) - now).total_seconds())
            return False, max(retry_after, 1)

        # Request allowed - record it
        history["minute"].append(now)
        history["hour"].append(now)
        history["day"].append(now)

        return True, None

    def _cleanup(self):
        """Remove old entries"""
        now = datetime.now()
        day_ago = now - timedelta(days=1)

        # Remove identifiers with no recent requests
        identifiers_to_remove = []
        for identifier, history in self.requests.items():
            if not history["day"] or max(history["day"]) < day_ago:
                identifiers_to_remove.append(identifier)

        for identifier in identifiers_to_remove:
            del self.requests[identifier]

        self.last_cleanup = now
        logger.info(f"Cleaned up {len(identifiers_to_remove)} old rate limit entries")

    def get_stats(self, identifier: str) -> Dict:
        """Get usage statistics for an identifier"""
        now = datetime.now()
        history = self.requests.get(identifier, {"minute": [], "hour": [], "day": []})

        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)

        return {
            "requests_last_minute": len([ts for ts in history["minute"] if ts > minute_ago]),
            "requests_last_hour": len([ts for ts in history["hour"] if ts > hour_ago]),
            "requests_last_day": len([ts for ts in history["day"] if ts > day_ago]),
            "limit_minute": self.requests_per_minute,
            "limit_hour": self.requests_per_hour,
            "limit_day": self.requests_per_day,
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting"""

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000,
        by_user: bool = True,
        by_ip: bool = True,
    ):
        """
        Initialize rate limit middleware

        Args:
            app: FastAPI application
            requests_per_minute: Max requests per minute
            requests_per_hour: Max requests per hour
            requests_per_day: Max requests per day
            by_user: Enable per-user rate limiting
            by_ip: Enable per-IP rate limiting
        """
        super().__init__(app)
        self.user_limiter = RateLimiter(
            requests_per_minute, requests_per_hour, requests_per_day
        ) if by_user else None
        self.ip_limiter = RateLimiter(
            requests_per_minute * 2,  # More generous for IP (multiple users)
            requests_per_hour * 2,
            requests_per_day * 2,
        ) if by_ip else None

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting"""

        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/", "/api/docs", "/api/redoc"]:
            return await call_next(request)

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Check IP-based rate limit
        if self.ip_limiter:
            allowed, retry_after = self.ip_limiter.is_allowed(client_ip)
            if not allowed:
                logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": f"Rate limit exceeded. Retry after {retry_after} seconds.",
                        "retry_after": retry_after
                    },
                    headers={"Retry-After": str(retry_after)}
                )

        # Check user-based rate limit (if authenticated)
        if self.user_limiter:
            # Try to get user ID from token
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.replace("Bearer ", "")
                # Simple extraction (in production, decode JWT properly)
                user_id = token.replace("demo_", "")

                allowed, retry_after = self.user_limiter.is_allowed(user_id)
                if not allowed:
                    logger.warning(f"Rate limit exceeded for user: {user_id}")
                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={
                            "detail": f"Rate limit exceeded. Retry after {retry_after} seconds.",
                            "retry_after": retry_after
                        },
                        headers={"Retry-After": str(retry_after)}
                    )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        if self.ip_limiter:
            stats = self.ip_limiter.get_stats(client_ip)
            response.headers["X-RateLimit-Limit-Minute"] = str(self.ip_limiter.requests_per_minute)
            response.headers["X-RateLimit-Remaining-Minute"] = str(
                max(0, self.ip_limiter.requests_per_minute - stats["requests_last_minute"])
            )

        return response


# Decorator for endpoint-specific rate limiting
def rate_limit(requests_per_minute: int = 10):
    """
    Decorator for endpoint-specific rate limiting

    Usage:
        @app.post("/expensive-operation")
        @rate_limit(requests_per_minute=5)
        async def expensive_operation():
            ...
    """
    limiter = RateLimiter(requests_per_minute=requests_per_minute)

    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            # Get identifier
            client_ip = request.client.host if request.client else "unknown"

            allowed, retry_after = limiter.is_allowed(client_ip)
            if not allowed:
                raise RateLimitExceeded(retry_after)

            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test rate limiter
    limiter = RateLimiter(requests_per_minute=5, requests_per_hour=20, requests_per_day=100)

    print("Testing rate limiter...")

    # Simulate requests
    for i in range(10):
        allowed, retry_after = limiter.is_allowed("user123")
        print(f"Request {i+1}: {'ALLOWED' if allowed else f'BLOCKED (retry after {retry_after}s)'}")

    # Get stats
    stats = limiter.get_stats("user123")
    print(f"\nStats: {stats}")
