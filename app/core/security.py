# app/core/security.py
import os
import secrets
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

load_dotenv()

# API Key configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Load valid API keys from environment
# Format: API_KEYS=key1,key2,key3
VALID_API_KEYS = set(
    key.strip() 
    for key in os.getenv("API_KEYS", "").split(",") 
    if key.strip()
)

# Add a default key for development (remove in production!)
if not VALID_API_KEYS and os.getenv("ENVIRONMENT") == "development":
    DEFAULT_KEY = "dev_test_key_12345"
    VALID_API_KEYS.add(DEFAULT_KEY)
    print(f"⚠️  WARNING: Using default API key for development: {DEFAULT_KEY}")


async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Verify API key from request header.
    Raises 401 if invalid or missing.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Include 'X-API-Key' header.",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    return api_key


def generate_api_key() -> str:
    """
    Generate a secure random API key.
    Use this to create new keys for users.
    """
    return secrets.token_urlsafe(32)


# Rate limiting storage (simple in-memory, use Redis for production)
from collections import defaultdict
from datetime import datetime, timedelta

class RateLimiter:
    """
    Simple in-memory rate limiter.
    For production, use Redis with sliding window.
    """
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    def is_allowed(self, api_key: str) -> bool:
        """Check if request is allowed under rate limit"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        
        # Clean old requests
        self.requests[api_key] = [
            req_time for req_time in self.requests[api_key]
            if req_time > cutoff
        ]
        
        # Check limit
        if len(self.requests[api_key]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.requests[api_key].append(now)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter(requests_per_minute=60)


async def verify_api_key_with_rate_limit(api_key: str = Security(api_key_header)):
    """
    Verify API key AND check rate limit.
    """
    # First verify the key
    verified_key = await verify_api_key(api_key)
    
    # Then check rate limit
    if not rate_limiter.is_allowed(verified_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Max 60 requests per minute.",
            headers={"Retry-After": "60"}
        )
    
    return verified_key