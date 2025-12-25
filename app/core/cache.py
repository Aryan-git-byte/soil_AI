# app/core/cache.py
import os
import redis
import json
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Use localhost for local dev, or your REDIS_URL for production (e.g., from Render)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

CACHE_ENABLED = False
cache = None

try:
    # Initialize Redis client with decoding enabled (returns strings, not bytes)
    cache = redis.from_url(REDIS_URL, decode_responses=True)
    # Quick connection test
    cache.ping()
    logger.info("✓ Redis connected successfully")
    CACHE_ENABLED = True
except Exception as e:
    logger.warning(f"⚠️ Redis connection failed: {e}. Caching is DISABLED.")
    CACHE_ENABLED = False

async def get_cached(key: str):
    """Retrieve JSON object from cache"""
    if not CACHE_ENABLED: return None
    try:
        data = cache.get(key)
        return json.loads(data) if data else None
    except Exception:
        return None

async def set_cached(key: str, data: dict, expire: int = 3600):
    """Save JSON object to cache with expiration (seconds)"""
    if not CACHE_ENABLED: return
    try:
        cache.setex(key, expire, json.dumps(data))
    except Exception as e:
        logger.error(f"Cache write failed: {e}")