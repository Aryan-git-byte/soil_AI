# app/main.py - Security Fixes Implemented
from fastapi import FastAPI, Depends, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.routers.ai import router as ai_router
from app.routers.location import router as location_router
from app.routers.image import router as image_router
from app.routers.weather import router as weather_router
from app.routers.sensors import router as sensors_router
from app.routers import videos
from app.core.security import verify_api_key_with_rate_limit, generate_api_key
import os
import logging
import secrets
import re
from datetime import datetime, timedelta
from collections import defaultdict

# Try to import ML router (optional)
try:
    from app.routers.crop_prediction import router as crop_router
    ML_ROUTER_AVAILABLE = True
except Exception as e:
    ML_ROUTER_AVAILABLE = False
    logging.warning(f"⚠️ ML Crop Prediction router not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO if os.getenv("ENVIRONMENT") == "production" else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check environment
IS_PRODUCTION = os.getenv("ENVIRONMENT") == "production"

app = FastAPI(
    title="FarmBot Nova Backend",
    description="AI-Powered Agricultural Assistant with RAG + ML Crop Prediction",
    version="1.1.0",
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json"
)

# CORS Configuration
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()]

if not ALLOWED_ORIGINS:
    raise RuntimeError(
        "❌ CRITICAL: ALLOWED_ORIGINS not configured.\n"
        "Set in .env: ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com"
    )

logger.info(f"✓ CORS enabled for origins: {ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# SECURITY MIDDLEWARES
# ---------------------------------------------------------------------------

MAX_REQUEST_SIZE = 15 * 1024 * 1024  # 15MB limit

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add essential security headers to all responses"""
    async def dispatch(self, request: Request, call_next):
        # 4. FIXED: Global File Upload Size Limit
        if request.method in ["POST", "PUT", "PATCH"]:
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > MAX_REQUEST_SIZE:
                return JSONResponse(
                    status_code=413,
                    content={"error": "Request body too large. Max 15MB allowed."}
                )

        response = await call_next(request)
        
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # XSS protection (legacy but still useful)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Force HTTPS (only in production)
        if IS_PRODUCTION:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Content Security Policy
        if request.url.path in ["/docs", "/redoc", "/openapi.json"]:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "img-src 'self' data: https://fastapi.tiangolo.com https://cdn.jsdelivr.net; "
                "font-src 'self' data:; "
                "connect-src 'self'"
            )
        else:
            response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        # Control referrer information
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions policy
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response


app.add_middleware(SecurityHeadersMiddleware)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.utcnow()
    logger.info(f"→ {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(
            f"← {request.method} {request.url.path} "
            f"status={response.status_code} duration={duration:.3f}s"
        )
        return response
    except Exception as e:
        logger.error(f"✗ Request failed: {request.method} {request.url.path} - {e}")
        raise

# 5. FIXED: Public Rate Limiter
public_rate_limits = defaultdict(list)

async def check_public_rate_limit(request: Request):
    """Simple in-memory IP-based rate limiter for public endpoints"""
    client_ip = request.client.host
    now = datetime.now()
    
    # Keep requests from the last minute
    cutoff = now - timedelta(minutes=1)
    public_rate_limits[client_ip] = [
        t for t in public_rate_limits[client_ip] if t > cutoff
    ]
    
    # Limit: 100 requests per minute
    if len(public_rate_limits[client_ip]) >= 100:
        logger.warning(f"Public rate limit exceeded for IP: {client_ip}")
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")
    
    public_rate_limits[client_ip].append(now)

# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

# Public routes (no authentication required)
@app.get("/")
async def root(request: Request):
    """Public health check endpoint"""
    await check_public_rate_limit(request) # Rate limited
    
    features = {
        "ai_chatbot": True,
        "rag_knowledge": True,
        "ml_predictions": ML_ROUTER_AVAILABLE,
        "image_analysis": True,
        "location_intelligence": True,
        "bilingual_support": True
    }
    
    return {
        "message": "FarmBot Nova Backend Running",
        "status": "healthy",
        "version": "1.1.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "features": features,
        "docs": "/docs" if not IS_PRODUCTION else "disabled"
    }


@app.get("/health")
async def health(request: Request):
    """Comprehensive health check for load balancers"""
    await check_public_rate_limit(request) # Rate limited
    
    from app.services.sensor_service import supabase
    from app.services.rag_service import qdrant_client, COLLECTION_NAME
    
    checks = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.1.0",
        "services": {},
        "features": {}
    }
    
    # Check Qdrant
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        checks["services"]["qdrant"] = "healthy"
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        checks["services"]["qdrant"] = "unhealthy"
        checks["status"] = "degraded"
    
    # Check Supabase
    try:
        supabase.table("sensor_data").select("id").limit(1).execute()
        checks["services"]["supabase"] = "healthy"
    except Exception as e:
        logger.error(f"Supabase health check failed: {e}")
        checks["services"]["supabase"] = "unhealthy"
        checks["status"] = "degraded"
    
    # Check ML availability
    checks["features"]["ml_predictions"] = ML_ROUTER_AVAILABLE
    checks["features"]["rag_knowledge"] = True
    checks["features"]["image_analysis"] = True
    checks["features"]["bilingual_support"] = True
    
    status_code = 200 if checks["status"] == "healthy" else 503
    return JSONResponse(content=checks, status_code=status_code)


# Protected routes (require API key + rate limiting)
app.include_router(ai_router, dependencies=[Depends(verify_api_key_with_rate_limit)])
app.include_router(location_router, dependencies=[Depends(verify_api_key_with_rate_limit)])
app.include_router(image_router, dependencies=[Depends(verify_api_key_with_rate_limit)])
app.include_router(weather_router, dependencies=[Depends(verify_api_key_with_rate_limit)])
app.include_router(sensors_router, dependencies=[Depends(verify_api_key_with_rate_limit)])
app.include_router(videos.router, dependencies=[Depends(verify_api_key_with_rate_limit)])

if ML_ROUTER_AVAILABLE:
    app.include_router(crop_router, dependencies=[Depends(verify_api_key_with_rate_limit)])
    logger.info("✓ ML Crop Prediction routes enabled")
else:
    logger.warning("⚠️ ML Crop Prediction routes disabled (service not available)")


# Admin endpoint
ADMIN_SECRET = os.getenv("ADMIN_SECRET")

if not ADMIN_SECRET:
    raise RuntimeError(
        "❌ CRITICAL: ADMIN_SECRET not set.\n"
        "Set a strong random secret in .env: ADMIN_SECRET=your-secret-here"
    )

# 2. FIXED: Admin Endpoint Security
@app.post("/admin/generate-api-key")
async def generate_new_api_key(
    x_admin_secret: str = Header(..., alias="X-Admin-Secret")
):
    """
    Generate a new API key (admin only).
    Uses Header authentication and constant-time comparison.
    """
    # 8. FIXED: Constant-time comparison
    if not secrets.compare_digest(x_admin_secret, ADMIN_SECRET):
        logger.warning("Unauthorized admin access attempt")
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid admin credentials"}
        )
    
    new_key = generate_api_key()
    logger.info("New API key generated by admin")
    
    return {
        "api_key": new_key,
        "instructions": "Add this key to API_KEYS in .env (comma-separated)",
        "example": f"API_KEYS=existing_key,{new_key}"
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    # 7. FIXED: Sanitize Error Messages
    safe_message = str(exc)
    
    # Clean sensitive patterns
    safe_message = re.sub(r'/[a-zA-Z0-9/_.-]+\.py', '[FILE]', safe_message)
    if 'sql' in safe_message.lower() or 'select' in safe_message.lower():
        safe_message = "Database operation failed"

    if IS_PRODUCTION:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred. Please try again later."
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": safe_message[:200], # Truncate for safety
                "type": type(exc).__name__
            }
        )


# HTTP error handlers
@app.exception_handler(401)
async def unauthorized_handler(request, exc):
    logger.warning(f"Unauthorized access attempt: {request.url.path}")
    return JSONResponse(
        status_code=401,
        content={
            "error": "Unauthorized",
            "message": "Valid API key required. Include 'X-API-Key' header."
        }
    )


@app.exception_handler(429)
async def rate_limit_handler(request, exc):
    logger.warning(f"Rate limit exceeded: {request.url.path}")
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "message": "Maximum 60 requests per minute. Please try again later."
        },
        headers={"Retry-After": "200"}
    )


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 50)
    logger.info("🚀 FarmBot Nova Backend v1.1.0 Starting")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"CORS Origins: {ALLOWED_ORIGINS}")
    logger.info(f"API Docs: {'ENABLED at /docs ✓' if not IS_PRODUCTION else 'DISABLED (production) ⚠️'}")
    logger.info(f"Security Headers: ENABLED")
    logger.info(f"ML Predictions: {'ENABLED ✓' if ML_ROUTER_AVAILABLE else 'DISABLED ⚠️'}")
    logger.info("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 FarmBot Nova Backend Shutting Down")
    logger.info("✓ Shutdown complete")