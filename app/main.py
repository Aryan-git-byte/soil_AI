# FarmBot Nova Backend - Secured with API Key Authentication
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.routers.ai import router as ai_router
from app.routers.location import router as location_router
from app.routers.image import router as image_router
from app.core.security import verify_api_key_with_rate_limit, generate_api_key
import os

app = FastAPI(
    title="FarmBot Nova Backend",
    description="AI-Powered Agricultural Assistant with RAG",
    version="1.0.0"
)

# CORS Configuration
# ⚠️ IMPORTANT: In production, replace "*" with your actual frontend domain
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Public routes (no authentication required)
@app.get("/")
async def root():
    """Public health check endpoint"""
    return {
        "message": "FarmBot Nova Backend Running",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Public health check for load balancers"""
    return {"status": "ok"}

# Protected routes (require API key)
# Option 1: Apply authentication globally to all routers
app.include_router(
    ai_router,
    dependencies=[Depends(verify_api_key_with_rate_limit)]
)

app.include_router(
    location_router,
    dependencies=[Depends(verify_api_key_with_rate_limit)]
)

app.include_router(
    image_router,
    dependencies=[Depends(verify_api_key_with_rate_limit)]
)

# Admin endpoint to generate new API keys (protect this!)
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "change_me_in_production")

@app.post("/admin/generate-api-key")
async def generate_new_api_key(admin_secret: str):
    """
    Generate a new API key (admin only).
    Call with: POST /admin/generate-api-key?admin_secret=YOUR_SECRET
    """
    if admin_secret != ADMIN_SECRET:
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid admin secret"}
        )
    
    new_key = generate_api_key()
    return {
        "api_key": new_key,
        "instructions": "Add this key to your .env file under API_KEYS"
    }

# Error handlers
@app.exception_handler(401)
async def unauthorized_handler(request, exc):
    return JSONResponse(
        status_code=401,
        content={
            "error": "Unauthorized",
            "message": "Valid API key required. Include 'X-API-Key' header.",
            "docs": f"{request.base_url}docs"
        }
    )

@app.exception_handler(429)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "message": "Maximum 60 requests per minute. Please try again later."
        },
        headers={"Retry-After": "60"}
    )