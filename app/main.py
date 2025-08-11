# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from dotenv import load_dotenv
import os

# Import our modules
from app.api import chat, data, admin
from app.services.supabase_client import SupabaseClient
from app.services.weather_service import WeatherService
from app.models.knowledge_base import KnowledgeBase

load_dotenv()

# Global instances
supabase_client = None
weather_service = None
knowledge_base = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global supabase_client, weather_service, knowledge_base
    
    print("🌱 Starting Agricultural AI System...")
    
    # Initialize services
    supabase_client = SupabaseClient()
    weather_service = WeatherService()
    knowledge_base = KnowledgeBase()
    
    # Store in app state
    app.state.supabase = supabase_client
    app.state.weather = weather_service
    app.state.knowledge_base = knowledge_base
    
    print("✅ All services initialized successfully!")
    
    yield
    
    # Shutdown
    print("🔄 Shutting down Agricultural AI System...")

app = FastAPI(
    title="Agricultural AI Advisory System",
    description="Smart farming AI that provides real-time agricultural advice using sensor data, weather, and expert knowledge",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(data.router, prefix="/api/v1/data", tags=["Data"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])

@app.get("/")
async def root():
    return {
        "message": "🌾 Agricultural AI Advisory System",
        "status": "active",
        "version": "1.0.0",
        "features": [
            "Real-time sensor data analysis",
            "Weather-based recommendations",
            "Persistent conversation memory",
            "Agricultural knowledge base",
            "Farmer-friendly advice"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        await app.state.supabase.health_check()
        
        return {
            "status": "healthy",
            "services": {
                "database": "connected",
                "weather": "active",
                "ai": "ready"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )