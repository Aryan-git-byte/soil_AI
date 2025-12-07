# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.ai import router as ai_router
from app.routers.location import router as location_router
from app.routers.image import router as image_router

app = FastAPI(title="FarmBot Nova Backend")

# ✅ ADD CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(location_router)
app.include_router(ai_router)
app.include_router(image_router)

@app.get("/")
async def root():
    return {"message": "FarmBot Nova Backend Running"}