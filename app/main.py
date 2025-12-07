from fastapi import FastAPI
from app.routers.ai import router as ai_router
from app.routers.location import router as location_router

app = FastAPI(title="FarmBot Nova Backend")

app.include_router(location_router)
app.include_router(ai_router)

@app.get("/")
async def root():
    return {"message": "FarmBot Nova Backend Running"}
