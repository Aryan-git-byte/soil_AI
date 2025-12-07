from fastapi import FastAPI
from app.routers.location import router as location_router

app = FastAPI(title="Farm AI Backend")

app.include_router(location_router)

@app.get("/")
def root():
    return {"message": "Farm AI Backend Running"}
