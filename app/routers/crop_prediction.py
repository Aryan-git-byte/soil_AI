# app/routers/crop_prediction.py
from fastapi import APIRouter, Query, Form
from app.services.crop_service import CropPredictionService

router = APIRouter(prefix="/api/crop", tags=["Crop Prediction"])
crop_service = CropPredictionService()

@router.post("/predict")
async def predict_crops(
    temperature: float = Form(...),
    humidity: float = Form(...),
    ph: float = Form(...),
    rainfall: float = Form(...),
    season: str = Form(...),
    state: str = Form(...),
    nitrogen: float = Form(...),
    phosphorus: float = Form(...),
    potassium: float = Form(...),
    top_k: int = Form(5)
):
    """
    ML-based crop recommendation
    (From Smart Harvest model)
    """
    return crop_service.predict_crops(
        temperature=temperature,
        humidity=humidity,
        ph=ph,
        rainfall=rainfall,
        season=season,
        state=state,
        nitrogen=nitrogen,
        phosphorus=phosphorus,
        potassium=potassium,
        top_k=top_k
    )


@router.post("/fertilizer")
async def recommend_fertilizer(
    crop: str = Form(...),
    nitrogen: float = Form(...),
    phosphorus: float = Form(...),
    potassium: float = Form(...)
):
    """
    Fertilizer recommendations for specific crop
    """
    return crop_service.get_fertilizer_recommendation(
        crop=crop,
        nitrogen=nitrogen,
        phosphorus=phosphorus,
        potassium=potassium
    )


@router.get("/info/{crop_name}")
async def get_crop_info(crop_name: str):
    """
    Get detailed crop information from allcrop.json
    """
    # Load and return crop details
    import json
    import os
    
    json_path = os.path.join(os.path.dirname(__file__), "../ml_models/allcrop.json")
    
    with open(json_path, 'r') as f:
        crop_data = json.load(f)
    
    crop_key = crop_name.title()
    
    if crop_key not in crop_data:
        return {"error": f"Crop '{crop_name}' not found"}
    
    return {
        "crop": crop_name,
        "info": crop_data[crop_key]
    }