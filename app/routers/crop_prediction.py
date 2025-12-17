# app/routers/crop_prediction.py
from fastapi import APIRouter, Query, Form, HTTPException
from app.services.crop_service import CropPredictionService

router = APIRouter(prefix="/api/crop", tags=["Crop Prediction"])
crop_service = CropPredictionService()

@router.post("/predict")
async def predict_crops(
    # 3. FIXED: Input Validation with ranges
    temperature: float = Form(..., ge=-50, le=60),
    humidity: float = Form(..., ge=0, le=100),
    ph: float = Form(..., ge=0, le=14),
    rainfall: float = Form(..., ge=0, le=5000),
    season: str = Form(..., min_length=1, max_length=50),
    state: str = Form(..., min_length=1, max_length=50),
    nitrogen: float = Form(..., ge=0, le=500),
    phosphorus: float = Form(..., ge=0, le=500),
    potassium: float = Form(..., ge=0, le=500),
    top_k: int = Form(5, ge=1, le=20)
):
    """
    ML-based crop recommendation with input validation
    """
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fertilizer")
async def recommend_fertilizer(
    crop: str = Form(..., min_length=1, max_length=100),
    nitrogen: float = Form(..., ge=0, le=500),
    phosphorus: float = Form(..., ge=0, le=500),
    potassium: float = Form(..., ge=0, le=500)
):
    """
    Fertilizer recommendations with input validation
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
    # Sanitize path traversal attempts
    if ".." in crop_name or "/" in crop_name:
         raise HTTPException(status_code=400, detail="Invalid crop name")

    import json
    import os
    
    json_path = os.path.join(os.path.dirname(__file__), "../ml_models/allcrop.json")
    
    if not os.path.exists(json_path):
         raise HTTPException(status_code=500, detail="Crop database not found")

    with open(json_path, 'r') as f:
        crop_data = json.load(f)
    
    crop_key = crop_name.title()
    
    if crop_key not in crop_data:
        return {"error": f"Crop '{crop_name}' not found"}
    
    return {
        "crop": crop_name,
        "info": crop_data[crop_key]
    }