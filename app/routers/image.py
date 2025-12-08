from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional, Dict

from app.services.image_service import ImageAnalysisService
from app.services.sensor_service import get_latest_sensor_data
from app.services.location_service import get_location_context


router = APIRouter(prefix="/api/image", tags=["Image Analysis"])


@router.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    query: str = Form("Analyze this image"),
    include_context: bool = Form(True)
):
    

    # Validate image
    content_type = file.content_type
    size = file.size or 0

    is_valid, error = ImageAnalysisService.validate_image(content_type, size)
    if not is_valid:
        return JSONResponse(status_code=400, content={"error": error})

    # Read + encode image
    image_bytes = await file.read()
    image_base64 = ImageAnalysisService.encode_image_to_base64(image_bytes)

    # Build optional context
    extra_context: Optional[Dict] = None

    if include_context:
        sensor = await get_latest_sensor_data(None, None)

        if sensor:
            lat = float(sensor["latitude"])
            lon = float(sensor["longitude"])
            location_context = get_location_context(lat, lon)

            extra_context = {
                "location_context": location_context,
                "sensor_data": sensor
            }

    # Analyze via Vision Model
    result = await ImageAnalysisService.analyze_image(
        image_base64=image_base64,
        media_type=content_type,
        query=query,
        context=extra_context
    )

    return result


@router.get("/suggestions")
async def image_suggestions(type: str = "general"):
    """
    Get suggested questions for image-based analysis.
    """
    return {
        "type": type,
        "suggestions": ImageAnalysisService.get_analysis_suggestions(type)
    }
