from fastapi import APIRouter, Query
from app.services.location_service import LocationService
from app.services.weather_service import WeatherService
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

router = APIRouter(prefix="/api/location", tags=["Location Intelligence"])

location_service = LocationService()
weather_service = WeatherService(api_key=os.getenv("OPENWEATHER_API_KEY"))
from app.services.soil_service import get_soil_physical

@router.get("/context")
def get_location_context(lat: float = Query(...), lon: float = Query(...)):
    # Build base context (includes weather if available)
    context = location_service.build_location_context(lat, lon, weather_service)

    # Attach soil physical properties (from raster datasets)
    try:
        soil = get_soil_physical(lat, lon)
        context["soil_physical"] = soil
        context["soil_available"] = bool(soil)
    except Exception as e:
        # Surface any error so caller can diagnose missing data
        context["soil_physical"] = None
        context["soil_available"] = False
        context["soil_error"] = str(e)
    return context
