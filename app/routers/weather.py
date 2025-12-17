# app/routers/weather.py
from fastapi import APIRouter, Query, HTTPException
from app.services.weather_service import WeatherService
from app.services.sensor_service import get_latest_sensor_data
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/weather", tags=["Weather"])

weather_service = WeatherService(api_key=os.getenv("OPENWEATHER_API_KEY"))


@router.get("")
async def get_current_weather(
    lat: float = Query(None, description="Latitude (optional, uses latest sensor if not provided)"),
    lon: float = Query(None, description="Longitude (optional, uses latest sensor if not provided)")
):
    """
    Get current weather snapshot for display in weather card.
    
    Returns:
    {
      "temp": 31.2,
      "humidity": 58,
      "wind_speed": 11.4,
      "rain": 0,
      "condition": "clear sky"
    }
    """
    # Use provided coords or fetch from latest sensor
    if lat is None or lon is None:
        sensor = await get_latest_sensor_data(None, None)
        if not sensor:
            raise HTTPException(
                status_code=400,
                detail="No coordinates provided and no sensor data available"
            )
        lat = float(sensor["latitude"])
        lon = float(sensor["longitude"])
    
    # Get current weather
    weather = weather_service.get_current_weather(lat, lon)
    
    if not weather:
        raise HTTPException(
            status_code=503,
            detail="Failed to fetch weather data from OpenWeather API"
        )
    
    # Format for frontend weather card
    return {
        "temp": weather["temperature"],
        "humidity": weather["humidity"],
        "wind_speed": weather["wind_speed"],
        "rain": 0,  # Current weather doesn't include rain amount
        "condition": weather["description"],
        "location": {
            "lat": lat,
            "lon": lon
        }
    }


@router.get("/forecast")
async def get_weather_forecast(
    lat: float = Query(None, description="Latitude (optional, uses latest sensor if not provided)"),
    lon: float = Query(None, description="Longitude (optional, uses latest sensor if not provided)")
):
    """
    Get 7-day weather forecast starting from tomorrow.
    
    Returns exactly 7 days of forecast data:
    [
      {
        "date": "2025-12-14",
        "temp_max": 32.1,
        "temp_min": 21.4,
        "avg_temp": 26.8,
        "humidity": 60,
        "rain": 0,
        "condition": "clear sky"
      },
      ...
    ]
    """
    # Use provided coords or fetch from latest sensor
    if lat is None or lon is None:
        sensor = await get_latest_sensor_data(None, None)
        if not sensor:
            raise HTTPException(
                status_code=400,
                detail="No coordinates provided and no sensor data available"
            )
        lat = float(sensor["latitude"])
        lon = float(sensor["longitude"])
    
    # Get 7-day forecast
    forecast = weather_service.get_7day_forecast(lat, lon)
    
    if not forecast or len(forecast) == 0:
        raise HTTPException(
            status_code=503,
            detail="Failed to fetch weather forecast from OpenWeather API"
        )
    
    return forecast