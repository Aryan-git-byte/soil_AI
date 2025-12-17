# app/routers/sensors.py
from fastapi import APIRouter, Query, HTTPException
from app.services.sensor_service import (
    get_latest_all_sensors,
    get_sensor_history,
    get_sensor_statistics,
    SENSOR_THRESHOLDS
)
from typing import Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sensors", tags=["Sensors"])


@router.get("/latest")
async def get_latest_sensors(limit: Optional[int] = Query(None, description="Limit number of sensors returned")):
    """
    Get latest readings for all sensor types with computed status.
    
    Returns one object per sensor type, sorted by priority (critical > warning > optimal).
    
    Response:
    [
      {
        "sensor_type": "soil_moisture",
        "value": 32,
        "unit": "%",
        "status": "warning",
        "timestamp": "2025-12-13T18:15:00",
        "latitude": 28.61,
        "longitude": 77.20
      },
      {
        "sensor_type": "ph",
        "value": 6.2,
        "unit": "",
        "status": "optimal",
        "timestamp": "2025-12-13T18:15:00"
      },
      ...
    ]
    
    Status values:
    - "optimal": Value is in ideal range
    - "warning": Value is outside ideal but not critical
    - "critical": Value requires immediate attention
    """
    try:
        sensors = await get_latest_all_sensors()
        
        if not sensors:
            raise HTTPException(
                status_code=404,
                detail="No sensor data available"
            )
        
        # Apply limit if specified
        if limit and limit > 0:
            sensors = sensors[:limit]
        
        return sensors
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching latest sensors: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch sensor data"
        )


@router.get("/history")
async def get_sensor_history_endpoint(
    sensor_type: str = Query(..., description="Sensor type (e.g., soil_moisture, ph, n, p, k)"),
    limit: int = Query(100, description="Number of records to return (max 1000)", ge=1, le=1000)
):
    """
    Get historical readings for a specific sensor type.
    
    Returns data sorted chronologically (oldest → newest) for easy graphing.
    
    Example:
    GET /api/sensors/history?sensor_type=soil_moisture&limit=100
    
    Response:
    [
      {
        "timestamp": "2025-12-13T10:00:00",
        "value": 35
      },
      {
        "timestamp": "2025-12-13T12:00:00",
        "value": 33
      },
      ...
    ]
    """
    # Validate sensor type
    if sensor_type not in SENSOR_THRESHOLDS:
        valid_types = list(SENSOR_THRESHOLDS.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sensor_type. Valid options: {', '.join(valid_types)}"
        )
    
    try:
        history = await get_sensor_history(sensor_type, limit=limit)
        
        if not history:
            raise HTTPException(
                status_code=404,
                detail=f"No historical data found for {sensor_type}"
            )
        
        return history
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching history for {sensor_type}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch history for {sensor_type}"
        )


@router.get("/statistics/{sensor_type}")
async def get_sensor_stats(
    sensor_type: str,
    days: int = Query(7, description="Number of days to analyze", ge=1, le=30)
):
    """
    Get statistical summary for a sensor over time.
    
    Returns min, max, avg, current value, and trend.
    
    Example:
    GET /api/sensors/statistics/soil_moisture?days=7
    
    Response:
    {
      "sensor_type": "soil_moisture",
      "min": 28,
      "max": 45,
      "avg": 36.5,
      "current": 32,
      "trend": "decreasing",
      "unit": "%",
      "data_points": 168
    }
    
    Trend values:
    - "increasing": Current value > 110% of average
    - "decreasing": Current value < 90% of average
    - "stable": Within ±10% of average
    """
    # Validate sensor type
    if sensor_type not in SENSOR_THRESHOLDS:
        valid_types = list(SENSOR_THRESHOLDS.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sensor_type. Valid options: {', '.join(valid_types)}"
        )
    
    try:
        stats = await get_sensor_statistics(sensor_type, days=days)
        
        if not stats:
            raise HTTPException(
                status_code=404,
                detail=f"Insufficient data to calculate statistics for {sensor_type}"
            )
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating statistics for {sensor_type}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate statistics for {sensor_type}"
        )


@router.get("/types")
async def get_sensor_types():
    """
    Get list of all available sensor types with their thresholds.
    
    Response:
    {
      "soil_moisture": {
        "unit": "%",
        "optimal_range": [35, 70],
        "description": "Soil moisture percentage"
      },
      "ph": {
        "unit": "",
        "optimal_range": [6.0, 7.5],
        "description": "Soil pH level"
      },
      ...
    }
    """
    sensor_info = {}
    
    for sensor_type, thresholds in SENSOR_THRESHOLDS.items():
        sensor_info[sensor_type] = {
            "unit": thresholds["unit"],
            "optimal_range": list(thresholds["optimal"]),
            "warning_low": list(thresholds["warning"]),
            "warning_high": list(thresholds["warning_high"]),
            "critical_low": list(thresholds["critical"]),
            "critical_high": list(thresholds["critical_high"])
        }
    
    return sensor_info