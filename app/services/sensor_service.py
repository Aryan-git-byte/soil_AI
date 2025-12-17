# app/services/sensor_service.py - Enhanced with Status Computation
import os
from typing import List, Dict, Optional
from datetime import datetime
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ========================================
# SENSOR THRESHOLDS & STATUS COMPUTATION
# ========================================

SENSOR_THRESHOLDS = {
    "soil_moisture": {
        "unit": "%",
        "critical": (0, 20),      # Below 20% = critical
        "warning": (20, 35),      # 20-35% = warning
        "optimal": (35, 70),      # 35-70% = optimal
        "warning_high": (70, 85), # 70-85% = warning (too wet)
        "critical_high": (85, 100) # Above 85% = critical (waterlogged)
    },
    "ph": {
        "unit": "",
        "critical": (0, 5.0),     # Too acidic
        "warning": (5.0, 6.0),    # Slightly acidic
        "optimal": (6.0, 7.5),    # Ideal for most crops
        "warning_high": (7.5, 8.5), # Slightly alkaline
        "critical_high": (8.5, 14)  # Too alkaline
    },
    "soil_temperature": {
        "unit": "°C",
        "critical": (0, 10),      # Too cold
        "warning": (10, 15),      # Cool
        "optimal": (15, 30),      # Ideal
        "warning_high": (30, 35), # Warm
        "critical_high": (35, 100) # Too hot
    },
    "ec": {  # Electrical Conductivity (Salinity)
        "unit": "dS/m",
        "critical": (0, 0.5),     # Too low (nutrient deficient)
        "warning": (0.5, 1.0),    # Low
        "optimal": (1.0, 2.5),    # Ideal
        "warning_high": (2.5, 4.0), # High salinity
        "critical_high": (4.0, 20)  # Excessive salinity
    },
    "n": {  # Nitrogen
        "unit": "mg/kg",
        "critical": (0, 150),
        "warning": (150, 250),
        "optimal": (250, 450),
        "warning_high": (450, 600),
        "critical_high": (600, 10000)
    },
    "p": {  # Phosphorus
        "unit": "mg/kg",
        "critical": (0, 10),
        "warning": (10, 20),
        "optimal": (20, 50),
        "warning_high": (50, 80),
        "critical_high": (80, 10000)
    },
    "k": {  # Potassium
        "unit": "mg/kg",
        "critical": (0, 100),
        "warning": (100, 150),
        "optimal": (150, 300),
        "warning_high": (300, 450),
        "critical_high": (450, 10000)
    }
}


def compute_sensor_status(sensor_type: str, value: float) -> str:
    """
    Compute sensor status based on agricultural thresholds.
    Returns: "optimal" | "warning" | "critical"
    """
    if sensor_type not in SENSOR_THRESHOLDS or value is None:
        return "unknown"
    
    thresholds = SENSOR_THRESHOLDS[sensor_type]
    
    # Check critical low
    if thresholds["critical"][0] <= value < thresholds["critical"][1]:
        return "critical"
    
    # Check warning low
    if thresholds["warning"][0] <= value < thresholds["warning"][1]:
        return "warning"
    
    # Check optimal
    if thresholds["optimal"][0] <= value <= thresholds["optimal"][1]:
        return "optimal"
    
    # Check warning high
    if thresholds["warning_high"][0] < value <= thresholds["warning_high"][1]:
        return "warning"
    
    # Check critical high
    if thresholds["critical_high"][0] < value <= thresholds["critical_high"][1]:
        return "critical"
    
    return "unknown"


# ========================================
# SENSOR DATA RETRIEVAL
# ========================================

async def get_latest_sensor_data(lat: Optional[float] = None, lon: Optional[float] = None):
    """
    Fetch the latest sensor reading near the coordinates.
    (Legacy function for backward compatibility)
    """
    resp = supabase.table("sensor_data") \
        .select("*") \
        .order("timestamp", desc=True) \
        .limit(1) \
        .execute()

    if resp.data and len(resp.data) > 0:
        row = resp.data[0]
        return {
            "soil_moisture": row.get("soil_moisture"),
            "ec": row.get("ec"),
            "soil_temperature": row.get("soil_temperature"),
            "n": row.get("n"),
            "p": row.get("p"),
            "k": row.get("k"),
            "ph": row.get("ph"),
            "latitude": row.get("latitude"),
            "longitude": row.get("longitude"),
            "timestamp": row.get("timestamp")
        }
    
    return None


async def get_latest_all_sensors() -> List[Dict]:
    """
    Get the latest reading for ALL sensor types.
    Returns a list of sensor objects with computed status.
    
    Returns:
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
      ...
    ]
    """
    try:
        # Get latest sensor record
        resp = supabase.table("sensor_data") \
            .select("*") \
            .order("timestamp", desc=True) \
            .limit(1) \
            .execute()
        
        if not resp.data or len(resp.data) == 0:
            return []
        
        row = resp.data[0]
        timestamp = row.get("timestamp")
        lat = row.get("latitude")
        lon = row.get("longitude")
        
        # Build sensor list
        sensors = []
        
        # Define sensor priority order
        sensor_fields = [
            "soil_moisture",
            "ph",
            "soil_temperature",
            "ec",
            "n",
            "p",
            "k"
        ]
        
        for sensor_type in sensor_fields:
            value = row.get(sensor_type)
            
            if value is not None:
                # Get unit from thresholds
                unit = SENSOR_THRESHOLDS.get(sensor_type, {}).get("unit", "")
                
                # Compute status
                status = compute_sensor_status(sensor_type, value)
                
                sensor_obj = {
                    "sensor_type": sensor_type,
                    "value": round(value, 2) if isinstance(value, float) else value,
                    "unit": unit,
                    "status": status,
                    "timestamp": timestamp
                }
                
                # Add location only to first sensor
                if len(sensors) == 0 and lat and lon:
                    sensor_obj["latitude"] = lat
                    sensor_obj["longitude"] = lon
                
                sensors.append(sensor_obj)
        
        # Sort by status priority (critical > warning > optimal)
        status_priority = {"critical": 0, "warning": 1, "optimal": 2, "unknown": 3}
        sensors.sort(key=lambda x: status_priority.get(x["status"], 3))
        
        return sensors
        
    except Exception as e:
        print(f"[Sensor] Error fetching latest sensors: {e}")
        import traceback
        traceback.print_exc()
        return []


async def get_sensor_history(
    sensor_type: str,
    limit: int = 100,
    lat: Optional[float] = None,
    lon: Optional[float] = None
) -> List[Dict]:
    """
    Get historical readings for a specific sensor type.
    
    Args:
        sensor_type: e.g., "soil_moisture", "ph", "n"
        limit: Number of records to return (default 100)
        lat/lon: Optional location filter (not implemented yet)
    
    Returns:
    [
      {
        "timestamp": "2025-12-13T10:00:00",
        "value": 35
      },
      ...
    ]
    Sorted chronologically (oldest first).
    """
    try:
        # Validate sensor type
        if sensor_type not in SENSOR_THRESHOLDS:
            return []
        
        # Fetch records
        resp = supabase.table("sensor_data") \
            .select(f"timestamp, {sensor_type}") \
            .order("timestamp", desc=True) \
            .limit(limit) \
            .execute()
        
        if not resp.data:
            return []
        
        # Format response
        history = []
        for row in resp.data:
            value = row.get(sensor_type)
            if value is not None:
                history.append({
                    "timestamp": row.get("timestamp"),
                    "value": round(value, 2) if isinstance(value, float) else value
                })
        
        return history
        
    except Exception as e:
        print(f"[Sensor] Error fetching history for {sensor_type}: {e}")
        import traceback
        traceback.print_exc()
        return []


async def get_sensor_statistics(sensor_type: str, days: int = 7) -> Optional[Dict]:
    """
    Get statistics for a sensor over the last N days.
    
    Returns:
    {
      "sensor_type": "soil_moisture",
      "min": 28,
      "max": 45,
      "avg": 36.5,
      "current": 32,
      "trend": "decreasing",
      "unit": "%"
    }
    """
    try:
        # Get recent data
        history = await get_sensor_history(sensor_type, limit=1000)
        
        if len(history) < 2:
            return None
        
        values = [h["value"] for h in history]
        
        # Calculate stats
        min_val = min(values)
        max_val = max(values)
        avg_val = sum(values) / len(values)
        current_val = values[-1]
        
        # Determine trend (compare current to average)
        if current_val > avg_val * 1.1:
            trend = "increasing"
        elif current_val < avg_val * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "sensor_type": sensor_type,
            "min": round(min_val, 2),
            "max": round(max_val, 2),
            "avg": round(avg_val, 2),
            "current": round(current_val, 2),
            "trend": trend,
            "unit": SENSOR_THRESHOLDS.get(sensor_type, {}).get("unit", ""),
            "data_points": len(values)
        }
        
    except Exception as e:
        print(f"[Sensor] Error calculating statistics for {sensor_type}: {e}")
        return None