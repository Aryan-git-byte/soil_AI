# app/services/sensor_service.py - Complete: Optimized + All Features Preserved
import os
import logging
from typing import List, Dict, Optional
from supabase import create_client, Client
from app.core.cache import get_cached, set_cached

logger = logging.getLogger(__name__)

# ========================================
# 1. OPTIMIZATION: Global Database Client
# ========================================
# We initialize this ONCE at startup to save 500ms on every request.
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    logger.warning("Supabase credentials missing. DB features disabled.")
    supabase = None

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
# SENSOR DATA RETRIEVAL (OPTIMIZED)
# ========================================

async def get_latest_sensor_data(lat: Optional[float] = None, lon: Optional[float] = None):
    """
    Fetch the latest sensor reading.
    ⚡ OPTIMIZED: Uses Redis Cache ("Read-Through" Strategy)
    """
    if not supabase: return None

    # 1. FAST PATH: Check Redis Cache (<5ms)
    cache_key = "sensor:latest"
    cached_data = await get_cached(cache_key)
    if cached_data:
        return cached_data

    # 2. SLOW PATH: Fetch from Database (~300ms)
    try:
        resp = supabase.table("sensor_data") \
            .select("*") \
            .order("timestamp", desc=True) \
            .limit(1) \
            .execute()

        if resp.data and len(resp.data) > 0:
            row = resp.data[0]
            result = {
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
            
            # 3. SAVE TO CACHE (Valid for 5 minutes)
            await set_cached(cache_key, result, expire=300)
            return result
    
    except Exception as e:
        logger.error(f"Supabase fetch failed: {e}")
        return None
    
    return None


async def get_latest_all_sensors() -> List[Dict]:
    """
    Get the latest reading for ALL sensor types.
    Returns a list of sensor objects with computed status.
    ⚡ OPTIMIZED: Now uses the cached 'get_latest_sensor_data' internally if possible.
    """
    try:
        # Reuse the cached data function to avoid a second DB hit
        raw_data = await get_latest_sensor_data()
        
        if not raw_data:
            return []
        
        timestamp = raw_data.get("timestamp")
        lat = raw_data.get("latitude")
        lon = raw_data.get("longitude")
        
        # Build sensor list
        sensors = []
        
        # Define sensor priority order
        sensor_fields = [
            "soil_moisture", "ph", "soil_temperature", "ec", "n", "p", "k"
        ]
        
        for sensor_type in sensor_fields:
            value = raw_data.get(sensor_type)
            
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
        logger.error(f"[Sensor] Error fetching latest sensors: {e}")
        return []


async def get_sensor_history(
    sensor_type: str,
    limit: int = 100,
    lat: Optional[float] = None,
    lon: Optional[float] = None
) -> List[Dict]:
    """
    Get historical readings for a specific sensor type.
    (This usually requires fresh data for charts, so we might skip caching or cache with short TTL)
    """
    try:
        if not supabase: return []

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
        logger.error(f"[Sensor] Error fetching history for {sensor_type}: {e}")
        return []


async def get_sensor_statistics(sensor_type: str, days: int = 7) -> Optional[Dict]:
    """
    Get statistics for a sensor over the last N days.
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
        current_val = values[0] # History is desc sorted, so [0] is latest
        
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
        logger.error(f"[Sensor] Error calculating statistics for {sensor_type}: {e}")
        return None