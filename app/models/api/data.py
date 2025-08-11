# app/api/data.py
from fastapi import APIRouter, HTTPException, Depends, Request, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

router = APIRouter()

# Data models
class SensorData(BaseModel):
    soil_moisture: Optional[float] = Field(None, ge=0, le=100, description="Soil moisture percentage")
    soil_temperature: Optional[float] = Field(None, ge=-20, le=60, description="Soil temperature in Celsius")
    ph: Optional[float] = Field(None, ge=0, le=14, description="Soil pH level")
    ec: Optional[float] = Field(None, ge=0, description="Electrical conductivity (dS/m)")
    n: Optional[float] = Field(None, ge=0, description="Nitrogen content (ppm)")
    p: Optional[float] = Field(None, ge=0, description="Phosphorus content (ppm)")
    k: Optional[float] = Field(None, ge=0, description="Potassium content (ppm)")
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    sensor_id: Optional[str] = None
    farm_id: Optional[str] = None

class ManualEntry(BaseModel):
    entry_type: str = Field(..., description="Type of entry (observation, treatment, harvest, etc.)")
    title: str = Field(..., max_length=200)
    description: str = Field(..., max_length=2000)
    data: Optional[Dict[str, Any]] = Field(None, description="Additional structured data")
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)

class WeatherRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    days: Optional[int] = Field(7, ge=1, le=14, description="Number of forecast days")

# Sensor data endpoints
@router.post("/sensor")
async def submit_sensor_data(request: Request, data: SensorData):
    """Submit sensor readings"""
    try:
        supabase = request.app.state.supabase
        
        # Convert to dict and add timestamp
        sensor_dict = data.dict(exclude_unset=True)
        sensor_dict["timestamp"] = datetime.now().isoformat()
        
        result = await supabase.insert_sensor_data(sensor_dict)
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to store sensor data")
        
        return {
            "status": "success",
            "message": "Sensor data stored successfully",
            "data_id": result.get("id"),
            "timestamp": sensor_dict["timestamp"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sensor data error: {str(e)}")

@router.get("/sensor/latest")
async def get_latest_sensor_data(
    request: Request,
    limit: int = Query(10, ge=1, le=100),
    lat: Optional[float] = Query(None, ge=-90, le=90),
    lng: Optional[float] = Query(None, ge=-180, le=180)
):
    """Get latest sensor readings"""
    try:
        supabase = request.app.state.supabase
        
        data = await supabase.get_latest_sensor_data(
            limit=limit,
            lat=lat,
            lng=lng
        )
        
        return {
            "status": "success",
            "data": data,
            "count": len(data),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sensor data retrieval error: {str(e)}")

@router.get("/sensor/trends")
async def get_sensor_trends(
    request: Request,
    hours: int = Query(24, ge=1, le=168),  # Max 1 week
    lat: Optional[float] = Query(None, ge=-90, le=90),
    lng: Optional[float] = Query(None, ge=-180, le=180)
):
    """Get sensor data trends over time"""
    try:
        supabase = request.app.state.supabase
        
        trends = await supabase.get_sensor_trends(
            hours=hours,
            lat=lat,
            lng=lng
        )
        
        # Calculate basic statistics if we have data
        analytics = {}
        if trends:
            analytics = _calculate_sensor_analytics(trends)
        
        return {
            "status": "success",
            "data": trends,
            "analytics": analytics,
            "period_hours": hours,
            "count": len(trends),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sensor trends error: {str(e)}")

# Manual entry endpoints
@router.post("/manual")
async def submit_manual_entry(request: Request, entry: ManualEntry):
    """Submit manual farm entry"""
    try:
        supabase = request.app.state.supabase
        
        result = await supabase.insert_manual_entry(
            entry_type=entry.entry_type,
            title=entry.title,
            description=entry.description,
            data=entry.data,
            lat=entry.latitude,
            lng=entry.longitude
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to store manual entry")
        
        return {
            "status": "success",
            "message": "Manual entry stored successfully",
            "entry_id": result.get("id"),
            "timestamp": result.get("timestamp")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Manual entry error: {str(e)}")

@router.get("/manual")
async def get_manual_entries(
    request: Request,
    entry_type: Optional[str] = Query(None),
    days: int = Query(30, ge=1, le=365)
):
    """Get manual farm entries"""
    try:
        supabase = request.app.state.supabase
        
        entries = await supabase.get_manual_entries(
            entry_type=entry_type,
            days=days
        )
        
        return {
            "status": "success",
            "data": entries,
            "count": len(entries),
            "filter": {"entry_type": entry_type, "days": days},
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Manual entries error: {str(e)}")

# Weather endpoints
@router.post("/weather/current")
async def get_current_weather(request: Request, weather_req: WeatherRequest):
    """Get current weather conditions"""
    try:
        weather_service = request.app.state.weather
        supabase = request.app.state.supabase
        
        weather_data = await weather_service.get_current_weather(
            weather_req.latitude,
            weather_req.longitude
        )
        
        if "error" in weather_data:
            raise HTTPException(status_code=503, detail=weather_data["error"])
        
        # Store weather data for future reference
        await supabase.store_api_data(
            source="weather_current",
            data=weather_data,
            lat=weather_req.latitude,
            lng=weather_req.longitude
        )
        
        return {
            "status": "success",
            "data": weather_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weather data error: {str(e)}")

@router.post("/weather/forecast")
async def get_weather_forecast(request: Request, weather_req: WeatherRequest):
    """Get weather forecast"""
    try:
        weather_service = request.app.state.weather
        supabase = request.app.state.supabase
        
        forecast_data = await weather_service.get_weather_forecast(
            weather_req.latitude,
            weather_req.longitude,
            weather_req.days
        )
        
        if "error" in forecast_data:
            raise HTTPException(status_code=503, detail=forecast_data["error"])
        
        # Store forecast data
        await supabase.store_api_data(
            source="weather_forecast",
            data=forecast_data,
            lat=weather_req.latitude,
            lng=weather_req.longitude
        )
        
        return {
            "status": "success",
            "data": forecast_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weather forecast error: {str(e)}")

# Farm summary endpoints
@router.get("/farm/summary")
async def get_farm_summary(
    request: Request,
    lat: Optional[float] = Query(None, ge=-90, le=90),
    lng: Optional[float] = Query(None, ge=-180, le=180),
    days: int = Query(7, ge=1, le=30)
):
    """Get comprehensive farm data summary"""
    try:
        supabase = request.app.state.supabase
        
        summary = await supabase.get_farm_summary(
            lat=lat,
            lng=lng,
            days=days
        )
        
        return {
            "status": "success",
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Farm summary error: {str(e)}")

# AI interaction history
@router.get("/ai/history")
async def get_ai_history(
    request: Request,
    days: int = Query(7, ge=1, le=30),
    limit: int = Query(50, ge=1, le=200)
):
    """Get AI interaction history"""
    try:
        supabase = request.app.state.supabase
        
        history = await supabase.get_ai_interaction_history(
            days=days,
            limit=limit
        )
        
        return {
            "status": "success",
            "data": history,
            "count": len(history),
            "period_days": days,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI history error: {str(e)}")

# Analytics endpoints
@router.get("/analytics/overview")
async def get_analytics_overview(
    request: Request,
    lat: Optional[float] = Query(None, ge=-90, le=90),
    lng: Optional[float] = Query(None, ge=-180, le=180),
    days: int = Query(7, ge=1, le=30)
):
    """Get analytics overview"""
    try:
        supabase = request.app.state.supabase
        
        # Get various data sources
        sensor_data = await supabase.get_sensor_trends(hours=days*24, lat=lat, lng=lng)
        manual_entries = await supabase.get_manual_entries(days=days)
        ai_history = await supabase.get_ai_interaction_history(days=days)
        
        # Calculate analytics
        analytics = {
            "sensor_analytics": _calculate_sensor_analytics(sensor_data) if sensor_data else {},
            "activity_summary": _calculate_activity_summary(manual_entries, ai_history),
            "trends": _calculate_trends(sensor_data) if sensor_data else {},
            "period": {"days": days, "start": (datetime.now() - timedelta(days=days)).isoformat()}
        }
        
        return {
            "status": "success",
            "analytics": analytics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics error: {str(e)}")

# Export endpoints
@router.get("/export/sensor")
async def export_sensor_data(
    request: Request,
    format: str = Query("json", regex="^(json|csv)$"),
    hours: int = Query(168, ge=1, le=720),  # Max 30 days
    lat: Optional[float] = Query(None, ge=-90, le=90),
    lng: Optional[float] = Query(None, ge=-180, le=180)
):
    """Export sensor data"""
    try:
        supabase = request.app.state.supabase
        
        data = await supabase.get_sensor_trends(
            hours=hours,
            lat=lat,
            lng=lng
        )
        
        if format == "json":
            return {
                "status": "success",
                "data": data,
                "export_info": {
                    "format": "json",
                    "records": len(data),
                    "period_hours": hours,
                    "exported_at": datetime.now().isoformat()
                }
            }
        else:  # CSV format
            # Convert to CSV format
            csv_data = _convert_to_csv(data)
            return {
                "status": "success",
                "csv_data": csv_data,
                "export_info": {
                    "format": "csv",
                    "records": len(data),
                    "period_hours": hours,
                    "exported_at": datetime.now().isoformat()
                }
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

# Helper functions
def _calculate_sensor_analytics(sensor_data: List[Dict]) -> Dict[str, Any]:
    """Calculate basic sensor data analytics"""
    if not sensor_data:
        return {}
    
    analytics = {"metrics": {}}
    
    # Analyze each metric
    metrics = ["soil_moisture", "soil_temperature", "ph", "ec", "n", "p", "k"]
    
    for metric in metrics:
        values = [float(reading.get(metric, 0)) for reading in sensor_data if reading.get(metric) is not None]
        
        if values:
            analytics["metrics"][metric] = {
                "current": values[-1] if values else None,
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "readings_count": len(values),
                "trend": _calculate_simple_trend(values[-10:])  # Last 10 readings
            }
    
    # Overall health assessment
    analytics["health_indicators"] = _assess_soil_health(analytics["metrics"])
    
    return analytics

def _calculate_simple_trend(values: List[float]) -> str:
    """Calculate simple trend direction"""
    if len(values) < 3:
        return "insufficient_data"
    
    # Compare first and last third of values
    first_third = values[:len(values)//3] if len(values) >= 6 else values[:2]
    last_third = values[-len(values)//3:] if len(values) >= 6 else values[-2:]
    
    first_avg = sum(first_third) / len(first_third)
    last_avg = sum(last_third) / len(last_third)
    
    change_percent = ((last_avg - first_avg) / first_avg) * 100 if first_avg != 0 else 0
    
    if abs(change_percent) < 5:
        return "stable"
    elif change_percent > 0:
        return "increasing"
    else:
        return "decreasing"

def _assess_soil_health(metrics: Dict) -> Dict[str, str]:
    """Assess overall soil health from metrics"""
    health = {}
    
    # pH assessment
    if "ph" in metrics and metrics["ph"]["current"]:
        ph = metrics["ph"]["current"]
        if 6.0 <= ph <= 7.0:
            health["ph_status"] = "optimal"
        elif 5.5 <= ph < 6.0 or 7.0 < ph <= 7.5:
            health["ph_status"] = "acceptable"
        else:
            health["ph_status"] = "needs_attention"
    
    # Moisture assessment
    if "soil_moisture" in metrics and metrics["soil_moisture"]["current"]:
        moisture = metrics["soil_moisture"]["current"]
        if 40 <= moisture <= 70:
            health["moisture_status"] = "optimal"
        elif 30 <= moisture < 40 or 70 < moisture <= 80:
            health["moisture_status"] = "acceptable"
        else:
            health["moisture_status"] = "needs_attention"
    
    # Overall assessment
    statuses = list(health.values())
    if all(status == "optimal" for status in statuses):
        health["overall"] = "excellent"
    elif all(status in ["optimal", "acceptable"] for status in statuses):
        health["overall"] = "good"
    else:
        health["overall"] = "needs_improvement"
    
    return health

def _calculate_activity_summary(manual_entries: List[Dict], ai_interactions: List[Dict]) -> Dict:
    """Calculate activity summary"""
    return {
        "manual_entries": {
            "total": len(manual_entries),
            "types": list(set(entry.get("entry_type", "unknown") for entry in manual_entries)),
            "recent": manual_entries[:5] if manual_entries else []
        },
        "ai_interactions": {
            "total": len(ai_interactions),
            "successful": len([i for i in ai_interactions if i.get("status") == "success"]),
            "recent_topics": _extract_recent_topics(ai_interactions[:10])
        }
    }

def _extract_recent_topics(interactions: List[Dict]) -> List[str]:
    """Extract recent topics from AI interactions"""
    topics = []
    agricultural_keywords = [
        "soil", "water", "fertilizer", "crop", "plant", "pest", "disease", 
        "weather", "irrigation", "harvest", "seed", "nutrition"
    ]
    
    for interaction in interactions:
        query = interaction.get("input_query", "").lower()
        for keyword in agricultural_keywords:
            if keyword in query and keyword not in topics:
                topics.append(keyword)
                if len(topics) >= 10:
                    break
    
    return topics

def _calculate_trends(sensor_data: List[Dict]) -> Dict:
    """Calculate trend analysis"""
    if len(sensor_data) < 10:
        return {"status": "insufficient_data"}
    
    # Group data by time periods
    now = datetime.now()
    day_ago = now - timedelta(days=1)
    week_ago = now - timedelta(days=7)
    
    recent_data = []
    older_data = []
    
    for reading in sensor_data:
        try:
            timestamp = datetime.fromisoformat(reading.get("timestamp", ""))
            if timestamp > day_ago:
                recent_data.append(reading)
            elif timestamp > week_ago:
                older_data.append(reading)
        except:
            continue
    
    trends = {}
    
    if recent_data and older_data:
        for metric in ["soil_moisture", "soil_temperature", "ph"]:
            recent_values = [r.get(metric) for r in recent_data if r.get(metric) is not None]
            older_values = [r.get(metric) for r in older_data if r.get(metric) is not None]
            
            if recent_values and older_values:
                recent_avg = sum(recent_values) / len(recent_values)
                older_avg = sum(older_values) / len(older_values)
                
                change = ((recent_avg - older_avg) / older_avg) * 100 if older_avg != 0 else 0
                
                trends[metric] = {
                    "change_percent": round(change, 2),
                    "direction": "up" if change > 2 else "down" if change < -2 else "stable",
                    "recent_average": round(recent_avg, 2),
                    "older_average": round(older_avg, 2)
                }
    
    return trends

def _convert_to_csv(data: List[Dict]) -> str:
    """Convert data to CSV format"""
    if not data:
        return ""
    
    # Get all possible headers
    headers = set()
    for row in data:
        headers.update(row.keys())
    
    headers = sorted(list(headers))
    
    # Create CSV content
    csv_lines = [",".join(headers)]
    
    for row in data:
        csv_row = []
        for header in headers:
            value = row.get(header, "")
            # Handle None values and escape commas
            if value is None:
                value = ""
            else:
                value = str(value).replace(",", ";")  # Replace commas to avoid CSV issues
            csv_row.append(value)
        csv_lines.append(",".join(csv_row))
    
    return "\n".join(csv_lines)