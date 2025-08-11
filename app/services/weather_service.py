# app/services/weather_service.py
import requests
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

class WeatherService:
    def __init__(self):
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        if not self.api_key:
            print("Warning: OPENWEATHER_API_KEY not set. Weather features will be limited.")
        
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.onecall_url = "https://api.openweathermap.org/data/3.0/onecall"
        
        # Cache for reducing API calls
        self._cache = {}
        self._cache_duration = 300  # 5 minutes cache
    
    def _get_cache_key(self, lat: float, lng: float, endpoint: str) -> str:
        """Generate cache key for location and endpoint"""
        return f"{endpoint}_{lat:.4f}_{lng:.4f}"
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        if not cache_entry:
            return False
        
        cache_time = datetime.fromisoformat(cache_entry["timestamp"])
        return (datetime.now() - cache_time).total_seconds() < self._cache_duration
    
    async def get_current_weather(self, lat: float, lng: float) -> Dict[str, Any]:
        """Get current weather conditions"""
        if not self.api_key:
            return {"error": "Weather API key not configured"}
        
        cache_key = self._get_cache_key(lat, lng, "current")
        
        # Check cache first
        if cache_key in self._cache and self._is_cache_valid(self._cache[cache_key]):
            return self._cache[cache_key]["data"]
        
        try:
            url = f"{self.base_url}/weather"
            params = {
                "lat": lat,
                "lon": lng,
                "appid": self.api_key,
                "units": "metric"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Structure weather data for agricultural use
            weather_info = {
                "location": {
                    "latitude": lat,
                    "longitude": lng,
                    "name": data.get("name", "Unknown"),
                    "country": data.get("sys", {}).get("country", "")
                },
                "current": {
                    "temperature": data["main"]["temp"],
                    "feels_like": data["main"]["feels_like"],
                    "humidity": data["main"]["humidity"],
                    "pressure": data["main"]["pressure"],
                    "description": data["weather"][0]["description"],
                    "wind_speed": data.get("wind", {}).get("speed", 0),
                    "wind_direction": data.get("wind", {}).get("deg", 0),
                    "visibility": data.get("visibility", 0) / 1000,  # Convert to km
                    "uv_index": None  # Not available in current weather
                },
                "agricultural_indicators": self._calculate_agricultural_indicators(data),
                "timestamp": datetime.now().isoformat(),
                "source": "OpenWeatherMap"
            }
            
            # Cache the result
            self._cache[cache_key] = {
                "data": weather_info,
                "timestamp": datetime.now().isoformat()
            }
            
            return weather_info
            
        except requests.RequestException as e:
            return {"error": f"Weather API request failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Weather processing error: {str(e)}"}
    
    async def get_weather_forecast(self, lat: float, lng: float, days: int = 7) -> Dict[str, Any]:
        """Get weather forecast for agricultural planning"""
        if not self.api_key:
            return {"error": "Weather API key not configured"}
        
        cache_key = self._get_cache_key(lat, lng, f"forecast_{days}")
        
        # Check cache
        if cache_key in self._cache and self._is_cache_valid(self._cache[cache_key]):
            return self._cache[cache_key]["data"]
        
        try:
            # Use OneCall API for detailed forecast
            url = self.onecall_url
            params = {
                "lat": lat,
                "lon": lng,
                "appid": self.api_key,
                "units": "metric",
                "exclude": "minutely,alerts"  # Focus on hourly and daily
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # Process forecast data
            forecast_info = {
                "location": {"latitude": lat, "longitude": lng},
                "current": self._process_current_detailed(data.get("current", {})),
                "hourly": self._process_hourly_forecast(data.get("hourly", [])[:24]),  # Next 24 hours
                "daily": self._process_daily_forecast(data.get("daily", [])[:days]),
                "agricultural_summary": self._generate_agricultural_forecast_summary(data),
                "timestamp": datetime.now().isoformat(),
                "source": "OpenWeatherMap OneCall"
            }
            
            # Cache the result
            self._cache[cache_key] = {
                "data": forecast_info,
                "timestamp": datetime.now().isoformat()
            }
            
            return forecast_info
            
        except requests.RequestException as e:
            return {"error": f"Forecast API request failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Forecast processing error: {str(e)}"}
    
    def _calculate_agricultural_indicators(self, weather_data: Dict) -> Dict[str, Any]:
        """Calculate agricultural indicators from weather data"""
        temp = weather_data["main"]["temp"]
        humidity = weather_data["main"]["humidity"]
        
        return {
            "frost_risk": "high" if temp <= 2 else "low",
            "heat_stress_risk": "high" if temp >= 35 else "medium" if temp >= 30 else "low",
            "humidity_level": "high" if humidity >= 80 else "medium" if humidity >= 60 else "low",
            "irrigation_recommendation": self._get_irrigation_recommendation(temp, humidity),
            "pest_disease_risk": self._assess_pest_disease_risk(temp, humidity),
            "field_work_suitability": self._assess_field_work_conditions(weather_data)
        }
    
    def _get_irrigation_recommendation(self, temp: float, humidity: float) -> str:
        """Recommend irrigation based on temperature and humidity"""
        if temp > 30 and humidity < 50:
            return "high_priority"
        elif temp > 25 and humidity < 60:
            return "moderate"
        elif temp < 15 or humidity > 80:
            return "low_priority"
        else:
            return "monitor"
    
    def _assess_pest_disease_risk(self, temp: float, humidity: float) -> str:
        """Assess pest and disease risk based on conditions"""
        if 20 <= temp <= 30 and humidity >= 70:
            return "high"  # Ideal conditions for many pests and diseases
        elif temp > 35 or humidity < 40:
            return "low"   # Too hot/dry for most pests
        else:
            return "medium"
    
    def _assess_field_work_conditions(self, weather_data: Dict) -> str:
        """Assess suitability for field work"""
        if "rain" in weather_data.get("weather", [{}])[0].get("main", "").lower():
            return "poor"
        
        wind_speed = weather_data.get("wind", {}).get("speed", 0)
        if wind_speed > 10:  # High wind
            return "caution"
        
        temp = weather_data["main"]["temp"]
        if temp < 5 or temp > 40:
            return "caution"
        
        return "good"
    
    def _process_current_detailed(self, current_data: Dict) -> Dict:
        """Process detailed current weather from OneCall API"""
        return {
            "temperature": current_data.get("temp", 0),
            "feels_like": current_data.get("feels_like", 0),
            "humidity": current_data.get("humidity", 0),
            "pressure": current_data.get("pressure", 0),
            "uv_index": current_data.get("uvi", 0),
            "wind_speed": current_data.get("wind_speed", 0),
            "wind_direction": current_data.get("wind_deg", 0),
            "visibility": current_data.get("visibility", 0) / 1000,
            "dew_point": current_data.get("dew_point", 0),
            "description": current_data.get("weather", [{}])[0].get("description", "")
        }
    
    def _process_hourly_forecast(self, hourly_data: List[Dict]) -> List[Dict]:
        """Process hourly forecast data"""
        processed = []
        for hour in hourly_data:
            processed.append({
                "time": datetime.fromtimestamp(hour.get("dt", 0)).isoformat(),
                "temperature": hour.get("temp", 0),
                "humidity": hour.get("humidity", 0),
                "precipitation_probability": hour.get("pop", 0) * 100,
                "wind_speed": hour.get("wind_speed", 0),
                "description": hour.get("weather", [{}])[0].get("description", "")
            })
        return processed
    
    def _process_daily_forecast(self, daily_data: List[Dict]) -> List[Dict]:
        """Process daily forecast data"""
        processed = []
        for day in daily_data:
            temp_data = day.get("temp", {})
            processed.append({
                "date": datetime.fromtimestamp(day.get("dt", 0)).strftime("%Y-%m-%d"),
                "temperature_min": temp_data.get("min", 0),
                "temperature_max": temp_data.get("max", 0),
                "humidity": day.get("humidity", 0),
                "precipitation_probability": day.get("pop", 0) * 100,
                "wind_speed": day.get("wind_speed", 0),
                "uv_index": day.get("uvi", 0),
                "description": day.get("weather", [{}])[0].get("description", ""),
                "agricultural_advice": self._get_daily_agricultural_advice(day)
            })
        return processed
    
    def _get_daily_agricultural_advice(self, day_data: Dict) -> str:
        """Generate daily agricultural advice based on forecast"""
        temp_min = day_data.get("temp", {}).get("min", 0)
        temp_max = day_data.get("temp", {}).get("max", 0)
        precipitation = day_data.get("pop", 0) * 100
        
        advice = []
        
        if temp_min <= 2:
            advice.append("Frost protection needed")
        if temp_max >= 35:
            advice.append("Heat stress management required")
        if precipitation >= 70:
            advice.append("Avoid field operations")
        elif precipitation <= 20 and temp_max > 25:
            advice.append("Consider irrigation")
        
        return "; ".join(advice) if advice else "Normal operations"
    
    def _generate_agricultural_forecast_summary(self, forecast_data: Dict) -> Dict:
        """Generate agricultural summary from forecast data"""
        daily_data = forecast_data.get("daily", [])[:7]
        
        if not daily_data:
            return {"error": "No forecast data available"}
        
        # Analyze patterns
        rainfall_days = sum(1 for day in daily_data if day.get("pop", 0) >= 0.3)
        hot_days = sum(1 for day in daily_data if day.get("temp", {}).get("max", 0) >= 30)
        frost_days = sum(1 for day in daily_data if day.get("temp", {}).get("min", 0) <= 2)
        
        return {
            "rainfall_days": rainfall_days,
            "hot_days": hot_days,
            "frost_days": frost_days,
            "irrigation_priority": "high" if hot_days >= 3 and rainfall_days <= 1 else "medium",
            "field_work_windows": self._identify_work_windows(daily_data),
            "pest_disease_alert": "high" if rainfall_days >= 3 else "low"
        }
    
    def _identify_work_windows(self, daily_data: List[Dict]) -> List[str]:
        """Identify good days for field work"""
        good_days = []
        for day in daily_data:
            date = datetime.fromtimestamp(day.get("dt", 0)).strftime("%Y-%m-%d")
            if (day.get("pop", 0) < 0.3 and  # Low rain chance
                day.get("wind_speed", 0) < 8 and  # Moderate wind
                day.get("temp", {}).get("max", 0) < 35):  # Not too hot
                good_days.append(date)
        
        return good_days[:3]  # Return up to 3 good days