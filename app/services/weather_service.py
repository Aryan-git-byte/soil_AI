# app/services/weather_service.py - Async Optimized
import httpx
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class WeatherService:
    BASE_URL = "https://api.openweathermap.org/data/2.5"
    HEADERS = {"User-Agent": "FarmAI/1.0"}

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def get_current_weather(self, lat: float, lon: float, client: Optional[httpx.AsyncClient] = None) -> Optional[Dict]:
        """Get current weather snapshot (Async)"""
        url = f"{self.BASE_URL}/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric"
        }

        try:
            # Use provided client or create a new one
            if client:
                res = await client.get(url, params=params)
            else:
                async with httpx.AsyncClient(headers=self.HEADERS, timeout=10.0) as local_client:
                    res = await local_client.get(url, params=params)
            
            data = res.json()

            return {
                "temperature": data["main"]["temp"],
                "temperature_unit": "°C",
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "humidity_unit": "%",
                "pressure": data["main"]["pressure"],
                "pressure_unit": "hPa",
                "wind_speed": data.get("wind", {}).get("speed"),
                "wind_speed_unit": "m/s",
                "wind_dir": data.get("wind", {}).get("deg"),
                "weather": data["weather"][0]["main"],
                "description": data["weather"][0]["description"]
            }
        except Exception as e:
            print(f"[Weather] Error fetching current weather: {e}")
            return None

    async def get_7day_forecast(self, lat: float, lon: float) -> List[Dict]:
        """
        Get 7-day weather forecast (Async)
        """
        url = f"{self.BASE_URL}/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",
            "cnt": 56
        }

        try:
            async with httpx.AsyncClient(headers=self.HEADERS, timeout=10.0) as client:
                res = await client.get(url, params=params)
            
            if res.status_code != 200:
                print(f"[Weather] Forecast API error: {res.status_code}")
                return []
            
            data = res.json()
            forecast_list = data.get("list", [])
            
            if not forecast_list:
                return []

            # Group by date and aggregate
            daily_data = {}
            today = datetime.now().date()
            
            for entry in forecast_list:
                dt = datetime.fromtimestamp(entry["dt"])
                date = dt.date()
                
                if date <= today:
                    continue
                
                date_str = date.strftime("%Y-%m-%d")
                
                if date_str not in daily_data:
                    daily_data[date_str] = {
                        "temps": [],
                        "humidity": [],
                        "rain": [],
                        "conditions": []
                    }
                
                daily_data[date_str]["temps"].append(entry["main"]["temp"])
                daily_data[date_str]["humidity"].append(entry["main"]["humidity"])
                
                rain_3h = entry.get("rain", {}).get("3h", 0)
                daily_data[date_str]["rain"].append(rain_3h)
                
                condition = entry["weather"][0]["description"]
                daily_data[date_str]["conditions"].append(condition)
            
            forecast = []
            sorted_dates = sorted(daily_data.keys())[:7]
            
            for date_str in sorted_dates:
                day = daily_data[date_str]
                temps = day["temps"]
                conditions = day["conditions"]
                most_common_condition = max(set(conditions), key=conditions.count)
                
                forecast.append({
                    "date": date_str,
                    "temp_max": round(max(temps), 1),
                    "temp_min": round(min(temps), 1),
                    "avg_temp": round(sum(temps) / len(temps), 1),
                    "humidity": round(sum(day["humidity"]) / len(day["humidity"])),
                    "rain": round(sum(day["rain"]), 1),
                    "condition": most_common_condition
                })
            
            return forecast
            
        except Exception as e:
            print(f"[Weather] Error fetching 7-day forecast: {e}")
            return []