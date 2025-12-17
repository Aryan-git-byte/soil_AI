# app/services/weather_service.py - Enhanced with 7-Day Forecast
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class WeatherService:
    BASE_URL = "https://api.openweathermap.org/data/2.5"
    HEADERS = {"User-Agent": "FarmAI/1.0"}

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_current_weather(self, lat: float, lon: float) -> Optional[Dict]:
        """Get current weather snapshot"""
        url = f"{self.BASE_URL}/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric"
        }

        try:
            res = requests.get(url, params=params, headers=self.HEADERS).json()

            return {
                "temperature": res["main"]["temp"],
                "temperature_unit": "°C",
                "feels_like": res["main"]["feels_like"],
                "humidity": res["main"]["humidity"],
                "humidity_unit": "%",
                "pressure": res["main"]["pressure"],
                "pressure_unit": "hPa",
                "wind_speed": res.get("wind", {}).get("speed"),
                "wind_speed_unit": "m/s",
                "wind_dir": res.get("wind", {}).get("deg"),
                "weather": res["weather"][0]["main"],
                "description": res["weather"][0]["description"]
            }
        except Exception as e:
            print(f"[Weather] Error fetching current weather: {e}")
            return None

    def get_7day_forecast(self, lat: float, lon: float) -> List[Dict]:
        """
        Get 7-day weather forecast (tomorrow onwards, excludes today)
        
        Returns exactly 7 days of forecast data in the format:
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
        url = f"{self.BASE_URL}/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",
            "cnt": 56  # Get 7 days worth (8 readings per day * 7 days)
        }

        try:
            res = requests.get(url, params=params, headers=self.HEADERS)
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
                # Parse timestamp
                dt = datetime.fromtimestamp(entry["dt"])
                date = dt.date()
                
                # Skip today - forecast starts tomorrow
                if date <= today:
                    continue
                
                date_str = date.strftime("%Y-%m-%d")
                
                # Initialize day if not exists
                if date_str not in daily_data:
                    daily_data[date_str] = {
                        "temps": [],
                        "humidity": [],
                        "rain": [],
                        "conditions": []
                    }
                
                # Collect data points for the day
                daily_data[date_str]["temps"].append(entry["main"]["temp"])
                daily_data[date_str]["humidity"].append(entry["main"]["humidity"])
                
                # Rain data (if available)
                rain_3h = entry.get("rain", {}).get("3h", 0)
                daily_data[date_str]["rain"].append(rain_3h)
                
                # Weather condition
                condition = entry["weather"][0]["description"]
                daily_data[date_str]["conditions"].append(condition)
            
            # Aggregate into final format
            forecast = []
            sorted_dates = sorted(daily_data.keys())[:7]  # Ensure exactly 7 days
            
            for date_str in sorted_dates:
                day = daily_data[date_str]
                
                temps = day["temps"]
                humidity_vals = day["humidity"]
                rain_vals = day["rain"]
                conditions = day["conditions"]
                
                # Most common condition for the day
                most_common_condition = max(set(conditions), key=conditions.count)
                
                forecast.append({
                    "date": date_str,
                    "temp_max": round(max(temps), 1),
                    "temp_min": round(min(temps), 1),
                    "avg_temp": round(sum(temps) / len(temps), 1),
                    "humidity": round(sum(humidity_vals) / len(humidity_vals)),
                    "rain": round(sum(rain_vals), 1),  # Total rainfall for the day
                    "condition": most_common_condition
                })
            
            # Ensure exactly 7 days (fill with None if API returns less)
            while len(forecast) < 7:
                last_date = datetime.strptime(sorted_dates[-1], "%Y-%m-%d") if sorted_dates else datetime.now()
                next_date = last_date + timedelta(days=1)
                forecast.append({
                    "date": next_date.strftime("%Y-%m-%d"),
                    "temp_max": None,
                    "temp_min": None,
                    "avg_temp": None,
                    "humidity": None,
                    "rain": None,
                    "condition": "No data"
                })
                sorted_dates.append(next_date.strftime("%Y-%m-%d"))
            
            return forecast[:7]  # Strictly return 7 days
            
        except Exception as e:
            print(f"[Weather] Error fetching 7-day forecast: {e}")
            import traceback
            traceback.print_exc()
            return []