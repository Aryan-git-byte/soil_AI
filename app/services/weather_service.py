import requests

class WeatherService:
    BASE_URL = "https://api.openweathermap.org/data/2.5"
    HEADERS = {"User-Agent": "FarmAI/1.0"}

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_current_weather(self, lat: float, lon: float):
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
        except:
            return None
