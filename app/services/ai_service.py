import json
import httpx
from app.core.config import OPENROUTER_KEYS, OPENWEATHER_API_KEY
from app.services.location_service import LocationService
from app.services.weather_service import WeatherService
from app.services.sensor_service import get_latest_sensor_data
from app.services.soil_service import get_soil_physical


# -----------------------------
# SYSTEM PROMPT (Nova tuned)
# -----------------------------
SYSTEM_PROMPT = """
You are FarmBot Nova — an agricultural assistant for Indian farming conditions.
Your job:
- use ONLY the <context> data given
- avoid assumptions or hallucinations
- give short, practical, actionable farming advice
- keep answers simple enough for farmers
- consider soil, weather, texture, location, and sensor data
- if data is missing, say it clearly

Answer in clear bullet points unless asked otherwise.
"""


# -----------------------------
# Build the final AI prompt
# -----------------------------
def build_prompt(query: str, full_context: dict):
    ctx = json.dumps(full_context, indent=2)
    return f"""
<context>
{ctx}
</context>

User Question:
{query}

Your answer:
"""


# -----------------------------
# OpenRouter fallback logic (FIXED)
# -----------------------------
async def call_openrouter(payload):
    """
    Async version using httpx instead of requests.
    Properly handles OpenRouter API with correct headers.
    """
    
    for i, key in enumerate(OPENROUTER_KEYS):
        try:
            print(f"[OpenRouter] Trying key {i+1}/{len(OPENROUTER_KEYS)}...")
            
            async with httpx.AsyncClient(timeout=40.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://farmbot.com",
                        "X-Title": "FarmBot Nova"
                    },
                    json=payload
                )

                if response.status_code == 200:
                    print(f"[OpenRouter] ✅ Key {i+1} succeeded")
                    return response.json()
                else:
                    error_msg = response.text[:200] if response.text else "No error body"
                    print(f"[OpenRouter] Key {i+1} failed: {response.status_code} - {error_msg}")

        except Exception as e:
            print(f"[OpenRouter] Key {i+1} exception: {str(e)[:150]}")

    raise Exception("❌ All OpenRouter keys failed. Check API keys and rate limits.")


# -----------------------------
# Main API: process AI query
# -----------------------------
async def process_ai_query(query: str, lat: float = None, lon: float = None):
    # Initialize services
    location_service = LocationService()
    weather_service = WeatherService(api_key=OPENWEATHER_API_KEY)
    sensor = None
    
    # 1️⃣ Get location (lat, lon)
    if lat is None or lon is None:
        # Fallback to latest sensor reading
        sensor = await get_latest_sensor_data(lat=None, lon=None)

        if not sensor:
            raise Exception("No sensor data available and no lat/lon provided.")

        lat = float(sensor.get("latitude", 0))
        lon = float(sensor.get("longitude", 0))
    
    # 2️⃣ Get full location context (weather, soil, wikipedia, monuments)
    location_context = location_service.build_location_context(lat, lon, weather_service)

    # Add soil physical properties
    try:
        soil_physical = get_soil_physical(lat, lon)
        location_context["soil_physical"] = soil_physical
    except Exception as e:
        location_context["soil_physical"] = None
        print(f"[Soil Physical] Error: {e}")

    # 3️⃣ If lat/lon were provided, fetch sensor data separately (if available)
    if lat is not None and lon is not None and sensor is None:
        sensor = await get_latest_sensor_data(lat=lat, lon=lon)
    
    # 4️⃣ Build unified context object
    full_context = {
        "coordinates": {"lat": lat, "lon": lon},
        "location_context": location_context,
    }
    
    # Add sensor data if available
    if sensor:
        full_context["sensor_data"] = {
            "soil_moisture": sensor.get("soil_moisture"),
            "ec": sensor.get("ec"),
            "soil_temperature": sensor.get("soil_temperature"),
            "n": sensor.get("n"),
            "p": sensor.get("p"),
            "k": sensor.get("k"),
            "ph": sensor.get("ph"),
            "timestamp": str(sensor.get("timestamp"))
        }
        full_context["sensor_available"] = True
    else:
        full_context["sensor_available"] = False

    # 5️⃣ Build prompt for Nova
    prompt = build_prompt(query, full_context)

    # 6️⃣ Prepare OpenRouter payload
    payload = {
        "model": "amazon/nova-2-lite-v1:free",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    }

    # 7️⃣ Call Nova 2 Lite (with fallback)
    ai_response = await call_openrouter(payload)

    return {
        "answer": ai_response["choices"][0]["message"]["content"],
        "context_used": full_context
    }