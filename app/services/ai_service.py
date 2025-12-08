# app/services/ai_service.py
import json
import httpx
import base64
from typing import Optional
from fastapi import UploadFile

from app.core.config import OPENROUTER_KEYS, OPENWEATHER_API_KEY
from app.services.location_service import LocationService
from app.services.weather_service import WeatherService
from app.services.sensor_service import get_latest_sensor_data
from app.services.soil_service import get_soil_physical, classify_indian_soil_type
from app.services.conversation_service import ConversationService
from app.services.rag_service import search_knowledge, format_context


# -----------------------------
# SYSTEM PROMPT (Nova tuned with memory)
# -----------------------------
SYSTEM_PROMPT = """
You are FarmBot Nova — an agricultural assistant for Indian farming conditions.
Your job:
- use ONLY the <context> data given
- remember previous conversations with the user
- avoid assumptions or hallucinations
- give short, practical, actionable farming advice
- keep answers simple enough for farmers
- consider soil, weather, texture, location, and sensor data
- if the Indian soil classification is provided, use it for crop recommendations
- mention the soil type (Alluvial, Black/Regur, Red & Yellow, Laterite, etc.) when giving advice
- if data is missing, say it clearly
- be friendly and remember user details they've shared (like their name, crops, location preferences)
- if an image is provided, analyze it thoroughly for diseases, pests, health issues, or other problems

Answer in clear bullet points unless asked otherwise.
"""


# -----------------------------
# Build the final AI prompt
# -----------------------------
def build_prompt(query: str, full_context: dict, has_image: bool = False):
    ctx = json.dumps(full_context, indent=2)
    
    if has_image:
        return f"""
<context>
{ctx}
</context>

User uploaded an image and asks:
{query}

Analyze the image carefully and provide detailed insights based on what you see.
Consider the context data (location, weather, soil) when giving recommendations.

Your answer:
"""
    else:
        return f"""
<context>
{ctx}
</context>

User Question:
{query}

Your answer:
"""


# -----------------------------
# Encode image to base64
# -----------------------------
def encode_image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode('utf-8')


# -----------------------------
# OpenRouter fallback logic
# -----------------------------
async def call_openrouter(messages: list):
    """
    Async version using httpx.
    Handles both text and vision (image) queries.
    """
    
    for i, key in enumerate(OPENROUTER_KEYS):
        try:
            print(f"[OpenRouter] Trying key {i+1}/{len(OPENROUTER_KEYS)}...")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://farmbot.com",
                        "X-Title": "FarmBot Nova"
                    },
                    json={
                        "model": "amazon/nova-2-lite-v1:free",
                        "messages": messages
                    }
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
# Main API: process AI query with optional image
# -----------------------------
async def process_ai_query(
    query: str,
    auth_id: str,
    conversation_id: str,
    lat: float = None,
    lon: float = None,
    image: Optional[UploadFile] = None  # 🆕 Optional image parameter
):
    # Initialize services
    location_service = LocationService()
    weather_service = WeatherService(api_key=OPENWEATHER_API_KEY)
    conversation_service = ConversationService()
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
        
        # Classify Indian soil type
        if soil_physical.get("sand_percent") and soil_physical.get("clay_percent") and soil_physical.get("silt_percent"):
            indian_soil = classify_indian_soil_type(
                sand_percent=soil_physical["sand_percent"],
                clay_percent=soil_physical["clay_percent"],
                silt_percent=soil_physical["silt_percent"],
                texture=soil_physical["texture"],
                lat=lat,
                lon=lon
            )
            location_context["indian_soil_classification"] = indian_soil
    except Exception as e:
        location_context["soil_physical"] = None
        location_context["indian_soil_classification"] = None
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

    # 5️⃣ Get conversation history
    history = await conversation_service.get_conversation_history(conversation_id, limit=20)
    formatted_history = conversation_service.format_history_for_ai(history)

    # 5.1️⃣ RAG vector search (knowledge base retrieval)
    # Run MiniLM embedding + vector search from Supabase
    try:
        rag_chunks = search_knowledge(query, top_k=8)
        rag_context_text = format_context(rag_chunks)
        full_context["rag_chunks"] = rag_chunks  # save raw results for debugging
    except Exception as e:
        print(f"[RAG ERROR] {e}")
        rag_context_text = ""
        full_context["rag_chunks"] = None

    # 5.2️⃣ If RAG returned chunks, attach them to prompt context
    if rag_context_text:
        full_context["retrieved_knowledge"] = rag_context_text
    else:
        full_context["retrieved_knowledge"] = "No relevant chunks found."


    # 6️⃣ Process image if provided
    has_image = image is not None
    image_metadata = None
    
    if has_image:
        # Read and encode image
        image_bytes = await image.read()
        image_base64 = encode_image_to_base64(image_bytes)
        media_type = image.content_type
        
        image_metadata = {
            "filename": image.filename,
            "content_type": media_type,
            "size": len(image_bytes)
        }
        print(f"[Image] Processing: {image.filename} ({len(image_bytes)} bytes)")

    # 7️⃣ Build messages array with history
    prompt = build_prompt(query, full_context, has_image)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    # Add conversation history
    messages.extend(formatted_history)
    
    # Add current user query (with or without image)
    if has_image:
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        })
    else:
        messages.append({"role": "user", "content": prompt})

    # 8️⃣ Call Nova 2 Lite (with fallback)
    ai_response = await call_openrouter(messages)
    answer = ai_response["choices"][0]["message"]["content"]

    # 9️⃣ Save conversation to database
    user_metadata = {
        "coordinates": {"lat": lat, "lon": lon}
    }
    if has_image:
        user_metadata["image"] = image_metadata
    
    await conversation_service.save_message(
        auth_id=auth_id,
        conversation_id=conversation_id,
        role="user",
        content=query,
        metadata=user_metadata
    )
    
    await conversation_service.save_message(
        auth_id=auth_id,
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        metadata={
            "context_used": full_context,
            "had_image": has_image
        }
    )

    return {
        "answer": answer,
        "context_used": full_context,
        "conversation_id": conversation_id,
        "message_count": len(history) + 2,  # +2 for current exchange
        "had_image": has_image
    }