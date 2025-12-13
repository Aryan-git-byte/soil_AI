# app/services/ai_service.py - Production Ready with ML Integration
import json
import httpx
import base64
import logging
from typing import Optional
from fastapi import UploadFile
from datetime import datetime

from app.core.config import OPENROUTER_KEYS, OPENWEATHER_API_KEY
from app.services.location_service import LocationService
from app.services.weather_service import WeatherService
from app.services.sensor_service import get_latest_sensor_data
from app.services.soil_service import get_soil_physical, classify_indian_soil_type
from app.services.conversation_service import ConversationService
from app.services.rag_service import hybrid_search, format_hybrid_context

logger = logging.getLogger(__name__)

# Try to import ML service (optional)
try:
    from app.services.crop_service import CropPredictionService
    crop_predictor = CropPredictionService()
    ML_AVAILABLE = True
    logger.info("✓ ML Crop Prediction Service loaded")
except Exception as e:
    ML_AVAILABLE = False
    logger.warning(f"⚠️ ML Service not available: {e}")

SYSTEM_PROMPT = """
You are FarmBot Nova — an advanced agricultural assistant for Indian farming.

Your capabilities:
- Use ONLY the <context> data provided
- Remember previous conversations
- Give short, practical, actionable advice
- Consider soil, weather, location, sensor data
- Use ML crop predictions when available
- Mention Indian soil type when giving advice
- Analyze images for diseases, pests, health issues
- If data is missing, say it clearly
- Be friendly and remember user details

When you cite knowledge from retrieved sources, mention the source briefly (e.g., "According to ICAR wheat guide...").

When ML predictions are available, explain WHY those crops were recommended based on the soil and weather conditions.

Answer in clear bullet points unless asked otherwise.
"""


def build_prompt(query: str, full_context: dict, has_image: bool = False):
    """Build AI prompt with context"""
    ctx = json.dumps(full_context, indent=2)
    
    if has_image:
        return f"""
<context>
{ctx}
</context>

User uploaded an image and asks:
{query}

Analyze the image carefully and provide detailed insights. If you reference information from the retrieved knowledge, cite the source.

Your answer:
"""
    else:
        return f"""
<context>
{ctx}
</context>

User Question:
{query}

Your answer (cite sources when using retrieved knowledge):
"""


def encode_image_to_base64(image_bytes: bytes) -> str:
    """Encode image to base64"""
    return base64.b64encode(image_bytes).decode('utf-8')


async def call_openrouter(messages: list):
    """OpenRouter API call with fallback"""
    
    for i, key in enumerate(OPENROUTER_KEYS):
        try:
            logger.info(f"Trying OpenRouter key {i+1}/{len(OPENROUTER_KEYS)}")
            
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
                    logger.info(f"OpenRouter key {i+1} succeeded")
                    return response.json()
                else:
                    error_msg = response.text[:200] if response.text else "No error body"
                    logger.warning(f"OpenRouter key {i+1} failed: {response.status_code} - {error_msg}")

        except Exception as e:
            logger.error(f"OpenRouter key {i+1} exception: {str(e)[:150]}")

    raise Exception("All OpenRouter keys failed")


def detect_season_from_date():
    """Detect Indian agricultural season from current date"""
    month = datetime.now().month
    
    # Kharif: June-November (monsoon crops)
    if 6 <= month <= 11:
        return "kharif"
    # Rabi: November-April (winter crops)
    elif month <= 4 or month >= 11:
        return "rabi"
    # Summer/Zaid: March-June
    else:
        return "perennial"


def should_use_ml_prediction(query: str, sensor: dict = None) -> bool:
    """
    Determine if ML crop prediction should be triggered
    """
    if not ML_AVAILABLE or not sensor:
        return False
    
    # Check if required sensor data is available
    required_fields = ["n", "p", "k", "ph"]
    if not all(sensor.get(field) is not None for field in required_fields):
        return False
    
    # Keywords that trigger ML prediction
    ml_keywords = [
        "crop", "grow", "plant", "recommend", "suggestion",
        "best", "suitable", "optimal", "should i", "what to"
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in ml_keywords)


async def process_ai_query(
    query: str,
    auth_id: str,
    conversation_id: str,
    lat: float = None,
    lon: float = None,
    image: Optional[UploadFile] = None
):
    """Main AI query processor with ML integration"""
    
    logger.info(f"Processing AI query for user {auth_id}, conversation {conversation_id}")
    
    # Initialize services
    location_service = LocationService()
    weather_service = WeatherService(api_key=OPENWEATHER_API_KEY)
    conversation_service = ConversationService()
    sensor = None
    
    # Get location coordinates
    if lat is None or lon is None:
        sensor = await get_latest_sensor_data(lat=None, lon=None)

        if not sensor:
            raise Exception("No sensor data and no lat/lon provided")

        lat = float(sensor.get("latitude", 0))
        lon = float(sensor.get("longitude", 0))
    
    logger.info(f"Using coordinates: ({lat}, {lon})")
    
    # Build location context
    location_context = location_service.build_location_context(lat, lon, weather_service)

    # Add soil physical properties
    try:
        soil_physical = get_soil_physical(lat, lon)
        location_context["soil_physical"] = soil_physical
        
        # Indian soil classification
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
        logger.error(f"Soil data error: {e}")

    # Get sensor data if needed
    if lat is not None and lon is not None and sensor is None:
        sensor = await get_latest_sensor_data(lat=lat, lon=lon)
    
    # Build unified context
    full_context = {
        "coordinates": {"lat": lat, "lon": lon},
        "location_context": location_context,
    }
    
    # Add sensor data
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

    # ========================================
    # 🚀 ML CROP PREDICTION INTEGRATION
    # ========================================
    ml_prediction_used = False
    
    if should_use_ml_prediction(query, sensor):
        try:
            logger.info("🌾 Triggering ML crop prediction...")
            
            # Get state from location context
            state = location_context.get("location_info", {}).get("state", "bihar")
            if state:
                state = state.lower().replace(" ", "")
            
            # Detect season
            season = detect_season_from_date()
            
            # Get weather data
            weather = location_context.get("weather", {})
            temperature = weather.get("temperature", 25.0)
            humidity = weather.get("humidity", 70.0)
            
            # Estimate rainfall (you might want to get actual data)
            rainfall = 100.0  # Default, you can enhance this
            
            # Run ML prediction
            ml_results = crop_predictor.predict_crops(
                temperature=temperature,
                humidity=humidity,
                ph=sensor.get("ph", 7.0),
                rainfall=rainfall,
                season=season,
                state=state,
                nitrogen=sensor.get("n", 50.0),
                phosphorus=sensor.get("p", 50.0),
                potassium=sensor.get("k", 50.0),
                top_k=5
            )
            
            # Add to context
            full_context["ml_crop_predictions"] = ml_results
            ml_prediction_used = True
            
            logger.info(f"✓ ML predictions: {[c['crop'] for c in ml_results['recommended_crops'][:3]]}")
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}", exc_info=True)
            full_context["ml_crop_predictions"] = {
                "error": "ML prediction unavailable",
                "reason": str(e)
            }

    # Get conversation history
    history = await conversation_service.get_conversation_history(conversation_id, limit=20)
    formatted_history = conversation_service.format_history_for_ai(history)

    # Hybrid RAG search
    rag_info = {
        "success": False,
        "rag_chunks": [],
        "web_results": [],
        "error": None
    }
    
    try:
        logger.info(f"Starting RAG search for: '{query[:50]}...'")
        
        hybrid_results = await hybrid_search(
            query=query,
            top_k_rag=8,
            top_k_web=5
        )
        
        rag_context_text = format_hybrid_context(hybrid_results)
        
        # Enhanced RAG info for frontend
        rag_info = {
            "success": True,
            "rag_chunks_count": hybrid_results["rag_count"],
            "web_results_count": hybrid_results["web_count"],
            "used_web_search": hybrid_results["used_web"],
            "search_decision": hybrid_results.get("search_decision"),
            "rag_sources": [
                {
                    "source": chunk.get("source"),
                    "crop": chunk.get("crop"),
                    "region": chunk.get("region"),
                    "similarity": round(chunk.get("similarity", 0), 4),
                    "text_preview": chunk.get("text", "")[:150] + "..."
                }
                for chunk in hybrid_results["rag_chunks"][:5]
            ],
            "web_sources": [
                {
                    "title": result.get("title"),
                    "url": result.get("url"),
                    "content_preview": result.get("content", "")[:150] + "..."
                }
                for result in hybrid_results["web_results"][:3]
            ]
        }
        
        logger.info(f"RAG search successful: rag={rag_info['rag_chunks_count']}, web={rag_info['web_results_count']}")
        
    except Exception as e:
        logger.error(f"RAG search failed: {e}", exc_info=True)
        rag_context_text = ""
        rag_info["error"] = str(e)

    # Attach retrieved knowledge
    full_context["knowledge_retrieval"] = rag_info
    
    if rag_context_text:
        full_context["retrieved_knowledge"] = rag_context_text
    else:
        full_context["retrieved_knowledge"] = "No relevant info found"

    # Process image if provided
    has_image = image is not None
    image_metadata = None
    
    if has_image:
        image_bytes = await image.read()
        image_base64 = encode_image_to_base64(image_bytes)
        media_type = image.content_type
        
        image_metadata = {
            "filename": image.filename,
            "content_type": media_type,
            "size": len(image_bytes)
        }
        logger.info(f"Processing image: {image.filename} ({len(image_bytes)} bytes)")

    # Build messages with history
    prompt = build_prompt(query, full_context, has_image)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    messages.extend(formatted_history)
    
    # Add current query
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

    # Call AI model
    logger.info("Calling OpenRouter API...")
    ai_response = await call_openrouter(messages)
    answer = ai_response["choices"][0]["message"]["content"]
    logger.info("Received AI response")

    # Save conversation
    user_metadata = {
        "coordinates": {"lat": lat, "lon": lon},
        "ml_prediction_used": ml_prediction_used
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
            "rag_info": rag_info,
            "had_image": has_image,
            "ml_prediction_used": ml_prediction_used
        }
    )

    logger.info(f"Query processed successfully for {auth_id}")

    return {
        "answer": answer,
        "context_used": full_context,
        "conversation_id": conversation_id,
        "message_count": len(history) + 2,
        "had_image": has_image,
        "rag_info": rag_info,
        "ml_prediction_used": ml_prediction_used
    }