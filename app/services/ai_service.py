# app/services/ai_service.py - Enhanced with Hindi/English Support
import json
import httpx
import base64
import logging
from typing import Optional
from fastapi import UploadFile
from datetime import datetime

from app.core.config import GROQ_API_KEY, OPENWEATHER_API_KEY
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

# ========================================
# 🌍 BILINGUAL SYSTEM PROMPT
# ========================================
SYSTEM_PROMPT_BILINGUAL = """
You are FarmBot Nova — an advanced agricultural assistant for Indian farming.

🌍 LANGUAGE SUPPORT:
- You can communicate in BOTH Hindi (हिंदी) and English
- Automatically detect the user's language from their query
- Respond in the SAME language the user used
- If user mixes languages, prioritize their dominant language
- Use Devanagari script for Hindi responses
- Keep farming terminology clear in both languages

Your capabilities:
- Primarily rely on <context> data. If missing, use general agricultural knowledge and clearly state assumptions.
- Remember previous conversations
- Give short, practical, actionable advice
- Consider soil, weather, location, sensor data
- Use ML crop predictions when available
- Mention Indian soil type when giving advice
- Analyze images for diseases, pests, health issues
- If data is missing, say it clearly
- Be friendly and remember user details

When you cite knowledge from retrieved sources, mention the source briefly (e.g., "According to ICAR wheat guide..." or "ICAR गेहूं गाइड के अनुसार...").

When ML predictions are available, explain WHY those crops were recommended based on the soil and weather conditions.

LANGUAGE EXAMPLES:
- Hindi Query: "गेहूं के लिए कौन सी खाद अच्छी है?"
  → Respond in Hindi: "गेहूं के लिए नाइट्रोजन, फॉस्फोरस और पोटाश (NPK) खाद उपयोगी है..."

- English Query: "What fertilizer is good for wheat?"
  → Respond in English: "For wheat, NPK fertilizer is beneficial..."

- Mixed Query: "What is best खाद for wheat?"
  → Respond in English (dominant language)

Answer in clear bullet points unless asked otherwise.
"""

# ========================================
# LANGUAGE DETECTION
# ========================================
def detect_language(text: str) -> str:
    """
    Detect if text is primarily Hindi or English.
    Returns: 'hindi' or 'english'
    """
    # Hindi Unicode range: \u0900-\u097F (Devanagari)
    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_chars = len([c for c in text if c.isalpha()])
    
    if total_chars == 0:
        return 'english'
    
    # If >30% Hindi characters, treat as Hindi
    hindi_ratio = hindi_chars / total_chars if total_chars > 0 else 0
    
    return 'hindi' if hindi_ratio > 0.3 else 'english'


# ========================================
# LANGUAGE-AWARE PROMPT BUILDER
# ========================================
def build_prompt_bilingual(query: str, full_context: dict, has_image: bool = False):
    """Build AI prompt with bilingual awareness"""
    
    detected_lang = detect_language(query)
    
    ctx = json.dumps(full_context, indent=2)
    
    # Language instruction based on detection
    if detected_lang == 'hindi':
        lang_instruction = "\n🌍 IMPORTANT: User is asking in HINDI. Respond in HINDI (हिंदी) using Devanagari script.\n"
    else:
        lang_instruction = "\n🌍 IMPORTANT: User is asking in ENGLISH. Respond in ENGLISH.\n"
    
    if has_image:
        return f"""
{lang_instruction}
<context>
{ctx}
</context>

User uploaded an image and asks:
{query}

Analyze the image carefully and provide detailed insights. If you reference information from the retrieved knowledge, cite the source.

Your answer (in {detected_lang.upper()}):
"""
    else:
        return f"""
{lang_instruction}
<context>
{ctx}
</context>

User Question:
{query}

Your answer (in {detected_lang.upper()}, cite sources when using retrieved knowledge):
"""


def encode_image_to_base64(image_bytes: bytes) -> str:
    """Encode image to base64"""
    return base64.b64encode(image_bytes).decode('utf-8')


async def call_groq(messages: list, model: str):
    """Groq API call"""
    
    if not GROQ_API_KEY:
        raise Exception("GROQ_API_KEY is not set")
        
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.7
                }
            )

            if response.status_code == 200:
                return response.json()
            else:
                error_msg = response.text[:200] if response.text else "No error body"
                logger.error(f"Groq API failed: {response.status_code} - {error_msg}")
                raise Exception(f"Groq API Error: {response.status_code}")

    except Exception as e:
        logger.error(f"Groq API exception: {str(e)}")
        raise


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
    Works with both Hindi and English queries
    """
    if not ML_AVAILABLE or not sensor:
        return False
    
    # Check if required sensor data is available
    required_fields = ["n", "p", "k", "ph"]
    if not all(sensor.get(field) is not None for field in required_fields):
        return False
    
    # Keywords in both English and Hindi
    ml_keywords_en = [
        "crop", "grow", "plant", "recommend", "suggestion",
        "best", "suitable", "optimal", "should i", "what to"
    ]
    
    ml_keywords_hi = [
        "फसल", "उगा", "लगा", "सुझाव", "सिफारिश",
        "अच्छा", "उपयुक्त", "बेहतर", "कौन सा", "क्या"
    ]
    
    query_lower = query.lower()
    
    return (
        any(keyword in query_lower for keyword in ml_keywords_en) or
        any(keyword in query for keyword in ml_keywords_hi)
    )


async def process_ai_query(
    query: str,
    auth_id: str,
    conversation_id: str,
    lat: float = None,
    lon: float = None,
    image: Optional[UploadFile] = None
):
    """Main AI query processor with bilingual support"""
    
    # Detect language
    detected_lang = detect_language(query)
    logger.info(f"Detected language: {detected_lang} for query: {query[:50]}...")
    
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
            # Return error message in detected language
            if detected_lang == 'hindi':
                raise Exception("कोई सेंसर डेटा उपलब्ध नहीं है और कोई स्थान प्रदान नहीं किया गया")
            else:
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
        "language_detected": detected_lang
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

    # ML Crop Prediction (works with both languages)
    ml_prediction_used = False
    
    if should_use_ml_prediction(query, sensor):
        try:
            logger.info("🌾 Triggering ML crop prediction...")
            
            state = location_context.get("location_info", {}).get("state", "bihar")
            if state:
                state = state.lower().replace(" ", "")
            
            season = detect_season_from_date()
            weather = location_context.get("weather", {})
            temperature = weather.get("temperature", 25.0)
            humidity = weather.get("humidity", 70.0)
            rainfall = 100.0
            
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

    # Build messages with bilingual system prompt
    prompt = build_prompt_bilingual(query, full_context, has_image)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_BILINGUAL}
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
    logger.info("Calling Groq API...")
    
    # Use Llama 4 Scout for both text and vision
    model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
    
    ai_response = await call_groq(messages, model_name)
    answer = ai_response["choices"][0]["message"]["content"]
    logger.info("Received AI response")

    # Save conversation
    user_metadata = {
        "coordinates": {"lat": lat, "lon": lon},
        "ml_prediction_used": ml_prediction_used,
        "language": detected_lang
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
            "ml_prediction_used": ml_prediction_used,
            "language": detected_lang
        }
    )

    logger.info(f"Query processed successfully for {auth_id} in {detected_lang}")

    return {
        "answer": answer,
        "context_used": full_context,
        "conversation_id": conversation_id,
        "message_count": len(history) + 2,
        "had_image": has_image,
        "rag_info": rag_info,
        "ml_prediction_used": ml_prediction_used,
        "detected_language": detected_lang
    }