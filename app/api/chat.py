# app/api/chat.py
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime

router = APIRouter()

# Request/Response models
class ChatMessage(BaseModel):
    message: str = Field(..., description="User's message/question")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    location: Optional[Dict[str, float]] = Field(None, description="User's location {lat, lng}")
    farm_context: Optional[Dict[str, Any]] = Field(None, description="Farm-specific context")

class ChatResponse(BaseModel):
    response: str
    session_id: str
    confidence: str
    action_items: List[str]
    data_sources: List[str]
    follow_up_questions: List[str]
    timestamp: str

class SessionInfo(BaseModel):
    session_id: str
    total_messages: int
    session_start: str
    last_interaction: str
    topics: List[str]

@router.post("/ask", response_model=ChatResponse)
async def ask_question(request: Request, message: ChatMessage):
    """Main chat endpoint - ask agricultural questions"""
    try:
        # Get or create session ID
        session_id = message.session_id or str(uuid.uuid4())
        
        # Get services from app state
        ai_provider = request.app.state.ai_provider if hasattr(request.app.state, 'ai_provider') else None
        memory = request.app.state.memory if hasattr(request.app.state, 'memory') else None
        weather_service = request.app.state.weather
        supabase = request.app.state.supabase
        knowledge_base = request.app.state.knowledge_base
        
        # Initialize AI provider if not available
        if not ai_provider:
            from app.models.ai_provider import GroqAIProvider
            ai_provider = GroqAIProvider()
            request.app.state.ai_provider = ai_provider
        
        # Initialize memory if not available
        if not memory:
            from app.models.memory import ConversationMemory
            memory = ConversationMemory()
            request.app.state.memory = memory
        
        # Gather context data
        context_data = await _gather_context_data(
            supabase, weather_service, knowledge_base, memory,
            message.location, session_id, message.farm_context
        )
        
        # Get conversation history for context
        conversation_history = await memory.get_conversation_history(session_id, limit=5)
        
        # Generate AI response
        ai_result = await ai_provider.generate_agricultural_advice(
            user_query=message.message,
            sensor_data=context_data.get("sensor_data"),
            weather_data=context_data.get("weather_data"),
            historical_data=conversation_history,
            farm_context=message.farm_context
        )
        
        if not ai_result["success"]:
            raise HTTPException(status_code=500, detail=ai_result.get("error", "AI processing failed"))
        
        # Store conversation in memory
        await memory.store_conversation(
            session_id=session_id,
            user_message=message.message,
            ai_response=ai_result["response"],
            context_data=context_data
        )
        
        # Log interaction
        await supabase.log_ai_interaction(
            input_query=message.message,
            output_advice=ai_result["response"],
            data_sources=context_data.get("sources", {}),
            status="success"
        )
        
        # Prepare response
        structured_advice = ai_result.get("structured_advice", {})
        
        return ChatResponse(
            response=ai_result["response"],
            session_id=session_id,
            confidence=structured_advice.get("confidence_level", "medium"),
            action_items=structured_advice.get("action_items", []),
            data_sources=list(context_data.get("sources", {}).keys()),
            follow_up_questions=structured_advice.get("follow_up_questions", []),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")

@router.get("/session/{session_id}", response_model=SessionInfo)
async def get_session_info(request: Request, session_id: str):
    """Get information about a chat session"""
    try:
        memory = request.app.state.memory if hasattr(request.app.state, 'memory') else None
        
        if not memory:
            from app.models.memory import ConversationMemory
            memory = ConversationMemory()
            request.app.state.memory = memory
        
        summary = await memory.get_conversation_summary(session_id)
        
        if summary["status"] == "no_history":
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionInfo(
            session_id=session_id,
            total_messages=summary.get("total_exchanges", 0),
            session_start=summary.get("session_start", ""),
            last_interaction=summary.get("last_interaction", ""),
            topics=summary.get("recent_topics", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session info error: {str(e)}")

@router.post("/session/{session_id}/context")
async def update_session_context(request: Request, session_id: str, context: Dict[str, Any]):
    """Update persistent session context"""
    try:
        memory = request.app.state.memory if hasattr(request.app.state, 'memory') else None
        
        if not memory:
            from app.models.memory import ConversationMemory
            memory = ConversationMemory()
            request.app.state.memory = memory
        
        success = await memory.store_session_context(
            session_id=session_id,
            farm_location=context.get("farm_location"),
            crop_info=context.get("crop_info"),
            farmer_preferences=context.get("farmer_preferences")
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update session context")
        
        return {"status": "success", "message": "Session context updated"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context update error: {str(e)}")

@router.delete("/session/{session_id}")
async def clear_session(request: Request, session_id: str):
    """Clear a chat session"""
    try:
        memory = request.app.state.memory if hasattr(request.app.state, 'memory') else None
        
        if not memory:
            from app.models.memory import ConversationMemory
            memory = ConversationMemory()
            request.app.state.memory = memory
        
        success = await memory.clear_session(session_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to clear session")
        
        return {"status": "success", "message": "Session cleared"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session clear error: {str(e)}")

@router.get("/sessions/stats")
async def get_memory_stats(request: Request):
    """Get memory system statistics"""
    try:
        memory = request.app.state.memory if hasattr(request.app.state, 'memory') else None
        
        if not memory:
            from app.models.memory import ConversationMemory
            memory = ConversationMemory()
            request.app.state.memory = memory
        
        stats = await memory.get_memory_stats()
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

@router.post("/quick-advice")
async def get_quick_advice(request: Request, 
                          question: str, 
                          location: Optional[Dict[str, float]] = None):
    """Get quick advice without session persistence"""
    try:
        # Get services
        weather_service = request.app.state.weather
        knowledge_base = request.app.state.knowledge_base
        
        ai_provider = request.app.state.ai_provider if hasattr(request.app.state, 'ai_provider') else None
        if not ai_provider:
            from app.models.ai_provider import GroqAIProvider
            ai_provider = GroqAIProvider()
        
        # Get weather data if location provided
        weather_data = None
        if location:
            weather_data = await weather_service.get_current_weather(
                location["lat"], location["lng"]
            )
        
        # Get relevant knowledge
        knowledge_results = knowledge_base.search(question, top_k=3)
        knowledge_context = "\n\n".join([doc["content"] for doc in knowledge_results])
        
        # Generate quick response
        messages = [
            {
                "role": "system", 
                "content": f"""You are an agricultural advisor. Use this knowledge base information to answer questions:
                
                {knowledge_context}
                
                Provide concise, actionable advice. Keep responses under 200 words."""
            },
            {
                "role": "user",
                "content": f"Weather: {weather_data}\n\nQuestion: {question}"
            }
        ]
        
        result = await ai_provider.generate_response(messages, model_hint="fast")
        
        return {
            "advice": result["response"],
            "confidence": "medium",
            "sources": [doc["doc_id"] for doc in knowledge_results],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick advice error: {str(e)}")

async def _gather_context_data(supabase, weather_service, knowledge_base, memory,
                             location: Dict = None, session_id: str = None, 
                             farm_context: Dict = None) -> Dict[str, Any]:
    """Gather all relevant context data for AI processing"""
    context_data = {"sources": {}}
    
    try:
        # Get sensor data
        if location:
            sensor_data = await supabase.get_latest_sensor_data(
                limit=5, 
                lat=location.get("lat"), 
                lng=location.get("lng")
            )
            if sensor_data:
                context_data["sensor_data"] = sensor_data
                context_data["sources"]["sensor"] = f"{len(sensor_data)} recent readings"
        
        # Get weather data
        if location:
            weather_data = await weather_service.get_current_weather(
                location["lat"], location["lng"]
            )
            if weather_data and "error" not in weather_data:
                context_data["weather_data"] = weather_data
                context_data["sources"]["weather"] = "Current conditions and forecast"
        
        # Get relevant knowledge from RAG system
        if session_id and memory:
            # Get recent conversation to understand context
            recent_conversations = await memory.get_conversation_history(session_id, limit=3)
            if recent_conversations:
                last_query = recent_conversations[-1].get("user_message", "")
                knowledge_results = knowledge_base.get_contextual_knowledge(
                    last_query, 
                    context_data.get("sensor_data"),
                    context_data.get("weather_data")
                )
                if knowledge_results:
                    context_data["knowledge"] = knowledge_results
                    context_data["sources"]["knowledge"] = f"{len(knowledge_results)} relevant topics"
        
        # Get farm summary from database
        if location:
            farm_summary = await supabase.get_farm_summary(
                lat=location.get("lat"),
                lng=location.get("lng"),
                days=7
            )
            if farm_summary:
                context_data["farm_summary"] = farm_summary
                context_data["sources"]["farm_data"] = "7-day farm summary"
        
    except Exception as e:
        print(f"Error gathering context data: {e}")
        # Continue with partial data
    
    return context_data