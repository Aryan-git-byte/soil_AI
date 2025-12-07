# app/routers/ai.py
from fastapi import APIRouter, Query, File, UploadFile, Form
from app.services.ai_service import process_ai_query
from app.services.conversation_service import ConversationService
from typing import Optional
import uuid

router = APIRouter(prefix="/api/ai", tags=["AI"])


@router.post("/ask")
async def ai_ask(
    query: str = Form(..., description="Agricultural question or image description"),
    auth_id: str = Form(..., description="User authentication ID"),
    conversation_id: str = Form(None, description="Conversation ID (optional)"),
    lat: float = Form(None, description="Optional latitude (overrides latest sensor)"),
    lon: float = Form(None, description="Optional longitude (overrides latest sensor)"),
    image: UploadFile = File(None, description="Optional image for analysis")
):
    """
    🌟 UNIFIED ENDPOINT: Handles both text-only and image+text queries
    
    POST /api/ai/ask (multipart/form-data)
    
    Features:
    - Text-only questions: Send query without image
    - Image analysis: Send query + image file
    - Mixed mode: "What's wrong with this plant?" + image
    - Conversation memory: Maintains full chat history
    - Context awareness: Includes sensor, weather, soil, location
    
    Example Usage:
    
    Text-only:
        FormData = { query: "Best crops for clay soil?", auth_id: "user123" }
    
    With image:
        FormData = { 
            query: "Identify this disease", 
            auth_id: "user123",
            image: [file]
        }
    
    Follow-up:
        FormData = { 
            query: "How do I treat it?", 
            auth_id: "user123",
            conversation_id: "conv_abc123"  // Same conversation
        }
    """
    
    # Generate new conversation_id if not provided
    if not conversation_id:
        conversation_id = f"conv_{auth_id}_{uuid.uuid4().hex[:8]}"
    
    # Pass everything to unified service
    return await process_ai_query(
        query=query,
        auth_id=auth_id,
        conversation_id=conversation_id,
        lat=lat,
        lon=lon,
        image=image  # Can be None for text-only
    )


@router.get("/conversations")
async def get_user_conversations(
    auth_id: str = Query(..., description="User authentication ID")
):
    """
    GET /api/ai/conversations?auth_id=<user_id>
    
    Get all conversations for a specific user.
    """
    conversation_service = ConversationService()
    conversations = await conversation_service.get_user_conversations(auth_id)
    
    return {
        "auth_id": auth_id,
        "conversations": conversations,
        "total": len(conversations)
    }


@router.get("/conversation/history")
async def get_conversation_history(
    conversation_id: str = Query(..., description="Conversation ID"),
    limit: int = Query(50, description="Number of messages to retrieve")
):
    """
    GET /api/ai/conversation/history?conversation_id=<conv_id>&limit=50
    
    Get full message history for a specific conversation.
    """
    conversation_service = ConversationService()
    history = await conversation_service.get_conversation_history(conversation_id, limit)
    
    return {
        "conversation_id": conversation_id,
        "messages": history,
        "total": len(history)
    }


@router.delete("/conversation")
async def delete_conversation(
    conversation_id: str = Query(..., description="Conversation ID to delete")
):
    """
    DELETE /api/ai/conversation?conversation_id=<conv_id>
    
    Delete a conversation and all its messages.
    """
    conversation_service = ConversationService()
    success = await conversation_service.delete_conversation(conversation_id)
    
    return {
        "success": success,
        "conversation_id": conversation_id,
        "message": "Conversation deleted" if success else "Failed to delete conversation"
    }