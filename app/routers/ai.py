from fastapi import APIRouter, Query, Header
from app.services.ai_service import process_ai_query
from app.services.conversation_service import ConversationService
import uuid

router = APIRouter(prefix="/api/ai", tags=["AI"])

@router.get("/ask")
async def ai_ask(
    query: str = Query(..., description="Agricultural question or query"),
    auth_id: str = Query(..., description="User authentication ID"),
    conversation_id: str = Query(None, description="Conversation ID (optional, will create new if not provided)"),
    lat: float = Query(None, description="Optional latitude (overrides latest sensor)"),
    lon: float = Query(None, description="Optional longitude (overrides latest sensor)")
):
    """
    GET /api/ai/ask?query=<question>&auth_id=<user_id>&conversation_id=<optional>&lat=<optional>&lon=<optional>
    
    Fetches sensor data + location context, retrieves conversation history,
    then asks Nova 2 Lite for insights with full context awareness.
    
    - auth_id: User identifier (required) - used to track user across conversations
    - conversation_id: Optional - if not provided, creates a new conversation
    - Each conversation_id maintains separate context
    """
    
    # Generate new conversation_id if not provided
    if not conversation_id:
        conversation_id = f"conv_{auth_id}_{uuid.uuid4().hex[:8]}"
    
    return await process_ai_query(
        query=query,
        auth_id=auth_id,
        conversation_id=conversation_id,
        lat=lat,
        lon=lon
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