# app/routers/ai.py
from fastapi import APIRouter, Query, File, UploadFile, Form, HTTPException, status, Depends, BackgroundTasks
from app.services.ai_service import process_ai_query
from app.services.conversation_service import ConversationService
from typing import Optional
import uuid
import re

router = APIRouter(prefix="/api/ai", tags=["AI"])

# Input Validation Helper
def validate_ai_inputs(
    query: str, 
    auth_id: str, 
    lat: Optional[float] = None, 
    lon: Optional[float] = None
):
    """Validate inputs to prevent DoS and Injection"""
    if len(query) > 5000:
        raise HTTPException(status_code=400, detail="Query too long (max 5000 chars)")
        
    if len(auth_id) > 100 or not re.match(r"^[a-zA-Z0-9_-]+$", auth_id):
        raise HTTPException(status_code=400, detail="Invalid auth_id format")
        
    if lat is not None and not (-90 <= lat <= 90):
        raise HTTPException(status_code=400, detail="Invalid latitude")
        
    if lon is not None and not (-180 <= lon <= 180):
        raise HTTPException(status_code=400, detail="Invalid longitude")

@router.post("/ask")
async def ai_ask(
    background_tasks: BackgroundTasks,  # <--- INJECTED HERE
    query: str = Form(..., description="Agricultural question or image description", min_length=1, max_length=5000),
    auth_id: str = Form(..., description="User authentication ID", min_length=1, max_length=100),
    conversation_id: str = Form(None, description="Conversation ID (optional)", max_length=100),
    lat: float = Form(None, description="Optional latitude", ge=-90.0, le=90.0),
    lon: float = Form(None, description="Optional longitude", ge=-180.0, le=180.0),
    image: UploadFile = File(None, description="Optional image for analysis")
):
    # Additional manual validation (just in case)
    validate_ai_inputs(query, auth_id, lat, lon)

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
        image=image,  # Can be None for text-only
        background_tasks=background_tasks # <--- PASSED TO SERVICE
    )


@router.get("/conversations")
async def get_user_conversations(
    auth_id: str = Query(..., description="User authentication ID", min_length=1, max_length=100)
):
    if not re.match(r"^[a-zA-Z0-9_-]+$", auth_id):
        raise HTTPException(status_code=400, detail="Invalid auth_id")

    conversation_service = ConversationService()
    conversations = await conversation_service.get_user_conversations(auth_id)
    
    return {
        "auth_id": auth_id,
        "conversations": conversations,
        "total": len(conversations)
    }


@router.get("/conversation/history")
async def get_conversation_history(
    conversation_id: str = Query(..., description="Conversation ID", max_length=100),
    limit: int = Query(50, description="Number of messages to retrieve", ge=1, le=100)
):
    conversation_service = ConversationService()
    history = await conversation_service.get_conversation_history(conversation_id, limit)
    
    return {
        "conversation_id": conversation_id,
        "messages": history,
        "total": len(history)
    }


@router.delete("/conversation")
async def delete_conversation(
    conversation_id: str = Query(..., description="Conversation ID to delete", max_length=100),
    auth_id: str = Query(..., description="User Auth ID for verification", min_length=1, max_length=100)
):
    """
    Deletes a conversation. 
    Requires auth_id to verify ownership.
    """
    conversation_service = ConversationService()
    
    # Verify ownership inside the service
    success = await conversation_service.delete_conversation_if_owner(conversation_id, auth_id)
    
    if not success:
        raise HTTPException(
            status_code=403, 
            detail="Failed to delete. Conversation not found or you are not the owner."
        )
    
    return {
        "success": True,
        "conversation_id": conversation_id,
        "message": "Conversation deleted"
    }