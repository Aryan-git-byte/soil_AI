# app/services/conversation_service.py
import os
import json
from datetime import datetime
from supabase import create_client
from typing import List, Dict, Optional
from app.core.cache import cache, CACHE_ENABLED

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class ConversationService:
    """
    Manages conversation history per user (auth_id) and per conversation (conversation_id).
    Optimization: Uses Redis List for O(1) retrieval and Supabase for persistence.
    """

    @staticmethod
    async def save_message(
        auth_id: str,
        conversation_id: str,
        role: str,  # 'user' or 'assistant'
        content: str,
        metadata: Optional[Dict] = None
    ):
        """
        Save a message to the conversation history.
        Writes to DB (Persistence) AND Redis (Cache).
        """
        try:
            timestamp = datetime.utcnow().isoformat()
            data = {
                "auth_id": auth_id,
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "metadata": metadata or {},
                "timestamp": timestamp
            }
            
            # 1. Write to Supabase (Source of Truth)
            # This is now running in the background, so latency doesn't block user
            result = supabase.table("conversation_history").insert(data).execute()
            
            # 2. Write to Redis Cache (Speed Layer)
            if CACHE_ENABLED and cache:
                redis_key = f"chat_history:{conversation_id}"
                
                # Push to end of list
                cache.rpush(redis_key, json.dumps(data))
                
                # Trim to keep only last 50 messages to save RAM
                cache.ltrim(redis_key, -50, -1)
                
                # Set expire to 7 days (conversations go stale)
                cache.expire(redis_key, 604800)

            return result.data[0] if result.data else None
            
        except Exception as e:
            print(f"[Conversation] Error saving message: {e}")
            return None

    @staticmethod
    async def get_conversation_history(
        conversation_id: str,
        limit: int = 20
    ) -> List[Dict]:
        """
        Get conversation history for a specific conversation_id.
        Strategy: Check Redis -> If Miss, Check DB & Populate Redis.
        """
        # 1. Try Redis First (<20ms)
        if CACHE_ENABLED and cache:
            try:
                redis_key = f"chat_history:{conversation_id}"
                
                # Get last 'limit' messages
                # lrange indices are inclusive. -limit to -1 gets the last N.
                cached_msgs = cache.lrange(redis_key, -limit, -1)
                
                if cached_msgs:
                    return [json.loads(msg) for msg in cached_msgs]
            except Exception as e:
                print(f"[Conversation] Redis read error: {e}")
        
        # 2. Fallback to Supabase (1s+)
        try:
            result = supabase.table("conversation_history") \
                .select("*") \
                .eq("conversation_id", conversation_id) \
                .order("timestamp", desc=False) \
                .execute() # Fetch all to cache them correctly
            
            messages = result.data if result.data else []
            
            # 3. Populate Redis for next time
            if CACHE_ENABLED and cache and messages:
                redis_key = f"chat_history:{conversation_id}"
                
                # Delete old key to ensure clean slate
                cache.delete(redis_key)
                
                # Push all messages
                # json.dumps ensures we store strings
                json_msgs = [json.dumps(m) for m in messages]
                cache.rpush(redis_key, *json_msgs)
                
                # Set Expiry
                cache.expire(redis_key, 604800)
                
                # Return only requested limit
                return messages[-limit:]
            
            return messages[-limit:]
            
        except Exception as e:
            print(f"[Conversation] Error fetching history: {e}")
            return []

    @staticmethod
    async def get_user_conversations(auth_id: str) -> List[Dict]:
        """
        Get all conversations for a user (grouped by conversation_id).
        """
        try:
            result = supabase.table("conversation_history") \
                .select("conversation_id, timestamp") \
                .eq("auth_id", auth_id) \
                .order("timestamp", desc=True) \
                .execute()
            
            if not result.data:
                return []
            
            # Group by conversation_id and get first message timestamp
            conversations = {}
            for msg in result.data:
                conv_id = msg["conversation_id"]
                if conv_id not in conversations:
                    conversations[conv_id] = {
                        "conversation_id": conv_id,
                        "last_message": msg["timestamp"]
                    }
            
            return list(conversations.values())
        except Exception as e:
            print(f"[Conversation] Error fetching user conversations: {e}")
            return []

    @staticmethod
    async def delete_conversation_if_owner(conversation_id: str, auth_id: str):
        """
        Delete all messages in a conversation ONLY IF the auth_id matches the owner.
        """
        try:
            # 1. Check ownership
            check_result = supabase.table("conversation_history") \
                .select("auth_id") \
                .eq("conversation_id", conversation_id) \
                .limit(1) \
                .execute()
            
            if not check_result.data:
                return False 
                
            if check_result.data[0]["auth_id"] != auth_id:
                print(f"[Security] Unauthorized delete attempt by {auth_id} on {conversation_id}")
                return False 

            # 2. Proceed with DB deletion
            result = supabase.table("conversation_history") \
                .delete() \
                .eq("conversation_id", conversation_id) \
                .execute()
            
            # 3. Clear Redis Cache
            if CACHE_ENABLED and cache:
                 cache.delete(f"chat_history:{conversation_id}")

            return True
        except Exception as e:
            print(f"[Conversation] Error deleting conversation: {e}")
            return False

    @staticmethod
    def format_history_for_ai(history: List[Dict]) -> List[Dict]:
        """
        Format conversation history for AI consumption.
        """
        return [
            {
                "role": msg["role"],
                "content": msg["content"]
            }
            for msg in history
        ]