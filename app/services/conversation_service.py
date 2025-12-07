# app/services/conversation_service.py
import os
from datetime import datetime
from supabase import create_client
from typing import List, Dict, Optional

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


class ConversationService:
    """
    Manages conversation history per user (auth_id) and per conversation (conversation_id).
    Each conversation_id has its own isolated context.
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
        """
        try:
            data = {
                "auth_id": auth_id,
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            result = supabase.table("conversation_history").insert(data).execute()
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
        Returns messages in chronological order (oldest first).
        """
        try:
            result = supabase.table("conversation_history") \
                .select("*") \
                .eq("conversation_id", conversation_id) \
                .order("timestamp", desc=False) \
                .limit(limit) \
                .execute()
            
            return result.data if result.data else []
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
    async def delete_conversation(conversation_id: str):
        """
        Delete all messages in a conversation.
        """
        try:
            result = supabase.table("conversation_history") \
                .delete() \
                .eq("conversation_id", conversation_id) \
                .execute()
            return True
        except Exception as e:
            print(f"[Conversation] Error deleting conversation: {e}")
            return False

    @staticmethod
    def format_history_for_ai(history: List[Dict]) -> List[Dict]:
        """
        Format conversation history for AI consumption.
        Converts database format to OpenAI/OpenRouter message format.
        """
        return [
            {
                "role": msg["role"],
                "content": msg["content"]
            }
            for msg in history
        ]