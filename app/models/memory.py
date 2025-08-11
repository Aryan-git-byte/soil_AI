# app/models/memory.py
import redis
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

class ConversationMemory:
    def __init__(self):
        # Redis connection for fast memory storage
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        
        # Sentence transformer for semantic similarity
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Memory configuration
        self.max_conversation_length = 20  # Keep last 20 exchanges
        self.memory_expiry_days = 30       # Expire memories after 30 days
        self.similarity_threshold = 0.7    # Threshold for similar conversations
        
    def _get_session_key(self, session_id: str) -> str:
        """Generate Redis key for session"""
        return f"agri_ai:session:{session_id}"
    
    def _get_context_key(self, session_id: str) -> str:
        """Generate Redis key for session context"""
        return f"agri_ai:context:{session_id}"
    
    def _get_memory_key(self, session_id: str) -> str:
        """Generate Redis key for long-term memory"""
        return f"agri_ai:memory:{session_id}"
    
    async def store_conversation(self, 
                               session_id: str,
                               user_message: str,
                               ai_response: str,
                               context_data: Dict = None) -> bool:
        """Store conversation exchange in memory"""
        try:
            conversation_key = self._get_session_key(session_id)
            
            # Create conversation record
            exchange = {
                "timestamp": datetime.now().isoformat(),
                "user_message": user_message,
                "ai_response": ai_response,
                "context_data": context_data or {},
                "embedding": self.embedder.encode(user_message).tolist()
            }
            
            # Get existing conversation
            existing = self.redis_client.get(conversation_key)
            conversation = json.loads(existing) if existing else []
            
            # Add new exchange
            conversation.append(exchange)
            
            # Keep only recent exchanges
            if len(conversation) > self.max_conversation_length:
                conversation = conversation[-self.max_conversation_length:]
            
            # Store back to Redis
            self.redis_client.setex(
                conversation_key,
                timedelta(days=self.memory_expiry_days),
                json.dumps(conversation)
            )
            
            # Also store in long-term searchable memory
            await self._store_long_term_memory(session_id, exchange)
            
            return True
            
        except Exception as e:
            print(f"Error storing conversation: {e}")
            return False
    
    async def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        try:
            conversation_key = self._get_session_key(session_id)
            existing = self.redis_client.get(conversation_key)
            
            if not existing:
                return []
            
            conversation = json.loads(existing)
            return conversation[-limit:] if conversation else []
            
        except Exception as e:
            print(f"Error getting conversation history: {e}")
            return []
    
    async def store_session_context(self, 
                                  session_id: str,
                                  farm_location: Dict = None,
                                  crop_info: Dict = None,
                                  farmer_preferences: Dict = None) -> bool:
        """Store persistent session context"""
        try:
            context_key = self._get_context_key(session_id)
            
            context = {
                "farm_location": farm_location,
                "crop_info": crop_info,
                "farmer_preferences": farmer_preferences,
                "last_updated": datetime.now().isoformat()
            }
            
            self.redis_client.setex(
                context_key,
                timedelta(days=self.memory_expiry_days),
                json.dumps(context)
            )
            
            return True
            
        except Exception as e:
            print(f"Error storing session context: {e}")
            return False
    
    async def get_session_context(self, session_id: str) -> Dict:
        """Get persistent session context"""
        try:
            context_key = self._get_context_key(session_id)
            existing = self.redis_client.get(context_key)
            
            if existing:
                return json.loads(existing)
            
            return {}
            
        except Exception as e:
            print(f"Error getting session context: {e}")
            return {}
    
    async def _store_long_term_memory(self, session_id: str, exchange: Dict) -> bool:
        """Store exchange in long-term searchable memory"""
        try:
            memory_key = self._get_memory_key(session_id)
            
            # Get existing memory
            existing = self.redis_client.get(memory_key)
            memory = json.loads(existing) if existing else []
            
            # Add new exchange
            memory.append(exchange)
            
            # Keep memory manageable (last 100 exchanges)
            if len(memory) > 100:
                memory = memory[-100:]
            
            # Store back
            self.redis_client.setex(
                memory_key,
                timedelta(days=self.memory_expiry_days * 2),  # Keep longer
                json.dumps(memory)
            )
            
            return True
            
        except Exception as e:
            print(f"Error storing long-term memory: {e}")
            return False
    
    async def search_similar_conversations(self, 
                                         session_id: str,
                                         query: str,
                                         limit: int = 5) -> List[Dict]:
        """Search for similar past conversations using semantic similarity"""
        try:
            memory_key = self._get_memory_key(session_id)
            existing = self.redis_client.get(memory_key)
            
            if not existing:
                return []
            
            memory = json.loads(existing)
            if not memory:
                return []
            
            # Get query embedding
            query_embedding = self.embedder.encode(query)
            
            # Calculate similarities
            similarities = []
            for exchange in memory:
                if "embedding" in exchange:
                    stored_embedding = np.array(exchange["embedding"])
                    similarity = np.dot(query_embedding, stored_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                    )
                    
                    if similarity > self.similarity_threshold:
                        similarities.append({
                            "exchange": exchange,
                            "similarity": float(similarity)
                        })
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return [item["exchange"] for item in similarities[:limit]]
            
        except Exception as e:
            print(f"Error searching similar conversations: {e}")
            return []
    
    async def get_conversation_summary(self, session_id: str) -> Dict:
        """Get a summary of the conversation session"""
        try:
            conversation = await self.get_conversation_history(session_id, limit=50)
            context = await self.get_session_context(session_id)
            
            if not conversation:
                return {"status": "no_history"}
            
            # Analyze conversation patterns
            total_exchanges = len(conversation)
            recent_topics = self._extract_topics(conversation[-10:])  # Last 10 exchanges
            common_questions = self._extract_common_questions(conversation)
            
            return {
                "status": "active",
                "total_exchanges": total_exchanges,
                "session_start": conversation[0]["timestamp"] if conversation else None,
                "last_interaction": conversation[-1]["timestamp"] if conversation else None,
                "recent_topics": recent_topics,
                "common_questions": common_questions,
                "context": context,
                "session_duration_hours": self._calculate_session_duration(conversation)
            }
            
        except Exception as e:
            print(f"Error getting conversation summary: {e}")
            return {"status": "error", "error": str(e)}
    
    def _extract_topics(self, conversations: List[Dict]) -> List[str]:
        """Extract main topics from recent conversations"""
        topics = []
        agricultural_keywords = [
            "soil", "water", "fertilizer", "crop", "plant", "seed", "harvest",
            "pest", "disease", "weather", "irrigation", "nutrition", "pH",
            "nitrogen", "phosphorus", "potassium", "moisture", "temperature"
        ]
        
        for conv in conversations:
            user_msg = conv.get("user_message", "").lower()
            for keyword in agricultural_keywords:
                if keyword in user_msg and keyword not in topics:
                    topics.append(keyword)
        
        return topics[:10]  # Return top 10 topics
    
    def _extract_common_questions(self, conversations: List[Dict]) -> List[str]:
        """Extract common question patterns"""
        questions = []
        for conv in conversations:
            user_msg = conv.get("user_message", "")
            if "?" in user_msg and len(user_msg) > 10:
                # Simplify question for pattern matching
                simplified = user_msg.lower().strip()
                if simplified not in [q.lower() for q in questions]:
                    questions.append(user_msg)
        
        return questions[:5]  # Return top 5 unique questions
    
    def _calculate_session_duration(self, conversations: List[Dict]) -> float:
        """Calculate session duration in hours"""
        if len(conversations) < 2:
            return 0
        
        try:
            start_time = datetime.fromisoformat(conversations[0]["timestamp"])
            end_time = datetime.fromisoformat(conversations[-1]["timestamp"])
            duration = end_time - start_time
            return round(duration.total_seconds() / 3600, 2)  # Convert to hours
        except:
            return 0
    
    async def clear_session(self, session_id: str) -> bool:
        """Clear all session data"""
        try:
            keys_to_delete = [
                self._get_session_key(session_id),
                self._get_context_key(session_id),
                self._get_memory_key(session_id)
            ]
            
            for key in keys_to_delete:
                self.redis_client.delete(key)
            
            return True
            
        except Exception as e:
            print(f"Error clearing session: {e}")
            return False
    
    async def get_memory_stats(self) -> Dict:
        """Get overall memory system statistics"""
        try:
            # Count different types of keys
            session_keys = len(self.redis_client.keys("agri_ai:session:*"))
            context_keys = len(self.redis_client.keys("agri_ai:context:*"))
            memory_keys = len(self.redis_client.keys("agri_ai:memory:*"))
            
            # Get Redis info
            redis_info = self.redis_client.info()
            
            return {
                "active_sessions": session_keys,
                "stored_contexts": context_keys,
                "memory_stores": memory_keys,
                "redis_memory_used": redis_info.get("used_memory_human", "unknown"),
                "total_keys": redis_info.get("db0", {}).get("keys", 0) if "db0" in redis_info else 0
            }
            
        except Exception as e:
            print(f"Error getting memory stats: {e}")
            return {"error": str(e)}