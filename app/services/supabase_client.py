# app/services/supabase_client.py
from supabase import create_client, Client
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

class SupabaseClient:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_ANON_KEY")
        
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment variables")
        
        self.client: Client = create_client(self.url, self.key)
        
    async def health_check(self) -> bool:
        """Check if Supabase connection is healthy"""
        try:
            result = self.client.table("sensor_data").select("id").limit(1).execute()
            return True
        except Exception as e:
            raise Exception(f"Supabase connection failed: {str(e)}")
    
    # SENSOR DATA OPERATIONS
    async def get_latest_sensor_data(self, limit: int = 10, lat: float = None, lng: float = None) -> List[Dict]:
        """Get latest sensor readings, optionally filtered by location"""
        try:
            query = self.client.table("sensor_data").select("*").order("timestamp", desc=True).limit(limit)
            
            # If location provided, filter nearby (within ~1km)
            if lat and lng:
                query = query.gte("latitude", lat - 0.01).lte("latitude", lat + 0.01)\
                           .gte("longitude", lng - 0.01).lte("longitude", lng + 0.01)
            
            result = query.execute()
            return result.data
        except Exception as e:
            print(f"Error fetching sensor data: {e}")
            return []
    
    async def insert_sensor_data(self, data: Dict[str, Any]) -> Dict:
        """Insert new sensor reading"""
        try:
            result = self.client.table("sensor_data").insert(data).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            print(f"Error inserting sensor data: {e}")
            return {}
    
    async def get_sensor_trends(self, hours: int = 24, lat: float = None, lng: float = None) -> List[Dict]:
        """Get sensor data trends over time"""
        try:
            cutoff = datetime.now() - timedelta(hours=hours)
            
            query = self.client.table("sensor_data").select("*")\
                   .gte("timestamp", cutoff.isoformat())\
                   .order("timestamp", desc=False)
            
            if lat and lng:
                query = query.gte("latitude", lat - 0.01).lte("latitude", lat + 0.01)\
                           .gte("longitude", lng - 0.01).lte("longitude", lng + 0.01)
            
            result = query.execute()
            return result.data
        except Exception as e:
            print(f"Error fetching sensor trends: {e}")
            return []
    
    # API DATA OPERATIONS  
    async def store_api_data(self, source: str, data: Dict, lat: float = None, lng: float = None) -> Dict:
        """Store data from external APIs (weather, soil, etc.)"""
        try:
            record = {
                "api_source": source,
                "data": data,
                "latitude": lat,
                "longitude": lng
            }
            result = self.client.table("api_data").insert(record).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            print(f"Error storing API data: {e}")
            return {}
    
    async def get_api_data(self, source: str = None, hours: int = 24) -> List[Dict]:
        """Get API data, optionally filtered by source and time"""
        try:
            cutoff = datetime.now() - timedelta(hours=hours)
            
            query = self.client.table("api_data").select("*")\
                   .gte("timestamp", cutoff.isoformat())\
                   .order("timestamp", desc=True)
            
            if source:
                query = query.eq("api_source", source)
            
            result = query.execute()
            return result.data
        except Exception as e:
            print(f"Error fetching API data: {e}")
            return []
    
    # MANUAL ENTRY OPERATIONS
    async def insert_manual_entry(self, entry_type: str, title: str, description: str, 
                                data: Dict = None, lat: float = None, lng: float = None) -> Dict:
        """Insert manual farm entry"""
        try:
            record = {
                "entry_type": entry_type,
                "title": title, 
                "description": description,
                "data": data or {},
                "latitude": lat,
                "longitude": lng
            }
            result = self.client.table("manual_entry").insert(record).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            print(f"Error inserting manual entry: {e}")
            return {}
    
    async def get_manual_entries(self, entry_type: str = None, days: int = 30) -> List[Dict]:
        """Get manual entries, optionally filtered by type"""
        try:
            cutoff = datetime.now() - timedelta(days=days)
            
            query = self.client.table("manual_entry").select("*")\
                   .gte("timestamp", cutoff.isoformat())\
                   .order("timestamp", desc=True)
            
            if entry_type:
                query = query.eq("entry_type", entry_type)
            
            result = query.execute()
            return result.data
        except Exception as e:
            print(f"Error fetching manual entries: {e}")
            return []
    
    # AI LOG OPERATIONS
    async def log_ai_interaction(self, input_query: str, output_advice: str, 
                               data_sources: Dict, status: str = "success") -> Dict:
        """Log AI interaction for learning and improvement"""
        try:
            record = {
                "input_query": input_query,
                "output_advice": output_advice,
                "data_sources_used": data_sources,
                "status": status
            }
            result = self.client.table("ai_log").insert(record).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            print(f"Error logging AI interaction: {e}")
            return {}
    
    async def get_ai_interaction_history(self, days: int = 7, limit: int = 50) -> List[Dict]:
        """Get recent AI interactions for context and learning"""
        try:
            cutoff = datetime.now() - timedelta(days=days)
            
            result = self.client.table("ai_log").select("*")\
                    .gte("timestamp", cutoff.isoformat())\
                    .order("timestamp", desc=True)\
                    .limit(limit).execute()
            
            return result.data
        except Exception as e:
            print(f"Error fetching AI history: {e}")
            return []
    
    # ANALYTICS OPERATIONS
    async def get_farm_summary(self, lat: float = None, lng: float = None, days: int = 7) -> Dict:
        """Get comprehensive farm data summary"""
        try:
            cutoff = datetime.now() - timedelta(days=days)
            
            # Get recent sensor data
            sensor_data = await self.get_latest_sensor_data(limit=100, lat=lat, lng=lng)
            
            # Get manual entries
            manual_entries = await self.get_manual_entries(days=days)
            
            # Get API data (weather, etc.)
            api_data = await self.get_api_data(hours=days*24)
            
            # Get AI interactions
            ai_history = await self.get_ai_interaction_history(days=days)
            
            return {
                "period_days": days,
                "location": {"latitude": lat, "longitude": lng} if lat and lng else None,
                "sensor_readings": len(sensor_data),
                "manual_entries": len(manual_entries),
                "api_data_points": len(api_data),
                "ai_interactions": len(ai_history),
                "latest_sensor": sensor_data[0] if sensor_data else None,
                "recent_entries": manual_entries[:5],
                "summary_generated": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error generating farm summary: {e}")
            return {}