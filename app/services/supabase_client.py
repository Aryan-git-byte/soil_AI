# app/services/enhanced_supabase_client.py
from supabase import create_client, Client
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

class EnhancedSupabaseClient:
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
                "longitude": lng,
                "timestamp": datetime.now().isoformat()
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
                "longitude": lng,
                "timestamp": datetime.now().isoformat()
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
                "status": status,
                "timestamp": datetime.now().isoformat()
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
    
    # FEEDBACK OPERATIONS
    async def store_feedback(self, feedback_record: Dict) -> Dict:
        """Store user feedback for system improvement"""
        try:
            result = self.client.table("user_feedback").insert(feedback_record).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            print(f"Error storing feedback: {e}")
            return {}
    
    async def get_feedback_analytics(self, days: int = 30) -> Dict:
        """Get feedback analytics for system improvement"""
        try:
            cutoff = datetime.now() - timedelta(days=days)
            
            # Get all feedback in the period
            result = self.client.table("user_feedback").select("*")\
                    .gte("timestamp", cutoff.isoformat()).execute()
            
            feedback_data = result.data
            
            if not feedback_data:
                return {"total_feedback": 0, "average_rating": 0}
            
            # Calculate analytics
            total_feedback = len(feedback_data)
            ratings = [f.get("rating", 0) for f in feedback_data if f.get("rating")]
            average_rating = sum(ratings) / len(ratings) if ratings else 0
            
            # Count feedback types
            feedback_types = {}
            for feedback in feedback_data:
                ftype = feedback.get("feedback_type", "unknown")
                feedback_types[ftype] = feedback_types.get(ftype, 0) + 1
            
            return {
                "total_feedback": total_feedback,
                "average_rating": round(average_rating, 2),
                "feedback_distribution": feedback_types,
                "period_days": days
            }
            
        except Exception as e:
            print(f"Error getting feedback analytics: {e}")
            return {}
    
    async def add_to_improvement_queue(self, feedback_record: Dict) -> Dict:
        """Add feedback to improvement queue for model retraining"""
        try:
            improvement_record = {
                "feedback_id": feedback_record.get("id"),
                "priority": "high" if feedback_record.get("feedback_type") == "incorrect" else "medium",
                "status": "pending",
                "feedback_data": feedback_record,
                "timestamp": datetime.now().isoformat()
            }
            
            result = self.client.table("improvement_queue").insert(improvement_record).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            print(f"Error adding to improvement queue: {e}")
            return {}
    
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
            
            # Calculate farm health metrics
            farm_health = self._calculate_farm_health(sensor_data)
            
            return {
                "period_days": days,
                "location": {"latitude": lat, "longitude": lng} if lat and lng else None,
                "sensor_readings": len(sensor_data),
                "manual_entries": len(manual_entries),
                "api_data_points": len(api_data),
                "ai_interactions": len(ai_history),
                "latest_sensor": sensor_data[0] if sensor_data else None,
                "recent_entries": manual_entries[:5],
                "farm_health": farm_health,
                "summary_generated": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error generating farm summary: {e}")
            return {}
    
    # LEARNING AND IMPROVEMENT OPERATIONS
    async def get_learning_data(self, limit: int = 1000) -> Dict:
        """Get data for model training and improvement"""
        try:
            # Get successful interactions with feedback
            successful_interactions = self.client.table("ai_log").select("*")\
                .eq("status", "success").limit(limit//2).execute()
            
            # Get interactions with positive feedback
            positive_feedback = self.client.table("user_feedback").select("*")\
                .in_("feedback_type", ["helpful"]).limit(limit//2).execute()
            
            return {
                "successful_interactions": successful_interactions.data,
                "positive_feedback": positive_feedback.data,
                "total_records": len(successful_interactions.data) + len(positive_feedback.data)
            }
            
        except Exception as e:
            print(f"Error getting learning data: {e}")
            return {}
    
    async def store_model_performance(self, model_version: str, metrics: Dict) -> Dict:
        """Store model performance metrics"""
        try:
            record = {
                "model_version": model_version,
                "performance_metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }
            
            result = self.client.table("model_performance").insert(record).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            print(f"Error storing model performance: {e}")
            return {}
    
    # DATA CLEANUP OPERATIONS
    async def cleanup_old_sensor_data(self, cutoff: datetime) -> int:
        """Clean up old sensor data"""
        try:
            result = self.client.table("sensor_data")\
                .delete().lt("timestamp", cutoff.isoformat()).execute()
            
            return len(result.data) if result.data else 0
        except Exception as e:
            print(f"Error cleaning up sensor data: {e}")
            return 0
    
    async def cleanup_old_api_data(self, cutoff: datetime) -> int:
        """Clean up old API data"""
        try:
            result = self.client.table("api_data")\
                .delete().lt("timestamp", cutoff.isoformat()).execute()
            
            return len(result.data) if result.data else 0
        except Exception as e:
            print(f"Error cleaning up API data: {e}")
            return 0
    
    # PREDICTIVE ANALYTICS
    async def get_predictive_data(self, lat: float = None, lng: float = None, days: int = 30) -> Dict:
        """Get data for predictive analytics"""
        try:
            cutoff = datetime.now() - timedelta(days=days)
            
            # Get sensor trends
            sensor_trends = await self.get_sensor_trends(hours=days*24, lat=lat, lng=lng)
            
            # Get weather history
            weather_data = await self.get_api_data(source="weather_current", hours=days*24)
            
            # Get crop performance data (from manual entries)
            crop_entries = await self.get_manual_entries(days=days)
            crop_performance = [e for e in crop_entries if e.get("entry_type") in ["harvest", "yield", "growth"]]
            
            return {
                "sensor_trends": sensor_trends,
                "weather_history": weather_data,
                "crop_performance": crop_performance,
                "analysis_period": days
            }
            
        except Exception as e:
            print(f"Error getting predictive data: {e}")
            return {}
    
    # Helper methods
    def _calculate_farm_health(self, sensor_data: List[Dict]) -> Dict:
        """Calculate overall farm health from sensor data"""
        if not sensor_data:
            return {"status": "no_data"}
        
        latest = sensor_data[0]
        health_scores = []
        
        # pH health (6.0-7.0 is optimal)
        if "ph" in latest:
            ph = latest["ph"]
            if 6.0 <= ph <= 7.0:
                health_scores.append(100)
            elif 5.5 <= ph < 6.0 or 7.0 < ph <= 7.5:
                health_scores.append(75)
            else:
                health_scores.append(50)
        
        # Moisture health (40-70% is optimal)
        if "soil_moisture" in latest:
            moisture = latest["soil_moisture"]
            if 40 <= moisture <= 70:
                health_scores.append(100)
            elif 30 <= moisture < 40 or 70 < moisture <= 80:
                health_scores.append(75)
            else:
                health_scores.append(50)
        
        # Temperature health (15-25°C is optimal for most crops)
        if "soil_temperature" in latest:
            temp = latest["soil_temperature"]
            if 15 <= temp <= 25:
                health_scores.append(100)
            elif 10 <= temp < 15 or 25 < temp <= 30:
                health_scores.append(75)
            else:
                health_scores.append(50)
        
        overall_health = sum(health_scores) / len(health_scores) if health_scores else 0
        
        return {
            "overall_score": round(overall_health, 1),
            "status": "excellent" if overall_health >= 90 else 
                     "good" if overall_health >= 75 else 
                     "fair" if overall_health >= 60 else "needs_attention",
            "individual_scores": {
                "ph": health_scores[0] if len(health_scores) > 0 else None,
                "moisture": health_scores[1] if len(health_scores) > 1 else None,
                "temperature": health_scores[2] if len(health_scores) > 2 else None
            }
        }