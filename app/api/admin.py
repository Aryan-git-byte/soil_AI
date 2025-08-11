# app/api/admin.py
from fastapi import APIRouter, HTTPException, Depends, Request, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import asyncio

router = APIRouter()

# Admin models
class FeedbackData(BaseModel):
    session_id: str
    interaction_id: str
    feedback_type: str = Field(..., regex="^(helpful|not_helpful|incorrect|request_clarification)$")
    rating: Optional[int] = Field(None, ge=1, le=5)
    comments: Optional[str] = Field(None, max_length=1000)
    suggested_improvement: Optional[str] = Field(None, max_length=2000)

class SystemStats(BaseModel):
    active_sessions: int
    total_interactions: int
    success_rate: float
    avg_response_time: float
    top_topics: List[str]
    user_satisfaction: float

class KnowledgeUpdate(BaseModel):
    doc_id: str
    content: str
    category: str
    priority: str = Field("medium", regex="^(low|medium|high)$")
    crops: List[str] = []
    metadata: Optional[Dict[str, Any]] = None

# Feedback endpoints
@router.post("/feedback")
async def submit_feedback(request: Request, feedback: FeedbackData):
    """Submit user feedback for system improvement"""
    try:
        supabase = request.app.state.supabase
        
        # Store feedback
        feedback_record = {
            "session_id": feedback.session_id,
            "interaction_id": feedback.interaction_id,
            "feedback_type": feedback.feedback_type,
            "rating": feedback.rating,
            "comments": feedback.comments,
            "suggested_improvement": feedback.suggested_improvement,
            "timestamp": datetime.now().isoformat()
        }
        
        result = await supabase.store_feedback(feedback_record)
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to store feedback")
        
        # If feedback indicates improvement needed, trigger learning process
        if feedback.feedback_type in ["not_helpful", "incorrect"]:
            # Add to improvement queue (could trigger retraining later)
            await supabase.add_to_improvement_queue(feedback_record)
        
        return {
            "status": "success",
            "message": "Feedback stored successfully",
            "feedback_id": result.get("id")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission error: {str(e)}")

@router.get("/feedback/analytics")
async def get_feedback_analytics(
    request: Request,
    days: int = Query(30, ge=1, le=90)
):
    """Get feedback analytics for system improvement"""
    try:
        supabase = request.app.state.supabase
        
        analytics = await supabase.get_feedback_analytics(days)
        
        return {
            "status": "success",
            "analytics": analytics,
            "period_days": days,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics error: {str(e)}")

# Knowledge base management
@router.post("/knowledge")
async def add_knowledge(request: Request, knowledge: KnowledgeUpdate):
    """Add or update knowledge base content"""
    try:
        knowledge_base = request.app.state.knowledge_base
        
        metadata = knowledge.metadata or {}
        metadata.update({
            "category": knowledge.category,
            "priority": knowledge.priority,
            "crops": knowledge.crops,
            "last_updated": datetime.now().isoformat()
        })
        
        knowledge_base.add_document(knowledge.doc_id, knowledge.content, metadata)
        
        return {
            "status": "success",
            "message": f"Knowledge document '{knowledge.doc_id}' added/updated",
            "doc_id": knowledge.doc_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Knowledge update error: {str(e)}")

@router.get("/knowledge")
async def get_knowledge_stats(request: Request):
    """Get knowledge base statistics"""
    try:
        knowledge_base = request.app.state.knowledge_base
        stats = knowledge_base.get_stats()
        
        return {
            "status": "success",
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Knowledge stats error: {str(e)}")

@router.delete("/knowledge/{doc_id}")
async def delete_knowledge(request: Request, doc_id: str):
    """Delete knowledge document"""
    try:
        knowledge_base = request.app.state.knowledge_base
        
        success = knowledge_base.delete_document(doc_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "status": "success",
            "message": f"Document '{doc_id}' deleted"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Knowledge deletion error: {str(e)}")

# System monitoring
@router.get("/stats", response_model=SystemStats)
async def get_system_stats(request: Request, days: int = Query(7, ge=1, le=30)):
    """Get comprehensive system statistics"""
    try:
        supabase = request.app.state.supabase
        memory = request.app.state.memory
        
        # Get various statistics
        ai_history = await supabase.get_ai_interaction_history(days=days, limit=1000)
        memory_stats = await memory.get_memory_stats()
        feedback_analytics = await supabase.get_feedback_analytics(days)
        
        # Calculate metrics
        total_interactions = len(ai_history)
        successful_interactions = len([i for i in ai_history if i.get("status") == "success"])
        success_rate = (successful_interactions / total_interactions * 100) if total_interactions > 0 else 0
        
        # Extract top topics
        top_topics = _extract_top_topics(ai_history)
        
        # Calculate user satisfaction from feedback
        user_satisfaction = feedback_analytics.get("average_rating", 0) * 20  # Convert 1-5 to 0-100
        
        return SystemStats(
            active_sessions=memory_stats.get("active_sessions", 0),
            total_interactions=total_interactions,
            success_rate=round(success_rate, 2),
            avg_response_time=2.5,  # Placeholder - could be calculated from logs
            top_topics=top_topics,
            user_satisfaction=round(user_satisfaction, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System stats error: {str(e)}")

# Data management
@router.post("/data/cleanup")
async def cleanup_old_data(request: Request, background_tasks: BackgroundTasks):
    """Clean up old data to maintain performance"""
    try:
        background_tasks.add_task(_cleanup_old_data_task, request.app.state.supabase)
        
        return {
            "status": "success",
            "message": "Data cleanup initiated in background"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup error: {str(e)}")

@router.post("/learning/retrain")
async def trigger_retraining(request: Request, background_tasks: BackgroundTasks):
    """Trigger model retraining based on feedback"""
    try:
        # This would trigger a background retraining process
        background_tasks.add_task(_retrain_model_task, request.app.state)
        
        return {
            "status": "success",
            "message": "Model retraining initiated",
            "note": "This may take several hours to complete"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")

# Export and backup
@router.get("/export/interactions")
async def export_interactions(
    request: Request,
    days: int = Query(30, ge=1, le=90),
    format: str = Query("json", regex="^(json|csv)$")
):
    """Export interaction data for analysis"""
    try:
        supabase = request.app.state.supabase
        
        interactions = await supabase.get_ai_interaction_history(days=days, limit=10000)
        
        if format == "csv":
            csv_data = _convert_interactions_to_csv(interactions)
            return {
                "status": "success",
                "format": "csv",
                "data": csv_data,
                "count": len(interactions)
            }
        else:
            return {
                "status": "success",
                "format": "json",
                "data": interactions,
                "count": len(interactions)
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

# Helper functions
def _extract_top_topics(ai_history: List[Dict]) -> List[str]:
    """Extract most common topics from AI interactions"""
    topic_counts = {}
    agricultural_keywords = [
        "soil", "water", "fertilizer", "crop", "plant", "pest", "disease",
        "weather", "irrigation", "harvest", "seed", "nutrition", "pH"
    ]
    
    for interaction in ai_history:
        query = interaction.get("input_query", "").lower()
        for keyword in agricultural_keywords:
            if keyword in query:
                topic_counts[keyword] = topic_counts.get(keyword, 0) + 1
    
    # Return top 10 topics
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
    return [topic for topic, count in sorted_topics[:10]]

async def _cleanup_old_data_task(supabase):
    """Background task to clean up old data"""
    try:
        # Clean up old sensor data (keep last 90 days)
        cutoff = datetime.now() - timedelta(days=90)
        await supabase.cleanup_old_sensor_data(cutoff)
        
        # Clean up old API data (keep last 30 days)
        api_cutoff = datetime.now() - timedelta(days=30)
        await supabase.cleanup_old_api_data(api_cutoff)
        
        print("Data cleanup completed successfully")
        
    except Exception as e:
        print(f"Data cleanup failed: {e}")

async def _retrain_model_task(app_state):
    """Background task for model retraining"""
    try:
        # This is a placeholder for actual retraining logic
        # In a real implementation, you would:
        # 1. Collect feedback data
        # 2. Prepare training data
        # 3. Fine-tune the model
        # 4. Validate performance
        # 5. Deploy updated model
        
        print("Model retraining initiated...")
        await asyncio.sleep(5)  # Simulate processing time
        print("Model retraining completed")
        
    except Exception as e:
        print(f"Model retraining failed: {e}")

def _convert_interactions_to_csv(interactions: List[Dict]) -> str:
    """Convert interactions to CSV format"""
    if not interactions:
        return ""
    
    headers = ["timestamp", "input_query", "status", "data_sources_used"]
    csv_lines = [",".join(headers)]
    
    for interaction in interactions:
        row = [
            interaction.get("timestamp", ""),
            interaction.get("input_query", "").replace(",", ";"),
            interaction.get("status", ""),
            str(interaction.get("data_sources_used", {})).replace(",", ";")
        ]
        csv_lines.append(",".join(row))
    
    return "\n".join(csv_lines)