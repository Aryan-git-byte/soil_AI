from fastapi import APIRouter, Query
from app.services.ai_service import process_ai_query

router = APIRouter(prefix="/api/ai", tags=["AI"])

@router.get("/ask")
async def ai_ask(
    query: str = Query(..., description="Agricultural question or query"),
    lat: float = Query(None, description="Optional latitude (overrides latest sensor)"),
    lon: float = Query(None, description="Optional longitude (overrides latest sensor)")
):
    """
    GET /api/ai/ask?query=<question>&lat=<optional>&lon=<optional>
    
    Fetches sensor data + location context, then asks Nova 2 Lite for insights.
    If lat/lon not provided, uses latest sensor location from DB.
    """
    return await process_ai_query(query, lat=lat, lon=lon)
