from fastapi import APIRouter, Query, HTTPException
from app.services.video_service import (
    get_videos_by_language,
    get_videos_by_topic,
    get_available_topics,
    get_video_by_id,
)

router = APIRouter(
    prefix="/videos",
    tags=["Educational Videos"]
)


# ===============================
# ROUTES
# ===============================

@router.get("")
def fetch_videos(
    language: str = Query("en", regex="^(en|hi)$"),
    topic: str | None = None
):
    """
    GET /videos
    GET /videos?language=en
    GET /videos?language=hi&topic=soil_health
    """
    if topic:
        return get_videos_by_topic(topic, language)
    return get_videos_by_language(language)


@router.get("/topics")
def fetch_topics(
    language: str = Query("en", regex="^(en|hi)$")
):
    """
    GET /videos/topics?language=en
    """
    return get_available_topics(language)


@router.get("/{video_id}")
def fetch_video_by_id(video_id: str):
    """
    GET /videos/{id}
    """
    video = get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video
