import os
from typing import List, Dict, Optional
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ===============================
# VIDEO SERVICE (DB ACCESS LAYERS)
# ===============================

def get_videos_by_language(language: str) -> List[Dict]:
    resp = (
        supabase.table("educational_videos")
        .select("*")
        .eq("language", language)
        .eq("is_active", True)
        .order("order_index", desc=False)
        .execute()
    )
    return resp.data or []


def get_videos_by_topic(topic: str, language: str) -> List[Dict]:
    resp = (
        supabase.table("educational_videos")
        .select("*")
        .eq("topic", topic)
        .eq("language", language)
        .eq("is_active", True)
        .order("order_index", desc=False)
        .execute()
    )
    return resp.data or []


def get_available_topics(language: str) -> List[str]:
    resp = (
        supabase.table("educational_videos")
        .select("topic")
        .eq("language", language)
        .eq("is_active", True)
        .execute()
    )

    topics = {row["topic"] for row in (resp.data or [])}
    return list(topics)


def get_video_by_id(video_id: str) -> Optional[Dict]:
    resp = (
        supabase.table("educational_videos")
        .select("*")
        .eq("id", video_id)
        .eq("is_active", True)
        .single()
        .execute()
    )
    return resp.data
