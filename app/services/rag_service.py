import os
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # use anon key for runtime

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load same model as embedding pipeline
print("[RAG] Loading MiniLM model for query embeddings...")
_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)


def embed_query(text: str) -> list[float]:
    """Embed user query with the same MiniLM model."""
    if not text:
        return []
    vec = _model.encode(text, show_progress_bar=False)
    return vec.tolist()


def search_knowledge(
    query: str,
    top_k: int = 8,
    crop: Optional[str] = None,
    region: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Vector search in knowledge_chunks via Supabase RPC."""
    embedding = embed_query(query)
    if not embedding:
        return []

    payload = {
        "query_embedding": embedding,
        "match_count": top_k,
        "filter_crop": crop,
        "filter_region": region,
    }

    resp = supabase.rpc("match_knowledge_chunks", payload).execute()
    # supabase-py returns .data as list of rows
    rows = getattr(resp, "data", []) or []
    return rows


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks as a context block for the LLM."""
    if not chunks:
        return ""

    lines = []
    for i, c in enumerate(chunks, start=1):
        header_parts = []
        if c.get("crop"):
            header_parts.append(f"Crop: {c['crop']}")
        if c.get("region"):
            header_parts.append(f"Region: {c['region']}")
        if c.get("source"):
            header_parts.append(f"Source: {c['source']}")

        header = " | ".join(header_parts) if header_parts else "Source chunk"

        lines.append(
            f"[{i}] {header}\n"
            f"Section: {c.get('section', '')}\n"
            f"Text: {c.get('text', '')}\n"
        )

    return "\n\n".join(lines)
