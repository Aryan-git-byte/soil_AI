import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
import httpx

load_dotenv()

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("[RAG] Loading MiniLM model...")
_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)


def embed_query(text: str) -> list[float]:
    """Generate query embedding"""
    if not text:
        return []
    vec = _model.encode(text, show_progress_bar=False)
    return vec.tolist()


def search_knowledge(
    query: str,
    top_k: int = 8,
    crop: Optional[str] = None,
    region: Optional[str] = None,
    similarity_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """Vector search in knowledge base"""
    embedding = embed_query(query)
    if not embedding:
        return []

    payload = {
        "query_embedding": embedding,
        "match_count": top_k * 2,  # Get more, filter later
        "filter_crop": crop,
        "filter_region": region,
    }

    try:
        resp = supabase.rpc("match_knowledge_chunks", payload).execute()
        rows = getattr(resp, "data", []) or []
        
        # Filter by similarity threshold
        filtered = [r for r in rows if r.get("similarity", 0) >= similarity_threshold]
        return filtered[:top_k]
    except Exception as e:
        print(f"[RAG ERROR] {e}")
        return []


async def search_tavily(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_domains: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Tavily web search integration"""
    if not TAVILY_API_KEY:
        print("[TAVILY] API key missing")
        return []

    # Agricultural domains priority
    ag_domains = [
        "icar.gov.in",
        "agricoop.gov.in", 
        "kvk.org.in",
        "tnau.ac.in",
        "agritech.tnau.ac.in"
    ]
    
    domains = include_domains or ag_domains

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": search_depth,
                    "max_results": max_results,
                    "include_domains": domains,
                    "include_answer": True
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("results", [])
            else:
                print(f"[TAVILY] Error {response.status_code}")
                return []
                
    except Exception as e:
        print(f"[TAVILY ERROR] {e}")
        return []


async def hybrid_search(
    query: str,
    top_k_rag: int = 5,
    top_k_web: int = 3,
    crop: Optional[str] = None,
    region: Optional[str] = None,
    force_web: bool = False
) -> Dict[str, Any]:
    """Combined RAG + Tavily search"""
    
    # Local RAG search
    rag_results = search_knowledge(query, top_k_rag, crop, region)
    
    # Web search conditions
    needs_web = (
        force_web or
        len(rag_results) < 3 or
        any(word in query.lower() for word in ["latest", "recent", "current", "new", "today"])
    )
    
    web_results = []
    if needs_web:
        web_results = await search_tavily(query, top_k_web)
    
    return {
        "rag_chunks": rag_results,
        "web_results": web_results,
        "rag_count": len(rag_results),
        "web_count": len(web_results),
        "used_web": needs_web
    }


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format RAG chunks for LLM"""
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


def format_web_context(results: List[Dict[str, Any]]) -> str:
    """Format Tavily results for LLM"""
    if not results:
        return ""

    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(
            f"[WEB-{i}] {r.get('title', 'Web Result')}\n"
            f"URL: {r.get('url', 'N/A')}\n"
            f"Content: {r.get('content', '')}\n"
        )

    return "\n\n".join(lines)


def format_hybrid_context(hybrid_results: Dict[str, Any]) -> str:
    """Format combined RAG + Web results"""
    sections = []
    
    # Knowledge base section
    if hybrid_results["rag_chunks"]:
        sections.append("=== KNOWLEDGE BASE ===")
        sections.append(format_context(hybrid_results["rag_chunks"]))
    
    # Web search section
    if hybrid_results["web_results"]:
        sections.append("\n=== RECENT WEB RESULTS ===")
        sections.append(format_web_context(hybrid_results["web_results"]))
    
    return "\n\n".join(sections)