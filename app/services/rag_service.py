import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import httpx

load_dotenv()

# Environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "farmbot_knowledge")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("Missing QDRANT_URL or QDRANT_API_KEY")

# Initialize Qdrant client
print(f"[RAG] Connecting to Qdrant at {QDRANT_URL}...")
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60
)

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
    """Vector search in Qdrant knowledge base"""
    embedding = embed_query(query)
    if not embedding:
        return []

    try:
        # Build filter conditions
        filter_conditions = []
        
        if crop:
            filter_conditions.append(
                FieldCondition(
                    key="crop",
                    match=MatchValue(value=crop)
                )
            )
        
        if region:
            filter_conditions.append(
                FieldCondition(
                    key="region",
                    match=MatchValue(value=region)
                )
            )
        
        # Create filter object if we have conditions
        query_filter = None
        if filter_conditions:
            query_filter = Filter(must=filter_conditions)
        
        # Search in Qdrant using query_points
        search_result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=top_k * 2,  # Get more, filter later
            query_filter=query_filter,
            score_threshold=similarity_threshold
        )
        
        # Convert Qdrant results to our format
        results = []
        for hit in search_result.points:
            result = {
                "id": hit.id,
                "similarity": hit.score,
                "source": hit.payload.get("source"),
                "crop": hit.payload.get("crop"),
                "region": hit.payload.get("region"),
                "section": hit.payload.get("section"),
                "text": hit.payload.get("text")
            }
            results.append(result)
        
        # Already filtered by score_threshold, just limit to top_k
        return results[:top_k]
        
    except Exception as e:
        print(f"[RAG ERROR] {e}")
        import traceback
        traceback.print_exc()
        return []


async def search_tavily(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Tavily web search integration"""
    if not TAVILY_API_KEY:
        print("[TAVILY] API key missing")
        return []

    # Detect query type for better search
    query_lower = query.lower()
    is_location_query = any(word in query_lower for word in ["near", "nearby", "kvk", "location", "find", "where"])
    is_recent_query = any(word in query_lower for word in ["latest", "recent", "new", "current", "today", "2024", "2025"])
    
    # Adjust search parameters based on query type
    if is_location_query:
        search_depth = "advanced"  # Better for specific searches
        domains = None  # Don't restrict domains for location searches
    elif is_recent_query:
        search_depth = "advanced"
        domains = include_domains
    else:
        # Agricultural domains priority
        ag_domains = [
            "icar.gov.in",
            "agricoop.gov.in", 
            "kvk.org.in",
        ]
        domains = include_domains or ag_domains
    
    # Exclude PDFs and irrelevant results
    pdf_domains = ["*.pdf", "agritech.tnau.ac.in/pdf/*"]
    exclude = exclude_domains or pdf_domains

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": search_depth,
                "max_results": max_results,
                "include_answer": True
            }
            
            # Only add domain filters if specified
            if domains:
                payload["include_domains"] = domains
            
            response = await client.post(
                "https://api.tavily.com/search",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                # Filter out PDF results manually
                filtered = [
                    r for r in results 
                    if not r.get("url", "").endswith(".pdf")
                ]
                
                return filtered
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
    
    # Local RAG search with Qdrant
    rag_results = search_knowledge(query, top_k_rag, crop, region)
    
    # Enhanced web search detection
    query_lower = query.lower()
    location_keywords = ["near", "nearby", "kvk", "location", "find", "where", "search"]
    time_keywords = ["latest", "recent", "current", "new", "today", "2024", "2025"]
    
    needs_web = (
        force_web or
        len(rag_results) < 2 or  # Lowered threshold
        any(word in query_lower for word in time_keywords) or
        any(word in query_lower for word in location_keywords)
    )
    
    web_results = []
    tavily_answer = None
    
    if needs_web:
        # Enhance query for better results
        enhanced_query = query
        if any(word in query_lower for word in location_keywords):
            # Add context for location queries
            if region:
                enhanced_query = f"{query} {region} India"
        
        web_data = await search_tavily(enhanced_query, top_k_web)
        web_results = web_data
    
    return {
        "rag_chunks": rag_results,
        "web_results": web_results,
        "rag_count": len(rag_results),
        "web_count": len(web_results),
        "used_web": needs_web,
        "query_enhanced": needs_web
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