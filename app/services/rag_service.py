# app/services/rag_service.py - Production Ready
import os
import logging
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import httpx

load_dotenv()

logger = logging.getLogger(__name__)

# Environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "farmbot_knowledge")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Validate required environment variables
if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("❌ CRITICAL: QDRANT_URL and QDRANT_API_KEY must be set in environment")

if not TAVILY_API_KEY:
    logger.warning("⚠️ TAVILY_API_KEY not set. Web search will be disabled.")

# Initialize Qdrant client
logger.info(f"Connecting to Qdrant at {QDRANT_URL}...")
try:
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60
    )
    # Test connection
    qdrant_client.get_collection(COLLECTION_NAME)
    logger.info(f"✓ Connected to Qdrant collection: {COLLECTION_NAME}")
except Exception as e:
    logger.error(f"❌ Failed to connect to Qdrant: {e}")
    raise

# Load embedding model
logger.info("Loading embedding model...")
try:
    _model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)
# ✅ Use half precision to save 50% memory
    import torch
    if hasattr(_model, 'half'):
        _model = _model.half()
    
    logger.info("✓ Embedding model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load embedding model: {e}")
    raise


def embed_query(text: str) -> list[float]:
    """Generate query embedding"""
    if not text:
        return []
    
    try:
        vec = _model.encode(text, show_progress_bar=False)
        return vec.tolist()
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return []


def search_knowledge(
    query: str,
    top_k: int = 8,
    similarity_threshold: float = 0.40
) -> List[Dict[str, Any]]:
    """Pure vector search in Qdrant knowledge base - NO FILTERS"""
    logger.info(f"RAG search: '{query[:50]}...' (top_k={top_k}, threshold={similarity_threshold})")
    
    embedding = embed_query(query)
    if not embedding:
        logger.warning("Failed to generate embedding for query")
        return []

    try:
        # Pure vector search - NO FILTERS!
        search_result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=top_k * 2,
            score_threshold=similarity_threshold
        )
        
        logger.info(f"Qdrant returned {len(search_result.points)} points")
        
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
        
        filtered_results = results[:top_k]
        logger.info(f"Returning {len(filtered_results)} RAG results")
        return filtered_results
        
    except Exception as e:
        logger.error(f"RAG search failed: {e}", exc_info=True)
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
        logger.warning("Tavily API key not configured, skipping web search")
        return []
    
    logger.info(f"Tavily search: '{query[:50]}...' (max_results={max_results})")
    
    # Detect query type for better search
    query_lower = query.lower()
    is_location_query = any(word in query_lower for word in ["near", "nearby", "kvk", "location", "find", "where"])
    is_recent_query = any(word in query_lower for word in ["latest", "recent", "new", "current", "today", "2024", "2025"])
    
    # Adjust search parameters
    if is_location_query or is_recent_query:
        search_depth = "advanced"
        domains = None
    else:
        ag_domains = [
            "icar.gov.in",
            "agricoop.gov.in", 
            "kvk.org.in",
        ]
        domains = include_domains or ag_domains
    
    # Exclude PDFs
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
            
            if domains:
                payload["include_domains"] = domains
            
            response = await client.post(
                "https://api.tavily.com/search",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                # Filter out PDFs
                filtered = [
                    r for r in results 
                    if not r.get("url", "").endswith(".pdf")
                ]
                
                logger.info(f"Tavily returned {len(filtered)} results")
                return filtered
            else:
                logger.error(f"Tavily error {response.status_code}: {response.text[:200]}")
                return []
                
    except Exception as e:
        logger.error(f"Tavily search failed: {e}", exc_info=True)
        return []


async def hybrid_search(
    query: str,
    top_k_rag: int = 8,
    top_k_web: int = 5,
    force_web: bool = False
) -> Dict[str, Any]:
    """Combined RAG + Tavily search - NO CROP/REGION FILTERS"""
    
    logger.info(f"Hybrid search: '{query[:50]}...'")
    
    # Phase 1: RAG Search
    rag_results = search_knowledge(query, top_k_rag)
    
    # Phase 2: Decide if web search is needed
    query_lower = query.lower()
    location_keywords = ["near", "nearby", "kvk", "location", "find", "where", "search"]
    time_keywords = ["latest", "recent", "current", "new", "today", "2024", "2025"]
    
    has_location_keyword = any(word in query_lower for word in location_keywords)
    has_time_keyword = any(word in query_lower for word in time_keywords)
    
    needs_web = (
        force_web or
        len(rag_results) < 3 or
        has_time_keyword or
        has_location_keyword
    )
    
    logger.info(f"Web search decision: needs_web={needs_web} (rag_count={len(rag_results)}, time={has_time_keyword}, location={has_location_keyword})")
    
    web_results = []
    if needs_web:
        web_data = await search_tavily(query, top_k_web)
        web_results = web_data
    
    result = {
        "rag_chunks": rag_results,
        "web_results": web_results,
        "rag_count": len(rag_results),
        "web_count": len(web_results),
        "used_web": needs_web,
        "query_enhanced": False,
        "search_decision": {
            "force_web": force_web,
            "rag_count": len(rag_results),
            "has_location_keyword": has_location_keyword,
            "has_time_keyword": has_time_keyword,
            "decided_web": needs_web
        }
    }
    
    logger.info(f"Hybrid search complete: rag={result['rag_count']}, web={result['web_count']}")
    return result


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
        if c.get("similarity"):
            header_parts.append(f"Relevance: {c['similarity']:.2%}")

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