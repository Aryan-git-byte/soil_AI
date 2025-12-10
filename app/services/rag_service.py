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
print(f"[RAG INIT] Connecting to Qdrant at {QDRANT_URL}...")
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60
)

print("[RAG INIT] Loading MiniLM model...")
_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)


def embed_query(text: str) -> list[float]:
    """Generate query embedding"""
    if not text:
        return []
    print(f"[RAG EMBED] Generating embedding for: '{text[:50]}...'")
    vec = _model.encode(text, show_progress_bar=False)
    print(f"[RAG EMBED] ✓ Generated {len(vec)}-dim vector")
    return vec.tolist()


def search_knowledge(
    query: str,
    top_k: int = 8,
    crop: Optional[str] = None,
    region: Optional[str] = None,
    similarity_threshold: float = 0.3  # LOWERED from 0.5
) -> List[Dict[str, Any]]:
    """Vector search in Qdrant knowledge base with detailed logging"""
    print("\n" + "="*80)
    print(f"[RAG SEARCH] Starting vector search")
    print(f"[RAG SEARCH] Query: '{query}'")
    print(f"[RAG SEARCH] Top K: {top_k}")
    print(f"[RAG SEARCH] Similarity threshold: {similarity_threshold}")
    print(f"[RAG SEARCH] Crop filter: {crop or 'None'}")
    print(f"[RAG SEARCH] Region filter: {region or 'None'}")
    print("="*80)
    
    embedding = embed_query(query)
    if not embedding:
        print("[RAG SEARCH] ✗ Failed to generate embedding")
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
            print(f"[RAG FILTER] Added crop filter: {crop}")
        
        if region:
            filter_conditions.append(
                FieldCondition(
                    key="region",
                    match=MatchValue(value=region)
                )
            )
            print(f"[RAG FILTER] Added region filter: {region}")
        
        # Create filter object if we have conditions
        query_filter = None
        if filter_conditions:
            query_filter = Filter(must=filter_conditions)
            print(f"[RAG FILTER] Applied {len(filter_conditions)} filter(s)")
        else:
            print("[RAG FILTER] No filters applied (searching all documents)")
        
        print(f"[RAG SEARCH] Querying Qdrant collection: {COLLECTION_NAME}")
        
        # Search in Qdrant using query_points
        search_result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=top_k * 2,  # Get more, filter later
            query_filter=query_filter,
            score_threshold=similarity_threshold
        )
        
        print(f"[RAG SEARCH] ✓ Qdrant returned {len(search_result.points)} points")
        
        # Convert Qdrant results to our format
        results = []
        for idx, hit in enumerate(search_result.points):
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
            
            # Detailed logging for each result
            print(f"\n[RAG RESULT #{idx+1}]")
            print(f"  Similarity: {hit.score:.4f}")
            print(f"  Source: {result['source']}")
            print(f"  Crop: {result['crop']}")
            print(f"  Region: {result['region']}")
            print(f"  Section: {result['section']}")
            print(f"  Text preview: {result['text'][:100]}...")
        
        # Filter by threshold and limit
        filtered_results = results[:top_k]
        
        print(f"\n[RAG SEARCH] ✓ Returning {len(filtered_results)} results (after filtering)")
        print("="*80 + "\n")
        
        return filtered_results
        
    except Exception as e:
        print(f"[RAG ERROR] ✗ Search failed: {e}")
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
    """Tavily web search integration with detailed logging"""
    print("\n" + "="*80)
    print(f"[TAVILY SEARCH] Starting web search")
    print(f"[TAVILY SEARCH] Query: '{query}'")
    print(f"[TAVILY SEARCH] Max results: {max_results}")
    print(f"[TAVILY SEARCH] Search depth: {search_depth}")
    print("="*80)
    
    if not TAVILY_API_KEY:
        print("[TAVILY] ✗ API key missing")
        return []

    # Detect query type for better search
    query_lower = query.lower()
    is_location_query = any(word in query_lower for word in ["near", "nearby", "kvk", "location", "find", "where"])
    is_recent_query = any(word in query_lower for word in ["latest", "recent", "new", "current", "today", "2024", "2025"])
    
    print(f"[TAVILY] Location query: {is_location_query}")
    print(f"[TAVILY] Recent query: {is_recent_query}")
    
    # Adjust search parameters based on query type
    if is_location_query:
        search_depth = "advanced"
        domains = None
        print("[TAVILY] Using advanced search for location query")
    elif is_recent_query:
        search_depth = "advanced"
        domains = include_domains
        print("[TAVILY] Using advanced search for recent query")
    else:
        ag_domains = [
            "icar.gov.in",
            "agricoop.gov.in", 
            "kvk.org.in",
        ]
        domains = include_domains or ag_domains
        print(f"[TAVILY] Using agricultural domains: {domains}")
    
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
            
            print(f"[TAVILY] Sending request to Tavily API...")
            
            response = await client.post(
                "https://api.tavily.com/search",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                print(f"[TAVILY] ✓ Received {len(results)} results")
                
                # Filter out PDFs
                filtered = [
                    r for r in results 
                    if not r.get("url", "").endswith(".pdf")
                ]
                
                print(f"[TAVILY] ✓ After PDF filter: {len(filtered)} results")
                
                # Log each result
                for idx, result in enumerate(filtered):
                    print(f"\n[TAVILY RESULT #{idx+1}]")
                    print(f"  Title: {result.get('title', 'N/A')}")
                    print(f"  URL: {result.get('url', 'N/A')}")
                    print(f"  Content preview: {result.get('content', '')[:100]}...")
                
                print("="*80 + "\n")
                return filtered
            else:
                print(f"[TAVILY] ✗ Error {response.status_code}")
                print(f"[TAVILY] Response: {response.text[:200]}")
                return []
                
    except Exception as e:
        print(f"[TAVILY ERROR] ✗ {e}")
        import traceback
        traceback.print_exc()
        return []


async def hybrid_search(
    query: str,
    top_k_rag: int = 5,
    top_k_web: int = 3,
    crop: Optional[str] = None,
    region: Optional[str] = None,
    force_web: bool = False
) -> Dict[str, Any]:
    """Combined RAG + Tavily search with detailed decision logging"""
    
    print("\n" + "#"*80)
    print(f"[HYBRID SEARCH] Starting hybrid search pipeline")
    print(f"[HYBRID SEARCH] Query: '{query}'")
    print(f"[HYBRID SEARCH] RAG top_k: {top_k_rag}, Web top_k: {top_k_web}")
    print(f"[HYBRID SEARCH] Force web: {force_web}")
    print("#"*80 + "\n")
    
    # Local RAG search with Qdrant
    print("[HYBRID] Phase 1: RAG Search")
    rag_results = search_knowledge(query, top_k_rag, crop, region)
    
    # Enhanced web search detection
    query_lower = query.lower()
    location_keywords = ["near", "nearby", "kvk", "location", "find", "where", "search"]
    time_keywords = ["latest", "recent", "current", "new", "today", "2024", "2025"]
    
    has_location_keyword = any(word in query_lower for word in location_keywords)
    has_time_keyword = any(word in query_lower for word in time_keywords)
    
    print(f"\n[HYBRID DECISION] RAG results count: {len(rag_results)}")
    print(f"[HYBRID DECISION] Force web: {force_web}")
    print(f"[HYBRID DECISION] Has location keyword: {has_location_keyword}")
    print(f"[HYBRID DECISION] Has time keyword: {has_time_keyword}")
    print(f"[HYBRID DECISION] RAG results below threshold: {len(rag_results) < 3}")
    
    needs_web = (
        force_web or
        len(rag_results) < 3 or  # Increased from 2
        has_time_keyword or
        has_location_keyword
    )
    
    print(f"[HYBRID DECISION] → Web search needed: {needs_web}")
    
    web_results = []
    
    if needs_web:
        print("\n[HYBRID] Phase 2: Web Search")
        
        # Enhance query for better results
        enhanced_query = query
        if has_location_keyword and region:
            enhanced_query = f"{query} {region} India"
            print(f"[HYBRID] Enhanced query: '{enhanced_query}'")
        
        web_data = await search_tavily(enhanced_query, top_k_web)
        web_results = web_data
    else:
        print("\n[HYBRID] Phase 2: Skipping web search (RAG sufficient)")
    
    result = {
        "rag_chunks": rag_results,
        "web_results": web_results,
        "rag_count": len(rag_results),
        "web_count": len(web_results),
        "used_web": needs_web,
        "query_enhanced": needs_web,
        "search_decision": {
            "force_web": force_web,
            "rag_count": len(rag_results),
            "has_location_keyword": has_location_keyword,
            "has_time_keyword": has_time_keyword,
            "decided_web": needs_web
        }
    }
    
    print(f"\n[HYBRID COMPLETE] Final stats:")
    print(f"  RAG chunks: {result['rag_count']}")
    print(f"  Web results: {result['web_count']}")
    print(f"  Used web: {result['used_web']}")
    print("#"*80 + "\n")
    
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