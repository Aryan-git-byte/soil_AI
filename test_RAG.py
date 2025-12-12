# RAG Diagnostic Script - Fixed Version
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

print("="*80)
print("🔍 FarmBot RAG Pipeline Diagnostic")
print("="*80)

# Check environment variables
print("\n[1] Checking Environment Variables...")
print("-" * 80)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "farmbot_knowledge")

if QDRANT_URL:
    print(f"✓ QDRANT_URL: {QDRANT_URL}")
else:
    print("✗ QDRANT_URL: NOT SET")

if QDRANT_API_KEY:
    print(f"✓ QDRANT_API_KEY: {'*' * 20}{QDRANT_API_KEY[-4:]}")
else:
    print("✗ QDRANT_API_KEY: NOT SET")

if TAVILY_API_KEY:
    print(f"✓ TAVILY_API_KEY: {'*' * 20}{TAVILY_API_KEY[-4:]}")
else:
    print("✗ TAVILY_API_KEY: NOT SET")

print(f"✓ COLLECTION_NAME: {COLLECTION_NAME}")

# Check Qdrant connection
print("\n[2] Testing Qdrant Connection...")
print("-" * 80)

try:
    from qdrant_client import QdrantClient
    
    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60
    )
    
    # ✅ FIXED: Get collection info correctly
    collection_info = qdrant.get_collection(COLLECTION_NAME)
    points_count = collection_info.points_count
    
    print(f"✓ Connected to Qdrant successfully!")
    print(f"✓ Collection: {COLLECTION_NAME}")
    print(f"✓ Total points: {points_count:,}")
    
    if points_count == 0:
        print("⚠️  WARNING: Collection is empty! No knowledge chunks found.")
        print("   → Run the knowledge pipeline scripts to populate the database")
    
except Exception as e:
    print(f"✗ Failed to connect to Qdrant: {e}")
    print("   → Check your QDRANT_URL and QDRANT_API_KEY")
    import traceback
    traceback.print_exc()

# Check embedding model
print("\n[3] Testing Embedding Model...")
print("-" * 80)

try:
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )
    
    test_text = "wheat fertilizer recommendations"
    embedding = model.encode(test_text)
    
    print(f"✓ Embedding model loaded successfully")
    print(f"✓ Model: all-MiniLM-L6-v2")
    print(f"✓ Embedding dimension: {len(embedding)}")
    print(f"✓ Test embedding for '{test_text}': {embedding[:5]}...")
    
except Exception as e:
    print(f"✗ Failed to load embedding model: {e}")

# Test RAG search
print("\n[4] Testing RAG Search...")
print("-" * 80)

try:
    from app.services.rag_service import search_knowledge
    
    test_queries = [
        "wheat fertilizer recommendations",
        "rice pest management",
        "soil preparation Bihar"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = search_knowledge(query, top_k=3, similarity_threshold=0.3)
        
        if results:
            print(f"✓ Found {len(results)} results")
            for i, result in enumerate(results, 1):
                sim = result.get('similarity', 0)
                src = result.get('source', 'Unknown')
                print(f"  [{i}] Similarity: {sim:.4f} | Source: {src}")
        else:
            print(f"⚠️  No results found (try lowering threshold)")
    
except Exception as e:
    print(f"✗ RAG search failed: {e}")
    import traceback
    traceback.print_exc()

# Test Tavily search
print("\n[5] Testing Tavily Web Search...")
print("-" * 80)

async def test_tavily():
    try:
        from app.services.rag_service import search_tavily
        
        results = await search_tavily("latest agricultural technology India", max_results=2)
        
        if results:
            print(f"✓ Tavily search working")
            print(f"✓ Found {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"  [{i}] {result.get('title', 'No title')}")
        else:
            print(f"⚠️  No results from Tavily (check API key or quota)")
    
    except Exception as e:
        print(f"✗ Tavily search failed: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(test_tavily())

# Test hybrid search
print("\n[6] Testing Hybrid Search (RAG + Web)...")
print("-" * 80)

async def test_hybrid():
    try:
        from app.services.rag_service import hybrid_search
        
        result = await hybrid_search(
            query="wheat varieties for Bihar",
            top_k_rag=5,
            top_k_web=3
        )
        
        print(f"✓ Hybrid search completed")
        print(f"  RAG chunks: {result['rag_count']}")
        print(f"  Web results: {result['web_count']}")
        print(f"  Used web: {result['used_web']}")
        
    except Exception as e:
        print(f"✗ Hybrid search failed: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(test_hybrid())

# Summary
print("\n" + "="*80)
print("📋 Diagnostic Summary")
print("="*80)

print("\n✅ If all tests passed:")
print("   → Your RAG pipeline is working correctly")
print("   → Check backend logs for detailed search information")
print("   → Use the enhanced frontend to see RAG sources")

print("\n⚠️  Common Issues:")
print("   1. Empty collection → Run knowledge pipeline scripts")
print("   2. No Tavily results → Check API key and quota")
print("   3. Low similarity scores → Lower threshold or improve data quality")
print("   4. Connection errors → Check URLs and API keys in .env")
print("   5. Missing 'text' field → Data ingestion issue, re-run pipeline")

print("\n📖 Next Steps:")
print("   1. Fix data ingestion (see Fix #2 below)")
print("   2. Update rag_service.py (see Fix #3)")
print("   3. Restart FastAPI server")
print("   4. Test queries and check console logs")

print("\n" + "="*80)