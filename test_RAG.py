# Test suite for enhanced RAG pipeline
import time
import asyncio
from app.services.rag_service import search_knowledge, search_tavily, hybrid_search

print("[TEST] Testing Enhanced RAG Pipeline...")
print("="*60)

# Test queries
test_queries = [
    {
        "query": "wheat fertilizer recommendations",
        "test_type": "RAG_ONLY"
    },
    {
        "query": "latest rice pest management techniques 2024",
        "test_type": "HYBRID"
    },
    {
        "query": "soil preparation for potato Bihar",
        "test_type": "RAG_ONLY"
    }
]

async def test_rag_only(query):
    """Test vector search"""
    print(f"\n{'='*60}")
    print(f"[RAG ONLY] Query: '{query}'")
    print('='*60)
    
    start = time.time()
    chunks = search_knowledge(query, top_k=5)
    elapsed = time.time() - start
    
    print(f"⏱️  Time: {elapsed:.3f}s")
    print(f"📊 Found: {len(chunks)} chunks")
    
    if elapsed < 2:
        print("✅ FAST - Index working")
    else:
        print("⚠️  SLOW - Check index")
    
    if chunks:
        print(f"\n🔍 Top result:")
        print(f"   Source: {chunks[0].get('source', 'N/A')}")
        print(f"   Crop: {chunks[0].get('crop', 'N/A')}")
        print(f"   Similarity: {chunks[0].get('similarity', 0):.4f}")
        print(f"   Text: {chunks[0].get('text', '')[:100]}...")


async def test_tavily(query):
    """Test Tavily search"""
    print(f"\n{'='*60}")
    print(f"[TAVILY] Query: '{query}'")
    print('='*60)
    
    start = time.time()
    results = await search_tavily(query, max_results=3)
    elapsed = time.time() - start
    
    print(f"⏱️  Time: {elapsed:.3f}s")
    print(f"📊 Found: {len(results)} results")
    
    if results:
        print(f"\n🌐 Top result:")
        print(f"   Title: {results[0].get('title', 'N/A')}")
        print(f"   URL: {results[0].get('url', 'N/A')}")
        print(f"   Content: {results[0].get('content', '')[:100]}...")


async def test_hybrid(query):
    """Test hybrid search"""
    print(f"\n{'='*60}")
    print(f"[HYBRID] Query: '{query}'")
    print('='*60)
    
    start = time.time()
    results = await hybrid_search(query, top_k_rag=5, top_k_web=3)
    elapsed = time.time() - start
    
    print(f"⏱️  Time: {elapsed:.3f}s")
    print(f"📊 RAG chunks: {results['rag_count']}")
    print(f"🌐 Web results: {results['web_count']}")
    print(f"🔄 Used web: {results['used_web']}")
    
    if results['rag_chunks']:
        print(f"\n🔍 Top RAG result:")
        chunk = results['rag_chunks'][0]
        print(f"   Source: {chunk.get('source', 'N/A')}")
        print(f"   Text: {chunk.get('text', '')[:100]}...")
    
    if results['web_results']:
        print(f"\n🌐 Top web result:")
        web = results['web_results'][0]
        print(f"   Title: {web.get('title', 'N/A')}")
        print(f"   URL: {web.get('url', 'N/A')}")


async def main():
    for test in test_queries:
        query = test["query"]
        test_type = test["test_type"]
        
        if test_type == "RAG_ONLY":
            await test_rag_only(query)
        elif test_type == "HYBRID":
            await test_hybrid(query)
        
        await asyncio.sleep(1)
    
    # Additional Tavily test
    print(f"\n{'='*60}")
    print("[BONUS] Testing Tavily directly")
    await test_tavily("latest agricultural technology India 2024")
    
    print(f"\n{'='*60}")
    print("✅ All tests complete!")


if __name__ == "__main__":
    asyncio.run(main())