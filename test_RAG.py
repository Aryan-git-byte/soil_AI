# test_rag.py
import time
from app.services.rag_service import search_knowledge

print("[TEST] Testing RAG with real embeddings...")

queries = [
    "wheat fertilizer recommendations",
    "rice pest management Bihar",
    "soil preparation for crops"
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"Query: '{query}'")
    print('='*60)
    
    start = time.time()
    chunks = search_knowledge(query, top_k=8)
    elapsed = time.time() - start
    
    print(f"⏱️  Time: {elapsed:.3f} seconds")
    print(f"📊 Found: {len(chunks)} chunks")
    
    if elapsed < 2:
        print("✅ FAST! Index is working")
    else:
        print("❌ SLOW! Index may not be used")
    
    if chunks:
        print(f"\n🔍 Top result:")
        print(f"   Source: {chunks[0].get('source', 'N/A')}")
        print(f"   Crop: {chunks[0].get('crop', 'N/A')}")
        print(f"   Similarity: {chunks[0].get('similarity', 0):.4f}")
        print(f"   Text: {chunks[0].get('text', '')[:150]}...")
    else:
        print("⚠️  No results returned")

print(f"\n{'='*60}")
print("Test complete!")