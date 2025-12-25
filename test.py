# fast_rag_test.py
import os
import time
import json
from dotenv import load_dotenv

# fastembed is what you've used before
from fastembed import TextEmbedding

# qdrant python client
from qdrant_client import QdrantClient

# optional: try to import typed search params (may not exist in all client versions)
try:
    from qdrant_client.models import HnswSearchParams
except Exception:
    HnswSearchParams = None

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "farmbot_knowledge")
MODEL_ID = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
LOCAL_CACHE = os.path.join(os.getcwd(), "model_cache")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("QDRANT_URL / QDRANT_API_KEY are required in .env")

print("🧠 Loading FastEmbed model (local cache)...")
embedder = TextEmbedding(MODEL_ID, cache_dir=LOCAL_CACHE)

print("📡 Connecting to Qdrant (gRPC preferred)...")
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=True,   # try gRPC for lower latency when available
    timeout=30
)

# ---------------------
# Helper: unify results
# ---------------------
def normalize_results(raw):
    """
    Turn various Qdrant method responses into a list of simple dicts:
    [{'id':..., 'score':..., 'payload': {...}}, ...]
    """
    out = []
    # if qdrant.search returned a list-like of hits (older/newer clients)
    try:
        # common structure: list of objects with .payload and .score
        for h in raw:
            item = {
                "id": getattr(h, "id", None) or h.get("id") if isinstance(h, dict) else None,
                "score": getattr(h, "score", None) or (h.get("score") if isinstance(h, dict) else None),
                "payload": getattr(h, "payload", None) or (h.get("payload") if isinstance(h, dict) else None),
            }
            out.append(item)
        if out:
            return out
    except Exception:
        pass

    # If response is QueryPointsResult-like with .points
    if hasattr(raw, "points"):
        for p in raw.points:
            out.append({
                "id": getattr(p, "id", None),
                "score": getattr(p, "score", None),
                "payload": getattr(p, "payload", None) or (p.payload if hasattr(p, "payload") else None)
            })
        return out

    # If qdrant._client returned a dict
    if isinstance(raw, dict):
        hits = raw.get("result") or raw.get("hits") or raw.get("points") or []
        for h in hits:
            out.append({
                "id": h.get("id"),
                "score": h.get("score"),
                "payload": h.get("payload")
            })
        return out

    # Fallback: return empty list
    return out

# -----------------------------
# Core: version-tolerant search
# -----------------------------
def qdrant_vector_search(client, collection, vector, limit=5, payload_fields=None, score_threshold=None, hnsw_ef=None):
    """
    Try the best available method on the installed qdrant client.
    - payload_fields: list or False (False -> no payload)
    - score_threshold: float or None (if supported)
    - hnsw_ef: int or None -> used if HnswSearchParams is available
    """
    # Prepare arguments for different client flavors
    # Some clients expect `query_vector`, others `vector` or `query`.
    # We'll attempt multiple call forms and normalize results.

    # Build "search params" if supported
    search_params = None
    if HnswSearchParams is not None and hnsw_ef is not None:
        try:
            search_params = HnswSearchParams(ef=hnsw_ef)
        except Exception:
            search_params = None

    # 1) Try modern high-level `search` (many versions expose this)
    try:
        if hasattr(client, "search"):
            kwargs = {
                "collection_name": collection,
                "query_vector": vector,
                "limit": limit,
            }
            if payload_fields is not None:
                kwargs["with_payload"] = payload_fields
            if score_threshold is not None:
                kwargs["score_threshold"] = score_threshold
            if search_params is not None:
                # argument name might be different across versions; try both
                try:
                    kwargs["search_params"] = search_params
                except Exception:
                    pass
            raw = client.search(**kwargs)
            return normalize_results(raw)
    except Exception as e:
        # fall through and try other signatures
        print("  - search() failed:", e)

    # 2) Try `search_points` (some releases)
    try:
        if hasattr(client, "search_points"):
            kwargs = {
                "collection_name": collection,
                "vector": vector,
                "limit": limit
            }
            if payload_fields is not None:
                kwargs["with_payload"] = payload_fields
            if score_threshold is not None:
                kwargs["score_threshold"] = score_threshold
            if search_params is not None:
                kwargs["params"] = search_params
            raw = client.search_points(**kwargs)
            return normalize_results(raw)
    except Exception as e:
        print("  - search_points() failed:", e)

    # 3) Try `query_points` (documented in many places)
    try:
        if hasattr(client, "query_points"):
            kwargs = {
                "collection_name": collection,
                "query": vector,
                "limit": limit
            }
            if payload_fields is not None:
                kwargs["with_payload"] = payload_fields
            if score_threshold is not None:
                kwargs["score_threshold"] = score_threshold
            raw = client.query_points(**kwargs)
            return normalize_results(raw)
    except Exception as e:
        print("  - query_points() failed:", e)

    # 4) Try to access underlying http/grpc client (last resort)
    if hasattr(client, "_client") and hasattr(client._client, "search"):
        try:
            raw = client._client.search(
                collection_name=collection,
                query_vector=vector,
                limit=limit
            )
            return normalize_results(raw)
        except Exception as e:
            print("  - internal _client.search() failed:", e)

    raise RuntimeError("No compatible search method found on Qdrant client instance.")


# -----------------------
# Warmup + Test function
# -----------------------
def run_test(query_text: str,
             top_k: int = 5,
             payload_fields: list | bool = False,
             score_threshold: float | None = None,
             hnsw_ef: int | None = None):
    print(f"\n🚀 QUERY: {query_text}")

    # 1) embed
    t0 = time.perf_counter()
    vec = list(embedder.embed([query_text]))[0].tolist()
    emb_ms = (time.perf_counter() - t0) * 1000
    print(f"   ✅ Embed time: {emb_ms:.2f} ms")

    # 2) warmup search (pay handshake/auth latency once)
    try:
        t1 = time.perf_counter()
        # do a tiny warmup - no payload to keep it tiny
        qdrant_vector_search(qdrant, COLLECTION, vec, limit=1, payload_fields=False, score_threshold=None, hnsw_ef=None)
        warm_ms = (time.perf_counter() - t1) * 1000
        print(f"   🐢 Warmup search (cold): {warm_ms:.2f} ms")
    except Exception as e:
        print("   ⚠ Warmup failed (non-fatal):", e)

    # 3) real search (with desired options)
    t2 = time.perf_counter()
    results = qdrant_vector_search(qdrant, COLLECTION, vec, limit=top_k, payload_fields=payload_fields, score_threshold=score_threshold, hnsw_ef=hnsw_ef)
    search_ms = (time.perf_counter() - t2) * 1000
    print(f"   🐇 Real search (warm): {search_ms:.2f} ms")

    print(f"   📦 Results: {len(results)}")
    for i, r in enumerate(results[:top_k], start=1):
        src = (r.get("payload") or {}).get("source") if r.get("payload") else None
        print(f"    [{i}] score={r.get('score')} | source={src}")

    total = emb_ms + search_ms
    print(f"\n✨ APP-LATENCY (embed + search): {total:.2f} ms")
    return {
        "embed_ms": emb_ms,
        "search_ms": search_ms,
        "total_ms": total,
        "hits": results
    }


if __name__ == "__main__":
    # warm local embedder once
    list(embedder.embed(["warmup"]))

    # Example: minimal payload + threshold + hnsw_ef tuning
    out = run_test(
        "how to grow rice in Bihar",
        top_k=5,
        payload_fields=["source", "crop"],  # only these payload fields -> smaller network size
        score_threshold=0.25,              # ignore low-similarity noise (adjust to your dataset)
        hnsw_ef=64                         # if supported, tune ef for speed/recall tradeoff
    )

    # Save a quick json output for inspection
    with open("last_rag_test.json", "w", encoding="utf-8") as f:
        json.dump(out, f, default=str, indent=2)
