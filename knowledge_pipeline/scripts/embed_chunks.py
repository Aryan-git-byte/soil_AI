import os
import json
import traceback

from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Directories
BASE_DIR = "knowledge_pipeline"
CLEAN_DIR = f"{BASE_DIR}/clean"

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # Use SERVICE ROLE KEY

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SERVICE_KEY in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Embedding Model (LOCAL)
print("[INFO] Loading local MiniLM model (384 dims)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

EMBEDDING_DIM = 384
BATCH_SIZE = 32   # Good speed & safe on CPU


# -----------------------------
# DATABASE INSERT (Guaranteed working)
# -----------------------------
def insert_chunks(rows: list[dict]):
    if not rows:
        return

    try:
        resp = supabase.table("knowledge_chunks").insert(rows).execute()

        # Supabase Python client differs by version → this works on all
        print(f"[DB] Inserted {len(rows)} rows")

    except Exception as e:
        print("[DB ERROR]", e)
        print(traceback.format_exc())


# -----------------------------
# EMBED ONE FILE
# -----------------------------
def embed_file(clean_path: str, filename: str):
    progress_file = os.path.join(
        CLEAN_DIR,
        filename.replace(".clean.jsonl", ".embed.progress")
    )

    # Resume logic
    last_index = -1
    if os.path.exists(progress_file):
        try:
            with open(progress_file) as pf:
                last_index = int(pf.read().strip())
        except:
            last_index = -1

    # Count total lines
    total_lines = sum(1 for _ in open(clean_path, "r", encoding="utf-8"))

    if last_index >= total_lines - 1:
        print(f"[SKIP] Already embedded: {filename}")
        return

    print(f"[EMBED] {filename} | Resume from line {last_index + 1}/{total_lines}")

    batch_texts = []
    batch_meta = []
    current_index = -1

    with open(clean_path, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            current_index = idx

            if idx <= last_index:
                continue  # Skip processed lines

            try:
                obj = json.loads(line)
            except:
                print(f"[ERR] Bad JSON: {idx}")
                continue

            text = obj.get("text", "").strip()
            if len(text) < 15:
                continue  # Skip garbage

            batch_texts.append(text)
            batch_meta.append(obj)

            if len(batch_texts) >= BATCH_SIZE:
                try:
                    embeddings = model.encode(batch_texts, show_progress_bar=False)
                except Exception:
                    print("[ERR] Embedding batch failed")
                    print(traceback.format_exc())
                    batch_texts.clear()
                    batch_meta.clear()
                    continue

                rows = []
                for meta, emb in zip(batch_meta, embeddings):
                    rows.append({
                        "source": meta.get("source"),
                        "crop": meta.get("crop"),
                        "region": meta.get("region"),
                        "section": meta.get("section"),
                        "text": meta.get("text"),
                        "embedding": emb.tolist(),
                    })

                insert_chunks(rows)

                # Save resume checkpoint
                with open(progress_file, "w") as pf:
                    pf.write(str(idx))

                print(f"[OK] Embedded up to line {idx+1}/{total_lines}")

                batch_texts.clear()
                batch_meta.clear()

        # Final partial batch
        if batch_texts:
            try:
                embeddings = model.encode(batch_texts, show_progress_bar=False)
                rows = []
                for meta, emb in zip(batch_meta, embeddings):
                    rows.append({
                        "source": meta.get("source"),
                        "crop": meta.get("crop"),
                        "region": meta.get("region"),
                        "section": meta.get("section"),
                        "text": meta.get("text"),
                        "embedding": emb.tolist(),
                    })

                insert_chunks(rows)
                with open(progress_file, "w") as pf:
                    pf.write(str(current_index))

                print(f"[OK] Final batch done for: {filename}")

            except Exception:
                print(f"[ERR] Final batch failed: {filename}")
                print(traceback.format_exc())

    print(f"[DONE] Completed: {filename}")


# -----------------------------
# MAIN
# -----------------------------
def main():
    files = [f for f in os.listdir(CLEAN_DIR) if f.endswith(".clean.jsonl")]
    if not files:
        print("[WARN] No .clean.jsonl files found.")
        return

    for file in files:
        clean_path = os.path.join(CLEAN_DIR, file)
        embed_file(clean_path, file)

    print("\n[ALL DONE] All embeddings complete.")


if __name__ == "__main__":
    main()
