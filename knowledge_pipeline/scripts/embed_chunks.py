import os
import json
import traceback
import re
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = "knowledge_pipeline"
CLEAN_DIR = f"{BASE_DIR}/clean"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("[INFO] Loading MiniLM model (CPU)...")
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

EMBEDDING_DIM = 384
BATCH_SIZE = 32  # small + safe for huge files

# Remove ALL control / null characters
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def clean_text(s: str | None) -> str:
    if not s:
        return ""
    # Kill control chars including \u0000
    s = CONTROL_CHARS.sub("", s)
    s = s.replace("\u0000", "")
    # Normalize spaces
    return " ".join(s.split())


def insert_chunks(rows: list[dict]) -> bool:
    """Insert rows into Supabase. Return True if success, False if error."""
    if not rows:
        return True

    # Final safety: sanitize all string fields again
    safe_rows: list[dict] = []
    for r in rows:
        safe_rows.append({
            "source": clean_text(r.get("source")),
            "crop": clean_text(r.get("crop")),
            "region": clean_text(r.get("region")),
            "section": clean_text(r.get("section")),
            "text": clean_text(r.get("text")),
            "embedding": r.get("embedding"),
        })

    try:
        supabase.table("knowledge_chunks").insert(safe_rows).execute()
        print(f"[DB] Inserted {len(safe_rows)} rows")
        return True
    except Exception as e:
        print("[DB ERROR]", e)
        print(traceback.format_exc())
        return False


def embed_file(path: str, filename: str):
    progress_file = os.path.join(
        CLEAN_DIR,
        filename.replace(".clean.jsonl", ".embed.progress")
    )

    # Resume index
    last_index = -1
    if os.path.exists(progress_file):
        try:
            last_index = int(open(progress_file).read().strip())
        except Exception:
            last_index = -1

    # Count total lines
    total_lines = sum(1 for _ in open(path, "r", encoding="utf-8"))
    if last_index >= total_lines - 1:
        print(f"[SKIP] Already embedded: {filename}")
        return

    print(f"[EMBED] {filename} | from line {last_index+1}/{total_lines}")

    batch_texts: list[str] = []
    batch_meta: list[dict] = []
    current_index = -1

    with open(path, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            current_index = idx

            if idx <= last_index:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                # bad JSON, skip
                continue

            raw_text = obj.get("text", "")
            cleaned = clean_text(raw_text)
            if len(cleaned) < 10:
                continue  # skip useless fragments

            batch_texts.append(cleaned)
            batch_meta.append(obj)

            if len(batch_texts) >= BATCH_SIZE:
                try:
                    embeddings = model.encode(batch_texts, show_progress_bar=False)
                except Exception:
                    print("[EMBED ERR] Skipping one faulty batch (encode)")
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
                        # IMPORTANT: store CLEANED text, not raw
                        "text": cleaned if meta is batch_meta[-1] else clean_text(meta.get("text", "")),
                        "embedding": emb.tolist(),
                    })

                ok = insert_chunks(rows)
                if ok:
                    with open(progress_file, "w") as pf:
                        pf.write(str(idx))
                    print(f"[OK] Embedded up to line {idx+1}/{total_lines}")
                else:
                    print("[WARN] Batch failed to insert, but continuing...")

                batch_texts.clear()
                batch_meta.clear()

        # Final leftover batch
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
                        "text": clean_text(meta.get("text", "")),
                        "embedding": emb.tolist(),
                    })
                ok = insert_chunks(rows)
                if ok:
                    with open(progress_file, "w") as pf:
                        pf.write(str(current_index))
                    print(f"[OK] Final batch done for {filename}")
            except Exception:
                print("[FINAL BATCH ERR] Skipped final batch")
                print(traceback.format_exc())

    print(f"[DONE] {filename}")


def main():
    files = [f for f in os.listdir(CLEAN_DIR) if f.endswith(".clean.jsonl")]
    if not files:
        print("[WARN] No cleaned files found.")
        return

    for file in files:
        path = os.path.join(CLEAN_DIR, file)
        embed_file(path, file)

    print("\n[ALL DONE] Embedding complete for all files.")


if __name__ == "__main__":
    main()
