import json
import os
import re
import traceback

RAW_DIR = "knowledge_pipeline/raw"
CLEAN_DIR = "knowledge_pipeline/clean"

os.makedirs(CLEAN_DIR, exist_ok=True)

# Simple crop detector
CROP_KEYWORDS = [
    "wheat", "rice", "maize", "paddy", "potato",
    "sugarcane", "millet", "soybean", "mustard"
]

# Region extractor
REGION_KEYWORDS = [
    "bihar", "punjab", "uttar pradesh", "west bengal",
    "haryana", "karnataka", "tamil nadu",
    "alluvial", "black soil", "deccan", "north india"
]


def detect_crop(text):
    text_l = text.lower()
    for crop in CROP_KEYWORDS:
        if crop in text_l:
            return crop
    return None


def detect_region(text):
    text_l = text.lower()
    for region in REGION_KEYWORDS:
        if region in text_l:
            return region
    return None


def clean_text(t: str):
    # Remove journal headers/footers
    t = re.sub(r"INDIAN JOURNAL.*?EDUCATION", "", t, flags=re.I)
    t = re.sub(r"Vol\..*?\d+,\s*\d+", "", t, flags=re.I)

    # Remove page numbers
    t = re.sub(r"\b\d{1,3}\b(?=\s+INDIAN|$)", "", t)

    # Remove references (pattern matches most references)
    t = re.sub(r"[A-Z][a-z]+,\s?[A-Z].+?\(\d{4}\).*", "", t)

    # Normalize spaces
    t = " ".join(t.split())

    return t.strip()


def split_into_blocks(text):
    """Split into 200–400 word meaningful blocks."""
    parts = re.split(r'\.\s+', text)
    blocks = []

    buffer = ""
    for sentence in parts:
        if len(buffer) + len(sentence) < 350:
            buffer += sentence + ". "
        else:
            blocks.append(buffer.strip())
            buffer = sentence + ". "

    if buffer.strip():
        blocks.append(buffer.strip())

    return blocks


def clean_single_file(raw_path, filename):
    """Clean a raw JSONL using resume logic."""

    out_file = os.path.join(CLEAN_DIR, filename.replace(".jsonl", ".clean.jsonl"))
    progress_file = os.path.join(CLEAN_DIR, filename.replace(".jsonl", ".progress"))

    # Resume: read last processed index
    last_index = -1
    if os.path.exists(progress_file):
        try:
            with open(progress_file) as pf:
                last_index = int(pf.read().strip())
        except:
            last_index = -1

    # Count total lines once (no reprocessing)
    total_lines = sum(1 for _ in open(raw_path, "r", encoding="utf-8"))

    # FULL SKIP
    if last_index >= total_lines - 1:
        print(f"[SKIP] Already cleaned: {filename} ({total_lines} blocks)")
        return

    print(f"[CLEAN] {filename} | Resume from block {last_index + 1}/{total_lines}")

    # Open raw + output for appending
    with open(raw_path, "r", encoding="utf-8") as fin, \
         open(out_file, "a", encoding="utf-8") as fout:

        for idx, line in enumerate(fin):

            if idx <= last_index:
                continue  # skip already processed lines

            try:
                obj = json.loads(line)
                cleaned = clean_text(obj["text"])

                crop = detect_crop(cleaned)
                region = detect_region(cleaned)

                blocks = split_into_blocks(cleaned)

                for b_i, block in enumerate(blocks):
                    new_obj = {
                        "source": obj["source"],
                        "crop": crop,
                        "region": region,
                        "section": f"{obj['section']}_p{b_i+1}",
                        "text": block
                    }
                    fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

                # Save progress
                with open(progress_file, "w") as pf:
                    pf.write(str(idx))

                print(f"[OK] Cleaned block {idx+1}/{total_lines}")

            except Exception as e:
                print(f"[ERR] Failed to clean block {idx} in {filename}")
                print(traceback.format_exc())
                continue

    print(f"[DONE] Clean completed: {filename}")


def main():
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".jsonl")]

    if not files:
        print("[WARN] No raw JSONL files to clean.")
        return

    for file in files:
        raw_path = os.path.join(RAW_DIR, file)
        clean_single_file(raw_path, file)

    print("\n[ALL DONE] Cleaning completed for all files.")


if __name__ == "__main__":
    main()
