import pdfplumber
import json
import os
import traceback

BASE_DIR = "knowledge_pipeline"
SOURCE_DIR = f"{BASE_DIR}/sources"
RAW_DIR = f"{BASE_DIR}/raw"

os.makedirs(RAW_DIR, exist_ok=True)

def extract_single_pdf(pdf_path, filename):
    out_file = os.path.join(RAW_DIR, f"{filename}.jsonl")
    progress_file = os.path.join(RAW_DIR, f"{filename}.progress")

    # Try to read progress file
    last_page_done = -1
    if os.path.exists(progress_file):
        try:
            with open(progress_file) as pf:
                last_page_done = int(pf.read().strip())
        except:
            last_page_done = -1

    print(f"\n[INFO] Processing: {filename}")

    # ---------------------------------------------------
    # FIRST: detect total pages WITHOUT scanning all pages
    # ---------------------------------------------------
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
    except:
        print(f"[FATAL] Could not open PDF: {filename}")
        return

    # ---------------------------------------------------
    # FULL SKIP LOGIC — Already completed?
    # ---------------------------------------------------
    if last_page_done >= total_pages - 1:
        print(f"[SKIP] {filename} already fully processed ({total_pages} pages).")
        return

    # ---------------------------------------------------
    # RESUME LOGIC (start from next page)
    # ---------------------------------------------------
    start_page = last_page_done + 1
    print(f"[RESUME] {filename}: starting from page {start_page+1}/{total_pages}")

    # Now extract only missing pages
    try:
        with pdfplumber.open(pdf_path) as pdf:
            with open(out_file, "a", encoding="utf-8") as fout:
                for page_index in range(start_page, total_pages):

                    try:
                        text = pdf.pages[page_index].extract_text() or ""
                        cleaned = " ".join(text.split())

                        block = {
                            "source": filename,
                            "section": f"page_{page_index+1}",
                            "text": cleaned
                        }

                        fout.write(json.dumps(block, ensure_ascii=False) + "\n")

                        # Update progress
                        with open(progress_file, "w") as pf:
                            pf.write(str(page_index))

                        print(f"[OK] {filename}: page {page_index+1}/{total_pages}")

                    except Exception as e:
                        print(f"[ERR] Failed on page {page_index} of {filename}")
                        print(traceback.format_exc())
                        continue

        print(f"[DONE] Completed: {filename}")

    except Exception as e:
        print(f"[FATAL] Error reading PDF {filename}")
        print(traceback.format_exc())


def main():
    pdf_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("[WARN] No PDFs found.")
        return

    for pdf in pdf_files:
        extract_single_pdf(
            os.path.join(SOURCE_DIR, pdf),
            pdf
        )

    print("\n[ALL DONE] All PDFs processed.")


if __name__ == "__main__":
    main()
