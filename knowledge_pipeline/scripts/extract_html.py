import requests
from bs4 import BeautifulSoup
import json
import yaml
import os

RAW_DIR = "knowledge_pipeline/raw/"

def extract_html(url, metadata):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    texts = []
    for tag in soup.find_all(["p", "li", "h2", "h3"]):
        content = tag.get_text().strip()
        if len(content) < 20:
            continue

        texts.append(content)

    blocks = []
    for idx, t in enumerate(texts):
        blocks.append({
            "source": metadata["name"],
            "url": metadata["url"],
            "crop": metadata.get("crop"),
            "region": metadata.get("region"),
            "section": f"block_{idx+1}",
            "text": t
        })

    return blocks


def main():
    with open("knowledge_pipeline/config.yaml") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(RAW_DIR, exist_ok=True)

    for item in cfg["sources"]:
        if item["type"] != "html":
            continue

        blocks = extract_html(item["url"], item)

        out_path = f"{RAW_DIR}/{item['id']}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for b in blocks:
                f.write(json.dumps(b, ensure_ascii=False) + "\n")

        print(f"[OK] Extracted HTML -> {out_path}")


if __name__ == "__main__":
    main()
