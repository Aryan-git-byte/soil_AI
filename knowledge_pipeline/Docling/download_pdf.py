import os
import hashlib
import requests

URLS = [u.strip() for u in open("links.txt") if u.strip()]

os.makedirs("pdfs", exist_ok=True)

seen = set()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (FarmBotAI PDF Harvester)"
}

for url in URLS:
    if url in seen:
        continue
    seen.add(url)

    name = hashlib.md5(url.encode()).hexdigest() + ".pdf"
    path = os.path.join("pdfs", name)

    print("⬇️", url)

    try:
        r = requests.get(
            url,
            headers=HEADERS,
            timeout=60,
            allow_redirects=True
        )

        # Basic sanity check
        if r.status_code != 200:
            print("   ❌ HTTP", r.status_code)
            continue

        if "pdf" not in r.headers.get("Content-Type", "").lower():
            print("   ⚠️ Not a PDF (skipped)")
            continue

        with open(path, "wb") as f:
            f.write(r.content)

    except Exception as e:
        print("   ❌ Failed:", e)
