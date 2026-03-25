import os
import re
import json
import httpx

FIXTURES_URL = "https://www.iplt20.com/matches/fixtures"

def main():
    html = httpx.get(FIXTURES_URL, timeout=30).text

    # Find embedded JSON-ish chunks
    for pattern in [
        r'window\.__INITIAL_STATE__\s*=\s*(\{.*?\})\s*;',
        r'window\.__data\s*=\s*(\{.*?\})\s*;',
        r'window\.appState\s*=\s*(\{.*?\})\s*;',
    ]:
        m = re.search(pattern, html, re.DOTALL)
        if m:
            print("FOUND EMBEDDED JSON")
            print(m.group(1)[:2000])
            return

    # fallback: inspect script tags
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.DOTALL | re.IGNORECASE)
    print(f"Found {len(scripts)} script tags")
    for i, s in enumerate(scripts[:20]):
        if "fixture" in s.lower() or "match" in s.lower() or "schedule" in s.lower():
            print(f"\n--- script {i} ---\n")
            print(s[:2000])

if __name__ == "__main__":
    main()