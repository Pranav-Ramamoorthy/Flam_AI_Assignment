#!/usr/bin/env python3
"""
prepare_corpus.py -- Fetch, normalize, and verify the multilingual evaluation corpus.

Downloads official FLORES-200 devtest parallel sentences for:
- English (eng_Latn)
- Hindi (hin_Deva)
- Kannada (kan_Knda) [Dravidian 1]
- Tamil (tam_Taml)   [Dravidian 2]

Normalizes text to Unicode NFC, cleans whitespace, checks alignment, and computes corpus statistics.
"""

import hashlib
import os
import sys
import unicodedata
import urllib.request

sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "https://raw.githubusercontent.com/AI4Bharat/CTQScorer/master/dataset/test"
LANG_MAP = {
    "eng": "eng_Latn.devtest",
    "hin": "hin_Deva.devtest",
    "kan": "kan_Knda.devtest",
    "tam": "tam_Taml.devtest",
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def fetch_file(lang: str, filename: str) -> list[str]:
    url = f"{BASE_URL}/{filename}"
    print(f"Fetching {lang} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read().decode("utf-8")
    
    raw_lines = content.splitlines()
    processed_lines = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        # Canonical Unicode normalization (NFC)
        line = unicodedata.normalize("NFC", line)
        processed_lines.append(line)
    return processed_lines


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stats = {}

    for lang, fname in LANG_MAP.items():
        lines = fetch_file(lang, fname)
        out_path = os.path.join(OUTPUT_DIR, f"{lang}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for l in lines:
                f.write(l + "\n")
        
        # Compute SHA-256 hash for reproducibility
        with open(out_path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
        
        total_words = sum(len(l.split()) for l in lines)
        total_chars = sum(len(l) for l in lines)
        total_bytes = sum(len(l.encode("utf-8")) for l in lines)

        stats[lang] = {
            "lines": len(lines),
            "words": total_words,
            "chars": total_chars,
            "utf8_bytes": total_bytes,
            "sha256": sha256,
            "path": out_path,
        }
        print(f"Saved {lang} -> {out_path} ({len(lines)} lines, {total_words} words, {total_chars} chars, {total_bytes} bytes)")

    # Integrity verification
    line_counts = {k: v["lines"] for k, v in stats.items()}
    if len(set(line_counts.values())) != 1:
        raise ValueError(f"Corpus alignment error: line counts differ across languages: {line_counts}")
    
    print("\nCorpus verification SUCCESS:")
    print(f"All {len(stats)} languages have exactly {next(iter(line_counts.values()))} parallel lines.")
    print("-" * 75)
    print(f"{'Lang':<6}{'Lines':<8}{'Words':<10}{'Chars':<10}{'UTF-8 Bytes':<14}{'SHA-256 (first 12)':<15}")
    print("-" * 75)
    for lang, s in stats.items():
        print(f"{lang:<6}{s['lines']:<8}{s['words']:<10}{s['chars']:<10}{s['utf8_bytes']:<14}{s['sha256'][:12]:<15}")
    print("-" * 75)


if __name__ == "__main__":
    main()

