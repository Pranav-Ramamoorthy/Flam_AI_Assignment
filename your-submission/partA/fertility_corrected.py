#!/usr/bin/env python3
"""
fertility_corrected.py -- Robust, multi-denominator, multi-tokenizer cross-lingual benchmark.

Computes tokenization metrics across multiple denominators:
1. per whitespace word (proper split)
2. per extended grapheme cluster (Unicode UAX #29, via regex \X)
3. per UTF-8 byte
4. per parallel sentence (semantic information unit)

Supports multiple tokenizers:
- gpt2 (tiktoken baseline)
- hf:xlm-roberta-base (SentencePiece 250k multilingual)
- hf:google/muril-base-cased (WordPiece 197k Indic-specialized)
- hf:Qwen/Qwen2.5-0.5B (Byte BPE 151k modern multilingual LLM)
"""

import argparse
import csv
import json
import os
import sys
import unicodedata
import regex  # Supports Unicode \X (extended grapheme clusters)

sys.stdout.reconfigure(encoding='utf-8')


def load_tokenizer(spec: str):
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer
        repo_id = spec[3:]
        tok = AutoTokenizer.from_pretrained(repo_id)
        return lambda s: tok.encode(s, add_special_tokens=False)
    elif spec == "gpt2":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return enc.encode
    else:
        raise ValueError(f"Unknown tokenizer spec: {spec}")


def read_lines(path: str) -> list[str]:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Canonical Unicode NFC normalization
            line = unicodedata.normalize("NFC", line)
            lines.append(line)
    return lines


def count_grapheme_clusters(text: str) -> int:
    # Unicode UAX #29 extended grapheme cluster match
    return len(regex.findall(r'\X', text))


def evaluate_corpus(lines: list[str], encode_fn):
    total_tokens = 0
    total_words = 0
    total_graphemes = 0
    total_bytes = 0
    total_chars = 0
    num_sentences = len(lines)

    per_line_tok_per_word = []
    per_line_tok_per_sent = []

    for line in lines:
        toks = encode_fn(line)
        tok_count = len(toks)
        words = line.split()
        word_count = len(words)
        grapheme_count = count_grapheme_clusters(line)
        byte_count = len(line.encode("utf-8"))
        char_count = len(line)

        total_tokens += tok_count
        total_words += word_count
        total_graphemes += grapheme_count
        total_bytes += byte_count
        total_chars += char_count

        per_line_tok_per_word.append(tok_count / word_count if word_count > 0 else 0)
        per_line_tok_per_sent.append(tok_count)

    return {
        "num_sentences": num_sentences,
        "total_tokens": total_tokens,
        "total_words": total_words,
        "total_graphemes": total_graphemes,
        "total_bytes": total_bytes,
        "total_chars": total_chars,
        # Micro-averages (true corpus ratio)
        "tok_per_sentence": total_tokens / num_sentences,
        "tok_per_word_micro": total_tokens / total_words,
        "tok_per_grapheme_micro": total_tokens / total_graphemes,
        "tok_per_byte_micro": total_tokens / total_bytes,
        "tok_per_char_micro": total_tokens / total_chars,
        # Macro-average for comparison
        "tok_per_word_macro": sum(per_line_tok_per_word) / len(per_line_tok_per_word),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="LANG=PATH",
        help="language code and path, e.g. eng=data/eng.txt",
    )
    ap.add_argument(
        "--tokenizer",
        action="append",
        required=True,
        help="tokenizer spec, e.g. gpt2, hf:xlm-roberta-base, hf:google/muril-base-cased, hf:Qwen/Qwen2.5-0.5B",
    )
    ap.add_argument("--output_csv", default=None, help="Path to save results as CSV")
    ap.add_argument("--output_json", default=None, help="Path to save results as JSON")
    args = ap.parse_args()

    # Load all corpora
    corpora = {}
    for spec in args.corpus:
        lang, path = spec.split("=", 1)
        corpora[lang] = read_lines(path)
        print(f"Loaded {lang} from {path}: {len(corpora[lang])} lines")

    all_results = {}
    csv_rows = []

    for tok_spec in args.tokenizer:
        print(f"\n=======================================================")
        print(f"Evaluating Tokenizer: {tok_spec}")
        print(f"=======================================================")
        encode_fn = load_tokenizer(tok_spec)
        all_results[tok_spec] = {}

        print(f"{'Lang':<6}{'Tok/Sent':>12}{'Tok/Word (Micro)':>18}{'Tok/Grapheme':>15}{'Tok/Byte':>12}{'Tok/Char':>12}")
        print("-" * 75)

        base_lang = None
        for lang, lines in corpora.items():
            if base_lang is None:
                base_lang = lang
            res = evaluate_corpus(lines, encode_fn)
            all_results[tok_spec][lang] = res

            print(f"{lang:<6}{res['tok_per_sentence']:>12.2f}{res['tok_per_word_micro']:>18.2f}{res['tok_per_grapheme_micro']:>15.3f}{res['tok_per_byte_micro']:>12.3f}{res['tok_per_char_micro']:>12.3f}")

            csv_rows.append({
                "tokenizer": tok_spec,
                "language": lang,
                "num_sentences": res["num_sentences"],
                "total_tokens": res["total_tokens"],
                "tok_per_sentence": round(res["tok_per_sentence"], 2),
                "tok_per_word_micro": round(res["tok_per_word_micro"], 2),
                "tok_per_word_macro": round(res["tok_per_word_macro"], 2),
                "tok_per_grapheme": round(res["tok_per_grapheme_micro"], 3),
                "tok_per_byte": round(res["tok_per_byte_micro"], 3),
                "tok_per_char": round(res["tok_per_char_micro"], 3),
            })

        print("-" * 75)
        print(f"Cost Multipliers relative to {base_lang} (Tok/Sentence vs Tok/Word):")
        base_sent = all_results[tok_spec][base_lang]["tok_per_sentence"]
        base_word = all_results[tok_spec][base_lang]["tok_per_word_micro"]
        for lang in corpora:
            if lang == base_lang:
                continue
            ratio_sent = all_results[tok_spec][lang]["tok_per_sentence"] / base_sent
            ratio_word = all_results[tok_spec][lang]["tok_per_word_micro"] / base_word
            print(f"  {lang:<4} -> Semantic Cost Ratio (Tok/Sent): {ratio_sent:.2f}x  |  Misleading Ratio (Tok/Word): {ratio_word:.2f}x")

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nSaved CSV results to {args.output_csv}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON results to {args.output_json}")


if __name__ == "__main__":
    main()

