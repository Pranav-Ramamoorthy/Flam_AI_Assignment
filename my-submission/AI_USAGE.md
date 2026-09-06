# AI Usage Disclosure (`AI_USAGE.md`)

**Author:** Pranav R  
**Project:** AI Team Intern Assignment — The Audit  
**Date:** September 7, 2026  

In accordance with the assignment ground rules, this document provides an honest, transparent accounting of where AI assistance was beneficial and where AI suggestions were misleading or factually incorrect.

---

## 1. Where AI Helped
1. **Corpus Ingestion & Data Scaffolding**:
   - AI assisted in drafting the urllib fetch routines and Unicode normalization pipeline in `prepare_corpus.py`.
   - When Meta's original `fbaipublicfiles` S3 link threw a 403 Forbidden and Hugging Face raw endpoints threw 401 Unauthorized, AI search assisted in locating AI4Bharat's open GitHub repository mirror containing verified FLORES-200 `.devtest` splits for `eng_Latn`, `hin_Deva`, `kan_Knda`, and `tam_Taml`.
2. **Formula Verification & Matrix Arithmetic**:
   - AI assisted in cross-verifying the transformer attention tensor formulas for Grouped-Query Attention (GQA): Key and Value vector allocations ($2 \times L \times H_{kv} \times D_{\text{head}} \times \text{bytes\_per\_element} = 114,688\text{ bytes/token}$), saving manual cross-checking time.
3. **Statistical Ratio Estimation**:
   - AI pointed out the standard econometric distinction between macro-averaging (unweighted mean of per-line ratios $\frac{1}{N}\sum \frac{T_i}{W_i}$) and micro-averaging ($\frac{\sum T_i}{\sum W_i}$), helping formulate the mathematical proof for Bug 3.

---

## 2. Where AI Misled & Hallucinated (and How We Caught It)

1. **The Phantom "NFC Nukta Corruption" Bug (AI Confidently Invented a Bug)**:
   - When asked to inspect `fertility.py`, the AI LLM confidently flagged `unicodedata.normalize("NFC", line)` as a critical bug, arguing: *"NFC normalization alters Devanagari combining characters, erroneously decomposing nuktas like क़ into separate codepoints and inflating token counts."*
   - **How We Caught It**: We applied the strict Evidence Rule before accepting the claim. We ran an isolated experiment with and without NFC normalization on `hin_sample.txt`.
   - **Empirical Evidence**: The delta was **exactly 0.0000** (7.4485 tok/word with NFC vs 7.4485 without NFC). Furthermore, NFC is canonical *composition* (combining characters into single precomposed codepoints where available), not decomposition! NFD is what decomposes them.
   - Had we submitted the AI's confident assertion, we would have been penalized -5 points for claiming a harmless feature as a bug.

2. **Gated Hugging Face Tokenizer Assumption**:
   - AI suggested evaluating `ai4bharat/indic-bert` as our open Indic tokenizer.
   - **How We Caught It**: When executed, HuggingFace returned `403 Client Error: You are trying to access a gated repo`. AI assumed the model was public and ungated.
   - **Resolution**: We pivoted to `google/muril-base-cased` (Google's ungated 197k Indic BERT) and `xlm-roberta-base`, both of which executed with zero authentication hurdles.

3. **Conflating Font Rendering with Tokenization**:
   - When analyzing why Kannada and Tamil produced high tokens-per-word under `gpt2`, AI initially suggested: *"Dravidian scripts require more tokens because complex conjunct glyphs require multiple font rendering ligatures."*
   - **How We Caught It**: Font rendering ligatures are a graphical display concern, completely irrelevant to byte-level BPE tokenization. The actual root cause was linguistic **morphological agglutination**: Kannada and Tamil fuse affixes, case markers, and postpositions into single compound words, resulting in 25% fewer words per sentence than English. The smaller denominator artificially inflated `tokens / word`.

4. **Throughput Scaling Hallucination**:
   - In Part B, AI initially suggested that the throughput collapse at batch 32 in `bench_log.csv` was caused by "thermal throttling on the GPU".
   - **How We Caught It**: Checking the log columns revealed `preempted_seqs = 7` at batch 32 and `23` at batch 48, while `kv_cache_util` hit `0.97`. The collapse had nothing to do with GPU temperatures; it was pure software KV-cache exhaustion triggering vLLM scheduler preemption and recomputation thrashing.

