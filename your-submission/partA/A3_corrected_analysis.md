# A3: Corrected Cross-Language Analysis & Denominator Reasoning

## 1. Empirical Cross-Language Benchmark Results

We benchmarked 1,012 parallel sentences from the FLORES-200 evaluation corpus across four distinct tokenizers and four distinct denominators:
1. **Per whitespace word** (proper `.split()`, micro-averaged)
2. **Per extended grapheme cluster** (Unicode UAX #29 aksharas, via regex `\X`)
3. **Per UTF-8 byte** (raw binary footprint)
4. **Per parallel sentence** (semantic content invariant)

### Summary Table across All Tokenizers and Languages

| Tokenizer | Vocab Size | Lang | Tok / Sent | Tok / Word (Micro) | Tok / Grapheme | Tok / Byte | Tok / Char | Semantic Cost Ratio (vs Eng) | Word Fertility Ratio (vs Eng) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gpt2`** *(v0 Baseline)* | 50,257 | **`eng`** | 26.72 | 1.23 | 0.205 | 0.205 | 0.205 | **1.00×** | **1.00×** |
| | | **`hin`** | 198.31 | 7.83 | 2.335 | 0.595 | 1.530 | **7.42×** | **6.34×** |
| | | **`kan`** | 363.01 | 22.82 | 4.065 | 0.979 | 2.662 | **13.58×** | **18.48×** |
| | | **`tam`** | 415.19 | 25.05 | 4.213 | 0.997 | 2.726 | **15.54×** | **20.28×** |
| **`xlm-roberta-base`** | 250,002 | **`eng`** | 30.30 | 1.40 | 0.232 | 0.232 | 0.232 | **1.00×** | **1.00×** |
| *(Multilingual S-Piece)* | | **`hin`** | 37.77 | 1.49 | 0.445 | 0.113 | 0.291 | **1.25×** | **1.06×** |
| | | **`kan`** | 40.97 | 2.58 | 0.459 | 0.110 | 0.300 | **1.35×** | **1.84×** |
| | | **`tam`** | 40.86 | 2.47 | 0.415 | 0.098 | 0.268 | **1.35×** | **1.76×** |
| **`google/muril-base`** | 197,285 | **`eng`** | 27.25 | 1.26 | 0.209 | 0.209 | 0.209 | **1.00×** | **1.00×** |
| *(Indic-Specialized)* | | **`hin`** | 31.55 | 1.24 | 0.371 | 0.095 | 0.243 | **1.16×** | **0.99×** |
| | | **`kan`** | 29.03 | 1.82 | 0.325 | 0.078 | 0.213 | **1.07×** | **1.45×** |
| | | **`tam`** | 28.86 | 1.74 | 0.293 | 0.069 | 0.189 | **1.06×** | **1.38×** |
| **`Qwen/Qwen2.5-0.5B`** | 151,643 | **`eng`** | 27.29 | 1.26 | 0.209 | 0.209 | 0.209 | **1.00×** | **1.00×** |
| *(Modern LLM Tokenizer)*| | **`hin`** | 120.51 | 4.76 | 1.419 | 0.361 | 0.930 | **4.42×** | **3.77×** |
| | | **`kan`** | 188.86 | 11.87 | 2.115 | 0.509 | 1.385 | **6.92×** | **9.41×** |
| | | **`tam`** | 166.83 | 10.06 | 1.693 | 0.400 | 1.095 | **6.11×** | **7.98×** |

---

## 2. Key Empirical Revelations

1. **Debunking the "Script Property" Hallucination**:
   - `REPORT_v0.md` asserted: *"Hindi simply has more Unicode characters per word, so any tokenizer will struggle. This is a property of the script, not the tokenizer."*
   - This claim is completely false. Under `gpt2`, Hindi required 198 tokens per sentence because GPT-2 has zero Indic vocabulary entries and falls back to byte-level BPE, emitting 2 to 3 tokens for every 3-byte Devanagari character.
   - When evaluated under `xlm-roberta-base`, Hindi requires only **37.77 tokens/sentence** (only **1.25×** English).
   - Under `google/muril-base-cased`, Hindi requires **31.55 tokens/sentence** (**1.16×** English), and Kannada and Tamil require only **29.03** and **28.86 tokens/sentence** (**1.07×** and **1.06×** English, virtually at parity!).
   - The token inflation in v0 was 100% an artifact of using an English-only tokenizer on Indic scripts, not an inherent property of the script.

2. **The Agglutinative Distortion of "Tokens Per Word"**:
   - Notice the divergence between `Tok / Sent` ratio and `Tok / Word` ratio for Kannada and Tamil under `google/muril-base`:
     - Kannada requires 29.03 tokens per sentence vs 27.25 in English (semantic cost ratio = **1.07×**).
     - However, Kannada's tokens per word is 1.82 vs 1.26 in English (word fertility ratio = **1.45×**).
   - Why? Because Kannada is agglutinative, packing multiple grammatical morphemes into single long words (15.9 words/sentence in Kannada vs 21.6 words/sentence in English).
   - Dividing tokens by whitespace words artificially inflates the cost metric for Dravidian languages by +35% simply because their grammar does not use spaces between particles!

---

## 3. Denominator Reasoning: What Does Each Denominator Hold Constant?

To choose the correct metric, we must analyze the physical and semantic invariance of each candidate denominator:

### A. Whitespace Word (`tokens / word`)
- **What it holds constant**: Arbitrary typographical whitespace separations.
- **Why it fails**: Language typology makes whitespace meaningless across language families. Analytic languages (English) use many short grammatical words; agglutinative languages (Kannada, Tamil) concatenate stems, case markers, and postpositions into composite single words. Holding words constant punishes agglutinative languages for having compact syntax.

### B. Unicode Characters / UTF-8 Bytes (`tokens / char`, `tokens / byte`)
- **What it holds constant**: Orthographic codepoint counts or binary storage bytes.
- **Why it fails**: In ASCII, English characters are 1 byte each. In UTF-8, Indic scripts (Devanagari, Kannada, Tamil) require 3 bytes per character. Furthermore, Indic scripts are abugidas where vowels attach as diacritic matras and consonant conjuncts fuse via viramas. Comparing tokens per byte or character conflates script encoding density with semantic information content.

### C. Extended Grapheme Clusters (`tokens / grapheme`)
- **What it holds constant**: User-perceived orthographic glyph units (UAX #29 aksharas).
- **Why it fails**: While linguistically superior to raw bytes, grapheme counts still vary widely based on phonological syllable structure across languages and do not correspond linearly to computational serving workload.

### D. Parallel Sentence / Semantic Intent (`tokens / parallel sentence`)
- **What it holds constant**: **THE SEMANTIC INFORMATION CONTENT AND USER INTENT.**
- **Why it is correct**: When a user submits a prompt to an AI assistant, they are expressing a specific semantic request. In a parallel corpus, each sentence pair conveys the exact same information. Serving cost (prefill FLOPS, KV-cache memory allocation, memory bandwidth for autoregressive decode, latency, and API token billing) is strictly driven by the **absolute number of tokens processed to fulfill that semantic request**.

---

## 4. The Single Decision Metric: Normalized Semantic Token Multiplier ($M_{\text{semantic}}$)

### Definition
$$\boxed{M_{\text{semantic}}(\text{lang}) = \frac{\text{Mean Tokens per Parallel Sentence}(\text{lang})}{\text{Mean Tokens per Parallel Sentence}(\text{English})}}$$

### Why this Single Number Should Drive Routing and Cost Decisions
1. **Direct Operational Alignment**: GPU memory allocation ($B \times S \times \text{bytes\_per\_token}$) and time-to-first-token (TTFT) depend strictly on total sequence length ($S$). If an Indic prompt requires 1.25× more tokens than English to convey the same instructions, the GPU must allocate exactly 1.25× more KV-cache and execute 1.25× more decode iterations.
2. **Eliminates Typological Bias**: It is invariant to whether a language is analytic, synthetic, agglutinative, or fusional.
3. **Actionable Financial Capacity**:
   - Under `REPORT_v0.md`'s flawed metric (7.45 tok/word with `gpt2`), leadership was told to budget **6.0× serving capacity** for Hindi.
   - Under the true semantic metric with an Indic-capable tokenizer (`xlm-roberta-base` or `google/muril-base`), the actual capacity multiplier is only **1.06× to 1.25×**.
   - Adopting $M_{\text{semantic}}$ prevents millions of dollars in unneeded GPU over-provisioning.

