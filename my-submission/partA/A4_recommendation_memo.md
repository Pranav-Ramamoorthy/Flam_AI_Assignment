# Executive Memo: Corrected Tokenizer Economics & Routing Policy

**To:** AI Leadership & Infrastructure Planning  
**From:** Pranav R (AI Team Intern — Audit Working Group)  
**Date:** September 7, 2026  
**Subject:** Correction of Tokenizer Benchmarks (`REPORT_v0.md`) & Indic Capacity Allocation  

---

### 1. Executive Summary & Corrected Headline Numbers
`REPORT_v0.md` concluded that Hindi fertility is 5.89× worse than English and advised budgeting a **6× serving cost multiplier** for Indic traffic. **This conclusion is fundamentally invalid and should be retracted immediately.**

The previous analysis suffered from two fatal flaws:
1. **Tokenizer Conflation**: It benchmarked `gpt2`, an English-only tokenizer (50k vocab) with zero Indic vocabulary, which fell back to byte-level BPE (emitting 2–3 tokens per character).
2. **Metric Invariance Failure**: It measured *tokens per whitespace word*. Dravidian languages (Kannada, Tamil) are agglutinative, packing multiple grammatical morphemes into single long words. This artificially shrank the word denominator, falsely inflating reported fertility.

When evaluated on an aligned 1,012-sentence parallel benchmark (FLORES-200) across semantic units (tokens per parallel sentence):
- Under an Indic-aware tokenizer (`google/muril-base`), Hindi requires only **1.16×** tokens relative to English; Kannada requires **1.07×**, and Tamil requires **1.06×** (near parity).
- Under a general multilingual model (`xlm-roberta-base`), the true cost multiplier across Indic languages is only **1.25× to 1.35×**.

| Language | v0 Claimed Cost Ratio | True Semantic Cost Ratio (`muril`) | True Semantic Cost Ratio (`xlm-r`) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **English** | 1.00× (baseline) | 1.00× | 1.00× | Baseline |
| **Hindi** | **6.00×** *(budgeted)* | **1.16×** | **1.25×** | Over-budgeted by **4.8×** |
| **Kannada** | Not measured | **1.07×** | **1.35×** | Managed within standard headroom |
| **Tamil** | Not measured | **1.06×** | **1.35×** | Managed within standard headroom |

---

### 2. Strategic Routing & Capacity Recommendation
- **Cancel Segregated Indic Infrastructure**: Do not provision a separate, 6× cost-allocated cluster for Indic traffic.
- **Enforce Multilingual Vocabulary in Base Model**: Ensure our serving model uses a modern multilingual tokenizer (minimum 128k vocabulary with native Devanagari and Dravidian subwords, as in our FLM-4B serving stack).
- **Correct Capacity Budget**: Budget an additional **+20% to +35% serving capacity headroom** for native Indic queries relative to English, rather than +500% (6×). This adjustment immediately recovers capital expenditure across planned GPU provisioning.

---

### 3. The Single Biggest Caveat
**The Literary-to-Colloquial Distribution Gap & Romanized Code-Mixing (Hinglish/Tanglish)**.  
Our audited numbers derive from FLORES-200, which features formal, grammatically edited prose written in native scripts. Production chatbot traffic in India heavily features **Romanized transliteration and code-mixing** (e.g., *"Aaj ka schedule update kar do"*). Tokenizers trained strictly on native Devanagari/Dravidian scripts often decompose Romanized Indic into single-character tokens. If production traffic shifts heavily to Romanized Indic, token inflation can spike unpredictably.

---

### 4. The One Production Metric to Monitor
$$\boxed{\mathbf{R}_{\text{prod}} = \frac{\mathbf{P95\text{ Total Tokens per Request}}(\text{Target Language})}{\mathbf{P95\text{ Total Tokens per Request}}(\text{English})}}$$

**Implementation**: Stream request telemetry into Prometheus grouped by detected language. Alert if $R_{\text{prod}} > 1.50\times$ over a rolling 1-hour window. This counter directly captures prompt length, generation verbosity, and subword fragmentation under live production traffic.

