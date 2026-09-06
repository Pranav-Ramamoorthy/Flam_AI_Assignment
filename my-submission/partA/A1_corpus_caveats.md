# A1: Multilingual Evaluation Corpus & Caveats

## 1. Corpus Selection & Overview
To replace the 10-sentence unaligned toy sample (`corpus_sample/`), we assembled a professional-grade parallel evaluation corpus using the **FLORES-200** benchmark (NLLB Consortium / Meta AI), specifically utilizing the verified `devtest` split.

The dataset covers four target languages:
1. **English (`eng`)**: `eng_Latn` — Germanic / Indo-European (analytic baseline).
2. **Hindi (`hin`)**: `hin_Deva` — Indo-Aryan / Indo-European (moderately inflected with postpositions).
3. **Kannada (`kan`)**: `kan_Knda` — Southern Dravidian (highly agglutinative).
4. **Tamil (`tam`)**: `tam_Taml` — Southern Dravidian (highly agglutinative).

All four language files are strictly aligned line-by-line across 1,012 parallel sentences, translated by professional translators from identical source materials.

---

## 2. Corpus Statistics & Quantitative Profile

| Language | ISO Code | Script | Sentence Count | Total Words (`.split()`) | Total Characters (Code Points) | Total UTF-8 Bytes | Bytes / Char | Words / Sentence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **English** | `eng_Latn` | Latin | 1,012 | 21,901 | 131,966 | 132,096 | 1.001 B | 21.64 |
| **Hindi** | `hin_Deva` | Devanagari | 1,012 | 25,643 | 131,180 | 337,439 | 2.572 B | 25.34 |
| **Kannada** | `kan_Knda` | Kannada | 1,012 | 16,100 | 138,027 | 375,341 | 2.719 B | 15.91 |
| **Tamil** | `tam_Taml` | Tamil | 1,012 | 16,775 | 154,131 | 421,635 | 2.736 B | 16.58 |

### File Integrity Hashes (SHA-256)
- `data/eng.txt`: `31b42bf432b350616b2cfd1e345fb691c95b08e2fce12d2745a55bc99d98e826`
- `data/hin.txt`: `ab36525552467b7da7f5a0346fb8cfd4ea6b9fcbcfecf1bc99955e88bbdbf5c1`
- `data/kan.txt`: `5d6c0a8dba394fb00dae3a479ff73552086e3f890250df7a2ae4e2c079dbf170`
- `data/tam.txt`: `d2b5192c60d70038891fc51c272bc194feefba10f545a0b77b1029c78b6da2b1`

---

## 3. Domain & Preprocessing Pipeline
- **Domain**: High-quality expository and narrative prose sampled across multiple domains: news journalism, Wikipedia encyclopedic articles, travel guides, science, culture, and history.
- **Preprocessing Applied**:
  1. **Unicode NFC Normalization**: Every line was passed through `unicodedata.normalize("NFC", line)` to compose canonical combining characters (e.g., base consonants and nuktas/matras), preventing artificial byte/token fragmentation.
  2. **Whitespace Stripping & Uniform Delimitation**: Lines were stripped of leading/trailing carriage returns and tabs, while preserving internal single-space word boundaries. Empty rows were purged.
  3. **Strict Alignment Check**: An automated assertion ensures that line $k$ in English corresponds exactly to line $k$ in Hindi, Kannada, and Tamil.

---

## 4. What This Corpus Cannot Tell You (Critical Caveats)
*(Sample-size and domain caveats are signal, not weakness.)*

While FLORES-200 provides a rigorous, gold-standard parallel baseline for benchmarking tokenization efficiency across standardized written registers, **leadership must not mistake this benchmark for a replica of production customer traffic**. Specifically, this corpus cannot tell us:

1. **Conversational Register & Colloquialisms**: FLORES-200 consists of professionally edited, grammatically pristine written prose. Production assistant traffic consists overwhelmingly of colloquial dialogue, conversational idioms, fragments, and informal register. Dravidian spoken colloquialisms collapse or drop formal case suffixes, which alters word and character distributions substantially compared to literary texts.
2. **Code-Mixing & Script Blending (Hinglish, Tanglish, Kanglish)**: In real-world Indian mobile and web usage, a massive portion of Indic traffic is written in Roman script (transliteration) or features dense code-mixing (e.g., mixing English technical nouns with Hindi verbs: *"Meeting reschedule kar do"*). FLORES-200 is 100% monolingual and native-script; it provides zero signal on Romanized Indic fertility or script-switching boundaries.
3. **Formatting & Non-Linguistic Artifacts**: Production chatbot prompts feature emojis, JSON/XML snippets, URLs, markdown formatting, numbered lists, and punctuation-dense tables. None of these operational structures appear in FLORES sentences.
4. **Prompt Length Distribution & Multi-Turn History**: FLORES sentences average 16–25 words per sentence (~100–150 characters). Real assistant prompts follow a bimodal distribution: ultra-short queries (2–5 words, e.g., *"Aaj ka mausam?"*) and very long contexts (system prompts, retrieved documents, and multi-turn chat history reaching 2,000–4,000 tokens). Tokenizer compression behavior on isolated 20-word sentences does not capture prefix caching efficiencies or subword merge reuse across multi-paragraph context windows.

