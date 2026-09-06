# B3: Misread Metric Audit & Honest Goodput Derivations

## 1. The Misread Column: `reported_tok_s` Conflates Prefill with Generation

In Section 2 of `REPORT_v0.md`, the intern observed:
> *"at batch 16, long prompts hit 1311 tok/s vs only 883 tok/s for short prompts. Longer prompts clearly give better GPU utilization... assume ~1600 tok/s per L4 (best observed) and scale linearly with batch size, so batch 48 should give us ~3200 tok/s."*

Both conclusions stem from a fundamental misunderstanding of the column **`reported_tok_s`**.

### What `reported_tok_s` Actually Measures
The benchmark harness calculates `reported_tok_s` by summing **ALL tokens processed (prompt tokens + generation tokens)** and dividing by the total run wall-clock time:
$$\text{reported\_tok\_s} = \frac{(\text{num\_requests} \times \text{prompt\_len}) + (\text{num\_requests} \times \text{gen\_len})}{\text{wall\_clock\_s}}$$

### Mathematical Proof for Batch-24 Long-Prompt (Row 12)
From `bench_log.csv`:
- `batch_size` = 24, `prompt_len` = 3584, `gen_len` = 512, `wall_clock_s` = 61.16
$$\text{Prompt Tokens} = 24 \times 3584 = 86,016\text{ tokens}$$
$$\text{Generated Tokens} = 24 \times 512 = 12,288\text{ tokens}$$
$$\text{Total Tokens Processed} = 86,016 + 12,288 = 98,304\text{ tokens}$$
$$\frac{98,304\text{ tokens}}{61.16\text{ s}} = \mathbf{1607.33\text{ tok/s}} \quad (\text{matches } 1607.4\text{ in the log})$$

In this run, **87.5% ($3584 / 4096 = 7/8$) of all tokens are prompt tokens**. Prompt tokens are processed during **prefill**, which is highly parallelized, compute-bound, and executes thousands of tokens per second (86,016 tokens in ~500 ms = ~172,000 tok/s during prefill). In contrast, **generation (decode)** is strictly autoregressive, memory-bandwidth bound, producing one token per step.

By dividing total tokens by elapsed time, `reported_tok_s` masks slow generation speed by padding the numerator with thousands of instantaneous prefill tokens. The illusion that "longer prompts give better throughput" is an artifact of measuring input processing speed rather than output generation speed.

---

## 2. Deriving Honest Generation "Goodput" for Batch-24 Long-Prompt

In LLM serving economics, **goodput** is defined as the rate of useful generated output tokens delivered to clients per second. We derive the honest goodput using **two independent methods**:

### Method 1: End-to-End Generation Throughput over Total Run Duration
$$\text{Goodput}_{\text{E2E}} = \frac{\text{Total Output (Generated) Tokens}}{\text{Total Elapsed Wall Clock Time}}$$
$$\text{Goodput}_{\text{E2E}} = \frac{24 \times 512\text{ tokens}}{61.16\text{ s}} = \frac{12,288}{61.16} = \mathbf{200.92\text{ tok/s}}$$

*Equivalent derivation by scaling the reported column:*
$$\text{Goodput}_{\text{E2E}} = \text{reported\_tok\_s} \times \frac{\text{gen\_len}}{\text{prompt\_len} + \text{gen\_len}} = 1607.4 \times \frac{512}{4096} = \mathbf{200.93\text{ tok/s}}$$

---

### Method 2: Steady-State Autoregressive Decode Rate from ITL
During the decode phase, each autoregressive step emits 1 token per active sequence across the batch ($24\text{ tokens/step}$). The time between successive tokens is measured by the median inter-token latency (`itl_ms_p50`):
- From Row 12: `itl_ms_p50` = $96.07\text{ ms} = 0.09607\text{ s}$ per token step.

$$\text{Goodput}_{\text{decode}} = \frac{\text{Concurrent Active Batch}}{\text{itl\_s\_p50}} = \frac{24\text{ tokens}}{0.09607\text{ s}} = \mathbf{249.82\text{ tok/s}}$$

*(Note: Method 2 measures pure steady-state generation speed during decode, whereas Method 1 includes the initial ~500 ms prefill latency. Excluding TTFT: $\frac{12,288}{61.16\text{s} - 0.5005\text{s}} = \mathbf{202.57\text{ tok/s}}$).*

---

## 3. Comparison of Claims vs Measured Reality

| Metric | Intern's Claim in `REPORT_v0` | Audited Engineering Reality | Delta / Overstatement |
| :--- | :--- | :--- | :--- |
| **Batch-24 Throughput** | 1607.4 tok/s | **200.9 tok/s** (Goodput E2E) | **8.0× Overstatement** |
| **Batch-48 Throughput** | ~3200 tok/s *(projected linear)*| **1298.5 tok/s** (Total) / **162.3 tok/s** (Goodput) | **16.0× Overstatement** |
| **Prompt Length Impact** | "Longer prompts give better throughput" | Longer prompts crash batch capacity and increase latency | Flawed Recommendation |

**Interactive Desmos Model:** An interactive visualization comparing the intern's reported throughput against honest generation goodput is available at [https://www.desmos.com/calculator/je5emtbbzy](https://www.desmos.com/calculator/je5emtbbzy).

---

## 4. What `REPORT_v0` Should Have Said

> **Revised Section 2 for Leadership:**  
> "The headline harness throughput of 1607 tok/s for long prompts is misleading: 87.5% of those tokens are prompt prefill tokens. True client generation goodput is only **~201 tok/s**.  
> Longer prompts do not improve serving economics; on the contrary, each 3584-token prompt consumes 392 MiB of KV cache before generation even begins. This caps our safe concurrency to **24 sequences on an NVIDIA L4**.  
> If clients pack 48 concurrent requests, the GPU runs out of KV memory, causing 23 sequences to be preempted and recomputed. This degrades generation goodput to **162 tok/s** (a 20% drop from peak) and blows P95 latency out to 105 seconds. Capacity planning must budget for **~200–250 output tok/s per L4**, with strict concurrency limits at batch 24."

