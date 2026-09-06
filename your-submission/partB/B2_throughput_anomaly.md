# B2: Long-Context Throughput Anomaly & Remediation

## 1. Identification of the Throughput Anomaly

In LLM serving, increasing batch size under memory-bandwidth-bound decode generally yields sub-linear or linear throughput scaling until GPU compute or memory is exhausted. In the short-prompt sweep (`prompt_len = 512`), throughput increases monotonically from batch 1 (70.2 tok/s) through batch 64 (2267.3 tok/s).

However, in the **long-context sweep (`prompt_len = 3584`, `gen_len = 512`, total 4096 tokens)**, a severe anomaly occurs:
- From batch 4 to batch 24, throughput scales smoothly from **565.4 tok/s** up to a peak of **1607.4 tok/s** (Row 12).
- At **Batch 32 (Row 13)**, throughput abruptly **collapses to 1384.0 tok/s** (a **13.9% drop**), while wall clock time explodes from 61.16s to **94.71s** (+54.9%).
- At **Batch 48 (Row 14)**, throughput degrades further to **1298.5 tok/s** (a **19.2% drop** below peak), while wall clock time explodes to **151.41s** (+147.6%).

Instead of scaling towards the intern's projected ~3200 tok/s, throughput degrades sharply past batch 24.

---

## 2. Explanation of the Mechanism (Row & Column Evidence)

The anomaly is caused by **KV-Cache Exhaustion leading to Scheduler Thrashing & Recomputation Preemption**.

| Row Index | `batch_size` | `reported_tok_s` | `wall_clock_s` | `preempted_seqs` | `kv_cache_util` | `ttft_ms_p50` | `e2e_ms_p95` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Row 12** | 24 | 1607.4 | 61.16 | **0** | **0.93** | 500.5 ms | 69,221 ms (69.2s) |
| **Row 13** | 32 | 1384.0 | 94.71 | **7** | **0.97** | 636.9 ms | 97,465 ms (97.5s) |
| **Row 14** | 48 | 1298.5 | 151.41 | **23** | **0.97** | 955.4 ms | 105,427 ms (105.4s) |

### Step-by-Step Mechanism
1. **Physical Capacity Limit**: As derived in B1, 1× NVIDIA L4 (24GB) can hold at most **25 concurrent 4096-token sequences** in fp16 KV cache. At batch 24, `kv_cache_util` hits **0.93** with 0 preemptions.
2. **Block Exhaustion & Preemption**: When 32 requests arrive simultaneously, the required KV blocks ($32 \times 448\text{ MiB} = 14.3\text{ GB}$) exceed the free KV cache pool (~11.3 GB). Once utilization hits 0.97, the vLLM scheduler cannot allocate blocks for decoding and is forced to **preempt running sequences** (`preempted_seqs = 7` at batch 32, and `preempted_seqs = 23` at batch 48).
3. **Recomputation Thrashing**: Under default vLLM swap/recompute policies without CPU offload, preempted sequences have their KV blocks freed and must undergo a **full re-prefill (recomputing all 3,584 prompt tokens)** once earlier sequences complete.
4. **Catastrophic Latency Inflation**:
   - Re-running prefills consumes dense tensor compute and memory bandwidth, stalling active decode steps.
   - Median TTFT (`ttft_ms_p50`) jumps from 500.5 ms (batch 24) to **636.9 ms** (batch 32) and **955.4 ms** (batch 48).
   - End-to-end P95 request latency (`e2e_ms_p95`) balloons from 69.2s to **105.4s**, degrading client SLA while burning power on redundant recomputations.

---

## 3. Proposed Config / Deployment Change & Quantitative Predictions

We propose two complementary interventions:

### Option A (Software Config): Enforce `max_num_seqs = 24` in vLLM Serving Configuration
- **Action**: In the vLLM server launch arguments, set:
  ```bash
  vllm serve FLM-4B-Instruct --max-num-seqs 24 --max-model-len 4096
  ```
- **Mechanism**: The engine schedules at most 24 concurrent active sequences. For a burst of 48 requests, the first 24 execute to completion without any memory pressure, and the remaining 24 execute immediately after in a clean second wave.
- **Predicted Quantitative Effect**:
  - `preempted_seqs` drops from 23 to **0**.
  - No prompt recomputations are performed.
  - Total wall-clock time for 48 requests drops from 151.41s to:
    $$\text{Wall Clock} \approx 2 \times 61.16\text{s} = \mathbf{122.32\text{s}} \quad (\mathbf{-19.2\%\text{ reduction in latency}})$$
  - Effective throughput increases from 1298.5 tok/s to **~1607 tok/s** (a **+23.8% throughput increase**).

### Option B (Architectural): Enable FP8 KV-Cache Quantization (`kv_cache_dtype="fp8"`)
- **Action**: Quantize Key and Value states to 8-bit float:
  ```bash
  vllm serve FLM-4B-Instruct --kv-cache-dtype fp8 --max-model-len 4096
  ```
- **Mechanism**: Cuts per-token KV memory in half, from 112 KiB down to **56 KiB** (and 4096-token sequence footprint from 448 MiB down to **224 MiB**).
- **Predicted Quantitative Effect**:
  - Maximum concurrent sequence capacity doubles from 25 to **$\approx 51$ sequences**.
  - All 48 sequences at batch 48 fit simultaneously in GPU VRAM without preemption (`preempted_seqs = 0`).
  - Batch 48 achieves true saturation scaling, delivering an estimated **~2200–2400 tok/s** with wall clock completing in **~85–90s** (a **40% latency reduction**).

