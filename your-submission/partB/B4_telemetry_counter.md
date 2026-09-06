# B4: Serving Stack Telemetry Counter

To directly confirm the B2 mechanism—that the throughput collapse beyond batch 24 is caused by KV-cache exhaustion triggering scheduler preemption and prompt recomputation—the single definitive metric to pull from the serving engine (vLLM / Triton) is **`vllm:num_preemptions_total`** (the cumulative count of preempted requests emitted via the Prometheus `/metrics` endpoint). 

In production or benchmark runs, we expect this counter to show a value of **exactly `0` for all batch sizes up to 24** (where peak KV cache utilization reaches 0.93 without memory pressure). The moment batch size is increased past physical capacity, we expect this counter to spike immediately to **`7` at batch 32** and **`23` at batch 48**, quantitatively proving that the engine was forced to evict and recompute active sequences due to block exhaustion.

