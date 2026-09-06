#!/usr/bin/env python3
"""
generate_desmos_plots.py -- Generate visual capacity frontier & throughput plots.

Produces:
1. my-submission/partB/capacity_frontier.png
2. Output equations and tables for Desmos Graphing Calculator
"""

import os
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.dirname(__file__)

# -----------------------------------------------------------------------------
# PLOT 1: Throughput vs Batch Size & KV-Cache Collapse (Long Context 4096)
# -----------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

batch_sizes = np.array([4, 8, 16, 24, 32, 48])
reported_tok_s = np.array([565.4, 902.6, 1311.4, 1607.4, 1384.0, 1298.5])
honest_goodput = reported_tok_s * (512.0 / 4096.0)  # Gen tok / wall clock
naive_batch = np.linspace(0, 48, 100)
naive_throughput = 66.975 * naive_batch  # Linear projection from batch 24

# Plot Reported Throughput vs Honest Goodput
ax1.plot(naive_batch, naive_throughput, 'r--', label="Intern's Naive Linear Scaling (hallucinates ~3200 tok/s at batch 48)", alpha=0.7)
ax1.plot(batch_sizes, reported_tok_s, 'bo-', linewidth=2.5, markersize=8, label="Harness Reported Throughput (Prefill + Decode Conflation)")
ax1.plot(batch_sizes, honest_goodput, 'gs-', linewidth=2.5, markersize=8, label="Honest Client Goodput (Generation Only: ~201 tok/s peak)")

# Preemption threshold line
ax1.axvline(x=25.8, color='crimson', linestyle=':', linewidth=2, label="L4 KV-Cache Physical Limit (B = 25.8 seqs)")
ax1.axvspan(25.8, 50, color='red', alpha=0.08, label="KV-Cache Thrashing & Preemption Zone")

# Annotations
ax1.annotate("Peak Safe Throughput\n(1607 tok/s, 0 Preemptions)", xy=(24, 1607.4), xytext=(12, 1800),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
ax1.annotate("Collapse: 7 Preemptions\n(1384 tok/s)", xy=(32, 1384.0), xytext=(33, 1550),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6))
ax1.annotate("Severe Collapse: 23 Preemptions\n(1298 tok/s)", xy=(48, 1298.5), xytext=(35, 1100),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6))
ax1.annotate("Honest Goodput: 200.9 tok/s\n(8x lower than reported!)", xy=(24, 200.9), xytext=(10, 400),
             arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=6))

ax1.set_xlabel("Concurrent Batch Size (Requests)", fontsize=12, fontweight='bold')
ax1.set_ylabel("Throughput (tokens / second)", fontsize=12, fontweight='bold')
ax1.set_title("NVIDIA L4 Serving Throughput Anomaly & Honest Goodput (FLM-4B)", fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_xlim(0, 50)
ax1.set_ylim(0, 3400)
ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "throughput_anomaly_plot.png")
plt.savefig(plot_path)
plt.close()
print(f"Saved throughput plot to {plot_path}")

# -----------------------------------------------------------------------------
# PLOT 2: Memory Frontier (Max Batch Size vs Sequence Length)
# -----------------------------------------------------------------------------
fig, ax2 = plt.subplots(figsize=(10, 6), dpi=300)

seq_lengths = np.linspace(256, 4096, 500)
# Available KV VRAM = 11.25 GiB = 11,520 MiB
# Bytes per token = 112 KiB = 0.109375 MiB
# Memory per seq = seq_len * 0.109375 MiB
max_batch_fp16 = 11520.0 / (seq_lengths * 0.109375)
max_batch_fp8 = 11520.0 / (seq_lengths * 0.0546875)

ax2.plot(seq_lengths, max_batch_fp16, 'b-', linewidth=2.5, label="FP16 KV Cache Frontier (Default)")
ax2.plot(seq_lengths, max_batch_fp8, 'g--', linewidth=2.5, label="FP8 KV Cache Frontier (2x Capacity)")

ax2.scatter([4096], [25.7], color='red', s=100, zorder=5, label="Benchmark Operating Point (S=4096, B_max=25.7)")
ax2.annotate("4096 Tokens -> B_max = 25 seqs", xy=(4096, 25.7), xytext=(3000, 50),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6))

ax2.set_xlabel("Sequence Length (tokens)", fontsize=12, fontweight='bold')
ax2.set_ylabel("Maximum Concurrent Batch Size", fontsize=12, fontweight='bold')
ax2.set_title("NVIDIA L4 (24GB) Serving Capacity Frontier (FLM-4B)", fontsize=14, fontweight='bold', pad=15)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_xlim(256, 4200)
ax2.set_ylim(0, 150)
ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)

plt.tight_layout()
plot_path2 = os.path.join(OUTPUT_DIR, "capacity_frontier_plot.png")
plt.savefig(plot_path2)
plt.close()
print(f"Saved capacity frontier plot to {plot_path2}")
