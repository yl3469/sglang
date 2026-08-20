#!/usr/bin/env python
"""Collect bench_serving jsonl outputs into a CSV + throughput/latency plot."""
import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = os.path.expanduser("~/hisparse_dp_run/bench")

rows = []
for path in sorted(
    glob.glob(f"{RUN}/*_rate*.jsonl") + glob.glob(f"{RUN}/*_conc*.jsonl")
):
    m = re.match(r"(.+)_(?:rate|conc)([\d.]+)\.jsonl", os.path.basename(path))
    arm, rate = m.group(1), float(m.group(2))
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            rows.append(
                dict(
                    arm=arm,
                    request_rate=rate,
                    input_throughput=r.get("input_throughput"),
                    output_throughput=r.get("output_throughput"),
                    mean_e2e_latency_ms=r.get("mean_e2e_latency_ms"),
                    median_e2e_latency_ms=r.get("median_e2e_latency_ms"),
                    median_ttft_ms=r.get("median_ttft_ms"),
                    p99_ttft_ms=r.get("p99_ttft_ms"),
                    median_tpot_ms=r.get("median_tpot_ms")
                    or r.get("median_itl_ms"),
                    p99_tpot_ms=r.get("p99_tpot_ms") or r.get("p99_itl_ms"),
                    completed=r.get("completed"),
                    duration=r.get("duration"),
                )
            )

if not rows:
    sys.exit("no bench outputs found")

import csv

out_csv = f"{RUN}/summary.csv"
with open(out_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out_csv} ({len(rows)} rows)")

arms = sorted({r["arm"] for r in rows})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
metrics = [
    ("median_tpot_ms", "Median TPOT (ms)"),
    ("median_e2e_latency_ms", "Median E2E latency (ms)"),
    ("median_ttft_ms", "Median TTFT (ms)"),
]
for ax, (key, label) in zip(axes, metrics):
    for arm in arms:
        pts = sorted(
            [
                (r["output_throughput"], r[key])
                for r in rows
                if r["arm"] == arm and r[key] is not None
            ]
        )
        ax.plot(*zip(*pts), marker="o", label=arm)
    ax.set_xlabel("Output throughput (tok/s)")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
axes[0].legend()
fig.suptitle("DSv4-Flash HiSparse: uniform vs DP per-layer partition")
fig.tight_layout()
out_png = f"{RUN}/throughput_latency.png"
fig.savefig(out_png, dpi=150)
print(f"wrote {out_png}")

for arm in arms:
    print(f"\n{arm}:")
    for r in sorted(
        (r for r in rows if r["arm"] == arm), key=lambda r: r["request_rate"]
    ):
        print(
            f"  rate={r['request_rate']:<4} out_tps={r['output_throughput']:.1f} "
            f"tpot={r['median_tpot_ms']:.1f}ms e2e={r['median_e2e_latency_ms']:.0f}ms "
            f"ttft={r['median_ttft_ms']:.0f}ms"
        )
