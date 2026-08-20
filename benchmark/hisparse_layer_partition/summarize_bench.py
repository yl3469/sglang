#!/usr/bin/env python
"""Collect bench_serving jsonl outputs into a CSV + throughput/latency plot."""
import csv
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
    m = re.match(r"(.+)_(rate|conc)([\d.]+)\.jsonl", os.path.basename(path))
    arm, load_kind, load = m.group(1), m.group(2), float(m.group(3))
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            rows.append(
                dict(
                    arm=arm,
                    load_kind=load_kind,
                    load=load,
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

out_csv = f"{RUN}/summary.csv"
with open(out_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out_csv} ({len(rows)} rows)")

# Two workloads on separate rows so the load axes are not conflated:
#   24k row: rate sweep (open-loop Poisson arrivals, req/s)
#   64k row: closed-loop concurrency sweep
WORKLOADS = [
    (
        "24k",
        "24k-in / 256-out random, rate sweep 0.5–8 req/s "
        "(32 prompts per point)",
        lambda a: a.endswith("3072"),
        "rate {g:g} req/s",
    ),
    (
        "64k",
        "64k-in / 256-out random, concurrency sweep 4/8/16 "
        "(16 prompts per point)",
        lambda a: a.endswith("64k"),
        "conc {g:g}",
    ),
]
ARM_STYLE = {  # arm prefix -> (label, color, marker)
    "uniform": ("Uniform buffer (3072 slots/layer)", "#1f77b4", "o"),
    "dp": ("DP partition (per-layer 1024–4096, mean 3072)", "#d62728", "s"),
}
METRICS = [
    ("median_tpot_ms", "Median TPOT (ms)"),
    ("median_e2e_latency_ms", "Median E2E latency (ms)"),
    ("median_ttft_ms", "Median TTFT (ms)"),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
for row_i, (wname, wdesc, warm, gfmt) in enumerate(WORKLOADS):
    for col_i, (key, label) in enumerate(METRICS):
        ax = axes[row_i][col_i]
        for prefix, (alabel, color, marker) in ARM_STYLE.items():
            pts = sorted(
                (r["load"], r["output_throughput"], r[key])
                for r in rows
                if r["arm"].startswith(prefix)
                and warm(r["arm"])
                and r[key] is not None
            )
            if not pts:
                continue
            ax.plot(
                [p[1] for p in pts],
                [p[2] for p in pts],
                marker=marker,
                color=color,
                label=alabel,
            )
            for g, x, y in pts:
                ax.annotate(
                    gfmt.format(g=g),
                    (x, y),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color=color,
                )
        ax.set_xlabel("Output throughput (tok/s)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if col_i == 0:
            ax.legend(fontsize=8, loc="upper left")
            ax.set_title(wdesc, fontsize=9, loc="left")
fig.suptitle(
    "DSv4-Flash FP8, 4×GB300 TP4, SGLang HiSparse — "
    "uniform vs DP per-layer partition at equal effective mean capacity\n"
    "(bench_serving random dataset; each point labeled with its request "
    "rate / concurrency)",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out_png = f"{RUN}/throughput_latency.png"
fig.savefig(out_png, dpi=150)
print(f"wrote {out_png}")

for wname, wdesc, warm, _ in WORKLOADS:
    print(f"\n== {wdesc}")
    for prefix, (alabel, _, _) in ARM_STYLE.items():
        sel = sorted(
            (r for r in rows if r["arm"].startswith(prefix) and warm(r["arm"])),
            key=lambda r: r["load"],
        )
        if not sel:
            continue
        print(f"  {alabel}:")
        for r in sel:
            print(
                f"    {r['load_kind']}={r['load']:<4g} "
                f"out_tps={r['output_throughput']:.1f} "
                f"tpot={r['median_tpot_ms']:.1f}ms "
                f"e2e={r['median_e2e_latency_ms']:.0f}ms "
                f"ttft={r['median_ttft_ms']:.0f}ms"
            )
