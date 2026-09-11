"""Aggregate A/B HiSparse sweep bench_serving JSONL outputs into one CSV.

Scans ``<results-dir>/<arm>/bench_c<concurrency>.jsonl`` (written by
``bench_serving --output-file``) for arms ``uniform`` and ``dp`` and emits a
tidy ``summary.csv`` with one row per (arm, concurrency): request + token
throughput and the P50(median)/P90/P99 of TTFT, TPOT (time-per-output-token =
inter-token / TBT), and E2E latency. Also prints a compact side-by-side table
and the DP-vs-uniform deltas so the serving benefit is immediately readable.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Optional, Sequence

ARMS = ("uniform", "dp")

# (csv_column, json_key)
COLUMNS = [
    ("request_throughput", "request_throughput"),
    ("output_throughput", "output_throughput"),
    ("total_throughput", "total_throughput"),
    ("completed", "completed"),
    ("total_output_tokens", "total_output_tokens"),
    ("sharegpt_output_len", "sharegpt_output_len"),
    ("p50_ttft_ms", "median_ttft_ms"),
    ("p90_ttft_ms", "p90_ttft_ms"),
    ("p99_ttft_ms", "p99_ttft_ms"),
    ("p50_tpot_ms", "median_tpot_ms"),
    ("p90_tpot_ms", "p90_tpot_ms"),
    ("p99_tpot_ms", "p99_tpot_ms"),
    ("p50_e2e_ms", "median_e2e_latency_ms"),
    ("p90_e2e_ms", "p90_e2e_latency_ms"),
    ("p99_e2e_ms", "p99_e2e_latency_ms"),
]


def _load_jsonl_last(path: str) -> Optional[dict]:
    """Return the last valid JSON object in a JSONL file (bench appends one)."""
    last = None
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return None
    return last


def collect(results_dir: str) -> List[dict]:
    rows: List[dict] = []
    for arm in ARMS:
        arm_dir = os.path.join(results_dir, arm)
        for path in sorted(glob.glob(os.path.join(arm_dir, "bench_c*.jsonl"))):
            data = _load_jsonl_last(path)
            if data is None:
                continue
            conc = data.get("max_concurrency")
            if conc is None:
                # derive from filename bench_c<N>.jsonl
                base = os.path.basename(path)
                try:
                    conc = int(base.split("bench_c", 1)[1].split(".", 1)[0])
                except (IndexError, ValueError):
                    conc = -1
            row = {"arm": arm, "concurrency": conc}
            for col, key in COLUMNS:
                row[col] = data.get(key)
            rows.append(row)
    rows.sort(key=lambda r: (r["concurrency"], r["arm"]))
    return rows


def write_csv(rows: Sequence[dict], out: str) -> None:
    fieldnames = ["arm", "concurrency"] + [c for c, _ in COLUMNS]
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def print_table(rows: Sequence[dict]) -> None:
    by_conc: Dict[int, Dict[str, dict]] = {}
    for r in rows:
        by_conc.setdefault(r["concurrency"], {})[r["arm"]] = r

    hdr = (
        f"{'conc':>5} {'arm':>7} {'req/s':>8} {'tok/s':>9} "
        f"{'ttft_p50':>9} {'ttft_p99':>9} {'tpot_p50':>9} {'tpot_p99':>9} "
        f"{'e2e_p50':>10} {'e2e_p99':>10}"
    )
    print("\n=== A/B serving sweep (uniform baseline vs DP per-layer buffer) ===")
    print(hdr)
    print("-" * len(hdr))
    for conc in sorted(by_conc):
        for arm in ARMS:
            r = by_conc[conc].get(arm)
            if not r:
                continue
            print(
                f"{conc:>5} {arm:>7} "
                f"{_fmt(r['request_throughput']):>8} {_fmt(r['output_throughput']):>9} "
                f"{_fmt(r['p50_ttft_ms']):>9} {_fmt(r['p99_ttft_ms']):>9} "
                f"{_fmt(r['p50_tpot_ms']):>9} {_fmt(r['p99_tpot_ms']):>9} "
                f"{_fmt(r['p50_e2e_ms']):>10} {_fmt(r['p99_e2e_ms']):>10}"
            )
        # DP vs uniform deltas (negative latency / positive throughput = DP better)
        u, d = by_conc[conc].get("uniform"), by_conc[conc].get("dp")
        if u and d:

            def delta_pct(dp_v, un_v):
                if dp_v is None or un_v is None or un_v == 0:
                    return "-"
                return f"{(dp_v - un_v) / un_v * 100:+.1f}%"

            print(
                f"{conc:>5} {'Δdp':>7} "
                f"{delta_pct(d['request_throughput'], u['request_throughput']):>8} "
                f"{delta_pct(d['output_throughput'], u['output_throughput']):>9} "
                f"{delta_pct(d['p50_ttft_ms'], u['p50_ttft_ms']):>9} "
                f"{delta_pct(d['p99_ttft_ms'], u['p99_ttft_ms']):>9} "
                f"{delta_pct(d['p50_tpot_ms'], u['p50_tpot_ms']):>9} "
                f"{delta_pct(d['p99_tpot_ms'], u['p99_tpot_ms']):>9} "
                f"{delta_pct(d['p50_e2e_ms'], u['p50_e2e_ms']):>10} "
                f"{delta_pct(d['p99_e2e_ms'], u['p99_e2e_ms']):>10}"
            )
    print(
        "\nΔdp = (dp - uniform)/uniform. For latency, negative = DP faster; "
        "for throughput, positive = DP higher.\n"
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    rows = collect(args.results_dir)
    if not rows:
        print(f"no bench_c*.jsonl found under {args.results_dir}/{{uniform,dp}}/")
        return
    write_csv(rows, args.out)
    print(f"Wrote {args.out} ({len(rows)} rows)")
    print_table(rows)


if __name__ == "__main__":
    main()
