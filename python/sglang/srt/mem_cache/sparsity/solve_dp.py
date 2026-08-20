"""Solve the HiSparse per-layer buffer partition from an online capture.

Pipeline (see selection_capture.py for the capture side):

1. Read the capped per-layer top-k selection streams recorded by a short
   profiling run (``selection_capture`` in ``--hisparse-config``) and the
   per-rank KV memory reports (``memory_report_path``).
2. Derive the affordable per-layer budget from the tightest rank's free
   memory (overridable with ``--target-mean``).
3. Replay exact per-request LRU at every candidate buffer size on the
   quantum grid -> per-layer miss curves. Misses are counted only for
   steps where the sequence exceeds the buffer (the kernel fast path does
   no host loads below it).
4. Exact min-plus DP: minimize total misses subject to
   sum(sizes) <= layer_num * target_mean, sizes in [top_k, physical].
5. Write a profile JSON consumable as
   ``--hisparse-config '{"layer_buffer_profile": "@profile.json", ...}'``.
   Weights are the solved sizes themselves; serving with
   ``device_buffer_size == max(sizes)`` reproduces them exactly (the
   loader normalizes the max-weight layer to the physical buffer).

Usage::

    python -m sglang.srt.mem_cache.sparsity.solve_dp \
        --capture /x/cap.rank0.pt --reports '/x/memreport.rank*.json' \
        --out /x/dp_profile.json [--target-mean 3072] [--quantum 256]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

import torch

_CAPTURE = None  # per-worker global (fork inheritance)


def _load_capture(path: str):
    return torch.load(path, map_location="cpu", weights_only=True)


def _layer_request_streams(layer_rec):
    """Group one layer's step records into per-request (seq_len, units) streams."""
    streams = {}
    for reqs, lens, topk in zip(
        layer_rec["req_pool_indices"], layer_rec["seq_lens"], layer_rec["top_k"]
    ):
        reqs_l = reqs.tolist()
        lens_l = lens.tolist()
        for row, (rid, seq) in enumerate(zip(reqs_l, lens_l)):
            units = topk[row]
            units = units[units >= 0].tolist()
            streams.setdefault(rid, []).append((seq, units))
    return streams


def _lru_misses(streams, capacity: int) -> int:
    """Exact LRU replay of per-step top-k sets; misses only above fast path."""
    misses = 0
    for stream in streams.values():
        cache: OrderedDict = OrderedDict()
        for seq, units in stream:
            counting = seq > capacity
            for u in units:
                if u in cache:
                    cache.move_to_end(u)
                else:
                    if counting:
                        misses += 1
                    cache[u] = None
                    if len(cache) > capacity:
                        cache.popitem(last=False)
    return misses


def _worker(job):
    layer_id, capacity = job
    layer_rec = _CAPTURE["layers"][layer_id]
    streams = _layer_request_streams(layer_rec)
    return layer_id, capacity, _lru_misses(streams, capacity)


def dp_partition(miss_curves, grid, budget_units, quantum):
    """Exact DP: choose one grid size per layer minimizing total misses.

    miss_curves: list (per layer) of {size: misses}. budget_units: total
    allowed sum(sizes)//quantum. Returns list of chosen sizes.
    """
    layer_num = len(miss_curves)
    grid_units = [s // quantum for s in grid]
    INF = float("inf")
    # dp[b] = (cost, choices) best over layers processed so far using b units
    dp = {0: (0.0, [])}
    for lid in range(layer_num):
        nxt = {}
        for used, (cost, picks) in dp.items():
            for s, su in zip(grid, grid_units):
                nb = used + su
                if nb > budget_units:
                    continue
                nc = cost + miss_curves[lid][s]
                cur = nxt.get(nb)
                if cur is None or nc < cur[0]:
                    nxt[nb] = (nc, picks + [s])
        dp = nxt
    best = min(dp.values(), key=lambda kv: kv[0])
    return best[1], best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--reports", required=True, help="glob of memory reports")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantum", type=int, default=256)
    ap.add_argument(
        "--target-mean",
        type=int,
        default=None,
        help="target mean buffer size; default derived from free memory",
    )
    ap.add_argument(
        "--memory-safety",
        type=float,
        default=0.5,
        help="fraction of reported free memory treated as spendable",
    )
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    global _CAPTURE
    _CAPTURE = _load_capture(args.capture)
    layer_num = _CAPTURE["layer_num"]
    top_k = _CAPTURE["top_k"]
    physical = _CAPTURE["device_buffer_size"]

    reports = [json.load(open(p)) for p in sorted(glob.glob(args.reports))]
    if not reports:
        raise SystemExit(f"no memory reports match {args.reports}")

    # Affordable mean size per layer: current physical size plus what the
    # tightest rank's free memory buys, spread over layers x request slots.
    afford = []
    for rep in reports:
        per_token = rep["layer_num"] * rep["max_num_req_slots"] * rep[
            "item_size_bytes"
        ]
        extra = int(rep["free_bytes"] * args.memory_safety / per_token)
        afford.append(rep["device_buffer_size"] + extra)
    afford_mean = min(afford)

    if args.target_mean is not None:
        target_mean = args.target_mean
    else:
        target_mean = min(afford_mean, physical)
    target_mean = max(top_k, min(target_mean, physical))
    target_mean = (target_mean // args.quantum) * args.quantum

    floor = max(top_k, args.quantum)
    floor = ((floor + args.quantum - 1) // args.quantum) * args.quantum
    grid = list(range(floor, physical + 1, args.quantum))
    if physical not in grid:
        grid.append(physical)

    print(
        f"layers={layer_num} top_k={top_k} physical={physical} "
        f"afford_mean={afford_mean} target_mean={target_mean} "
        f"grid={grid[0]}..{grid[-1]}x{args.quantum} "
        f"steps/layer={_CAPTURE['steps_per_layer'][:4]}..."
    )

    jobs = [(lid, c) for lid in range(layer_num) for c in grid]
    miss_curves = [dict() for _ in range(layer_num)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for lid, cap, misses in pool.map(_worker, jobs, chunksize=4):
            miss_curves[lid][cap] = misses

    budget_units = (target_mean // args.quantum) * layer_num
    sizes, dp_cost = dp_partition(miss_curves, grid, budget_units, args.quantum)

    uniform_size = (target_mean // args.quantum) * args.quantum
    uniform_cost = sum(
        curve[min(grid, key=lambda s: abs(s - uniform_size))]
        for curve in miss_curves
    )
    rel = (uniform_cost - dp_cost) / uniform_cost if uniform_cost else 0.0

    out = {
        "layer_buffer_profile": {str(i): s for i, s in enumerate(sizes)},
        "serve_device_buffer_size": max(sizes),
        "uniform_mean": uniform_size,
        "quantum": args.quantum,
        "predicted": {
            "uniform_misses": uniform_cost,
            "dp_misses": dp_cost,
            "relative_reduction": rel,
        },
        "miss_curves": {
            str(i): {str(s): m for s, m in curve.items()}
            for i, curve in enumerate(miss_curves)
        },
        "memory": {
            "afford_mean": afford_mean,
            "free_bytes_min": min(r["free_bytes"] for r in reports),
            "ranks": len(reports),
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=2)
    print(
        f"sizes mean={sum(sizes)/len(sizes):.0f} min={min(sizes)} "
        f"max={max(sizes)}\npredicted misses: uniform={uniform_cost} "
        f"dp={dp_cost} (-{100*rel:.1f}%)\nwrote {args.out}"
    )
    # The profile passed to the server must contain ONLY layer weights:
    profile_path = args.out.replace(".json", "") + ".profile.json"
    with open(profile_path, "w") as handle:
        json.dump({str(i): s for i, s in enumerate(sizes)}, handle)
    print(
        f"serve with: --hisparse-config '"
        + json.dumps(
            {
                "device_buffer_size": max(sizes),
                "layer_buffer_profile": f"@{profile_path}",
            }
        )
        + "'"
    )


if __name__ == "__main__":
    main()
