"""Stage 4: per-layer DP capacity allocation + reference-schema CSV.

Ties the collected indexer trace to the exact min-plus DP allocator
(:func:`sglang.srt.mem_cache.sparsity.per_layer_budget_dp.dp_allocate`) and
emits ``dp_allocations.csv`` in the same schema as the reference
``dsa-offloading`` output::

    total_ratio,predicted_uniform_miss_rate,predicted_dp_miss_rate,layer_<id>...

Flow:

1. Load ``req_*.npz`` traces (Stage 1 output) and build per-layer cost curves in
   integer RATIO units (Stage 3, :mod:`trace_to_curves`).
2. For each total budget (a *sweep* of mean capacity ratios, PLUS an optional
   budget DERIVED from recorded free KV memory), run ``dp_allocate`` with
   ``grid_step = 1`` (one grid unit == ``RATIO_STEP`` of context) and
   ``mean_budget = round(total_ratio / RATIO_STEP)``.
3. Report, per total budget, the predicted uniform vs DP mean miss rate and the
   per-layer allocated ratio; also emit an LRU->Belady gap report.

The DP objective and semantics are identical to the reference (sum of per-layer
miss cost, min-plus convolution over a discretized budget), but the cost curves
here are MEASURED empirically from the real sglang decode trace (LRU / Belady
replay) rather than an analytical log-log fit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Sequence

from sglang.srt.mem_cache.sparsity.per_layer_budget_dp import (
    LayerCostCurve,
    dp_allocate,
)
from sglang.srt.mem_cache.sparsity.trace.trace_to_curves import (
    DEFAULT_MEASURE_RATIOS,
    RATIO_STEP,
    build_ratio_cost_curves,
    load_request_topk,
    measure_miss_rate_curves,
    ratio_units,
)

__all__ = [
    "uniform_miss_rate",
    "allocate_for_total",
    "kv_budget_to_ratio",
    "build_dp_rows",
    "write_dp_allocations_csv",
]


def uniform_miss_rate(curves: Sequence[LayerCostCurve], mean_grid: int) -> float:
    """Mean per-layer miss rate when every layer gets exactly ``mean_grid``.

    Mirrors the reference ``predicted_uniform_miss_rate`` (average over layers
    of the cost at the uniform per-layer ratio).
    """
    return sum(c.cost_at(mean_grid) for c in curves) / len(curves)


def allocate_for_total(
    curves: Sequence[LayerCostCurve],
    total_ratio: float,
    floors_units: Optional[Sequence[int]] = None,
):
    """Run the DP at one total (mean) capacity ratio.

    Returns ``(allocation, predicted_dp_miss_rate, predicted_uniform_miss_rate)``.
    ``mean_budget`` is the mean ratio in grid units; the DP distributes
    ``mean_budget * num_layers`` across layers. ``predicted_dp_miss_rate`` is the
    summed DP cost divided by the layer count (mean miss rate, matching the
    reference).
    """
    mean_grid = ratio_units(total_ratio)
    if mean_grid <= 0:
        raise ValueError(f"total_ratio {total_ratio} rounds to zero grid units")
    alloc = dp_allocate(curves, mean_budget=mean_grid, grid_step=1, floors=floors_units)
    predicted_dp = alloc.total_cost / len(curves)
    predicted_uniform = uniform_miss_rate(curves, mean_grid)
    return alloc, predicted_dp, predicted_uniform


def kv_budget_to_ratio(
    kv_budget: Dict[str, object],
    num_layers: int,
    slot_ctx0: int,
    compress_ratio: int = 4,
) -> Optional[float]:
    """Derive a mean per-layer capacity ratio from recorded free KV memory.

    The recorded ``token_capacity`` (per-DP ``max_total_num_tokens``) is the
    number of KV slots the engine can hold. Split evenly across the indexer
    layers gives a mean per-layer device-buffer slot budget; dividing by the
    request's context slot count (``slot_ctx0``, already in compressed slots)
    yields the mean capacity RATIO the DP consumes.

    Returns ``None`` if the budget json lacks a usable capacity field.
    """
    cap = None
    for key in (
        "per_dp_token_capacity",
        "token_capacity",
        "max_total_num_tokens",
        "kv_available_tokens",
    ):
        val = kv_budget.get(key)
        if isinstance(val, list) and val:
            val = min(v for v in val if v)
        if isinstance(val, (int, float)) and val > 0:
            cap = float(val)
            break
    if cap is None or slot_ctx0 <= 0 or num_layers <= 0:
        return None
    mean_slots_per_layer = cap / num_layers
    ratio = mean_slots_per_layer / float(slot_ctx0)
    return ratio


def build_dp_rows(
    curves: Sequence[LayerCostCurve],
    total_ratios: Sequence[float],
    layer_ids: Sequence[int],
    kv_derived_ratio: Optional[float] = None,
) -> List[Dict[str, object]]:
    """Build the ``dp_allocations.csv`` rows for each total budget.

    Each row: ``total_ratio``, ``predicted_uniform_miss_rate``,
    ``predicted_dp_miss_rate``, and one ``layer_<id>`` per layer (allocated
    ratio). The KV-derived budget, if given and inside the measured grid, is
    appended as an extra row.
    """
    if len(layer_ids) != len(curves):
        raise ValueError(
            f"layer_ids ({len(layer_ids)}) must match curve count ({len(curves)})"
        )
    max_grid = max(c.max_size for c in curves)

    totals = list(total_ratios)
    kv_rows: set = set()
    if kv_derived_ratio is not None:
        # The DP needs the grid to extend ABOVE the mean; clamp the derived
        # ratio just below the largest measured ratio so the allocation stays
        # feasible while still reflecting the hardware headroom.
        derived_units = ratio_units(kv_derived_ratio)
        if derived_units >= max_grid:
            derived_units = max_grid - 1
        derived_ratio = max(RATIO_STEP, derived_units * RATIO_STEP)
        totals.append(derived_ratio)
        kv_rows.add(derived_ratio)

    rows: List[Dict[str, object]] = []
    for total in totals:
        alloc, predicted_dp, predicted_uniform = allocate_for_total(curves, total)
        row: Dict[str, object] = {
            "total_ratio": total,
            "predicted_uniform_miss_rate": predicted_uniform,
            "predicted_dp_miss_rate": predicted_dp,
            "kv_budget_derived": int(total in kv_rows),
        }
        for lid, grid in zip(layer_ids, alloc.per_layer_grid):
            row[f"layer_{lid}"] = grid * RATIO_STEP
        rows.append(row)
    return rows


def write_dp_allocations_csv(
    path: str, rows: Sequence[Dict[str, object]], layer_ids: Sequence[int]
) -> None:
    """Write rows to CSV using the reference column order.

    Columns: ``total_ratio, predicted_uniform_miss_rate,
    predicted_dp_miss_rate, kv_budget_derived, layer_<id>...`` (layers sorted).
    """
    fieldnames = [
        "total_ratio",
        "predicted_uniform_miss_rate",
        "predicted_dp_miss_rate",
        "kv_budget_derived",
    ] + [f"layer_{lid}" for lid in layer_ids]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _resolve_layer_ids(
    arg_ids: Optional[Sequence[int]],
    num_layers: int,
    offset: int,
    stride: int,
) -> List[int]:
    if arg_ids:
        if len(arg_ids) != num_layers:
            raise ValueError(
                f"--layer-ids has {len(arg_ids)} entries but there are "
                f"{num_layers} indexer layers"
            )
        return list(arg_ids)
    return [offset + stride * i for i in range(num_layers)]


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="req_*.npz files, a glob, or a directory of traces.",
    )
    parser.add_argument("--policy", default="lru", choices=("lru", "belady"))
    parser.add_argument(
        "--measure-ratios",
        nargs="*",
        type=float,
        default=list(DEFAULT_MEASURE_RATIOS),
    )
    parser.add_argument(
        "--total-ratios",
        nargs="*",
        type=float,
        default=[0.1, 0.15, 0.2, 0.3],
        help="Uniform-equivalent mean budgets; total = num_layers * ratio.",
    )
    parser.add_argument("--min-buffer-slots", type=int, default=513)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Uniformly subsample each trace to at most this many decode steps "
        "before replay (locality is stationary across decode; bounds the "
        "pure-Python replay cost on long traces). Default: use all steps.",
    )
    parser.add_argument(
        "--kv-budget",
        default=None,
        help="kv_budget.json (Stage 2) to derive an extra total-ratio row.",
    )
    parser.add_argument(
        "--compress-ratio",
        type=int,
        default=4,
        help="DSv4 indexer compression ratio (geometry only).",
    )
    parser.add_argument(
        "--layer-ids",
        nargs="*",
        type=int,
        default=None,
        help="Explicit model-layer id per indexer layer (for CSV columns).",
    )
    parser.add_argument("--layer-offset", type=int, default=0)
    parser.add_argument("--layer-stride", type=int, default=1)
    parser.add_argument(
        "--skip-gap",
        action="store_true",
        help="Skip the (expensive) LRU->Belady gap report; emit only "
        "dp_allocations.csv. Belady replay is offline-optimal but much slower "
        "than LRU on large traces.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/per_layer_budget_dp",
        help="Directory for dp_allocations.csv + dp_gap_report.json.",
    )
    args = parser.parse_args(argv)

    from sglang.srt.mem_cache.sparsity.trace.trace_to_curves import (
        _gather_request_files,
    )

    files = _gather_request_files(args.inputs)
    requests = [load_request_topk(f) for f in files]
    num_layers = requests[0].num_layers
    mean_slot_ctx0 = int(sum(r.slot_ctx0 for r in requests) / max(1, len(requests)))
    print(
        f"Loaded {len(requests)} requests, {num_layers} indexer layers, "
        f"mean slot_ctx0 {mean_slot_ctx0}"
    )

    rates, ratios = measure_miss_rate_curves(
        requests,
        args.measure_ratios,
        policy=args.policy,
        min_buffer_slots=args.min_buffer_slots,
        max_steps=args.max_steps,
    )
    curves = build_ratio_cost_curves(rates, ratios)

    layer_ids = _resolve_layer_ids(
        args.layer_ids, num_layers, args.layer_offset, args.layer_stride
    )

    kv_derived_ratio = None
    if args.kv_budget:
        with open(args.kv_budget) as f:
            kv_budget = json.load(f)
        kv_derived_ratio = kv_budget_to_ratio(
            kv_budget,
            num_layers=num_layers,
            slot_ctx0=mean_slot_ctx0,
            compress_ratio=args.compress_ratio,
        )
        print(f"KV-derived mean capacity ratio: {kv_derived_ratio}")

    rows = build_dp_rows(
        curves, args.total_ratios, layer_ids, kv_derived_ratio=kv_derived_ratio
    )
    csv_path = os.path.join(args.out_dir, "dp_allocations.csv")
    write_dp_allocations_csv(csv_path, rows, layer_ids)
    print(f"Wrote {csv_path}")
    for row in rows:
        tag = " [kv-derived]" if row.get("kv_budget_derived") else ""
        print(
            f"  total {row['total_ratio']:g}: uniform="
            f"{row['predicted_uniform_miss_rate'] * 100:.2f}% "
            f"dp={row['predicted_dp_miss_rate'] * 100:.2f}%{tag}"
        )

    # LRU->Belady gap report at the sweep totals (only meaningful for LRU curves;
    # rebuild Belady curves to quantify the achievable headroom).
    if args.policy == "lru" and not args.skip_gap:
        belady_rates, belady_ratios = measure_miss_rate_curves(
            requests,
            args.measure_ratios,
            policy="belady",
            min_buffer_slots=args.min_buffer_slots,
            max_steps=args.max_steps,
        )
        belady_curves = build_ratio_cost_curves(belady_rates, belady_ratios)
        gap_rows = []
        for total in args.total_ratios:
            mean_grid = ratio_units(total)
            lru_u = uniform_miss_rate(curves, mean_grid)
            bel_u = uniform_miss_rate(belady_curves, mean_grid)
            _, lru_dp, _ = allocate_for_total(curves, total)
            _, bel_dp, _ = allocate_for_total(belady_curves, total)
            gap_rows.append(
                {
                    "total_ratio": total,
                    "lru_uniform_miss_rate": lru_u,
                    "belady_uniform_miss_rate": bel_u,
                    "lru_dp_miss_rate": lru_dp,
                    "belady_dp_miss_rate": bel_dp,
                    "uniform_gap": lru_u - bel_u,
                    "dp_gap": lru_dp - bel_dp,
                }
            )
        gap_path = os.path.join(args.out_dir, "dp_gap_report.json")
        with open(gap_path, "w") as f:
            json.dump(gap_rows, f, indent=2)
        print(f"Wrote {gap_path}")


if __name__ == "__main__":
    main()
