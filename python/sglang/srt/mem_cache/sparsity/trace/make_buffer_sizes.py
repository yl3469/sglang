"""Convert a per-layer DP allocation CSV into HiSparse device-buffer configs.

Reads ``dp_allocations.csv`` (produced by :mod:`run_per_layer_dp`) and emits two
per-layer device-buffer-size vectors for an A/B serving comparison:

* **uniform** — every layer gets the same buffer ``B_mean`` (the baseline: one
  scalar capacity shared by all layers, matching stock HiSparse).
* **dp**      — the DP per-layer allocation ``B_l``, scaled to the SAME total
  budget as the uniform arm so the two configs use identical device memory and
  differ only in how that memory is *distributed* across layers.

Both are written as JSON files (a bare list) that
``--hisparse-config '{"device_buffer_sizes_path": "..."}'`` can load, and the
matching ``--hisparse-config`` JSON strings are printed for convenience.

Buffer sizes are in COMPRESSED KV slots (the unit HiSparse's device buffer uses)
and every entry is floored at ``top_k`` (index_topk) and rounded to a multiple of
``page_size`` so the physical layout stays page-aligned. The uniform and DP
totals are matched exactly (any rounding remainder is added to the layer with
the largest DP ratio).

Example::

    python -m sglang.srt.mem_cache.sparsity.trace.make_buffer_sizes \
        --dp-csv results/.../per_layer_budget_dp/dp_allocations.csv \
        --total-ratio 0.3 --b-mean 1024 --top-k 512 --page-size 64 \
        --out-dir results/.../buffer_configs
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple


def _round_to_page(x: int, page_size: int) -> int:
    """Round ``x`` up to the nearest positive multiple of ``page_size``."""
    if page_size <= 1:
        return int(x)
    return int(((x + page_size - 1) // page_size) * page_size)


# A fitted row is only trusted for a served budget within this relative
# distance of its total_ratio; beyond it the DP shape is extrapolated onto a
# budget it was never optimized for (the 0.1-row-served-at-0.04 mistake).
AUTO_MAX_REL_DISTANCE = 0.25


def served_total_ratio(csv_path: str, b_mean: int) -> float:
    """Served budget as a ratio of context: ``b_mean / mean_slot_ctx0`` from the
    ``dp_meta.json`` sidecar run_per_layer_dp writes next to the CSV."""
    meta_path = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "dp_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found; rerun run_per_layer_dp (it writes the sidecar) "
            "or pass an explicit --total-ratio"
        )
    with open(meta_path) as f:
        meta = json.load(f)
    return b_mean / float(meta["mean_slot_ctx0"])


def read_dp_row(
    csv_path: str,
    total_ratio: Optional[float],
    *,
    served_ratio: Optional[float] = None,
) -> Tuple[Dict[int, float], float]:
    """Read one row of dp_allocations.csv.

    Returns ``(per_layer_ratio, chosen_total_ratio)`` where ``per_layer_ratio``
    maps model-layer id -> allocated capacity ratio. If ``total_ratio`` is None
    the row with the LARGEST DP improvement over uniform is chosen (the most
    informative A/B point); otherwise the row whose ``total_ratio`` matches
    (closest) is used. With ``served_ratio`` (auto mode) the closest row is
    used only if it lies within ``AUTO_MAX_REL_DISTANCE`` of the served budget;
    otherwise a UNIFORM shape is returned, because no fitted row covers the
    budget actually served and the DP itself predicts no gain there.
    """
    rows: List[dict] = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        raise ValueError(f"{csv_path} has no data rows")

    if served_ratio is not None:
        chosen = min(rows, key=lambda r: abs(float(r["total_ratio"]) - served_ratio))
        rel = abs(float(chosen["total_ratio"]) - served_ratio) / served_ratio
        if rel > AUTO_MAX_REL_DISTANCE:
            print(
                f"WARNING: served budget ratio {served_ratio:.3f} is {rel:.0%} away "
                f"from the nearest fitted row ({float(chosen['total_ratio']):g}); "
                "using a UNIFORM allocation (dp == uniform). Serve above the fitted "
                "rows' budget or refit with --total-ratios covering it."
            )
            layer_ratio = {
                int(k.split("_", 1)[1]): served_ratio
                for k in chosen
                if k.startswith("layer_")
            }
            return layer_ratio, served_ratio
    elif total_ratio is None:
        # Largest (uniform - dp) miss-rate gap.
        def gap(r: dict) -> float:
            return float(r["predicted_uniform_miss_rate"]) - float(
                r["predicted_dp_miss_rate"]
            )

        chosen = max(rows, key=gap)
    else:
        chosen = min(rows, key=lambda r: abs(float(r["total_ratio"]) - total_ratio))

    layer_ratio: Dict[int, float] = {}
    for k, v in chosen.items():
        if k.startswith("layer_"):
            layer_ratio[int(k.split("_", 1)[1])] = float(v)
    if not layer_ratio:
        raise ValueError(f"no layer_<id> columns found in {csv_path}")
    return layer_ratio, float(chosen["total_ratio"])


def build_buffer_sizes(
    layer_ratio: Dict[int, float],
    b_mean: int,
    top_k: int,
    page_size: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Build (uniform, dp, layer_ids) buffer-size vectors at matched total.

    ``b_mean`` is the uniform per-layer buffer (compressed slots). The DP arm
    distributes the SAME total (``b_mean * L``) across layers in proportion to
    the DP ratios, floored at ``top_k`` and page-aligned, with the rounding
    remainder assigned to preserve the exact total.
    """
    layer_ids = sorted(layer_ratio)
    L = len(layer_ids)
    ratios = [layer_ratio[l] for l in layer_ids]
    mean_ratio = sum(ratios) / L

    # Floor every layer at top_k, page-aligned.
    floor = _round_to_page(top_k, page_size)
    b_mean = max(_round_to_page(b_mean, page_size), floor)

    uniform = [b_mean] * L
    total_budget = b_mean * L

    # DP arm: proportional to ratio, floored, page-aligned.
    raw = [b_mean * (r / mean_ratio) if mean_ratio > 0 else b_mean for r in ratios]
    dp = [max(_round_to_page(int(round(x)), page_size), floor) for x in raw]

    # Reconcile to the exact uniform total by nudging the largest-ratio layers
    # in page-sized steps (keeps page alignment and the >=floor guarantee).
    order = sorted(range(L), key=lambda i: ratios[i], reverse=True)
    step = max(1, page_size)
    guard = 100000
    while sum(dp) > total_budget and guard > 0:
        # remove a page from the currently-largest that can afford it
        for i in sorted(range(L), key=lambda i: dp[i], reverse=True):
            if dp[i] - step >= floor:
                dp[i] -= step
                break
        else:
            break
        guard -= 1
    while sum(dp) < total_budget and guard > 0:
        dp[order[0]] += step
        # re-sort so we don't pile everything on one layer
        order = sorted(range(L), key=lambda i: (ratios[i], -dp[i]), reverse=True)
        guard -= 1

    return uniform, dp, layer_ids


def _write(out_dir: str, name: str, sizes: Sequence[int]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"buffer_sizes_{name}.json")
    with open(path, "w") as f:
        json.dump({"device_buffer_sizes": list(sizes)}, f)
    return path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp-csv", required=True)
    parser.add_argument(
        "--total-ratio",
        default=None,
        help="Which CSV row to use: a ratio, 'auto' (= b_mean / mean context "
        "slots from dp_meta.json, falling back to uniform if no fitted row is "
        "within 25%% of it), or omitted (row with the largest DP gain).",
    )
    parser.add_argument(
        "--b-mean",
        type=int,
        required=True,
        help="Uniform per-layer device buffer in COMPRESSED slots (baseline).",
    )
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    if args.total_ratio is None or args.total_ratio == "auto":
        served = (
            served_total_ratio(args.dp_csv, args.b_mean)
            if args.total_ratio == "auto"
            else None
        )
        layer_ratio, chosen = read_dp_row(args.dp_csv, None, served_ratio=served)
        if served is not None:
            print(f"auto: served budget ratio {served:.3f} (b_mean={args.b_mean})")
    else:
        layer_ratio, chosen = read_dp_row(args.dp_csv, float(args.total_ratio))
    uniform, dp, layer_ids = build_buffer_sizes(
        layer_ratio, args.b_mean, args.top_k, args.page_size
    )

    up = _write(args.out_dir, "uniform", uniform)
    dpp = _write(args.out_dir, "dp", dp)

    print(f"Using dp_allocations row total_ratio={chosen:g}, {len(layer_ids)} layers")
    print(f"  uniform total={sum(uniform)} sizes[:5]={uniform[:5]}  -> {up}")
    print(
        f"  dp      total={sum(dp)} min={min(dp)} max={max(dp)} "
        f"sizes[:5]={dp[:5]}  -> {dpp}"
    )
    assert sum(uniform) == sum(dp), (sum(uniform), sum(dp))
    print("  (totals matched: uniform and DP use identical device memory)")
    print("\nUse with:")
    print(
        f'  UNIFORM: --hisparse-config \'{{"top_k":{args.top_k},'
        f'"device_buffer_sizes_path":"{up}"}}\''
    )
    print(
        f'  DP:      --hisparse-config \'{{"top_k":{args.top_k},'
        f'"device_buffer_sizes_path":"{dpp}"}}\''
    )


if __name__ == "__main__":
    main()
