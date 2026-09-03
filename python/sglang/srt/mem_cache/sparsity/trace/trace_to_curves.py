"""Stage 3: decoded indexer topk trace -> per-layer LRU/Belady cost curves.

Consumes the per-request ``.npz`` files written by
:mod:`indexer_trace_sink` (each holds the base64-decoded ``indexer.topk``
int32 array plus request geometry) and produces, per model layer, a
:class:`~sglang.srt.mem_cache.sparsity.per_layer_budget_dp.LayerCostCurve`
that the DP allocator consumes.

Design (mirrors the reference ``per_layer_budget_dp`` measurement):

* The demand stream for a layer is the per-decode-step SELECTION of context
  slot ids (``indexer.topk``); a negative entry is padding and is dropped
  (reference uses ``topk >= 0``).
* Capacity is expressed as a RATIO of the request's context length, so one
  allocation profile applies to requests of different lengths. For each
  request and each ratio we replay LRU (or Belady) at a fixed
  ``buffer = int(ratio * slot_ctx0)`` and record the mean demand-miss RATE.
* Miss rates are averaged across requests per ``(layer, ratio)`` (equal-request
  mean), then packaged as a cost curve whose ``sizes`` are integer ratio units
  (``ratio / RATIO_STEP``) and whose ``costs`` are mean miss rates. This lets
  the token-unit :func:`dp_allocate` run directly on ratio units (grid_step=1),
  and the allocated ``size`` reads back as an integer ratio unit.

Everything here is pure host-side analysis (numpy only for I/O); the replay
primitives come from :mod:`per_layer_budget_replay`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sglang.srt.mem_cache.sparsity.per_layer_budget_dp import LayerCostCurve
from sglang.srt.mem_cache.sparsity.per_layer_budget_replay import (
    replay_belady_fast,
    replay_lru_fast,
    SelectionTrace,
)

__all__ = [
    "RATIO_STEP",
    "RequestTopk",
    "topk_array_to_selection_traces",
    "load_request_topk",
    "measure_miss_rate_curves",
    "build_ratio_cost_curves",
    "ratio_units",
]

# Ratio discretization: the DP runs in integer ratio units, one unit ==
# RATIO_STEP of a request's context length. 0.001 (permille) matches the
# reference dp_step granularity closely while staying integer-exact.
RATIO_STEP = 0.001

# Default capacity ratios to measure per layer (mirrors reference MEASURE_RATIOS,
# trimmed to what a short interactive decode can exercise). Overridable via CLI.
DEFAULT_MEASURE_RATIOS: Tuple[float, ...] = (
    0.04,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
)


def ratio_units(ratio: float) -> int:
    """Convert a capacity ratio to integer DP grid units (round to RATIO_STEP)."""
    return int(round(ratio / RATIO_STEP))


@dataclass
class RequestTopk:
    """One request's decoded indexer topk trace.

    ``topk`` has shape ``(steps, num_indexer_layers, index_topk)``; entries are
    context slot ids, negative entries are padding. ``slot_ctx0`` is the context
    slot count used to convert a ratio into an absolute buffer size.
    """

    topk: np.ndarray  # int, (steps, layers, topk)
    slot_ctx0: int
    meta: Dict[str, object]

    @property
    def num_layers(self) -> int:
        return int(self.topk.shape[1])


def _reshape_topk(flat: np.ndarray, num_layers: int, index_topk: int) -> np.ndarray:
    """Reshape a flat int32 topk buffer to ``(steps, num_layers, index_topk)``.

    Mirrors ``extract_indexer_topk_from_meta_info`` + the documented reshape
    ``(seqlen-1, num_indexer_layers, index_topk)``.
    """
    if num_layers <= 0 or index_topk <= 0:
        raise ValueError(
            f"num_layers ({num_layers}) and index_topk ({index_topk}) must be "
            "positive to reshape the topk buffer"
        )
    per_step = num_layers * index_topk
    if flat.size % per_step != 0:
        raise ValueError(
            f"topk buffer of size {flat.size} is not divisible by "
            f"num_layers*index_topk ({per_step}); geometry mismatch"
        )
    steps = flat.size // per_step
    return flat.reshape(steps, num_layers, index_topk)


def _stride_steps(topk: np.ndarray, max_steps: Optional[int]) -> np.ndarray:
    """Uniformly subsample the step axis of a ``(steps, layers, topk)`` array.

    Per-layer selection locality is stationary across decode, so a uniform
    stride preserves the miss-curve shape while bounding the pure-Python replay
    cost (the full trace can be tens of thousands of steps). Returns the array
    unchanged when ``max_steps`` is None or already covers every step.
    """
    if max_steps is None or topk.shape[0] <= max_steps:
        return topk
    idx = np.linspace(0, topk.shape[0] - 1, num=max_steps).round().astype(np.int64)
    idx = np.unique(idx)
    return topk[idx]


def topk_array_to_selection_traces(
    topk: np.ndarray,
    max_steps: Optional[int] = None,
) -> List[SelectionTrace]:
    """Split a ``(steps, layers, topk)`` array into one SelectionTrace/layer.

    Negative ids (padding) are dropped. Returns a list of length ``layers``;
    each entry is a list over steps of the selected slot ids at that step.
    ``max_steps`` uniformly subsamples the step axis first (see
    :func:`_stride_steps`) to bound replay cost on very long traces.
    """
    if topk.ndim != 3:
        raise ValueError(f"expected 3D (steps, layers, topk), got {topk.shape}")
    topk = _stride_steps(topk, max_steps)
    steps, num_layers, _ = topk.shape
    per_layer: List[SelectionTrace] = [[] for _ in range(num_layers)]
    for layer in range(num_layers):
        layer_view = topk[:, layer, :]
        for step in range(steps):
            row = layer_view[step]
            # Vectorized mask + tolist() is far cheaper than a per-element
            # Python int() comprehension on wide (index_topk=512) rows.
            sel = row[row >= 0].tolist()
            per_layer[layer].append(sel)
    return per_layer


def _slot_ctx0_from_traces(
    per_layer: Sequence[SelectionTrace], explicit: Optional[int]
) -> int:
    """Context slot count used to size buffers from ratios.

    Prefer the explicit value stored by the sink (prefill-derived). Otherwise
    fall back to the largest slot id observed anywhere in the trace + 1, an
    upper bound on the context reached during decode (decode adds only a few
    slots on top of a large prefill, so this closely tracks the initial
    context the reference uses).
    """
    if explicit is not None and explicit > 0:
        return int(explicit)
    max_id = -1
    for trace in per_layer:
        for step in trace:
            for tok in step:
                if tok > max_id:
                    max_id = tok
    return max_id + 1


def load_request_topk(path: str) -> RequestTopk:
    """Load one ``req_*.npz`` written by :mod:`indexer_trace_sink`."""
    with np.load(path, allow_pickle=False) as npz:
        keys = set(npz.files)
        if "topk_indices" in keys:
            # Preferred: already reshaped 3D int array.
            topk = np.asarray(npz["topk_indices"])
            if topk.ndim == 1:
                num_layers = int(npz["num_layers"])
                index_topk = int(npz["index_topk"])
                topk = _reshape_topk(topk, num_layers, index_topk)
        elif "indexer_topk_flat" in keys:
            num_layers = int(npz["num_layers"])
            index_topk = int(npz["index_topk"])
            topk = _reshape_topk(
                np.asarray(npz["indexer_topk_flat"]), num_layers, index_topk
            )
        else:
            raise KeyError(
                f"{path} has none of topk_indices/indexer_topk_flat; keys={keys}"
            )
        meta: Dict[str, object] = {}
        for k in ("prompt_len", "output_len", "index_topk", "num_layers"):
            if k in keys:
                meta[k] = int(npz[k])
        explicit_ctx = int(npz["slot_ctx0"]) if "slot_ctx0" in keys else None
    if explicit_ctx is not None and explicit_ctx > 0:
        slot_ctx0 = int(explicit_ctx)
    else:
        # Fall back to the largest slot id observed + 1 (vectorized; avoids
        # building the full per-step Python lists just to size buffers).
        max_id = int(topk.max()) if topk.size else -1
        slot_ctx0 = max_id + 1
    return RequestTopk(topk=topk.astype(np.int64), slot_ctx0=slot_ctx0, meta=meta)


def _demand_count(trace: SelectionTrace) -> int:
    """Total demanded (de-duplicated per step) tokens across the trace."""
    return sum(len(set(step)) for step in trace)


def measure_miss_rate_curves(
    requests: Sequence[RequestTopk],
    ratios: Sequence[float],
    policy: str = "lru",
    min_buffer_slots: int = 513,
    max_steps: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[float]]:
    """Measure per-layer mean miss RATE at each capacity ratio.

    Returns ``(rate_per_layer, ratios_used)`` where ``rate_per_layer[l]`` is a
    float array aligned to ``ratios_used`` giving the equal-request mean
    demand-miss rate for layer ``l``. A ratio is skipped for a request when its
    buffer would fall below ``min_buffer_slots`` (matches the reference floor:
    ``index_topk`` worst-case misses + 1 new KV, no prefetch). ``max_steps``
    uniformly subsamples the decode step axis to bound replay cost on long
    traces (locality is stationary across decode).
    """
    if policy not in ("lru", "belady"):
        raise ValueError("policy must be 'lru' or 'belady'")
    replay = replay_lru_fast if policy == "lru" else replay_belady_fast
    if not requests:
        raise ValueError("no requests to measure")
    num_layers = requests[0].num_layers
    for r in requests:
        if r.num_layers != num_layers:
            raise ValueError(
                f"inconsistent layer count across requests: "
                f"{r.num_layers} vs {num_layers}"
            )
    ratios = sorted(ratios)

    # Accumulate (sum_rate, count) per (layer, ratio) for the equal-request mean.
    sum_rate = np.zeros((num_layers, len(ratios)), dtype=np.float64)
    count = np.zeros((num_layers, len(ratios)), dtype=np.int64)

    for req in requests:
        per_layer = topk_array_to_selection_traces(req.topk, max_steps=max_steps)
        for layer, trace in enumerate(per_layer):
            demands = _demand_count(trace)
            if demands == 0:
                continue
            for ri, ratio in enumerate(ratios):
                buffer_size = int(req.slot_ctx0 * ratio)
                if buffer_size < min_buffer_slots:
                    continue
                misses = replay(trace, buffer_size)
                sum_rate[layer, ri] += misses / demands
                count[layer, ri] += 1

    rate_per_layer: List[np.ndarray] = []
    for layer in range(num_layers):
        with np.errstate(invalid="ignore", divide="ignore"):
            rates = np.where(count[layer] > 0, sum_rate[layer] / count[layer], np.nan)
        rate_per_layer.append(rates)
    return rate_per_layer, ratios


def build_ratio_cost_curves(
    rate_per_layer: Sequence[np.ndarray],
    ratios: Sequence[float],
) -> List[LayerCostCurve]:
    """Package per-layer miss-rate-vs-ratio into DP cost curves (ratio units).

    ``sizes`` are integer ratio units (``ratio / RATIO_STEP``); ``costs`` are
    mean miss rates. Ratios with no valid measurement (NaN) are dropped. Costs
    are made non-increasing in size (miss rate should not rise with capacity;
    tiny measurement noise is clamped) so the DP sees a well-formed curve.
    """
    curves: List[LayerCostCurve] = []
    for rates in rate_per_layer:
        sizes: List[int] = []
        costs: List[float] = []
        for ratio, rate in zip(ratios, rates):
            if np.isnan(rate):
                continue
            unit = ratio_units(ratio)
            if sizes and unit == sizes[-1]:
                continue
            sizes.append(unit)
            costs.append(float(rate))
        if not sizes:
            raise ValueError(
                "a layer had no valid ratio measurements; lower the ratios or "
                "raise the decode length / context so buffers clear the floor"
            )
        # Enforce non-increasing cost with increasing size (clamp noise).
        for i in range(1, len(costs)):
            if costs[i] > costs[i - 1]:
                costs[i] = costs[i - 1]
        curves.append(LayerCostCurve(sizes=sizes, costs=costs))
    return curves


def _gather_request_files(inputs: Sequence[str]) -> List[str]:
    files: List[str] = []
    for item in inputs:
        if os.path.isdir(item):
            files.extend(sorted(glob.glob(os.path.join(item, "req_*.npz"))))
        elif any(ch in item for ch in "*?["):
            files.extend(sorted(glob.glob(item)))
        else:
            files.append(item)
    if not files:
        raise FileNotFoundError(f"no req_*.npz found under {inputs}")
    return files


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
        "--min-buffer-slots",
        type=int,
        default=513,
        help="Skip a ratio for a request if its buffer is below this floor.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Uniformly subsample each trace to at most this many decode steps "
        "before replay (bounds pure-Python replay cost on long traces).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to dump the curves as JSON (sizes in ratio units).",
    )
    args = parser.parse_args(argv)

    files = _gather_request_files(args.inputs)
    requests = [load_request_topk(f) for f in files]
    print(
        f"Loaded {len(requests)} requests, "
        f"{requests[0].num_layers} indexer layers, "
        f"slot_ctx0 range "
        f"[{min(r.slot_ctx0 for r in requests)}, "
        f"{max(r.slot_ctx0 for r in requests)}]"
    )
    rate_per_layer, ratios = measure_miss_rate_curves(
        requests,
        args.measure_ratios,
        policy=args.policy,
        min_buffer_slots=args.min_buffer_slots,
        max_steps=args.max_steps,
    )
    curves = build_ratio_cost_curves(rate_per_layer, ratios)
    print(f"Built {len(curves)} cost curves ({args.policy}).")
    for layer, curve in enumerate(curves):
        pretty = ", ".join(
            f"{s * RATIO_STEP:g}:{c:.3f}" for s, c in zip(curve.sizes, curve.costs)
        )
        print(f"  layer {layer:>3}: {pretty}")

    if args.out:
        payload = {
            "policy": args.policy,
            "ratio_step": RATIO_STEP,
            "ratios": list(ratios),
            "curves": [
                {"sizes": list(c.sizes), "costs": list(c.costs)} for c in curves
            ],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
