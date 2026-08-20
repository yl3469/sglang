"""Per-layer HiSparse device-buffer partition from a pre-solved DP profile.

Layers differ several-fold in KV miss rate at equal capacity, so a single
``device_buffer_size`` wastes device memory on easy layers. This module
turns a pre-solved per-layer allocation profile (an exact min-plus DP over
measured per-layer miss curves; solved offline in the dsa-offloading repo,
``offline_analysis/per_layer_budget_dp.py``) into per-layer effective
buffer sizes for :class:`HiSparseCoordinator`.

Design constraints (v1, deliberately minimal):

- The physical per-request buffer stays uniform at
  ``device_buffer_size + page`` per layer; per-layer sizes only bound the
  LRU region the swap-in kernel manages, so ``B_l <= device_buffer_size``.
  Weights are therefore normalized so the LARGEST layer gets the full
  physical buffer; the effective mean is reported by the caller.
- Sizes are quantized down to a coarse quantum so the number of distinct
  JIT kernel specializations (one per distinct ``hot_buffer_size``) stays
  small.
- Every size is floored at the top-k width (the kernel requires
  ``hot_buffer_size >= num_top_k``).

The profile maps model layer ids to relative capacity weights. Profiles
solved on a subset of layers extend to all layers by nearest relative
depth, the same convention as the offline studies.
"""

from __future__ import annotations

import json
from typing import Dict, List, Mapping, Optional, Sequence, Union

# Pre-solved profile for DeepSeek-V4-Flash (SWE-bench 64k/100k traces,
# 12 logged C4A layers, LRU replay; exact DP at total ratio 0.2; see
# dsa-offloading offline_analysis/results/per_layer_budget_dp/
# dp_allocations.csv). Values are relative capacity weights.
DSV4_FLASH_SWE_PROFILE: Dict[int, float] = {
    2: 0.25,
    4: 0.115,
    10: 0.24,
    12: 0.18,
    14: 0.205,
    22: 0.235,
    24: 0.24,
    26: 0.18,
    32: 0.205,
    34: 0.2,
    40: 0.255,
    42: 0.095,
}

NAMED_PROFILES: Dict[str, Dict[int, float]] = {
    "dsv4_flash_swe": DSV4_FLASH_SWE_PROFILE,
}

ProfileSpec = Union[str, Mapping[Union[int, str], float]]


def load_layer_profile(spec: ProfileSpec) -> Dict[int, float]:
    """Resolve a profile spec to ``{model_layer_id: weight}``.

    Accepts a named built-in profile (``"dsv4_flash_swe"``), a mapping,
    a JSON object string, or ``"@/path/to/profile.json"``.
    """
    if isinstance(spec, str):
        if spec in NAMED_PROFILES:
            profile = NAMED_PROFILES[spec]
        elif spec.startswith("@"):
            with open(spec[1:], encoding="utf-8") as handle:
                profile = json.load(handle)
        else:
            profile = json.loads(spec)
    else:
        profile = spec
    out = {int(k): float(v) for k, v in profile.items()}
    if not out:
        raise ValueError("layer profile is empty")
    if min(out.values()) <= 0:
        raise ValueError(f"layer profile weights must be positive: {out}")
    return out


def map_profile_to_layers(
    profile: Mapping[int, float], num_layers: int
) -> List[float]:
    """Weights for ``num_layers`` cache layers by nearest relative depth.

    Cache layer ``i`` (0-based, depth-ordered) takes the weight of the
    profile layer whose relative depth is closest — the convention used
    when transferring per-layer budgets across models in the offline
    studies.
    """
    ids = sorted(profile)
    if num_layers == len(ids):
        return [profile[i] for i in ids]
    max_id = max(ids)
    weights = []
    for i in range(num_layers):
        rel = i / max(num_layers - 1, 1)
        nearest = min(ids, key=lambda lid: abs(lid / max_id - rel))
        weights.append(profile[nearest])
    return weights


def compute_layer_buffer_sizes(
    weights: Sequence[float],
    device_buffer_size: int,
    floor: int,
    quantum: int = 256,
) -> List[int]:
    """Per-layer effective buffer sizes from relative weights.

    The largest-weight layer receives the full physical
    ``device_buffer_size``; other layers scale proportionally, quantized
    DOWN to ``quantum`` (bounding distinct JIT kernel variants) and
    floored at ``floor`` (the kernel's ``num_top_k`` requirement).
    """
    if device_buffer_size < floor:
        raise ValueError(
            f"device_buffer_size ({device_buffer_size}) < floor ({floor})"
        )
    top = max(weights)
    sizes = []
    for w in weights:
        size = int(device_buffer_size * (w / top))
        size = (size // quantum) * quantum
        size = max(min(size, device_buffer_size), floor)
        sizes.append(size)
    return sizes


def resolve_layer_buffer_sizes(
    profile_spec: Optional[ProfileSpec],
    num_layers: int,
    device_buffer_size: int,
    floor: int,
    quantum: int = 256,
) -> Optional[List[int]]:
    """One-call helper: spec -> per-layer sizes (None if no profile)."""
    if profile_spec is None:
        return None
    profile = load_layer_profile(profile_spec)
    weights = map_profile_to_layers(profile, num_layers)
    return compute_layer_buffer_sizes(
        weights, device_buffer_size, floor, quantum
    )
