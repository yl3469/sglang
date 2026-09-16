"""Exact DP capacity partition for HiSparse per-layer device buffers (T1).

HiSparse today uses a single per-request-per-layer GPU cache size ``B`` shared
by every layer (``device_buffer_size`` in :class:`SparseConfig`). Layers differ
several-fold in miss rate at equal capacity, so one uniform ``B`` wastes memory.

This module replaces the single ``B`` with per-layer sizes ``B_l`` chosen by an
exact dynamic program over MEASURED per-layer miss curves::

    F_l(b) = min_x [ cost_l(x) + F_{l-1}(b - x) ]

This is a min-plus convolution, exact on the grid, with no convexity
assumption. Complexity is ``O(L * (B/Δ)^2)`` — linear in the layer count.

The routine is entirely host-side and offline: it consumes cost curves produced
by a replay harness (see :mod:`per_layer_budget_replay`) and emits an allocation
that can be applied at page-table sizing time. It touches no kernel code and
cannot affect output correctness.

Critical API semantics (must match the reference implementation
``per_layer_budget_dp.dp_allocate``):

* The total budget is the LAYER-SUM, expressed as ``mean_budget * num_layers``.
* The size grid must extend ABOVE the target mean, otherwise the DP degenerates
  to the uniform split (every layer pinned at the mean). :func:`dp_allocate`
  raises if the supplied curves do not offer any grid point above the mean.
* Each layer has a floor ``B_l >= floor_l`` where the caller sets
  ``floor_l = max_misses_per_step + 1 + prefetch_budget``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

__all__ = ["LayerCostCurve", "DPAllocation", "dp_allocate", "dp_allocate_for_target"]

_INF = float("inf")


@dataclass(frozen=True)
class LayerCostCurve:
    """Miss cost as a function of cache size for one layer.

    ``sizes`` are cache sizes expressed as integer multiples of ``grid_step``
    (i.e. grid indices), and ``costs[i]`` is the measured miss cost at
    ``sizes[i]``. ``sizes`` must be strictly increasing. Lower cost is better;
    the DP minimizes the summed cost. Costs are treated as a step function:
    for a budget between two measured grid points the lower (smaller-size)
    point's cost is used, so any granted capacity is fully accounted for.
    """

    sizes: Sequence[int]
    costs: Sequence[float]

    def __post_init__(self) -> None:
        if len(self.sizes) != len(self.costs):
            raise ValueError("sizes and costs must have equal length")
        if len(self.sizes) == 0:
            raise ValueError("cost curve must have at least one grid point")
        for a, b in zip(self.sizes, self.sizes[1:]):
            if b <= a:
                raise ValueError("sizes must be strictly increasing")
        if any(s < 0 for s in self.sizes):
            raise ValueError("sizes must be non-negative grid multiples")

    @property
    def min_size(self) -> int:
        return self.sizes[0]

    @property
    def max_size(self) -> int:
        return self.sizes[-1]

    def cost_at(self, size: int) -> float:
        """Step-function cost lookup at grid multiple ``size``.

        Returns the cost of the largest measured grid point ``<= size``.
        Raises if ``size`` is below the smallest measured point (the curve does
        not describe that regime, which usually means the floor is misconfigured).
        """
        if size < self.sizes[0]:
            raise ValueError(
                f"size {size} below smallest measured grid point {self.sizes[0]}"
            )
        # sizes is small (grid points per layer); linear scan is fine and keeps
        # the dependency surface at zero.
        best = self.costs[0]
        for s, c in zip(self.sizes, self.costs):
            if s <= size:
                best = c
            else:
                break
        return best


@dataclass(frozen=True)
class DPAllocation:
    """Result of :func:`dp_allocate`."""

    # Per-layer budget in TOKENS (grid multiple * grid_step).
    per_layer_budget: List[int]
    # Per-layer budget as grid multiples.
    per_layer_grid: List[int]
    # Summed miss cost of the chosen allocation.
    total_cost: float
    grid_step: int
    mean_budget: int

    @property
    def num_layers(self) -> int:
        return len(self.per_layer_budget)

    @property
    def total_budget(self) -> int:
        return sum(self.per_layer_budget)


def dp_allocate(
    cost_curves: Sequence[LayerCostCurve],
    mean_budget: int,
    grid_step: int,
    floors: Optional[Sequence[int]] = None,
) -> DPAllocation:
    """Solve the exact per-layer budget partition.

    Args:
        cost_curves: One :class:`LayerCostCurve` per layer, with sizes given as
            integer grid multiples (units of ``grid_step`` tokens).
        mean_budget: Target MEAN per-layer budget in tokens. The total budget
            distributed by the DP is ``mean_budget * num_layers`` (the
            layer-sum). Must be a multiple of ``grid_step``.
        grid_step: Token granularity of one grid unit (must be > 0).
        floors: Optional per-layer floor in TOKENS. Each granted ``B_l`` is at
            least the floor. Defaults to each curve's smallest measured size.

    Returns:
        A :class:`DPAllocation` with per-layer budgets summing to at most the
        total budget (it can be less only if floors force it higher — see the
        over-subscription check below, which raises in that case).

    Raises:
        ValueError: on invalid grid alignment, when no curve offers a grid point
            strictly above the mean (the degenerate/uniform case the reference
            explicitly guards against), or when the floors alone exceed the
            total budget.
    """
    if grid_step <= 0:
        raise ValueError("grid_step must be positive")
    if mean_budget <= 0:
        raise ValueError("mean_budget must be positive")
    if mean_budget % grid_step != 0:
        raise ValueError(
            f"mean_budget {mean_budget} must be a multiple of grid_step {grid_step}"
        )
    num_layers = len(cost_curves)
    if num_layers == 0:
        raise ValueError("cost_curves must be non-empty")

    mean_grid = mean_budget // grid_step
    total_grid = mean_grid * num_layers

    # Guard against the degenerate case: if no layer's curve extends above the
    # mean, the DP has no freedom and collapses to the uniform split. The
    # reference implementation treats this as a configuration error.
    if not any(curve.max_size > mean_grid for curve in cost_curves):
        raise ValueError(
            "no cost curve extends above the target mean grid size "
            f"({mean_grid}); grid must extend ABOVE the mean or the DP "
            "degenerates to the uniform split"
        )

    floor_grid = _resolve_floor_grid(cost_curves, floors, grid_step)

    floor_total = sum(floor_grid)
    if floor_total > total_grid:
        raise ValueError(
            f"per-layer floors sum to {floor_total * grid_step} tokens, exceeding "
            f"the total budget {total_grid * grid_step} tokens; lower the floors "
            "or raise mean_budget"
        )

    # Per-layer max usable grid size, capped by both the curve and the total
    # budget (a single layer can never need more than the whole budget).
    cap_grid = [min(curve.max_size, total_grid) for curve in cost_curves]
    dp, choice = _min_plus_sweep(cost_curves, total_grid, floor_grid, cap_grid)

    # The optimum uses the full budget when more capacity never hurts (costs are
    # non-increasing in size for real miss curves), but we take the min over the
    # feasible tail [floor_total, total_grid] to stay correct for arbitrary curves.
    best_b = None
    best_cost = _INF
    for b in range(floor_total, total_grid + 1):
        if dp[b] < best_cost:
            best_cost = dp[b]
            best_b = b
    if best_b is None or best_cost == _INF:
        raise ValueError(
            "DP found no feasible allocation; check floors, caps, and curves"
        )
    per_layer_grid = _backtrack(choice, best_b)

    per_layer_budget = [g * grid_step for g in per_layer_grid]
    return DPAllocation(
        per_layer_budget=per_layer_budget,
        per_layer_grid=per_layer_grid,
        total_cost=best_cost,
        grid_step=grid_step,
        mean_budget=mean_budget,
    )


def _resolve_floor_grid(
    cost_curves: Sequence[LayerCostCurve],
    floors: Optional[Sequence[int]],
    grid_step: int,
) -> List[int]:
    """Per-layer floor in grid units (curve minimum when no floors are given)."""
    if floors is None:
        return [curve.min_size for curve in cost_curves]
    if len(floors) != len(cost_curves):
        raise ValueError("floors must have one entry per layer")
    floor_grid = []
    for f, curve in zip(floors, cost_curves):
        # Round the floor UP to a grid multiple so B_l stays grid-aligned
        # while never dropping below the requested floor.
        fg = math.ceil(f / grid_step)
        fg = max(fg, curve.min_size)
        floor_grid.append(fg)
    return floor_grid


def _min_plus_sweep(
    cost_curves: Sequence[LayerCostCurve],
    total_grid: int,
    floor_grid: Sequence[int],
    cap_grid: Sequence[int],
):
    """Min-plus convolution over layers.

    Returns ``(dp, choice)``: ``dp[b]`` is the least summed cost using EXACTLY
    ``b`` grid units over all layers (inf when infeasible) and
    ``choice[l][b]`` the grid size given to layer ``l`` on the path to ``dp[b]``
    after processing layers ``0..l``.
    """
    num_layers = len(cost_curves)
    dp = [_INF] * (total_grid + 1)
    choice: List[List[int]] = [[0] * (total_grid + 1) for _ in range(num_layers)]

    curve0 = cost_curves[0]
    lo0, hi0 = floor_grid[0], cap_grid[0]
    if lo0 > hi0:
        raise ValueError(
            f"layer 0 floor ({lo0}) exceeds its max curve size ({hi0}) in grid units"
        )
    for x in _candidate_sizes(curve0, lo0, hi0):
        dp[x] = curve0.cost_at(x)
        choice[0][x] = x

    for l in range(1, num_layers):
        curve = cost_curves[l]
        lo, hi = floor_grid[l], cap_grid[l]
        if lo > hi:
            raise ValueError(
                f"layer {l} floor ({lo}) exceeds its max curve size ({hi}) in grid units"
            )
        prev = dp
        cur = [_INF] * (total_grid + 1)
        candidates = _candidate_sizes(curve, lo, hi)
        for b in range(total_grid + 1):
            best = _INF
            best_x = lo
            # Give layer l a size x in [lo, hi]; the rest (b - x) goes to prior
            # layers, which needed at least their cumulative floors.
            for x in candidates:
                if x > b:
                    break
                prev_cost = prev[b - x]
                if prev_cost == _INF:
                    continue
                total = prev_cost + curve.cost_at(x)
                if total < best:
                    best = total
                    best_x = x
            cur[b] = best
            choice[l][b] = best_x
        dp = cur
    return dp, choice


def _candidate_sizes(curve: LayerCostCurve, lo: int, hi: int) -> List[int]:
    """Grid sizes worth granting to one layer: its measured points in [lo, hi]
    plus the floor itself. Costs are step functions, so a size strictly between
    two measured points costs the same as the lower one while using more
    budget; it can never be part of a minimal-budget or minimal-cost optimum.
    Restricting the inner loop to these keeps the sweep exact and turns
    O(L * B * B) into O(L * B * points)."""
    sizes = [lo] + [x for x in curve.sizes if lo < x <= hi]
    return sizes


def _backtrack(choice: Sequence[Sequence[int]], b: int) -> List[int]:
    num_layers = len(choice)
    per_layer_grid = [0] * num_layers
    for l in range(num_layers - 1, -1, -1):
        x = choice[l][b]
        per_layer_grid[l] = x
        b -= x
    return per_layer_grid


def dp_allocate_for_target(
    cost_curves: Sequence[LayerCostCurve],
    target_mean_cost: float,
    grid_step: int,
    floors: Optional[Sequence[int]] = None,
) -> DPAllocation:
    """Smallest layer-sum budget whose optimal split meets a target mean cost.

    The dual of :func:`dp_allocate`: instead of "best misses for a fixed
    budget" this answers "fewest buffer rows (bytes) for at most
    ``target_mean_cost`` mean miss cost". With physically per-layer buffers
    every row costs the same bytes in every layer, so minimizing the grid sum
    minimizes device memory. One min-plus sweep over the full grid gives
    ``dp[b]`` for every total ``b``; the answer is the smallest feasible ``b``
    with ``dp[b] <= target_mean_cost * num_layers``.

    Raises ValueError when even the largest measured sizes cannot reach the
    target (the target lies below the traces' compulsory-miss floor).
    """
    if grid_step <= 0:
        raise ValueError("grid_step must be positive")
    if target_mean_cost < 0:
        raise ValueError("target_mean_cost must be non-negative")
    num_layers = len(cost_curves)
    if num_layers == 0:
        raise ValueError("cost_curves must be non-empty")

    floor_grid = _resolve_floor_grid(cost_curves, floors, grid_step)
    cap_grid = [curve.max_size for curve in cost_curves]
    total_grid = sum(cap_grid)
    dp, choice = _min_plus_sweep(cost_curves, total_grid, floor_grid, cap_grid)

    target_total = target_mean_cost * num_layers
    best_b = None
    for b in range(sum(floor_grid), total_grid + 1):
        if dp[b] <= target_total:
            best_b = b
            break
    if best_b is None:
        floor_cost = min(c for c in dp if c != _INF) / num_layers
        raise ValueError(
            f"target mean cost {target_mean_cost} is unreachable; the measured "
            f"curves bottom out at {floor_cost:.4f} even at their largest sizes"
        )
    per_layer_grid = _backtrack(choice, best_b)
    return DPAllocation(
        per_layer_budget=[g * grid_step for g in per_layer_grid],
        per_layer_grid=per_layer_grid,
        total_cost=dp[best_b],
        grid_step=grid_step,
        mean_budget=math.ceil(best_b / num_layers) * grid_step,
    )
