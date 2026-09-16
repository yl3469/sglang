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

__all__ = ["LayerCostCurve", "DPAllocation", "dp_allocate"]


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

    if floors is None:
        floor_grid = [curve.min_size for curve in cost_curves]
    else:
        if len(floors) != num_layers:
            raise ValueError("floors must have one entry per layer")
        floor_grid = []
        for f, curve in zip(floors, cost_curves):
            # Round the floor UP to a grid multiple so B_l stays grid-aligned
            # while never dropping below the requested floor.
            fg = math.ceil(f / grid_step)
            fg = max(fg, curve.min_size)
            floor_grid.append(fg)

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

    # DP over layers. f[b] = min summed cost using layers processed so far with
    # exactly b grid units allocated. We track "exactly b" (not "<= b") and read
    # the optimum over the feasible tail at the end, which keeps the recurrence a
    # clean min-plus convolution.
    NEG = float("inf")
    # After processing layer 0..l, dp[b] holds the best cost; choice[l][b] holds
    # the grid size given to layer l to reach that optimum (for backtracking).
    dp = [NEG] * (total_grid + 1)
    choice: List[List[int]] = [[0] * (total_grid + 1) for _ in range(num_layers)]

    # Base: layer 0.
    curve0 = cost_curves[0]
    lo0, hi0 = floor_grid[0], cap_grid[0]
    if lo0 > hi0:
        raise ValueError(
            f"layer 0 floor ({lo0 * grid_step}) exceeds its max curve size "
            f"({hi0 * grid_step})"
        )
    for x in range(lo0, hi0 + 1):
        dp[x] = curve0.cost_at(x)
        choice[0][x] = x

    # Layers 1..L-1.
    for l in range(1, num_layers):
        curve = cost_curves[l]
        lo, hi = floor_grid[l], cap_grid[l]
        if lo > hi:
            raise ValueError(
                f"layer {l} floor ({lo * grid_step}) exceeds its max curve size "
                f"({hi * grid_step})"
            )
        prev = dp
        cur = [NEG] * (total_grid + 1)
        for b in range(total_grid + 1):
            best = NEG
            best_x = lo
            # Give layer l a size x in [lo, hi]; the rest (b - x) goes to prior
            # layers, which needed at least their cumulative floors.
            x_max = min(hi, b)
            for x in range(lo, x_max + 1):
                prev_cost = prev[b - x]
                if prev_cost == NEG:
                    continue
                total = prev_cost + curve.cost_at(x)
                if total < best:
                    best = total
                    best_x = x
            cur[b] = best
            choice[l][b] = best_x
        dp = cur

    # The optimum uses the full budget when more capacity never hurts (costs are
    # non-increasing in size for real miss curves), but we take the min over the
    # feasible tail [floor_total, total_grid] to stay correct for arbitrary curves.
    best_b = None
    best_cost = NEG
    for b in range(floor_total, total_grid + 1):
        if dp[b] < best_cost:
            best_cost = dp[b]
            best_b = b
    if best_b is None or best_cost == NEG:
        raise ValueError(
            "DP found no feasible allocation; check floors, caps, and curves"
        )

    # Backtrack the per-layer grid sizes.
    per_layer_grid = [0] * num_layers
    b = best_b
    for l in range(num_layers - 1, -1, -1):
        x = choice[l][b]
        per_layer_grid[l] = x
        b -= x

    per_layer_budget = [g * grid_step for g in per_layer_grid]
    return DPAllocation(
        per_layer_budget=per_layer_budget,
        per_layer_grid=per_layer_grid,
        total_cost=best_cost,
        grid_step=grid_step,
        mean_budget=mean_budget,
    )
