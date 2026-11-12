"""Offline trace-replay for HiSparse per-layer miss curves (T1 support).

The DP allocator (:mod:`per_layer_budget_dp`) needs a miss cost per
``(layer, cache_size)``. This module produces those curves by replaying a
per-layer selection log — the top-k token ids the indexer selected at each
decode step — through two reference cache policies:

* **LRU** — the policy the HiSparse Resolve kernel runs today.
* **Belady** (offline optimal / MIN) — evict the resident whose next use is
  farthest in the future. Belady is EXACTLY computable from a logged top-k
  selection stream because future selections are known, so it turns every
  result into "x% of the achievable LRU->Belady gap".

Everything here is pure Python (stdlib only). It is offline analysis tooling,
not a serving hot path, so clarity beats micro-optimization. The synthetic
trace helper makes the module testable without real models or GPUs.

Miss model: at each step the policy must serve a set of "demanded" token ids
(the step's top-k selection). A demanded token that is not resident is a miss
and is then admitted (evicting if the cache is full). We count demand misses,
matching HiSparse's per-step top-k demand-miss metric.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence

from sglang.srt.mem_cache.sparsity.per_layer_budget_dp import LayerCostCurve

__all__ = [
    "SelectionTrace",
    "replay_lru",
    "replay_belady",
    "replay_lru_fast",
    "replay_belady_fast",
    "build_cost_curves",
    "lru_to_belady_gap",
    "make_synthetic_trace",
]


# A per-layer selection log: steps[i] is the list of token ids selected at
# decode step i. Token ids are arbitrary hashable ints.
SelectionTrace = List[List[int]]


def _demand_stream(trace: SelectionTrace) -> List[List[int]]:
    """Normalize each step's selection to a de-duplicated list (order kept)."""
    out: List[List[int]] = []
    for step in trace:
        seen = set()
        uniq = []
        for tok in step:
            if tok not in seen:
                seen.add(tok)
                uniq.append(tok)
        out.append(uniq)
    return out


def replay_lru(trace: SelectionTrace, cache_size: int) -> int:
    """Return the number of demand misses under LRU at ``cache_size``."""
    if cache_size <= 0:
        raise ValueError("cache_size must be positive")
    steps = _demand_stream(trace)

    # OrderedDict-style recency via a dict of token -> logical clock; evict the
    # smallest clock. For clarity we keep an explicit recency counter.
    resident: Dict[int, int] = {}
    clock = 0
    misses = 0
    for step in steps:
        for tok in step:
            clock += 1
            if tok in resident:
                resident[tok] = clock  # touch -> most recent
                continue
            misses += 1
            if len(resident) >= cache_size:
                # Evict least-recently-used.
                lru_tok = min(resident, key=resident.__getitem__)
                del resident[lru_tok]
            resident[tok] = clock
    return misses


def replay_belady(trace: SelectionTrace, cache_size: int) -> int:
    """Return demand misses under Belady (offline optimal) at ``cache_size``.

    On a miss with a full cache, evict the resident whose NEXT demand is
    farthest in the future (or never demanded again). Future demands are known
    because the whole trace is logged.
    """
    if cache_size <= 0:
        raise ValueError("cache_size must be positive")
    steps = _demand_stream(trace)

    # Flatten to a linear demand sequence, remembering step boundaries is not
    # needed for Belady counting (a miss is per-token). Precompute, for each
    # position, the token, then answer "next occurrence" lazily via per-token
    # occurrence lists + a cursor.
    flat: List[int] = [tok for step in steps for tok in step]
    n = len(flat)

    occurrences: Dict[int, List[int]] = {}
    for pos, tok in enumerate(flat):
        occurrences.setdefault(tok, []).append(pos)
    # Cursor into each token's occurrence list pointing at the current position.
    cursor: Dict[int, int] = {tok: 0 for tok in occurrences}

    def next_use(tok: int, after: int) -> int:
        """Smallest occurrence position of tok strictly greater than ``after``."""
        occ = occurrences[tok]
        c = cursor[tok]
        while c < len(occ) and occ[c] <= after:
            c += 1
        cursor[tok] = c
        return occ[c] if c < len(occ) else n  # n == "never again"

    resident = set()
    misses = 0
    for pos, tok in enumerate(flat):
        if tok in resident:
            continue
        misses += 1
        if len(resident) >= cache_size:
            # Evict the resident used farthest in the future.
            victim = max(resident, key=lambda t: next_use(t, pos))
            resident.discard(victim)
        resident.add(tok)
    return misses


def replay_lru_fast(trace: SelectionTrace, cache_size: int) -> int:
    """O(1)-eviction LRU miss count; identical result to :func:`replay_lru`.

    Uses an ``OrderedDict`` as the recency structure (move-to-end on touch,
    pop-from-front on evict) instead of the O(cache_size) ``min`` scan per miss.
    This is the hot path for large real traces (millions of demands).
    """
    if cache_size <= 0:
        raise ValueError("cache_size must be positive")
    from collections import OrderedDict

    resident: "OrderedDict[int, None]" = OrderedDict()
    misses = 0
    for step in trace:
        seen = set()
        for tok in step:
            if tok in seen:
                continue
            seen.add(tok)
            if tok in resident:
                resident.move_to_end(tok)
                continue
            misses += 1
            if len(resident) >= cache_size:
                resident.popitem(last=False)
            resident[tok] = None
    return misses


def replay_belady_fast(trace: SelectionTrace, cache_size: int) -> int:
    """Belady (offline optimal) miss count with a lazy max-heap eviction.

    Identical result to :func:`replay_belady`, but replaces the per-miss
    O(cache_size) ``max(..., key=next_use)`` scan with a max-heap (via negated
    keys) of next-use positions plus lazy deletion. The heap can hold stale
    entries because a resident token's next-use advances every time it is hit;
    we therefore track each resident's CURRENT next-use in ``cur_nu`` and, on
    each hit or admission, push a fresh ``(-next_use, tok)`` entry. When popping
    a victim we accept it only if it is still resident AND its heap key equals
    its current next-use; otherwise the entry is stale and skipped. Complexity
    ~O(demands log demands).
    """
    if cache_size <= 0:
        raise ValueError("cache_size must be positive")
    import heapq

    steps = _demand_stream(trace)
    flat: List[int] = [tok for step in steps for tok in step]
    n = len(flat)

    # next_occ[pos] = smallest occurrence position of flat[pos] strictly after
    # pos (or n == "never again"). One right-to-left pass.
    next_occ = [n] * n
    last_seen: Dict[int, int] = {}
    for pos in range(n - 1, -1, -1):
        tok = flat[pos]
        next_occ[pos] = last_seen.get(tok, n)
        last_seen[tok] = pos

    resident: set = set()
    cur_nu: Dict[int, int] = {}  # token -> its current next-use position
    heap: List = []  # (-next_use, tok); may contain stale entries
    misses = 0
    for pos, tok in enumerate(flat):
        nu = next_occ[pos]
        if tok in resident:
            # Hit: next-use advanced; refresh the recorded value and push a new
            # (larger) heap entry. The old entry becomes stale.
            cur_nu[tok] = nu
            heapq.heappush(heap, (-nu, tok))
            continue
        misses += 1
        if len(resident) >= cache_size:
            while heap:
                neg_nu, victim = heapq.heappop(heap)
                if victim in resident and -neg_nu == cur_nu[victim]:
                    resident.discard(victim)
                    del cur_nu[victim]
                    break
                # else: stale (evicted, or superseded by a later push); skip.
        resident.add(tok)
        cur_nu[tok] = nu
        heapq.heappush(heap, (-nu, tok))
    return misses


def _grid_sizes(max_size: int, grid_step: int) -> List[int]:
    if grid_step <= 0:
        raise ValueError("grid_step must be positive")
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    sizes = list(range(grid_step, max_size + 1, grid_step))
    if not sizes or sizes[-1] != max_size:
        sizes.append(max_size)
    return sizes


def build_cost_curves(
    per_layer_traces: Sequence[SelectionTrace],
    max_size: int,
    grid_step: int,
    policy: str = "lru",
) -> List[LayerCostCurve]:
    """Build :class:`LayerCostCurve` per layer for the DP allocator.

    Sizes are emitted as GRID MULTIPLES (units of ``grid_step``) to match
    :func:`per_layer_budget_dp.dp_allocate`. Costs are demand-miss counts under
    ``policy`` ("lru" or "belady"). The grid extends up to ``max_size`` so it
    can lie ABOVE the mean the DP is later given.
    """
    if policy not in ("lru", "belady"):
        raise ValueError("policy must be 'lru' or 'belady'")
    replay = replay_lru if policy == "lru" else replay_belady
    token_sizes = _grid_sizes(max_size, grid_step)

    curves: List[LayerCostCurve] = []
    for trace in per_layer_traces:
        grid_multiples = [s // grid_step for s in token_sizes]
        costs = [float(replay(trace, s)) for s in token_sizes]
        # Collapse potential duplicate grid multiples (only the trailing
        # max_size entry can collide with the last stepped point).
        dedup_sizes: List[int] = []
        dedup_costs: List[float] = []
        for gm, c in zip(grid_multiples, costs):
            if dedup_sizes and dedup_sizes[-1] == gm:
                continue
            dedup_sizes.append(gm)
            dedup_costs.append(c)
        curves.append(LayerCostCurve(sizes=dedup_sizes, costs=dedup_costs))
    return curves


def lru_to_belady_gap(trace: SelectionTrace, cache_size: int) -> Dict[str, float]:
    """Report LRU vs Belady miss rate and the achievable gap at one size.

    Returns a dict with ``lru_misses``, ``belady_misses``, ``demands``,
    ``lru_miss_rate``, ``belady_miss_rate`` and ``gap`` (LRU - Belady miss
    rate, in fraction). This is the denominator for "x% of the gap closed"
    claims required by the goal's replay gate.
    """
    demands = sum(len(set(step)) for step in trace)
    lru = replay_lru(trace, cache_size)
    belady = replay_belady(trace, cache_size)
    lru_rate = lru / demands if demands else 0.0
    belady_rate = belady / demands if demands else 0.0
    return {
        "lru_misses": float(lru),
        "belady_misses": float(belady),
        "demands": float(demands),
        "lru_miss_rate": lru_rate,
        "belady_miss_rate": belady_rate,
        "gap": lru_rate - belady_rate,
    }


def make_synthetic_trace(
    num_steps: int,
    top_k: int,
    vocab: int,
    hot_fraction: float = 0.2,
    hot_bias: float = 0.8,
    seed: int = 0,
) -> SelectionTrace:
    """Generate a reproducible synthetic per-layer selection trace.

    A ``hot_fraction`` of the vocabulary is selected with probability
    ``hot_bias`` (locality), the rest uniformly. Useful as a test fixture and
    for exercising the DP end-to-end without real indexer logs.
    """
    if not 0 < hot_fraction <= 1:
        raise ValueError("hot_fraction must be in (0, 1]")
    if not 0 <= hot_bias <= 1:
        raise ValueError("hot_bias must be in [0, 1]")
    if top_k > vocab:
        raise ValueError("top_k cannot exceed vocab")
    rng = random.Random(seed)
    num_hot = max(1, int(vocab * hot_fraction))
    hot = list(range(num_hot))
    cold = list(range(num_hot, vocab))

    trace: SelectionTrace = []
    for _ in range(num_steps):
        chosen = set()
        while len(chosen) < top_k:
            if cold and rng.random() >= hot_bias:
                chosen.add(rng.choice(cold))
            else:
                chosen.add(rng.choice(hot))
        trace.append(list(chosen))
    return trace
