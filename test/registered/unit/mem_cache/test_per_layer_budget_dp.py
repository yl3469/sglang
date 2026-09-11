import unittest

from sglang.srt.mem_cache.sparsity.per_layer_budget_dp import (
    LayerCostCurve,
    dp_allocate,
)
from sglang.srt.mem_cache.sparsity.per_layer_budget_replay import (
    build_cost_curves,
    lru_to_belady_gap,
    make_synthetic_trace,
    replay_belady,
    replay_lru,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestLayerCostCurve(CustomTestCase):
    def test_validates_monotonic_sizes(self):
        with self.assertRaises(ValueError):
            LayerCostCurve(sizes=[2, 2], costs=[1.0, 0.5])
        with self.assertRaises(ValueError):
            LayerCostCurve(sizes=[1, 2], costs=[1.0])

    def test_step_function_lookup(self):
        c = LayerCostCurve(sizes=[1, 3, 5], costs=[9.0, 4.0, 1.0])
        self.assertEqual(c.cost_at(1), 9.0)
        self.assertEqual(c.cost_at(2), 9.0)  # largest measured <= 2 is size 1
        self.assertEqual(c.cost_at(3), 4.0)
        self.assertEqual(c.cost_at(6), 1.0)
        with self.assertRaises(ValueError):
            c.cost_at(0)


class TestDPAllocate(CustomTestCase):
    def _flat_curve(self, sizes, cost_per_layer):
        # Cost strictly decreasing so more capacity is always weakly better.
        costs = [cost_per_layer / s for s in sizes]
        return LayerCostCurve(sizes=list(sizes), costs=costs)

    def test_uniform_curves_give_uniform_split(self):
        # Identical layers -> each should get the mean.
        grid_step = 10
        mean_budget = 30  # mean grid = 3
        sizes = [1, 2, 3, 4, 5]
        curves = [self._flat_curve(sizes, 100.0) for _ in range(4)]
        alloc = dp_allocate(curves, mean_budget, grid_step)
        self.assertEqual(alloc.num_layers, 4)
        self.assertEqual(alloc.total_budget, mean_budget * 4)
        # All equal to the mean.
        self.assertEqual(alloc.per_layer_budget, [30, 30, 30, 30])

    def test_skewed_curves_favor_needy_layer(self):
        # Layer 1 has a much steeper miss curve -> should get more capacity.
        grid_step = 1
        mean_budget = 4  # mean grid 4, total grid 8 over 2 layers
        sizes = [1, 2, 3, 4, 5, 6, 7]
        cheap = LayerCostCurve(sizes=sizes, costs=[7, 6, 5, 4, 3, 2, 1])
        steep = LayerCostCurve(sizes=sizes, costs=[700, 300, 120, 40, 20, 10, 5])
        alloc = dp_allocate([cheap, steep], mean_budget, grid_step)
        self.assertEqual(alloc.total_budget, 8)
        # Steep layer (index 1) must receive strictly more than the cheap one.
        self.assertGreater(alloc.per_layer_budget[1], alloc.per_layer_budget[0])

    def test_grid_must_extend_above_mean(self):
        grid_step = 1
        mean_budget = 5
        # All curves cap at the mean (5) -> degenerate, must raise.
        sizes = [1, 2, 3, 4, 5]
        curves = [LayerCostCurve(sizes=sizes, costs=[5, 4, 3, 2, 1]) for _ in range(3)]
        with self.assertRaises(ValueError):
            dp_allocate(curves, mean_budget, grid_step)

    def test_floor_is_respected(self):
        grid_step = 1
        mean_budget = 4
        sizes = [1, 2, 3, 4, 5, 6, 7, 8]
        cheap = LayerCostCurve(sizes=sizes, costs=[8, 7, 6, 5, 4, 3, 2, 1])
        steep = LayerCostCurve(sizes=sizes, costs=[80, 40, 20, 10, 5, 3, 2, 1])
        # Force the cheap layer to keep at least 3 tokens even though the DP
        # would otherwise starve it in favor of the steep layer.
        alloc = dp_allocate([cheap, steep], mean_budget, grid_step, floors=[3, 1])
        self.assertGreaterEqual(alloc.per_layer_budget[0], 3)
        self.assertEqual(alloc.total_budget, 8)

    def test_floors_exceeding_budget_raise(self):
        grid_step = 1
        mean_budget = 2  # total grid 4 over 2 layers
        sizes = [1, 2, 3, 4, 5]
        curves = [LayerCostCurve(sizes=sizes, costs=[5, 4, 3, 2, 1])] * 2
        with self.assertRaises(ValueError):
            dp_allocate(curves, mean_budget, grid_step, floors=[3, 3])

    def test_mean_budget_grid_alignment(self):
        sizes = [1, 2, 3]
        curves = [LayerCostCurve(sizes=sizes, costs=[3, 2, 1])]
        with self.assertRaises(ValueError):
            dp_allocate(curves, mean_budget=5, grid_step=2)

    def test_optimality_against_brute_force(self):
        # Small enough to brute-force every allocation and confirm the DP is
        # exactly optimal (not just heuristic).
        import itertools

        grid_step = 1
        mean_budget = 3
        num_layers = 3
        total = mean_budget * num_layers  # 9
        sizes = [1, 2, 3, 4, 5, 6, 7]
        rng_costs = [
            [30, 18, 12, 9, 7, 6, 5],
            [50, 20, 8, 5, 4, 3, 2],
            [10, 9, 8, 7, 6, 5, 4],
        ]
        curves = [LayerCostCurve(sizes=sizes, costs=c) for c in rng_costs]
        alloc = dp_allocate(curves, mean_budget, grid_step)

        # Brute force: all (a,b,c) with a+b+c <= total, each in [1, 7].
        best = float("inf")
        for combo in itertools.product(range(1, 8), repeat=num_layers):
            if sum(combo) > total:
                continue
            cost = sum(curve.cost_at(x) for curve, x in zip(curves, combo))
            best = min(best, cost)
        self.assertAlmostEqual(alloc.total_cost, best)


class TestReplay(CustomTestCase):
    def test_belady_never_worse_than_lru(self):
        trace = make_synthetic_trace(num_steps=80, top_k=8, vocab=40, seed=3)
        for size in (10, 16, 24):
            lru = replay_lru(trace, size)
            belady = replay_belady(trace, size)
            self.assertLessEqual(belady, lru)

    def test_full_cache_no_misses_beyond_unique(self):
        # Cache >= unique tokens -> misses equal number of unique tokens (each
        # admitted exactly once, never evicted).
        trace = [[1, 2, 3], [2, 3, 4], [1, 4, 5]]
        unique = len({t for step in trace for t in step})
        self.assertEqual(replay_lru(trace, 100), unique)
        self.assertEqual(replay_belady(trace, 100), unique)

    def test_lru_known_small_case(self):
        # size 1 cache, alternating a/b forces a miss every access.
        trace = [[1], [2], [1], [2]]
        self.assertEqual(replay_lru(trace, 1), 4)
        # Belady with size 1 also misses every time here.
        self.assertEqual(replay_belady(trace, 1), 4)

    def test_gap_report_fields(self):
        trace = make_synthetic_trace(num_steps=60, top_k=6, vocab=30, seed=1)
        rep = lru_to_belady_gap(trace, cache_size=12)
        self.assertGreaterEqual(rep["gap"], 0.0)
        self.assertAlmostEqual(rep["lru_miss_rate"], rep["lru_misses"] / rep["demands"])
        self.assertGreaterEqual(rep["lru_miss_rate"], rep["belady_miss_rate"])

    def test_build_cost_curves_shape_and_grid(self):
        traces = [make_synthetic_trace(50, 6, 30, seed=i) for i in range(3)]
        curves = build_cost_curves(traces, max_size=20, grid_step=5, policy="lru")
        self.assertEqual(len(curves), 3)
        for c in curves:
            # grid multiples of step 5 up to 20 -> {1,2,3,4}
            self.assertEqual(c.sizes[0], 1)
            self.assertEqual(c.max_size, 4)
            # Miss cost is non-increasing in size.
            self.assertTrue(all(a >= b for a, b in zip(c.costs, c.costs[1:])))


class TestReplayIntoDP(CustomTestCase):
    def test_end_to_end_replay_then_allocate(self):
        # Build LRU curves from synthetic per-layer traces with differing
        # locality, then let the DP partition a fixed mean budget.
        traces = [
            make_synthetic_trace(120, top_k=8, vocab=64, hot_fraction=hf, seed=7 + i)
            for i, hf in enumerate([0.1, 0.5, 0.3])
        ]
        grid_step = 4
        max_size = 48  # extends above the mean below
        curves = build_cost_curves(traces, max_size, grid_step, policy="lru")
        alloc = dp_allocate(curves, mean_budget=24, grid_step=grid_step)
        self.assertEqual(alloc.num_layers, 3)
        self.assertEqual(alloc.total_budget, 24 * 3)
        # DP allocation must be at least as good as the uniform split.
        uniform_cost = sum(c.cost_at(24 // grid_step) for c in curves)
        self.assertLessEqual(alloc.total_cost, uniform_cost + 1e-9)


if __name__ == "__main__":
    unittest.main()
