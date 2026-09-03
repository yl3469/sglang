import os
import tempfile
import unittest

import numpy as np
import pybase64

from sglang.srt.mem_cache.sparsity.per_layer_budget_dp import dp_allocate
from sglang.srt.mem_cache.sparsity.trace.run_per_layer_dp import (
    allocate_for_total,
    build_dp_rows,
    kv_budget_to_ratio,
    uniform_miss_rate,
    write_dp_allocations_csv,
)
from sglang.srt.mem_cache.sparsity.trace.trace_to_curves import (
    RATIO_STEP,
    build_ratio_cost_curves,
    load_request_topk,
    measure_miss_rate_curves,
    ratio_units,
    topk_array_to_selection_traces,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _make_topk(
    steps: int,
    hot_sets,
    index_topk: int,
    ctx: int,
    seed: int = 0,
) -> np.ndarray:
    """Build a synthetic (steps, layers, index_topk) int32 topk array.

    ``hot_sets[l]`` bounds layer l's selectable slot ids (small == high
    locality). Padding is emulated by writing -1 into unused slots when a hot
    set is smaller than index_topk.
    """
    rng = np.random.default_rng(seed)
    num_layers = len(hot_sets)
    topk = np.full((steps, num_layers, index_topk), -1, dtype=np.int32)
    for layer, hot in enumerate(hot_sets):
        hot = min(hot, ctx)
        width = min(index_topk, hot)
        for s in range(steps):
            picks = rng.integers(0, hot, size=width)
            topk[s, layer, :width] = picks
    return topk


def _encode_meta_info(topk: np.ndarray) -> str:
    """Emulate meta_info['indexer_topk']: base64 of flat int32 bytes."""
    flat = topk.astype(np.int32).ravel()
    return pybase64.b64encode(flat.tobytes()).decode("utf-8")


class TestTopkDecode(CustomTestCase):
    def test_base64_roundtrip_and_reshape(self):
        topk = _make_topk(6, [50, 400, 2000], index_topk=8, ctx=5000, seed=1)
        b64 = _encode_meta_info(topk)
        # Mirror extract_indexer_topk_from_meta_info: decode flat int32.
        flat = np.frombuffer(pybase64.b64decode(b64.encode("utf-8")), dtype=np.int32)
        restored = flat.reshape(6, 3, 8)
        np.testing.assert_array_equal(restored, topk)

    def test_selection_traces_drop_padding(self):
        topk = np.array(
            [[[1, 2, -1], [3, -1, -1]]], dtype=np.int32
        )  # steps=1, layers=2, topk=3
        per_layer = topk_array_to_selection_traces(topk)
        self.assertEqual(per_layer[0], [[1, 2]])
        self.assertEqual(per_layer[1], [[3]])

    def test_reshape_geometry_mismatch_raises(self):
        from sglang.srt.mem_cache.sparsity.trace.trace_to_curves import (
            _reshape_topk,
        )

        with self.assertRaises(ValueError):
            _reshape_topk(np.arange(10, dtype=np.int32), num_layers=3, index_topk=8)


class TestNpzRoundTrip(CustomTestCase):
    def _write_req(self, path, topk, ctx, index_topk):
        # Store both the reshaped 3D array and geometry, as the sink would.
        np.savez(
            path,
            topk_indices=topk.astype(np.int64),
            slot_ctx0=np.int64(ctx),
            index_topk=np.int64(index_topk),
            num_layers=np.int64(topk.shape[1]),
            prompt_len=np.int64(ctx),
            output_len=np.int64(topk.shape[0]),
        )

    def test_load_request_topk(self):
        topk = _make_topk(20, [100, 800, 4000], index_topk=8, ctx=20000, seed=2)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "req_0.npz")
            self._write_req(path, topk, ctx=20000, index_topk=8)
            req = load_request_topk(path)
        self.assertEqual(req.num_layers, 3)
        self.assertEqual(req.slot_ctx0, 20000)
        self.assertEqual(req.topk.shape, (20, 3, 8))

    def test_load_flat_buffer(self):
        topk = _make_topk(12, [100, 800], index_topk=8, ctx=10000, seed=3)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "req_0.npz")
            np.savez(
                path,
                indexer_topk_flat=topk.astype(np.int32).ravel(),
                num_layers=np.int64(2),
                index_topk=np.int64(8),
                slot_ctx0=np.int64(10000),
            )
            req = load_request_topk(path)
        self.assertEqual(req.topk.shape, (12, 2, 8))


class TestCurvesAndDP(CustomTestCase):
    def _requests(self):
        from sglang.srt.mem_cache.sparsity.trace.trace_to_curves import (
            RequestTopk,
        )

        # Three layers with sharply different locality -> the DP should not
        # give them equal budget.
        ctx = 40000
        topk = _make_topk(60, [300, 3000, 30000], index_topk=16, ctx=ctx, seed=7)
        return [RequestTopk(topk=topk.astype(np.int64), slot_ctx0=ctx, meta={})]

    def test_lru_curves_non_increasing(self):
        reqs = self._requests()
        rates, ratios = measure_miss_rate_curves(
            reqs,
            [0.02, 0.05, 0.1, 0.2, 0.3, 0.4],
            policy="lru",
            min_buffer_slots=1,
        )
        curves = build_ratio_cost_curves(rates, ratios)
        self.assertEqual(len(curves), 3)
        for c in curves:
            self.assertTrue(all(a >= b for a, b in zip(c.costs, c.costs[1:])))

    def test_belady_never_worse_than_lru(self):
        reqs = self._requests()
        ratios_in = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4]
        lru_rates, _ = measure_miss_rate_curves(
            reqs, ratios_in, policy="lru", min_buffer_slots=1
        )
        bel_rates, _ = measure_miss_rate_curves(
            reqs, ratios_in, policy="belady", min_buffer_slots=1
        )
        for lr, br in zip(lru_rates, bel_rates):
            for a, b in zip(lr, br):
                if not (np.isnan(a) or np.isnan(b)):
                    self.assertLessEqual(b, a + 1e-9)

    def _decreasing_curves(self):
        # Strictly-decreasing per-layer curves in RATIO units so the DP always
        # benefits from extra capacity (full budget used) and the layer with
        # the steeper curve earns more. Mirrors the tested DP module's fixtures
        # but expressed on the ratio grid this driver consumes.
        from sglang.srt.mem_cache.sparsity.per_layer_budget_dp import (
            LayerCostCurve,
        )

        sizes = [ratio_units(r) for r in (0.05, 0.1, 0.2, 0.3, 0.4)]
        cheap = LayerCostCurve(sizes=sizes, costs=[0.5, 0.45, 0.4, 0.38, 0.37])
        steep = LayerCostCurve(sizes=sizes, costs=[0.9, 0.5, 0.25, 0.15, 0.1])
        mid = LayerCostCurve(sizes=sizes, costs=[0.7, 0.6, 0.5, 0.44, 0.4])
        return [cheap, steep, mid]

    def test_dp_allocation_is_non_uniform_and_budget_conserved(self):
        curves = self._decreasing_curves()
        total_ratio = 0.1
        alloc, dp_rate, uniform_rate = allocate_for_total(curves, total_ratio)
        mean_grid = ratio_units(total_ratio)
        # Budget conserved: with strictly-decreasing curves more capacity always
        # helps, so the DP spends the whole budget.
        self.assertEqual(alloc.total_budget, mean_grid * len(curves))
        # DP is at least as good as the uniform split.
        self.assertLessEqual(dp_rate, uniform_rate + 1e-9)
        # The steep layer (index 1) must receive strictly more than the cheap.
        self.assertGreater(alloc.per_layer_grid[1], alloc.per_layer_grid[0])

    def test_dp_under_allocates_flat_curves(self):
        # Realistic uniform-random traces plateau: once the buffer clears the
        # working set, extra capacity does not help, so the DP (argmin over the
        # feasible budget tail, matching the reference) allocates <= full budget
        # and never worse than uniform. This guards that contract.
        reqs = self._requests()
        ratios_in = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4]
        rates, ratios = measure_miss_rate_curves(
            reqs, ratios_in, policy="lru", min_buffer_slots=1
        )
        curves = build_ratio_cost_curves(rates, ratios)
        alloc, dp_rate, uniform_rate = allocate_for_total(curves, 0.1)
        self.assertLessEqual(alloc.total_budget, ratio_units(0.1) * len(curves))
        self.assertLessEqual(dp_rate, uniform_rate + 1e-9)

    def test_build_dp_rows_schema_and_kv_row(self):
        reqs = self._requests()
        ratios_in = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4]
        rates, ratios = measure_miss_rate_curves(
            reqs, ratios_in, policy="lru", min_buffer_slots=1
        )
        curves = build_ratio_cost_curves(rates, ratios)
        layer_ids = [2, 22, 42]  # e.g. DSv4 C4A layers
        rows = build_dp_rows(
            curves,
            [0.1, 0.2],
            layer_ids,
            kv_derived_ratio=0.15,
        )
        # Two sweep rows + one kv-derived row.
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertIn("total_ratio", row)
            self.assertIn("predicted_uniform_miss_rate", row)
            self.assertIn("predicted_dp_miss_rate", row)
            for lid in layer_ids:
                self.assertIn(f"layer_{lid}", row)
        self.assertEqual(sum(r["kv_budget_derived"] for r in rows), 1)

    def test_write_csv_reference_schema(self):
        reqs = self._requests()
        ratios_in = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4]
        rates, ratios = measure_miss_rate_curves(
            reqs, ratios_in, policy="lru", min_buffer_slots=1
        )
        curves = build_ratio_cost_curves(rates, ratios)
        layer_ids = [2, 22, 42]
        rows = build_dp_rows(curves, [0.1, 0.2], layer_ids)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dp_allocations.csv")
            write_dp_allocations_csv(path, rows, layer_ids)
            with open(path) as f:
                header = f.readline().strip().split(",")
        self.assertEqual(header[0], "total_ratio")
        self.assertEqual(header[1], "predicted_uniform_miss_rate")
        self.assertEqual(header[2], "predicted_dp_miss_rate")
        self.assertIn("layer_2", header)
        self.assertIn("layer_42", header)


class TestKvBudget(CustomTestCase):
    def test_kv_budget_to_ratio(self):
        # 40k slots / 3 layers / 20k ctx -> ~0.667 mean ratio.
        budget = {"per_dp_token_capacity": 40000}
        ratio = kv_budget_to_ratio(
            budget, num_layers=3, slot_ctx0=20000, compress_ratio=4
        )
        self.assertAlmostEqual(ratio, 40000 / 3 / 20000, places=6)

    def test_kv_budget_missing_capacity(self):
        self.assertIsNone(kv_budget_to_ratio({}, num_layers=3, slot_ctx0=20000))

    def test_kv_budget_list_takes_min(self):
        budget = {"per_dp_token_capacity": [50000, 40000, 60000]}
        ratio = kv_budget_to_ratio(budget, num_layers=4, slot_ctx0=10000)
        self.assertAlmostEqual(ratio, 40000 / 4 / 10000, places=6)


class TestKvBudgetParsers(CustomTestCase):
    def test_parse_server_info_min_across_dp(self):
        from sglang.srt.mem_cache.sparsity.trace.record_kv_budget import (
            parse_server_info,
        )

        info = {
            "internal_states": [
                {
                    "memory_usage": {"token_capacity": 50000, "kvcache": 12.5},
                    "max_total_num_tokens": 50000,
                    "startup_available_gpu_memory_gb": 70.1,
                },
                {"memory_usage": {"token_capacity": 48000}},
            ]
        }
        b = parse_server_info(info)
        self.assertEqual(b["per_dp_token_capacity"], [50000, 48000])
        self.assertEqual(b["token_capacity"], 48000)
        self.assertEqual(b["num_dp"], 2)

    def test_parse_server_info_pd_nested(self):
        from sglang.srt.mem_cache.sparsity.trace.record_kv_budget import (
            parse_server_info,
        )

        pd = {
            "decode": [{"internal_states": [{"memory_usage": {"token_capacity": 999}}]}]
        }
        b = parse_server_info(pd)
        self.assertEqual(b["token_capacity"], 999)

    def test_parse_metrics_text(self):
        from sglang.srt.mem_cache.sparsity.trace.record_kv_budget import (
            parse_metrics_text,
        )

        text = (
            "# HELP\n"
            'sglang:max_total_num_tokens{model="x"} 50000.0\n'
            'sglang:kv_available_tokens{model="x"} 12345\n'
        )
        m = parse_metrics_text(text)
        self.assertEqual(m["max_total_num_tokens"], 50000.0)
        self.assertEqual(m["kv_available_tokens"], 12345.0)


if __name__ == "__main__":
    unittest.main()
