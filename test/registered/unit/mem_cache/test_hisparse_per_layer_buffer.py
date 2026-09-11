"""Unit tests for HiSparse per-layer device-buffer sizing (CPU, no model).

Exercises the pure-logic pieces added for per-layer device buffers without
constructing a real HiSparseCoordinator (which needs GPU pools):

* factory parsing of ``device_buffer_sizes`` / ``device_buffer_sizes_path``
* the per-layer capacity resolve + validation logic
* the reserved-token-loc write (uniform broadcast vs per-layer scatter), and the
  key invariant that a UNIFORM per-layer config writes exactly what the original
  single-broadcast write did.
"""

import json
import os
import tempfile
import unittest

import torch

from sglang.srt.mem_cache.sparsity.factory import _parse_sparse_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeArgs:
    def __init__(self, s):
        self.hisparse_config = s


class TestFactoryParse(CustomTestCase):
    def test_none_default(self):
        c = _parse_sparse_config(_FakeArgs(None))
        self.assertIsNone(c.device_buffer_sizes)

    def test_inline_list_sets_physical_max(self):
        c = _parse_sparse_config(
            _FakeArgs(
                json.dumps(
                    {
                        "top_k": 512,
                        "device_buffer_size": 1024,
                        "device_buffer_sizes": [512, 1024, 2048],
                    }
                )
            )
        )
        self.assertEqual(c.device_buffer_sizes, [512, 1024, 2048])
        self.assertEqual(c.device_buffer_size, 2048)  # physical width = max

    def test_path_bare_list(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "b.json")
            with open(p, "w") as f:
                json.dump([512, 768, 900], f)
            c = _parse_sparse_config(
                _FakeArgs(json.dumps({"top_k": 512, "device_buffer_sizes_path": p}))
            )
        self.assertEqual(c.device_buffer_sizes, [512, 768, 900])

    def test_path_dict_form(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "b.json")
            with open(p, "w") as f:
                json.dump({"device_buffer_sizes": [640, 512]}, f)
            c = _parse_sparse_config(
                _FakeArgs(json.dumps({"top_k": 512, "device_buffer_sizes_path": p}))
            )
        self.assertEqual(c.device_buffer_sizes, [640, 512])

    def test_path_wrong_schema_raises(self):
        # A file that exists but lacks the expected key must fail loudly, not
        # silently fall back to the uniform path with the feature disabled.
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "b.json")
            with open(p, "w") as f:
                json.dump({"buffer_sizes": [640, 512]}, f)  # wrong key
            with self.assertRaisesRegex(ValueError, "does not contain"):
                _parse_sparse_config(
                    _FakeArgs(json.dumps({"top_k": 512, "device_buffer_sizes_path": p}))
                )

    def test_below_topk_raises(self):
        with self.assertRaises(ValueError):
            _parse_sparse_config(
                _FakeArgs(json.dumps({"top_k": 512, "device_buffer_sizes": [256]}))
            )

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            _parse_sparse_config(
                _FakeArgs(json.dumps({"top_k": 512, "device_buffer_sizes": []}))
            )


class _ResolveShim:
    """Replicates HiSparseCoordinator's per-layer resolve + reserved-write logic.

    We can't build a real coordinator on CPU (it needs GPU KV pools), so this
    shim carries the exact same fields and methods copied structurally, letting
    us test the logic in isolation. If the coordinator logic changes, these
    tests should be updated in lockstep.
    """

    def __init__(self, device_buffer_size, top_k, layer_num, device_buffer_sizes):
        self.device_buffer_size = device_buffer_size
        self.top_k = top_k
        self.layer_num = layer_num
        if device_buffer_sizes is None:
            self.device_buffer_sizes = [device_buffer_size] * layer_num
        else:
            sizes = list(device_buffer_sizes)
            if len(sizes) != layer_num:
                raise ValueError("length mismatch")
            for b in sizes:
                if b < top_k:
                    raise ValueError("below top_k")
                if b > device_buffer_size:
                    raise ValueError("exceeds physical")
            self.device_buffer_sizes = sizes
        self._uniform_buffer = len(set(self.device_buffer_sizes)) == 1
        self._reserved_col_uniform = self.device_buffer_sizes[0]
        padded = device_buffer_size + 4  # + page_size
        self.req_device_buffer_token_locs = torch.full(
            (layer_num, 8, padded), -1, dtype=torch.int32
        )

    # Copied structurally from HiSparseCoordinator._write_reserved_token_locs.
    def write_reserved(self, req_indices, reserved_buffer_loc):
        loc_i32 = reserved_buffer_loc.to(torch.int32)
        if self._uniform_buffer:
            self.req_device_buffer_token_locs[
                :, req_indices, self._reserved_col_uniform
            ] = loc_i32
        else:
            for layer_id in range(self.layer_num):
                self.req_device_buffer_token_locs[
                    layer_id, req_indices, self.device_buffer_sizes[layer_id]
                ] = loc_i32


class TestResolveAndReservedWrite(CustomTestCase):
    def test_none_gives_uniform(self):
        s = _ResolveShim(1024, 512, 21, None)
        self.assertEqual(s.device_buffer_sizes, [1024] * 21)
        self.assertTrue(s._uniform_buffer)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            _ResolveShim(1024, 512, 21, [1024] * 20)

    def test_exceeds_physical_raises(self):
        with self.assertRaises(ValueError):
            _ResolveShim(1024, 512, 3, [1024, 2048, 1024])

    def test_uniform_write_matches_stock_broadcast(self):
        # The uniform per-layer write must equal the original single broadcast
        # write at column device_buffer_size.
        layer_num, dbs = 4, 1024
        s = _ResolveShim(dbs, 512, layer_num, [dbs] * layer_num)
        req = torch.tensor([0, 2, 5], dtype=torch.int64)
        loc = torch.tensor([11, 22, 33], dtype=torch.int32)
        s.write_reserved(req, loc)

        # Reference: stock broadcast at column dbs.
        ref = torch.full_like(s.req_device_buffer_token_locs, -1)
        ref[:, req, dbs] = loc
        self.assertTrue(torch.equal(s.req_device_buffer_token_locs, ref))

    def test_per_layer_write_targets_correct_columns(self):
        sizes = [512, 768, 1024]
        s = _ResolveShim(1024, 512, 3, sizes)
        self.assertFalse(s._uniform_buffer)
        req = torch.tensor([1, 3], dtype=torch.int64)
        loc = torch.tensor([7, 9], dtype=torch.int32)
        s.write_reserved(req, loc)
        for layer_id, b in enumerate(sizes):
            col = s.req_device_buffer_token_locs[layer_id, req, b]
            self.assertTrue(torch.equal(col, loc))
            # A different column (the physical max) must remain untouched (-1)
            # for the shorter-capacity layers.
            if b < 1024:
                self.assertTrue(
                    torch.all(s.req_device_buffer_token_locs[layer_id, req, 1024] == -1)
                )


if __name__ == "__main__":
    unittest.main()
