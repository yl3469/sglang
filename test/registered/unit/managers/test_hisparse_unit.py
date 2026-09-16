"""Unit tests for HiSparse hierarchical sparse KV cache system.

Tests cover:
- CUDA kernel correctness (swap_in_selected_pages vs naive_load_topk oracle)
- Memory allocator lifecycle (alloc / free / available_size)
- Request lifecycle (staging path, direct-to-host path)
- Batch multi-request correctness
"""

import os
import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

# ---------------------------------------------------------------------------
# Test configuration (small-scale for fast CI runs)
# ---------------------------------------------------------------------------
SIZE = 2048  # device buffer pool size (tokens)
PAGE_SIZE = 64  # page size (must be 64 for CUDA, 1 for ROCm)
TOP_K = 256  # top-k selection count
DEVICE_BUFFER_SIZE = 512  # device buffer per request
HOST_TO_DEVICE_RATIO = 2
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
KV_CACHE_DIM = 576  # MLA dim (DeepSeek-style)
LAYER_NUM = 2
MAX_NUM_REQS = 8
MAX_CONTEXT_LEN = 2048


def _make_req(rid="test-req-0", origin_input_ids=None, output_ids=None):
    """Create a minimal mock Req object with the fields HiSparseCoordinator uses."""
    if origin_input_ids is None:
        origin_input_ids = list(range(64))
    if output_ids is None:
        output_ids = []
    req = SimpleNamespace(
        rid=rid,
        origin_input_ids=origin_input_ids,
        output_ids=output_ids,
        fill_ids=origin_input_ids + output_ids,
        seqlen=len(origin_input_ids) + len(output_ids),
        req_pool_idx=None,
        kv=SimpleNamespace(kv_allocated_len=0),
        kv_committed_len=0,
        finished_reason=None,
        hisparse_staging=False,
        staging=False,
        inflight_middle_chunks=0,
    )
    req.finished = lambda: req.finished_reason is not None
    req.set_extend_range = lambda start, end: setattr(
        req, "extend_range", Range(start, end)
    )
    return req


class TestHiSparseUnit(unittest.TestCase):
    """Test class that builds a minimal HiSparse component stack."""

    # ==================================================================
    # Fixture
    # ==================================================================

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for HiSparse tests.")
        if is_npu() or is_xpu():
            raise unittest.SkipTest("HiSparse tests only support CUDA/ROCm.")
        if not (is_cuda() or is_hip()):
            raise unittest.SkipTest("CUDA/ROCm not available.")

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29599")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
        cls.tp_group = torch.distributed.group.WORLD

        from sglang.srt.mem_cache.pool_host.common import (
            ALLOC_MEMORY_FUNCS,
            alloc_with_pin_memory,
        )

        cls._original_alloc = ALLOC_MEMORY_FUNCS["cuda"]
        ALLOC_MEMORY_FUNCS["cuda"] = alloc_with_pin_memory

        if is_hip():
            from sglang.srt.layers.attention.dsa.utils import (
                aiter_can_use_preshuffle_paged_mqa,
            )

            global_page_size = 64 if aiter_can_use_preshuffle_paged_mqa() else 1
        else:
            global_page_size = PAGE_SIZE

        from sglang.srt.mem_cache.allocator.hisparse import (
            HiSparseTokenToKVPoolAllocator,
        )
        from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool

        cls.device_pool = HiSparseDSATokenToKVPool(
            size=SIZE,
            page_size=global_page_size,
            kv_lora_rank=KV_LORA_RANK,
            dtype=torch.bfloat16,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            layer_num=LAYER_NUM,
            device="cuda",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=KV_CACHE_DIM,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )
        cls.allocator = HiSparseTokenToKVPoolAllocator(
            size=SIZE,
            page_size=global_page_size,
            dtype=torch.bfloat16,
            device="cuda",
            kvcache=cls.device_pool,
            need_sort=False,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )

        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

        cls.req_to_token_pool = ReqToTokenPool(
            size=MAX_NUM_REQS,
            max_context_len=MAX_CONTEXT_LEN,
            device="cuda",
            enable_memory_saver=False,
        )

        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        cls.page_size = global_page_size
        cls.coordinator = HiSparseCoordinator(
            req_to_token_pool=cls.req_to_token_pool,
            token_to_kv_pool_allocator=cls.allocator,
            top_k=TOP_K,
            device_buffer_size=DEVICE_BUFFER_SIZE,
            device="cuda",
            tp_group=cls.tp_group,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )

    @classmethod
    def tearDownClass(cls):
        from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS

        ALLOC_MEMORY_FUNCS["cuda"] = cls._original_alloc
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def setUp(self):
        """Reset shared allocator / coordinator state so tests are isolated.

        Without this, a mid-test assertion failure skips cleanup and leaks
        resources, causing unrelated failures in later tests.
        """
        self.allocator.clear()
        self.req_to_token_pool.clear()
        self.coordinator.mem_pool_host.clear()
        # Reset per-request coordinator bookkeeping
        self.coordinator.req_to_device_buffer.zero_()
        self.coordinator.req_device_buffer_size.zero_()
        self.coordinator.req_to_host_pool.fill_(-1)
        self.coordinator.req_to_host_pool_allocated_len.zero_()
        self.coordinator.req_device_buffer_tokens.fill_(-1)
        self.coordinator.req_device_buffer_token_locs.fill_(-1)
        self.coordinator.lru_slots[:] = self.coordinator._lru_init.view(1, 1, -1)
        self.coordinator.ack_staging_queue.clear()
        self.coordinator._has_pending_backup = False
        for i in range(len(self.coordinator._skip_first_backup)):
            self.coordinator._skip_first_backup[i] = False

    # ==================================================================
    # Low-level helpers
    # ==================================================================

    def _alloc_req_slot(self, req):
        """Allocate a req_pool_idx for the request."""
        indices = self.req_to_token_pool.alloc([req])
        self.assertIsNotNone(indices, "Failed to allocate req pool slot")
        return req.req_pool_idx

    def _free_req_slot(self, req):
        """Free the req_pool_idx."""
        if req.req_pool_idx is not None:
            self.req_to_token_pool.free(req)

    def _alloc_kv(self, req, fill_len, *, logical_only=False):
        """Allocate KV indices, write req_to_token_pool, update req fields.
        If logical_only=True, uses alloc_logical_only (PD-separated path).
        Returns kv_loc tensor."""
        device = self.allocator.device
        alloc_fn = (
            self.allocator.alloc_logical_only
            if logical_only
            else self.allocator.alloc_extend
        )
        kv_loc = alloc_fn(
            prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
            extend_num_tokens=fill_len,
        )
        self.assertIsNotNone(kv_loc, "KV alloc failed")
        self.req_to_token_pool.write((req.req_pool_idx, slice(0, len(kv_loc))), kv_loc)
        req.kv.kv_allocated_len = fill_len
        req.kv_committed_len = fill_len
        req.full_untruncated_fill_ids = array("q", range(fill_len))
        req.extend_range = Range(0, fill_len)
        return kv_loc

    # ==================================================================
    # Mid-level helpers
    # ==================================================================

    @staticmethod
    def _kv_pattern(layer_id, token_id):
        """Deterministic KV value for (layer, token) — used by write & verify."""
        v = (layer_id * 10000 + token_id + 1) * 0.001
        return float(torch.tensor(v, dtype=torch.bfloat16))

    def _write_device_patterns(self, kv_loc, fill_len):
        """Write distinguishable patterns into device KV buffer for all layers.

        kv_loc contains *logical* indices; we must translate them to hisparse
        device indices before indexing kv_buffer (which is sized for the
        hisparse pool, not the larger logical space).
        """
        hisparse_locs = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        for lid in range(LAYER_NUM):
            for i in range(fill_len):
                self.device_pool.kv_buffer[lid][hisparse_locs[i]] = self._kv_pattern(
                    lid, i
                )

    def _populate_host_pool(self, req, fill_len):
        """Allocate host slots, write known patterns, register in coordinator.
        Returns host_indices (cuda tensor)."""
        host_pool = self.coordinator.mem_pool_host
        host_indices = host_pool.alloc(fill_len)
        self.assertIsNotNone(host_indices, "Host alloc failed")
        host_indices = host_indices.to(device="cuda")
        self.coordinator.req_to_host_pool[req.req_pool_idx, :fill_len] = host_indices
        self.coordinator.req_to_host_pool_allocated_len[req.req_pool_idx] = fill_len
        for lid in range(LAYER_NUM):
            for i in range(fill_len):
                host_pool.kv_buffer[lid][host_indices[i]] = self._kv_pattern(lid, i)
        return host_indices

    def _build_topk_tokens(self, fill_len, *, include_newest=False):
        """Build a 1-D [TOP_K] int32 cuda tensor of token positions.

        If include_newest=True, fill_len-1 is guaranteed as the last valid slot.
        Pads with -1 when fill_len (or fill_len-1) < TOP_K.

        For long-sequence tests (fill_len > DEVICE_BUFFER_SIZE) where the
        "newest token" reserved slot is not populated (it requires an actual
        decode step + map_last_loc_to_buffer), callers should pass
        ``fill_len - 1`` as the effective pool size so position fill_len-1 is
        never randomly selected.
        """
        n = min(fill_len, TOP_K)
        if include_newest and n > 1:
            tokens = torch.randperm(fill_len - 1, device="cuda")[: n - 1].to(
                torch.int32
            )
            tokens = torch.cat(
                [tokens, torch.tensor([fill_len - 1], dtype=torch.int32, device="cuda")]
            )
        else:
            tokens = torch.randperm(fill_len, device="cuda")[:n].to(torch.int32)
        if n < TOP_K:
            pad = torch.full((TOP_K - n,), -1, dtype=torch.int32, device="cuda")
            tokens = torch.cat([tokens, pad])
        return tokens

    def _make_batch_tensors(self, reqs, fill_lens):
        """Build (req_pool_indices [int64], seq_lens [int32]) on cuda."""
        rpi = torch.tensor(
            [r.req_pool_idx for r in reqs], dtype=torch.int64, device="cuda"
        )
        sls = torch.tensor(fill_lens, dtype=torch.int32, device="cuda")
        return rpi, sls

    def _assert_kv_correct(self, locs_row, tokens_row, layer_id, count, msg=""):
        """Assert device KV data at *locs_row[:count]* matches the written
        pattern for the corresponding *tokens_row[:count]* positions."""
        for i in range(count):
            tok = int(tokens_row[i].item())
            if tok < 0:
                continue
            expected = self._kv_pattern(layer_id, tok)
            actual = self.device_pool.kv_buffer[layer_id][locs_row[i].long()]
            self.assertTrue(
                torch.allclose(
                    actual.float(),
                    torch.full_like(actual.float(), expected),
                    atol=1e-2,
                ),
                f"{msg}layer {layer_id}, token {tok}: KV data mismatch",
            )

    def _assert_matches_naive(self, rpi, sls, batch, kernel_locs, layer_id, msg=""):
        """Assert kernel swap_in KV data matches naive_load_topk KV data."""
        naive_locs = self.coordinator.naive_load_topk(rpi, sls, batch, layer_id)
        for b in range(batch.shape[0]):
            for i in range(TOP_K):
                if batch[b, i] < 0:
                    continue
                naive_data = self.device_pool.kv_buffer[layer_id][
                    naive_locs[b, i].long()
                ]
                kernel_data = self.device_pool.kv_buffer[layer_id][
                    kernel_locs[b, i].long()
                ]
                self.assertTrue(
                    torch.allclose(naive_data.float(), kernel_data.float(), atol=1e-2),
                    f"{msg}layer {layer_id}, b{b} idx {i}: naive != kernel",
                )

    def _swap_in_selected_pages(
        self,
        rpi: torch.Tensor,
        sls: torch.Tensor,
        batch: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Wrapper that sets num_real_reqs before calling swap_in_selected_pages.

        In production, model_runner sets num_real_reqs before each forward
        pass.  Tests must replicate that to get correct kernel behaviour.
        """
        self.coordinator.num_real_reqs[0] = rpi.shape[0]
        return self.coordinator.swap_in_selected_pages(rpi, sls, batch, layer_id)

    def _cleanup_req(self, req, kv_loc, *, logical_only=False):
        """request_finished -> free KV -> free req slot."""
        self.coordinator.request_finished(req)
        if logical_only:
            self.allocator.logical_attn_allocator.free(kv_loc)
        else:
            self.allocator.free(kv_loc)
        self._free_req_slot(req)

    def _get_initial_sizes(self):
        """Snapshot allocator available sizes."""
        return (
            self.allocator.logical_attn_allocator.available_size(),
            self.allocator.hisparse_attn_allocator.available_size(),
            self.coordinator.mem_pool_host.available_size(),
        )

    def _assert_sizes_restored(self, initial_sizes, msg=""):
        """Assert allocator sizes match the snapshot."""
        logical, hisparse, host = self._get_initial_sizes()
        self.assertEqual(logical, initial_sizes[0], f"Logical leak {msg}")
        self.assertEqual(hisparse, initial_sizes[1], f"HiSparse leak {msg}")
        self.assertEqual(host, initial_sizes[2], f"Host leak {msg}")

    # ==================================================================
    # Test: Kernel correctness — short sequence (fast path)
    # ==================================================================
    def test_kernel_correctness_short_seq(self):
        """Short seq (len <= device_buffer_size): kernel fast path returns
        device buffer locs, matching naive_load_topk."""
        initial = self._get_initial_sizes()
        req = _make_req("short-seq", list(range(self.page_size)))
        self._alloc_req_slot(req)

        fill_len = self.page_size
        kv_loc = self._alloc_kv(req, fill_len)
        self._write_device_patterns(kv_loc, fill_len)
        self.coordinator.alloc_device_buffer(req)

        tokens = self._build_topk_tokens(fill_len)
        batch = tokens.unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])

        for lid in range(LAYER_NUM):
            naive_locs = self.coordinator.naive_load_topk(rpi, sls, batch, lid)
            kernel_locs = self._swap_in_selected_pages(rpi, sls, batch, lid)
            valid = batch[0] >= 0
            self.assertTrue(
                torch.equal(naive_locs[0][valid].cpu(), kernel_locs[0][valid].cpu()),
                f"Layer {lid}: kernel locs != naive oracle",
            )

        self._cleanup_req(req, kv_loc)
        self._assert_sizes_restored(initial, "short_seq")

    # ==================================================================
    # Test: Kernel correctness — long sequence (cache miss + host DMA)
    # ==================================================================
    def test_kernel_correctness_long_seq(self):
        """Long seq (len > device_buffer_size): kernel loads from host,
        matching naive_load_topk for data correctness."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size * 2
        req = _make_req("long-seq", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self.coordinator.admit_request_direct(req)

        # Pass fill_len-1 so position fill_len-1 ("newest token") is never
        # randomly selected — its reserved device-buffer slot is only valid
        # after map_last_loc_to_buffer in a real decode step.
        tokens = self._build_topk_tokens(fill_len - 1)
        batch = tokens.unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])

        for lid in range(LAYER_NUM):
            naive_locs = self.coordinator.naive_load_topk(rpi, sls, batch, lid)
            kernel_locs = self._swap_in_selected_pages(rpi, sls, batch, lid)
            self.assertTrue(torch.all(naive_locs[0, :TOP_K] >= 0))
            self.assertTrue(torch.all(kernel_locs[0, :TOP_K] >= 0))
            # Verify both return correct KV data independently
            self._assert_kv_correct(naive_locs[0], tokens, lid, TOP_K, msg="Naive: ")
            self._assert_kv_correct(kernel_locs[0], tokens, lid, TOP_K, msg="Kernel: ")

        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "long_seq")

    # ==================================================================
    # Test: Kernel LRU replacement across multiple decode steps
    # ==================================================================
    def test_kernel_lru_replacement(self):
        """Multi-step swap-in: second call hits cached tokens, only
        evicts/loads new misses."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size * 2
        req = _make_req("lru-test", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self.coordinator.admit_request_direct(req)

        rpi, sls = self._make_batch_tensors([req], [fill_len])

        # Step 1: load the first TOP_K positions from host (no newest token —
        # the reserved slot is only valid after map_last_loc_to_buffer which is
        # called during an actual decode step, not modelled here).
        tokens_s1 = torch.arange(TOP_K, dtype=torch.int32, device="cuda")
        locs1 = self._swap_in_selected_pages(
            rpi, sls, tokens_s1.unsqueeze(0), layer_id=0
        )
        self.assertTrue(torch.all(locs1[0, :TOP_K] >= 0))

        # Step 2: half overlap (hit) + half new (miss).
        # Choose new tokens from a range safely below fill_len.
        half = TOP_K // 2
        new_start = TOP_K  # first position not in step-1
        tokens_s2 = torch.cat(
            [
                tokens_s1[:half],  # hits
                torch.arange(
                    new_start, new_start + half, dtype=torch.int32, device="cuda"
                ),  # misses
            ]
        )
        locs2 = self._swap_in_selected_pages(
            rpi, sls, tokens_s2.unsqueeze(0), layer_id=0
        )
        self.assertTrue(torch.all(locs2[0, :TOP_K] >= 0))

        # Verify repeated (hit) tokens still have correct KV data
        self._assert_kv_correct(
            locs2[0], tokens_s2, layer_id=0, count=half, msg="LRU hit: "
        )
        # Also verify new (miss) tokens loaded correctly
        self._assert_kv_correct(
            locs2[0, half:],
            tokens_s2[half:],
            layer_id=0,
            count=half,
            msg="LRU miss: ",
        )

        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "lru_replacement")

    # ==================================================================
    # Test: Allocator alloc/free lifecycle
    # ==================================================================
    def test_allocator_alloc_free_cycle(self):
        """alloc_extend / alloc_device_buffer / free restores available_size."""
        initial = self._get_initial_sizes()
        device = self.allocator.device
        fill_len = self.page_size * 2

        kv_loc = self.allocator.alloc_extend(
            prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
            extend_num_tokens=fill_len,
        )
        self.assertIsNotNone(kv_loc)
        self.assertEqual(len(kv_loc), fill_len)

        mapping = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        self.assertTrue(torch.all(mapping > 0), "Mapping should be non-zero")
        self.assertLess(self.allocator.available_size(), initial[0])

        need_size = min(
            ((fill_len + self.page_size - 1) // self.page_size) * self.page_size,
            DEVICE_BUFFER_SIZE,
        )
        buf_idx = self.allocator.alloc_device_buffer(kv_loc, need_size)
        self.assertIsNotNone(buf_idx)
        mapping_after = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        self.assertTrue(torch.all(mapping_after == 0), "Mapping should be cleared")

        self.allocator.free_hisparse_indices(buf_idx)
        self.allocator.logical_attn_allocator.free(kv_loc)
        self._assert_sizes_restored(initial, "alloc_free_cycle")

    def test_allocator_page_size_one_alloc_free_cycle(self):
        """alloc() maps logical to hisparse indices for ROCm page_size=1."""
        if self.page_size != 1:
            self.skipTest("page_size=1 alloc path is ROCm-specific")

        initial = self._get_initial_sizes()
        need_size = 16

        kv_loc = self.allocator.alloc(need_size)
        self.assertIsNotNone(kv_loc)
        self.assertEqual(len(kv_loc), need_size)

        mapping = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        self.assertTrue(torch.all(mapping > 0), "Mapping should be non-zero")
        self.assertLess(self.allocator.available_size(), initial[0])

        self.allocator.free(kv_loc)
        mapping_after = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        self.assertTrue(torch.all(mapping_after == 0), "Mapping should be cleared")
        self._assert_sizes_restored(initial, "page_size_one_alloc_free_cycle")

    def test_decode_remap_frees_stale_page_size_one_mapping(self):
        """map_last_loc_to_buffer frees the temporary alloc() hisparse slot."""
        if self.page_size != 1:
            self.skipTest("page_size=1 decode remap path is ROCm-specific")

        initial = self._get_initial_sizes()
        device = self.allocator.device
        fill_len = 2
        req = _make_req("decode-remap", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self.coordinator.alloc_device_buffer(req)
        self.coordinator._skip_first_backup[req.req_pool_idx] = True

        out_loc = self.allocator.alloc(1)
        self.assertIsNotNone(out_loc)
        stale_loc = self.allocator.full_to_hisparse_device_index_mapping[
            out_loc
        ].clone()
        self.assertTrue(torch.all(stale_loc > 0), "Temporary mapping should exist")

        seq_len = fill_len + 1
        self.req_to_token_pool.write((req.req_pool_idx, fill_len), out_loc)
        req.kv.kv_allocated_len = seq_len
        req.kv_committed_len = seq_len

        self.coordinator.map_last_loc_to_buffer(
            seq_lens=torch.tensor([seq_len], dtype=torch.int64, device=device),
            out_cache_loc=out_loc,
            req_pool_indices=torch.tensor(
                [req.req_pool_idx], dtype=torch.int64, device=device
            ),
            seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([req.req_pool_idx], dtype=torch.int64),
        )

        remapped_loc = self.allocator.full_to_hisparse_device_index_mapping[out_loc]
        self.assertTrue(torch.all(remapped_loc > 0), "Remapped loc should exist")
        self.assertFalse(
            torch.equal(stale_loc, remapped_loc),
            "Decode loc should move from temporary mapping to device buffer",
        )
        self.assertEqual(
            self.allocator.hisparse_attn_allocator.available_size(),
            initial[1] - seq_len,
        )

        self.coordinator.request_finished(req)
        self.allocator.logical_attn_allocator.free(torch.cat([kv_loc, out_loc]))
        self._free_req_slot(req)
        self._assert_sizes_restored(initial, "decode_remap")

    # ==================================================================
    # Test: Staging (PD Colocate) path
    # ==================================================================
    def test_request_lifecycle_staging_path(self):
        """prefill -> staging DMA -> collect_ready -> swap-in -> finish."""
        initial = self._get_initial_sizes()
        fill_len = self.page_size
        req = _make_req("staging-req", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self._write_device_patterns(kv_loc, fill_len)

        self.coordinator.admit_request_into_staging(req)
        self.assertTrue(req.hisparse_staging)

        torch.cuda.synchronize()
        ready = self.coordinator.collect_ready_reqs()
        self.assertEqual(len(ready), 1)
        self.assertFalse(req.hisparse_staging)
        self.assertTrue(self.coordinator._skip_first_backup[req.req_pool_idx])

        tokens = self._build_topk_tokens(fill_len)
        batch = tokens.unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])

        locs = self._swap_in_selected_pages(rpi, sls, batch, layer_id=0)
        valid_n = min(fill_len, TOP_K)
        self.assertTrue(torch.all(locs[0, :valid_n] >= 0))
        self._assert_kv_correct(
            locs[0], tokens, layer_id=0, count=valid_n, msg="Staging: "
        )
        self._assert_matches_naive(rpi, sls, batch, locs, layer_id=0, msg="Staging: ")

        self._cleanup_req(req, kv_loc)
        self._assert_sizes_restored(initial, "staging_path")

    # ==================================================================
    # Test: Single-node staging host page allocation
    # ==================================================================
    def test_single_node_staging_allocates_paged_host_slots(self):
        """Single-node staging should allocate host slots at page granularity."""
        initial = self._get_initial_sizes()
        fill_len = self.page_size * 2 + 1
        rounded_len = (fill_len + self.page_size - 1) // self.page_size * self.page_size
        req = _make_req("single-node-staging-pages", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self._write_device_patterns(kv_loc, fill_len)

        self.coordinator.admit_request_into_staging(req)
        torch.cuda.synchronize()
        ready = self.coordinator.collect_ready_reqs()
        self.assertEqual(ready, [req])

        host_row = self.coordinator.req_to_host_pool[req.req_pool_idx, :rounded_len]
        self.assertTrue(torch.all(host_row >= 0))
        self.assertEqual(torch.unique(host_row).numel(), rounded_len)
        self.assertEqual(
            int(self.coordinator.req_to_host_pool_allocated_len[req.req_pool_idx]),
            rounded_len,
        )

        available_size = self.coordinator.mem_pool_host.available_size()
        next_host_index = self.coordinator.mem_pool_host.alloc_paged_token_slots(
            self.coordinator.req_to_host_pool,
            self.coordinator.req_to_host_pool_allocated_len,
            req.req_pool_idx,
            fill_len,
            1,
        )
        # With page_size>1 the rounded-up staging allocation provides headroom,
        # so no new pages are needed.  With page_size=1 there is no headroom and
        # exactly one new page is allocated for the next token.
        expected_new_pages = 0 if fill_len < rounded_len else 1
        self.assertEqual(
            self.coordinator.mem_pool_host.available_size(),
            available_size - expected_new_pages,
        )
        self.assertTrue(torch.all(next_host_index >= 0))

        expected_total = rounded_len + expected_new_pages * self.page_size
        allocated_host_indices = self.coordinator.mem_pool_host.allocated_host_indices(
            self.coordinator.req_to_host_pool,
            req.req_pool_idx,
            int(self.coordinator.req_to_host_pool_allocated_len[req.req_pool_idx]),
        )
        self.assertEqual(allocated_host_indices.numel(), expected_total)

        self._cleanup_req(req, kv_loc)
        self._assert_sizes_restored(initial, "single_node_staging_pages")

    # ==================================================================
    # Test: Direct-to-host (PD separated) path
    # ==================================================================
    def test_request_lifecycle_direct_path(self):
        """alloc_logical_only -> host write -> admit_direct -> swap-in -> finish."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size
        req = _make_req("direct-req", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self.coordinator.admit_request_direct(req)

        self.assertFalse(req.staging)
        self.assertTrue(self.coordinator._skip_first_backup[req.req_pool_idx])
        buf_tokens = self.coordinator.req_device_buffer_tokens[
            :, req.req_pool_idx, :DEVICE_BUFFER_SIZE
        ]
        self.assertTrue(torch.all(buf_tokens == -1))

        tokens = self._build_topk_tokens(fill_len - 1)
        batch = tokens.unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])

        locs = self._swap_in_selected_pages(rpi, sls, batch, layer_id=0)
        self.assertTrue(torch.all(locs[0, :TOP_K] >= 0))
        self._assert_kv_correct(
            locs[0], tokens, layer_id=0, count=TOP_K, msg="Direct: "
        )
        self._assert_matches_naive(rpi, sls, batch, locs, layer_id=0, msg="Direct: ")

        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "direct_path")

    # ==================================================================
    # Test: PD decode prealloc host page allocation
    # ==================================================================
    def test_pd_decode_prealloc_hisparse_host_slots(self):
        """PD decode prealloc should allocate RDMA targets through the host pool."""
        initial = self._get_initial_sizes()
        fill_len = self.page_size * 2 + 1
        req = _make_req("pd-decode-prealloc", list(range(fill_len)))

        from sglang.srt.disaggregation.decode import DecodePreallocQueue

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.req_to_token_pool = self.req_to_token_pool
        queue.token_to_kv_pool_allocator = self.allocator
        queue.token_to_kv_pool = self.allocator.get_kvcache()
        queue.tree_cache = SimpleNamespace(
            evictable_size=lambda: 0,
            protected_size=lambda: 0,
        )
        queue.scheduler = SimpleNamespace(
            enable_hisparse=True,
            hisparse_coordinator=self.coordinator,
            server_args=SimpleNamespace(disaggregation_decode_enable_radix_cache=False),
        )

        host_indices = queue._pre_alloc(req)
        self.assertEqual(host_indices.numel(), fill_len)
        self.assertTrue(torch.all(host_indices >= 0))
        self.assertTrue(
            torch.equal(
                host_indices,
                self.coordinator.req_to_host_pool[req.req_pool_idx, :fill_len],
            )
        )
        self.assertEqual(req.kv.kv_allocated_len, fill_len)
        self.assertEqual(req.kv_committed_len, fill_len)
        self.assertEqual(req.extend_range.length, fill_len)

        rounded_len = (fill_len + self.page_size - 1) // self.page_size * self.page_size
        self.assertEqual(
            int(self.coordinator.req_to_host_pool_allocated_len[req.req_pool_idx]),
            rounded_len,
        )
        allocated_host_indices = self.coordinator.mem_pool_host.allocated_host_indices(
            self.coordinator.req_to_host_pool,
            req.req_pool_idx,
            int(self.coordinator.req_to_host_pool_allocated_len[req.req_pool_idx]),
        )
        self.assertEqual(allocated_host_indices.numel(), rounded_len)

        kv_loc = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.kv.kv_allocated_len
        ].clone()
        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "pd_decode_prealloc_hisparse")

    # ==================================================================
    # Test: Batch multiple requests
    # ==================================================================
    def test_batch_multiple_requests(self):
        """Mix of short & long requests in batch: kernel correct + no leaks."""
        initial = self._get_initial_sizes()

        configs = [
            ("batch-short-0", self.page_size),
            ("batch-short-1", self.page_size),
            ("batch-long-0", DEVICE_BUFFER_SIZE + self.page_size),
            ("batch-long-1", DEVICE_BUFFER_SIZE + self.page_size * 2),
        ]

        reqs, kv_locs = [], []
        for rid, fl in configs:
            req = _make_req(rid, list(range(fl)))
            self._alloc_req_slot(req)
            is_long = fl > DEVICE_BUFFER_SIZE
            kv_loc = self._alloc_kv(req, fl, logical_only=is_long)
            if is_long:
                self._populate_host_pool(req, fl)
                self.coordinator.admit_request_direct(req)
            else:
                self._write_device_patterns(kv_loc, fl)
                self.coordinator.alloc_device_buffer(req)
            reqs.append(req)
            kv_locs.append(kv_loc)

        rpi, sls = self._make_batch_tensors(reqs, [c[1] for c in configs])
        top_k_batch = torch.stack(
            [
                # For long sequences pass fl-1 to exclude the "newest token" position
                # whose reserved device-buffer slot is not populated in unit tests.
                self._build_topk_tokens(fl - 1 if fl > DEVICE_BUFFER_SIZE else fl)
                for _, fl in configs
            ]
        )

        for lid in range(LAYER_NUM):
            locs = self._swap_in_selected_pages(rpi, sls, top_k_batch, lid)
            for i, (rid, fl) in enumerate(configs):
                vn = min(fl, TOP_K)
                self.assertTrue(
                    torch.all(locs[i, :vn] >= 0),
                    f"Req {rid}, layer {lid}: negative locs",
                )
                self._assert_kv_correct(
                    locs[i], top_k_batch[i], lid, vn, msg=f"{rid}: "
                )

        for i, req in enumerate(reqs):
            is_long = configs[i][1] > DEVICE_BUFFER_SIZE
            self._cleanup_req(req, kv_locs[i], logical_only=is_long)

        self._assert_sizes_restored(initial, "batch_multiple")


class TestHiSparsePerLayerBufferRegions(unittest.TestCase):
    """The pool appends num_slots * B_l private rows to layer l only, so two
    layouts with equal slot totals allocate equal bytes."""

    def _pool(self, sizes):
        from sglang.srt.mem_cache.hisparse_buffer_layout import HiSparseBufferLayout
        from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool

        layout = HiSparseBufferLayout.build(
            context_rows=SIZE,
            page_size=PAGE_SIZE,
            layer_num=LAYER_NUM,
            buffer_sizes=sizes,
            num_slots=MAX_NUM_REQS,
        )
        return HiSparseDSATokenToKVPool(
            size=SIZE,
            page_size=PAGE_SIZE,
            kv_lora_rank=KV_LORA_RANK,
            dtype=torch.bfloat16,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            layer_num=LAYER_NUM,
            device="cuda",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=KV_CACHE_DIM,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
            buffer_layout=layout,
        )

    def test_layer_rows_and_byte_equality(self):
        dp = self._pool((256, 512))
        self.assertEqual(
            dp.kv_buffer[0].shape[0], SIZE + PAGE_SIZE + MAX_NUM_REQS * 256
        )
        self.assertEqual(
            dp.kv_buffer[1].shape[0], SIZE + PAGE_SIZE + MAX_NUM_REQS * 512
        )
        uniform = self._pool((384, 384))
        self.assertEqual(uniform.get_kv_size_bytes(), dp.get_kv_size_bytes())

    def test_copy_rows_per_layer_moves_only_that_layer(self):
        pool = self._pool((256, 512))
        src = torch.tensor([3, 4], device="cuda")
        dst = torch.tensor([SIZE + PAGE_SIZE, SIZE + PAGE_SIZE + 1], device="cuda")
        pool.kv_buffer[0][src] = torch.randn_like(pool.kv_buffer[0][src])
        pool.kv_buffer[1][src] = torch.randn_like(pool.kv_buffer[1][src])
        pool.copy_rows_per_layer(0, dst, src)
        self.assertTrue(torch.equal(pool.kv_buffer[0][dst], pool.kv_buffer[0][src]))
        self.assertTrue(torch.all(pool.kv_buffer[1][dst] == 0))


class TestHiSparseResidentLruSplit(TestHiSparseUnit):
    """Run the whole coordinator suite on a heterogeneous per-layer layout
    (B = (256, 512)) and add the resident/LRU split checks.

    A layer stays on the request's shared rows while ``seq_len <= B_l`` and
    moves to its private rows (with a warm copy of tokens ``[0, B_l)``) once
    ``seq_len > B_l``; the shared rows go back to the pool after every layer
    has moved and the last resident-slot token has been backed up.
    """

    BUFFER_SIZES = (256, 512)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from sglang.srt.mem_cache.allocator.hisparse import (
            HiSparseTokenToKVPoolAllocator,
        )
        from sglang.srt.mem_cache.hisparse_buffer_layout import HiSparseBufferLayout
        from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        cls.layout = HiSparseBufferLayout.build(
            context_rows=SIZE,
            page_size=cls.page_size,
            layer_num=LAYER_NUM,
            buffer_sizes=cls.BUFFER_SIZES,
            num_slots=cls.req_to_token_pool.req_to_token.shape[0],
        )
        cls.device_pool = HiSparseDSATokenToKVPool(
            size=SIZE,
            page_size=cls.page_size,
            kv_lora_rank=KV_LORA_RANK,
            dtype=torch.bfloat16,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            layer_num=LAYER_NUM,
            device="cuda",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=KV_CACHE_DIM,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
            buffer_layout=cls.layout,
        )
        cls.allocator = HiSparseTokenToKVPoolAllocator(
            size=SIZE,
            page_size=cls.page_size,
            dtype=torch.bfloat16,
            device="cuda",
            kvcache=cls.device_pool,
            need_sort=False,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )
        cls.coordinator = HiSparseCoordinator(
            req_to_token_pool=cls.req_to_token_pool,
            token_to_kv_pool_allocator=cls.allocator,
            top_k=TOP_K,
            device_buffer_size=DEVICE_BUFFER_SIZE,
            device_buffer_sizes=list(cls.BUFFER_SIZES),
            device="cuda",
            tp_group=cls.tp_group,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )

    def setUp(self):
        super().setUp()
        for req_idx in range(MAX_NUM_REQS):
            self.coordinator.layer_buffers.reset(req_idx)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _admit_with_patterns(self, req, fill_len):
        """Staging-style admission of a prefilled request whose device rows
        and host slots hold the per-layer test patterns."""
        self._alloc_req_slot(req)
        kv_loc = self._alloc_kv(req, fill_len)
        self._write_device_patterns(kv_loc, fill_len)
        self._populate_host_pool(req, fill_len)
        self.coordinator.alloc_device_buffer(req)
        self.coordinator._skip_first_backup[req.req_pool_idx] = True
        return kv_loc

    def _decode_step(self, req, seq_len):
        """Allocate the logical slot for token ``seq_len-1`` and run the
        scheduler-side decode bookkeeping for it."""
        device = self.allocator.device
        last_loc = self.req_to_token_pool.req_to_token[req.req_pool_idx, seq_len - 2]
        out_loc = self.allocator.alloc_decode(
            seq_lens=torch.tensor([seq_len], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int64),
            last_loc=last_loc.view(1),
        )
        self.assertIsNotNone(out_loc)
        self.req_to_token_pool.write((req.req_pool_idx, seq_len - 1), out_loc)
        req.kv.kv_allocated_len = seq_len
        req.kv_committed_len = seq_len
        self.coordinator.map_last_loc_to_buffer(
            seq_lens=torch.tensor([seq_len], dtype=torch.int64, device=device),
            out_cache_loc=out_loc,
            req_pool_indices=torch.tensor(
                [req.req_pool_idx], dtype=torch.int64, device=device
            ),
            seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([req.req_pool_idx], dtype=torch.int64),
        )
        return out_loc

    def _assert_private_warm(self, layer_id, req_idx, n_check):
        """Layer ``layer_id`` is in LRU mode on its private rows and slots
        ``[0, n_check)`` hold tokens ``[0, n_check)`` with the right KV."""
        b = self.BUFFER_SIZES[layer_id]
        locs = self.coordinator.req_device_buffer_token_locs[layer_id, req_idx, :b]
        self.assertTrue(torch.all(locs >= self.layout.region_start))
        self.assertTrue(
            torch.equal(
                locs.to(torch.int64),
                self.layout.slot_row_start(layer_id, req_idx)
                + torch.arange(b, dtype=torch.int64, device=locs.device),
            )
        )
        tokens = self.coordinator.req_device_buffer_tokens[layer_id, req_idx, :b]
        self.assertTrue(
            torch.equal(tokens.cpu(), torch.arange(b, dtype=torch.int32)),
            f"layer {layer_id} warm tokens",
        )
        got = self.device_pool.kv_buffer[layer_id][locs[:n_check].to(torch.int64)]
        got = got.reshape(n_check, -1)[:, 0].float().cpu()
        want = torch.tensor([self._kv_pattern(layer_id, i) for i in range(n_check)])
        self.assertTrue(torch.equal(got, want), f"layer {layer_id} warm KV")

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_admission_splits_layers_by_capacity(self):
        """seq_len=320 with B=(256,512): layer 0 goes LRU (warm), layer 1 stays
        resident on the shared rows."""
        initial = self._get_initial_sizes()
        fill_len = self.page_size * 5  # 320 > 256, <= 512
        req = _make_req("split-admit", list(range(fill_len)))
        kv_loc = self._admit_with_patterns(req, fill_len)
        c = self.coordinator
        req_idx = req.req_pool_idx

        self.assertEqual(c.layer_buffers.lru_layer_count(req_idx), 1)
        self.assertTrue(c.layer_buffers.is_lru(0, req_idx))
        self.assertFalse(c.layer_buffers.is_lru(1, req_idx))
        self.assertFalse(c.layer_buffers.shared_released(req_idx))
        self._assert_private_warm(0, req_idx, n_check=256)

        locs1 = c.req_device_buffer_token_locs[1, req_idx, :fill_len]
        self.assertTrue(torch.all(locs1 < self.layout.region_start))
        self.assertTrue(
            torch.equal(
                locs1.to(torch.int64),
                c.req_to_device_buffer[req_idx, :fill_len].to(torch.int64),
            )
        )

        # The kernel must serve both layers correctly from their own rows.
        rpi, sls = self._make_batch_tensors([req], [fill_len])
        for layer_id in range(LAYER_NUM):
            tokens = self._build_topk_tokens(fill_len - 1)
            batch = tokens.unsqueeze(0)
            locs = self._swap_in_selected_pages(rpi, sls, batch, layer_id)
            self._assert_kv_correct(
                locs[0], tokens, layer_id, count=TOP_K, msg=f"split L{layer_id}: "
            )
            if fill_len <= self.BUFFER_SIZES[layer_id]:
                # The oracle is only slot-stable for resident layers; in LRU
                # mode it reloads into the first slots and clobbers the rows
                # the kernel just filled (true for shared rows as well).
                self._assert_matches_naive(
                    rpi, sls, batch, locs, layer_id, msg=f"split L{layer_id}: "
                )

        self._cleanup_req(req, kv_loc)
        self._assert_sizes_restored(initial, "split_admit")

    def test_transition_and_release_return_shared_rows(self):
        """Decode from 320 to 580: layer 1 moves to its private rows at
        seq_len 513 and the shared rows are freed at seq_len 514."""
        initial = self._get_initial_sizes()
        fill_len = self.page_size * 5
        req = _make_req("split-decode", list(range(fill_len)))
        kv_loc = self._admit_with_patterns(req, fill_len)
        c = self.coordinator
        req_idx = req.req_pool_idx
        shared_width = self.layout.shared_width
        self.assertEqual(shared_width, DEVICE_BUFFER_SIZE)

        decode_locs = []
        for seq_len in range(fill_len + 1, shared_width + 69):
            hisparse_free_before = (
                self.allocator.hisparse_attn_allocator.available_size()
            )
            decode_locs.append(self._decode_step(req, seq_len))
            if seq_len <= shared_width:
                self.assertEqual(c.layer_buffers.lru_layer_count(req_idx), 1, seq_len)
                self.assertFalse(c.layer_buffers.shared_released(req_idx), seq_len)
            elif seq_len == shared_width + 1:
                self.assertEqual(c.layer_buffers.lru_layer_count(req_idx), 2, seq_len)
                self.assertFalse(c.layer_buffers.shared_released(req_idx), seq_len)
                self._assert_private_warm(1, req_idx, n_check=fill_len)
            elif seq_len == shared_width + 2:
                self.assertTrue(c.layer_buffers.shared_released(req_idx), seq_len)
                torch.cuda.synchronize()
                self.assertEqual(
                    self.allocator.hisparse_attn_allocator.available_size(),
                    hisparse_free_before + shared_width,
                    "shared rows must return to the pool on release",
                )
                self.assertTrue(
                    torch.all(c.req_to_device_buffer[req_idx, :shared_width] == 0)
                )
                self.assertTrue(
                    torch.all(c.req_to_device_buffer[req_idx, shared_width] > 0)
                )
            else:
                self.assertTrue(c.layer_buffers.shared_released(req_idx), seq_len)

        # Post-release: the newest token still lands in the reserved row and
        # both layers serve top-k from private rows + the reserved slot.
        seq_len = shared_width + 68
        rpi, sls = self._make_batch_tensors([req], [seq_len])
        for layer_id in range(LAYER_NUM):
            tokens = self._build_topk_tokens(fill_len)
            batch = tokens.unsqueeze(0)
            locs = self._swap_in_selected_pages(rpi, sls, batch, layer_id)
            self.assertTrue(torch.all(locs[0, :TOP_K] >= 0))
            self._assert_kv_correct(
                locs[0],
                tokens,
                layer_id,
                count=TOP_K,
                msg=f"post-release L{layer_id}: ",
            )

        self._cleanup_req(req, torch.cat([kv_loc, torch.cat(decode_locs)]))
        self._assert_sizes_restored(initial, "split_decode")

    def test_long_admission_goes_straight_to_lru_and_releases(self):
        """seq_len > shared_width at admission: every layer is LRU (warm) and
        the shared rows are released immediately."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size
        req = _make_req("split-long", list(range(fill_len)))
        kv_loc = self._admit_with_patterns(req, fill_len)
        c = self.coordinator
        req_idx = req.req_pool_idx

        self.assertEqual(c.layer_buffers.lru_layer_count(req_idx), LAYER_NUM)
        self.assertTrue(c.layer_buffers.shared_released(req_idx))
        self._assert_private_warm(0, req_idx, n_check=256)
        self._assert_private_warm(1, req_idx, n_check=512)
        # Only the reserved page is still held by the request.
        self.assertEqual(
            self.allocator.hisparse_attn_allocator.available_size(),
            initial[1] - self.page_size,
        )

        self._cleanup_req(req, kv_loc)
        self._assert_sizes_restored(initial, "split_long")


if __name__ == "__main__":
    unittest.main()
