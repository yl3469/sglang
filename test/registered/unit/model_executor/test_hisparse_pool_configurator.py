import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHiSparsePoolConfigurator(CustomTestCase):
    def _kvc(
        self,
        kv_cache_dtype: torch.dtype,
        *,
        enable_hisparse: bool,
        host_to_device_ratio: int = 1,
        device_buffer_sizes=None,
        max_running_requests=None,
        attn_dp_size: int = 1,
    ):
        hisparse_config = {"host_to_device_ratio": host_to_device_ratio, "top_k": 256}
        if device_buffer_sizes is not None:
            hisparse_config["device_buffer_sizes"] = list(device_buffer_sizes)
        hf_config = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            index_topk=2048,
            index_head_dim=128,
        )
        hf_config.get_text_config = lambda: hf_config
        return MagicMock(
            use_mla_backend=True,
            kv_cache_dtype=kv_cache_dtype,
            model_config=SimpleNamespace(
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                hf_config=hf_config,
            ),
            ps=SimpleNamespace(attn_dp_size=attn_dp_size),
            server_args=SimpleNamespace(
                enable_hisparse=enable_hisparse,
                hisparse_config=json.dumps(hisparse_config),
                dsa_prefill_backend="flashmla_sparse",
                dsa_decode_backend="flashmla_sparse",
                max_running_requests=max_running_requests,
                disaggregation_mode="null",
                disaggregation_decode_extra_slots=0,
            ),
        )

    def _compute_cell_size(
        self,
        kv_cache_dtype: torch.dtype,
        *,
        enable_hisparse: bool,
        host_to_device_ratio: int = 1,
    ) -> int:
        hf_config = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            index_topk=2048,
            index_head_dim=128,
        )
        hf_config.get_text_config = lambda: hf_config
        kvc = MagicMock(
            use_mla_backend=True,
            kv_cache_dtype=kv_cache_dtype,
            model_config=SimpleNamespace(
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                hf_config=hf_config,
            ),
            server_args=SimpleNamespace(
                enable_hisparse=enable_hisparse,
                hisparse_config=(f'{{"host_to_device_ratio": {host_to_device_ratio}}}'),
                dsa_prefill_backend="flashmla_sparse",
                dsa_decode_backend="flashmla_sparse",
            ),
        )

        with get_parallel().override(attn_tp_size=1):
            configurator = object.__new__(DefaultPoolConfigurator)
            return configurator._compute_cell_size(kvc, num_layers=2)

    def test_mla_layout_without_hisparse(self):
        for kv_cache_dtype, expected_cell_size in (
            (torch.bfloat16, 2568),
            (torch.float8_e4m3fn, 1576),
        ):
            with self.subTest(kv_cache_dtype=kv_cache_dtype):
                cell_size = self._compute_cell_size(
                    kv_cache_dtype,
                    enable_hisparse=False,
                )
                self.assertEqual(cell_size, expected_cell_size)

    def test_hisparse_indexer_scales_with_ratio(self):
        for host_to_device_ratio, expected_cell_size in (
            (2, 1840),
            (4, 2368),
        ):
            with self.subTest(host_to_device_ratio=host_to_device_ratio):
                cell_size = self._compute_cell_size(
                    torch.float8_e4m3fn,
                    enable_hisparse=True,
                    host_to_device_ratio=host_to_device_ratio,
                )
                self.assertEqual(cell_size, expected_cell_size)

    def test_per_layer_buffer_bytes_are_slots_times_sum_of_sizes(self):
        # fp8 DSA latent row = 656 B (576 latent + 80 scale bytes, the same
        # kv_cache_dim the pool allocates); slots = 16 // 1 + 1.
        kvc = self._kvc(
            torch.float8_e4m3fn,
            enable_hisparse=True,
            host_to_device_ratio=2,
            device_buffer_sizes=(256, 512),
            max_running_requests=16,
        )
        with get_parallel().override(attn_tp_size=1):
            configurator = object.__new__(DefaultPoolConfigurator)
            got = configurator._hisparse_buffer_bytes_for(kvc, num_layers=2)
        self.assertEqual(got, 17 * (256 + 512) * 656)
        # Equal slot totals -> equal reserved bytes regardless of the split.
        kvc_u = self._kvc(
            torch.float8_e4m3fn,
            enable_hisparse=True,
            host_to_device_ratio=2,
            device_buffer_sizes=(384, 384),
            max_running_requests=16,
        )
        with get_parallel().override(attn_tp_size=1):
            self.assertEqual(
                configurator._hisparse_buffer_bytes_for(kvc_u, num_layers=2), got
            )

    def test_per_layer_buffers_require_explicit_max_running_requests(self):
        kvc = self._kvc(
            torch.float8_e4m3fn,
            enable_hisparse=True,
            host_to_device_ratio=2,
            device_buffer_sizes=(256, 512),
            max_running_requests=None,
        )
        configurator = object.__new__(DefaultPoolConfigurator)
        with self.assertRaises(ValueError):
            configurator._hisparse_buffer_bytes_for(kvc, num_layers=2)

    def test_no_reservation_without_hisparse(self):
        kvc = self._kvc(torch.float8_e4m3fn, enable_hisparse=False)
        configurator = object.__new__(DefaultPoolConfigurator)
        self.assertEqual(configurator._hisparse_buffer_bytes_for(kvc, num_layers=2), 0)


if __name__ == "__main__":
    unittest.main()
