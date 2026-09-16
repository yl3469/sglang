# mapping on device memory, host memory and memory allocator

import logging
from typing import Optional

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.hisparse_buffer_layout import HiSparseBufferLayout
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.utils import is_cuda, is_hip

logger = logging.getLogger(__name__)

# sgl_kernel.kvcacheio is only available in CUDA/ROCm sgl-kernel builds (not XPU/MPS/NPU/CPU).
_is_cuda = is_cuda()
_is_hip = is_hip()
if _is_cuda or _is_hip:
    from sgl_kernel.kvcacheio import transfer_kv_all_layer_mla
else:

    def transfer_kv_all_layer_mla(*args, **kwargs):
        raise RuntimeError(
            "HiSparse device KV transfer requires sgl_kernel.kvcacheio (CUDA/ROCm). "
            "It is not available on this backend."
        )


class HiSparseDSATokenToKVPool(DSATokenToKVPool):
    def __init__(
        self,
        size: int,
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        kv_cache_dim: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        host_to_device_ratio: int = 2,
        buffer_layout: Optional[HiSparseBufferLayout] = None,
    ):
        # Per-layer private device-buffer regions appended to every layer's KV
        # tensor (see HiSparseBufferLayout). None keeps the stock uniform shape.
        # Set before super().__init__: the base constructor calls
        # _create_buffers(), which reads it.
        if buffer_layout is not None:
            if (
                buffer_layout.context_rows != size
                or buffer_layout.page_size != page_size
            ):
                raise ValueError(
                    f"buffer_layout (context_rows={buffer_layout.context_rows}, "
                    f"page_size={buffer_layout.page_size}) does not match the pool "
                    f"(size={size}, page_size={page_size})"
                )
            if buffer_layout.layer_num != layer_num:
                raise ValueError(
                    f"buffer_layout has {buffer_layout.layer_num} layers, pool has {layer_num}"
                )
        self.buffer_layout = buffer_layout
        super().__init__(
            size=size,
            page_size=page_size,
            kv_lora_rank=kv_lora_rank,
            dtype=dtype,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=layer_num,
            device=device,
            index_head_dim=index_head_dim,
            enable_memory_saver=enable_memory_saver,
            kv_cache_dim=kv_cache_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            index_buf_size=size * host_to_device_ratio,
        )
        self.bytes_per_token = self.kv_cache_dim * self.dtype.itemsize

    def _create_buffers(self):
        if self.buffer_layout is None:
            super()._create_buffers()
            return
        from contextlib import nullcontext

        from sglang.srt.mem_cache.memory_pool import GPU_MEMORY_TYPE_KV_CACHE

        # Same allocation context as MLATokenToKVPool._create_buffers, but layer
        # l gets (size + page_size) shared rows + num_slots * B_l private rows.
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self.kv_buffer = [
                    torch.zeros(
                        (self.buffer_layout.layer_rows(layer), 1, self.kv_cache_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for layer in range(self.layer_num)
                ]

    def copy_rows_per_layer(
        self, layer_id: int, dst_indices: torch.Tensor, src_indices: torch.Tensor
    ) -> None:
        """Device-to-device row copy within one layer's KV tensor (disjoint rows).

        Used once per request and layer when a buffer leaves resident mode, so
        plain indexing is fine (the JIT hicache kernel requires element sizes
        that are multiples of 128 B, which DSA fp8 rows are not).
        """
        buf = self.kv_buffer[layer_id - self.start_layer]
        buf[dst_indices] = buf[src_indices]

    def register_mapping(self, full_to_hisparse_device_index_mapping: torch.Tensor):
        self.full_to_hisparse_device_index_mapping = (
            full_to_hisparse_device_index_mapping
        )

    def translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def _translate_loc_to_hisparse_device(self, compressed_indices: torch.Tensor):
        return self.full_to_hisparse_device_index_mapping[compressed_indices]

    def translate_loc_from_full_to_hisparse_device(self, full_indices: torch.Tensor):
        return self._translate_loc_to_hisparse_device(full_indices)

    def translate_loc_from_full_to_compressed(self, full_indices: torch.Tensor):
        return full_indices

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_kv_buffer(layer, loc, cache_k, cache_v)

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        super().set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        loc = self.translate_loc_to_hisparse_device(loc)
        return super().get_mla_kv_buffer(layer, loc, dst_dtype)

    def transfer_values_on_device(self, dst_indices, src_indices):
        transfer_kv_all_layer_mla(
            src_layers=self.data_ptrs,
            dst_layers=self.data_ptrs,
            src_indices=src_indices,
            dst_indices=dst_indices,
            item_size=self.bytes_per_token,
            num_layers=self.layer_num,
        )

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseDevicePool does not support get_cpu_copy")

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError("HiSparseDevicePool does not support load_cpu_copy")
