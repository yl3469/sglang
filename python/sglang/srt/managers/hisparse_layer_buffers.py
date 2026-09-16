"""Per-request, per-layer device-buffer mode tracking for HiSparse.

Each (request, layer) buffer is either *resident* (``seq_len <= B_l``: slots
point at the request's SHARED pool rows, the pre-existing behavior the swap-in
kernel's fast path relies on) or in *LRU* mode (``seq_len > B_l``: slots point
at layer ``l``'s PRIVATE region rows owned by this request slot). A layer
switches exactly once per request, with a device-to-device warm copy of tokens
``[0, B_l)`` so hit/miss behavior matches the shared-row implementation.

This collaborator owns only the mode bookkeeping and the writes into the
coordinator's per-layer tensors; the coordinator decides *when* to call it and
handles stream ordering and shared-row release.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from sglang.srt.mem_cache.hisparse_buffer_layout import HiSparseBufferLayout
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool


class HiSparseLayerBuffers:
    def __init__(
        self,
        *,
        layout: HiSparseBufferLayout,
        mem_pool_device: HiSparseDSATokenToKVPool,
        device: str,
        token_locs: torch.Tensor,
        tokens: torch.Tensor,
        lru_slots: torch.Tensor,
    ):
        if token_locs.shape[1] > layout.num_slots:
            raise ValueError(
                f"coordinator has {token_locs.shape[1]} request slots but the "
                f"buffer layout was built for {layout.num_slots}"
            )
        self.layout = layout
        self._pool = mem_pool_device
        self._token_locs = token_locs  # [layer_num, req_slots, padded]
        self._tokens = tokens  # [layer_num, req_slots, padded]
        self._lru_slots = lru_slots  # [layer_num, req_slots, shared_width] int16
        num_req_slots = token_locs.shape[1]
        # Private rows of every (layer, request slot): [num_slots, B_l] int32.
        self._slot_rows: List[torch.Tensor] = [
            torch.arange(
                layout.region_start,
                layout.region_start + layout.num_slots * b,
                dtype=torch.int32,
                device=device,
            ).view(layout.num_slots, b)
            for b in layout.buffer_sizes
        ]
        self._arange: Dict[int, torch.Tensor] = {
            b: torch.arange(b, dtype=torch.int32, device=device)
            for b in set(layout.buffer_sizes)
        }
        self._lru_init: Dict[int, torch.Tensor] = {
            b: torch.arange(b, dtype=torch.int16, device=device)
            for b in set(layout.buffer_sizes)
        }
        # Number of layers (in layout.layers_by_size order) in LRU mode per request.
        self._lru_layer_count: List[int] = [0] * num_req_slots
        self._shared_released: List[bool] = [False] * num_req_slots

    # ---- queries -------------------------------------------------------
    def lru_layer_count(self, req_idx: int) -> int:
        return self._lru_layer_count[req_idx]

    def is_lru(self, layer_id: int, req_idx: int) -> bool:
        n = self._lru_layer_count[req_idx]
        return layer_id in self.layout.layers_by_size[:n]

    def private_rows(self, layer_id: int, req_idx: int) -> torch.Tensor:
        return self._slot_rows[layer_id][req_idx]

    def needs_transition(self, req_idx: int, seq_len: int) -> bool:
        return self.layout.lru_layers_for(seq_len) > self._lru_layer_count[req_idx]

    def shared_released(self, req_idx: int) -> bool:
        return self._shared_released[req_idx]

    def should_release_shared(self, req_idx: int, seq_len: int) -> bool:
        """All layers are LRU and the last resident-slot token has been backed
        up (position shared_width-1 is written at seq_len == shared_width and
        backed up one step later), so the shared rows can go back to the pool."""
        return (
            not self._shared_released[req_idx]
            and seq_len >= self.layout.shared_width + 2
        )

    # ---- transitions ---------------------------------------------------
    def admit(
        self, req_idx: int, seq_len: int, shared_rows: torch.Tensor, *, warm: bool
    ) -> Tuple[int, ...]:
        """Enter LRU mode for every layer with ``B_l < seq_len`` at admission.

        ``shared_rows`` are the request's shared slot rows (token t at index t);
        with ``warm`` the first B_l of them are copied into the private rows.
        Returns the layer ids that entered LRU mode.
        """
        self._lru_layer_count[req_idx] = 0
        self._shared_released[req_idx] = False
        return self.advance(req_idx, seq_len, shared_rows, warm=warm)

    def advance(
        self,
        req_idx: int,
        seq_len: int,
        shared_rows: torch.Tensor,
        *,
        warm: bool = True,
    ) -> Tuple[int, ...]:
        """Move newly-overflowed layers (``B_l < seq_len``) into LRU mode."""
        target = self.layout.lru_layers_for(seq_len)
        current = self._lru_layer_count[req_idx]
        if target <= current:
            return ()
        layer_ids = self.layout.layers_by_size[current:target]
        for layer_id in layer_ids:
            self._enter_lru(layer_id, req_idx, shared_rows, warm=warm)
        self._lru_layer_count[req_idx] = target
        return layer_ids

    def mark_released(self, req_idx: int) -> None:
        self._shared_released[req_idx] = True

    def reset(self, req_idx: int) -> None:
        self._lru_layer_count[req_idx] = 0
        self._shared_released[req_idx] = False

    def _enter_lru(
        self, layer_id: int, req_idx: int, shared_rows: torch.Tensor, *, warm: bool
    ) -> None:
        b = self.layout.buffer_sizes[layer_id]
        rows = self._slot_rows[layer_id][req_idx]
        self._token_locs[layer_id, req_idx, :b] = rows
        if warm and shared_rows.numel() >= b:
            # Tokens [0, B_l) live at shared slots [0, B_l) while resident.
            self._pool.copy_rows_per_layer(
                layer_id, rows.to(torch.int64), shared_rows[:b].to(torch.int64)
            )
            self._tokens[layer_id, req_idx, :b] = self._arange[b]
        else:
            self._tokens[layer_id, req_idx, :b] = -1
        self._lru_slots[layer_id, req_idx, :b] = self._lru_init[b]
