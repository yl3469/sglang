"""Geometry of physically per-layer HiSparse device buffers.

A HiSparse KV pool has, per layer, ``context_rows + page_size`` SHARED rows (the
stock pool: context tokens plus the reserved newest-token page; one row index
means the same row in every layer) followed by a PRIVATE region of
``num_slots * B_l`` rows that only layer ``l`` uses, where ``B_l`` is that
layer's device buffer capacity. Request slot ``s`` owns rows
``[region_start + s * B_l, region_start + (s + 1) * B_l)`` of layer ``l``.

Per request and layer, a buffer is *resident* while ``seq_len <= B_l`` (slots
point at the request's shared rows, exactly the pre-existing behavior the
swap-in kernel's fast path relies on) and switches to *LRU* mode once
``seq_len > B_l`` (slots point at the private rows). This struct only holds
the geometry; the coordinator drives the mode transitions.
"""

from __future__ import annotations

import bisect
from typing import Tuple

import msgspec

# LRU slot indices (host bookkeeping and the swap-in kernel) are int16 and
# address slots 0..B_l-1, so a per-layer buffer holds at most this many slots.
MAX_BUFFER_SLOTS = 32768


class HiSparseBufferLayout(msgspec.Struct, frozen=True, kw_only=True):
    context_rows: int
    page_size: int
    layer_num: int
    buffer_sizes: Tuple[int, ...]
    num_slots: int
    # Derived at build() time; kept as fields so the struct stays frozen/pure.
    max_buffer_size: int
    shared_width: int
    padded_buffer_size: int
    region_start: int
    layers_by_size: Tuple[int, ...]
    sorted_sizes: Tuple[int, ...]

    @classmethod
    def build(
        cls,
        *,
        context_rows: int,
        page_size: int,
        layer_num: int,
        buffer_sizes: Tuple[int, ...],
        num_slots: int,
    ) -> HiSparseBufferLayout:
        sizes = tuple(int(b) for b in buffer_sizes)
        if len(sizes) != layer_num:
            raise ValueError(
                f"buffer_sizes has {len(sizes)} entries but layer_num is {layer_num}"
            )
        if context_rows < 1 or page_size < 1 or num_slots < 1:
            raise ValueError(
                f"context_rows ({context_rows}), page_size ({page_size}) and "
                f"num_slots ({num_slots}) must be >= 1"
            )
        for b in sizes:
            if b < 1 or b > MAX_BUFFER_SLOTS:
                raise ValueError(
                    f"per-layer buffer size {b} outside [1, {MAX_BUFFER_SLOTS}] "
                    "(int16 LRU slot indices)"
                )
        max_buffer_size = max(sizes)
        # The shared (resident-mode) width is the page-aligned max capacity;
        # this is the alignment the pre-existing single-width code assumed.
        shared_width = ((max_buffer_size + page_size - 1) // page_size) * page_size
        order = sorted(range(layer_num), key=lambda l: (sizes[l], l))
        return cls(
            context_rows=context_rows,
            page_size=page_size,
            layer_num=layer_num,
            buffer_sizes=sizes,
            num_slots=num_slots,
            max_buffer_size=max_buffer_size,
            shared_width=shared_width,
            padded_buffer_size=shared_width + page_size,
            region_start=context_rows + page_size,
            layers_by_size=tuple(order),
            sorted_sizes=tuple(sizes[l] for l in order),
        )

    def layer_rows(self, layer_id: int) -> int:
        """Total rows of layer ``layer_id``'s KV tensor (shared + private)."""
        return self.region_start + self.num_slots * self.buffer_sizes[layer_id]

    def slot_row_start(self, layer_id: int, slot: int) -> int:
        """First private row of request slot ``slot`` in layer ``layer_id``."""
        if not 0 <= slot < self.num_slots:
            raise ValueError(f"slot {slot} outside [0, {self.num_slots})")
        return self.region_start + slot * self.buffer_sizes[layer_id]

    def buffer_rows_total(self) -> int:
        """Private rows summed over layers and slots (the DP's slot total x K)."""
        return self.num_slots * sum(self.buffer_sizes)

    def buffer_bytes(self, bytes_per_row: int) -> int:
        return self.buffer_rows_total() * bytes_per_row

    def lru_layers_for(self, seq_len: int) -> int:
        """Number of layers in LRU mode at ``seq_len``: those with ``B_l < seq_len``.

        Layers are ordered by ``layers_by_size``, so the first ``n`` entries of
        that tuple are exactly the LRU-mode layers.
        """
        return bisect.bisect_left(self.sorted_sizes, seq_len)
