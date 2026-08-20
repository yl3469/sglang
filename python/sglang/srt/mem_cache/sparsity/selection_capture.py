"""Capped capture of HiSparse per-layer top-k selections + KV memory report.

Two small utilities used to solve the per-layer DP partition ONLINE from a
short profiling run of the served model (no offline trace infrastructure):

- :class:`SelectionCapture` records, per layer, the top-k unit indices the
  indexer selected at each decode step (the full logits are sequence-sized
  and are deliberately NOT captured — the selection stream is all the LRU
  miss-curve replay needs). A hard cap (``max_steps_per_layer``) bounds
  memory and disk; capture disables itself and flushes once every layer
  reaches the cap, or on explicit flush at server shutdown.
- :func:`write_memory_report` records how much device memory is left for
  KV cache on this rank's GPU after the pools are allocated, plus the
  HiSparse geometry, so the DP solver can pick a per-layer budget that
  actually fits.

Enable both through ``--hisparse-config``::

    --hisparse-config '{"selection_capture": {"path": "/x/cap", "max_steps_per_layer": 1024},
                        "memory_report_path": "/x/memreport"}'

The capture hook lives in ``HiSparseCoordinator.swap_in_selected_pages``,
which runs under CUDA graph capture in normal serving — so capture REQUIRES
``--disable-cuda-graph`` (enforced at init). Profiling runs are short and
eager; benchmark runs keep graphs on with capture off.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


class SelectionCapture:
    """Accumulates per-layer top-k selection streams with a hard cap."""

    def __init__(
        self,
        path: str,
        layer_num: int,
        top_k: int,
        device_buffer_size: int,
        max_steps_per_layer: int = 1024,
        rank: int = 0,
    ):
        self.path = f"{path}.rank{rank}.pt"
        self.layer_num = layer_num
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        self.max_steps_per_layer = max_steps_per_layer
        # per-layer lists of (req_pool_indices int32 [R], seq_lens int32 [R],
        # top_k int32 [R, top_k]) CPU tuples
        self._steps: List[list] = [[] for _ in range(layer_num)]
        self._flushed = False
        logger.info(
            "HiSparse selection capture active: path=%s cap=%d steps/layer",
            self.path,
            max_steps_per_layer,
        )

    @property
    def active(self) -> bool:
        return not self._flushed

    def record(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
    ) -> None:
        if self._flushed or torch.cuda.is_current_stream_capturing():
            return
        steps = self._steps[layer_id]
        if len(steps) >= self.max_steps_per_layer:
            if all(len(s) >= self.max_steps_per_layer for s in self._steps):
                self.flush()
            return
        steps.append(
            (
                req_pool_indices.to("cpu", torch.int32, non_blocking=False),
                seq_lens.to("cpu", torch.int32, non_blocking=False),
                top_k_result.to("cpu", torch.int32, non_blocking=False),
            )
        )

    def flush(self) -> None:
        if self._flushed:
            return
        self._flushed = True
        payload = {
            "layer_num": self.layer_num,
            "top_k": self.top_k,
            "device_buffer_size": self.device_buffer_size,
            "steps_per_layer": [len(s) for s in self._steps],
            "layers": [
                {
                    "req_pool_indices": [s[0] for s in steps],
                    "seq_lens": [s[1] for s in steps],
                    "top_k": [s[2] for s in steps],
                }
                for steps in self._steps
            ],
        }
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        torch.save(payload, self.path)
        self._steps = [[] for _ in range(self.layer_num)]
        logger.info(
            "HiSparse selection capture flushed to %s (steps/layer=%s)",
            self.path,
            payload["steps_per_layer"],
        )


def write_memory_report(
    path: str,
    rank: int,
    coordinator,
) -> None:
    """Record per-GPU free memory + HiSparse geometry after pool allocation.

    One JSON per TP rank (``<path>.rank<k>.json``); the DP solver reads all
    of them so the layer budget is sized to the tightest GPU.
    """
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    pool = coordinator.mem_pool_device
    report = {
        "rank": rank,
        "device": torch.cuda.get_device_name(),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "layer_num": int(pool.layer_num),
        "page_size": int(coordinator.page_size),
        "top_k": int(coordinator.top_k),
        "device_buffer_size": int(coordinator.device_buffer_size),
        "padded_buffer_size": int(coordinator.padded_buffer_size),
        "layer_buffer_sizes": list(coordinator.layer_buffer_sizes),
        "item_size_bytes": int(coordinator.item_size_bytes),
        "max_num_req_slots": int(
            coordinator.req_to_token_pool.req_to_token.shape[0]
        ),
        "host_pool_size": int(coordinator.mem_pool_host.size),
        # bytes the per-request device buffers consume at the CURRENT
        # physical size; scales linearly in device_buffer_size for sizing
        "device_buffer_bytes_total": int(
            pool.layer_num
            * coordinator.req_to_token_pool.req_to_token.shape[0]
            * coordinator.padded_buffer_size
            * coordinator.item_size_bytes
        ),
    }
    out = f"{path}.rank{rank}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info(
        "HiSparse memory report written to %s (free %.1f GiB / total %.1f GiB)",
        out,
        free_bytes / 2**30,
        total_bytes / 2**30,
    )


def maybe_create_selection_capture(
    config: Optional[dict],
    layer_num: int,
    top_k: int,
    device_buffer_size: int,
    rank: int,
) -> Optional[SelectionCapture]:
    """Rank-0-only factory from the ``selection_capture`` config dict."""
    if not config or rank != 0:
        return None
    return SelectionCapture(
        path=config["path"],
        layer_num=layer_num,
        top_k=top_k,
        device_buffer_size=device_buffer_size,
        max_steps_per_layer=int(config.get("max_steps_per_layer", 1024)),
        rank=rank,
    )
