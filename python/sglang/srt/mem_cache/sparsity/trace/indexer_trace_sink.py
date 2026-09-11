"""Stage 1 helper: persist per-request indexer topk traces from bench_serving.

``bench_serving`` already forwards ``--extra-request-body`` (so a request can
ask for ``return_indexer_topk=true``) and receives the response ``meta_info``,
but it does not persist the returned ``indexer_topk`` payload. This module is
the small, isolated sink that does so, called from the sglang backend response
path ONLY when ``--indexer-trace-out`` is set (otherwise a complete no-op).

Each captured request is written to ``<out_dir>/req_<idx>.npz`` holding the
reshaped int32 topk array ``(steps, num_indexer_layers, index_topk)`` plus the
request geometry needed by the offline replay
(:mod:`sglang.srt.mem_cache.sparsity.trace.trace_to_curves`). The base64 int32
decode mirrors ``state_capturer/indexer_topk.extract_indexer_topk_from_meta_info``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["IndexerTraceSink", "decode_indexer_topk"]


def decode_indexer_topk(indexer_topk_base64: str):
    """Decode a base64 int32 ``meta_info['indexer_topk']`` to a flat np array.

    Mirrors :func:`extract_indexer_topk_from_meta_info` without importing the
    server module (keeps this helper lightweight for the bench client).
    """
    import numpy as np
    import pybase64

    return np.frombuffer(
        pybase64.b64decode(indexer_topk_base64.encode("utf-8")), dtype=np.int32
    )


class IndexerTraceSink:
    """Collects ``meta_info['indexer_topk']`` payloads and writes ``.npz`` files.

    Thread-safety: bench_serving runs requests on a single asyncio event loop,
    so calls are serialized; no locking is required. The sink is deliberately
    tolerant — a missing/empty payload is skipped with a warning rather than
    aborting the benchmark.
    """

    def __init__(
        self,
        out_dir: str,
        num_indexer_layers: Optional[int] = None,
        index_topk: Optional[int] = None,
    ):
        self.out_dir = out_dir
        self.num_indexer_layers = num_indexer_layers
        self.index_topk = index_topk
        self._count = 0
        self._skipped = 0
        os.makedirs(out_dir, exist_ok=True)

    def capture(
        self,
        req_idx: int,
        meta_info: dict,
        prompt_len: int = 0,
        output_len: int = 0,
    ) -> bool:
        """Persist one request's topk trace. Returns True if written.

        The topk buffer is reshaped to ``(steps, num_indexer_layers,
        index_topk)`` when both dimensions are known (from server args or the
        payload); otherwise the flat int32 buffer is stored with the geometry
        fields so :mod:`trace_to_curves` can reshape it later.
        """
        import numpy as np

        payload = (meta_info or {}).get("indexer_topk")
        if not payload:
            self._skipped += 1
            return False
        try:
            flat = decode_indexer_topk(payload)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("failed to decode indexer_topk for req %d: %s", req_idx, exc)
            self._skipped += 1
            return False

        num_layers = self.num_indexer_layers
        index_topk = self.index_topk
        path = os.path.join(self.out_dir, f"req_{req_idx}.npz")

        arrays = {
            "prompt_len": np.int64(prompt_len),
            "output_len": np.int64(output_len),
        }
        # slot_ctx0 (compressed context slots) is used to size ratio buffers.
        # Best available proxy from the client is prompt_len; the offline stage
        # also falls back to max observed slot id when this is absent.
        if prompt_len:
            arrays["slot_ctx0"] = np.int64(prompt_len)

        if num_layers and index_topk and flat.size % (num_layers * index_topk) == 0:
            steps = flat.size // (num_layers * index_topk)
            topk = flat.reshape(steps, num_layers, index_topk).astype(np.int32)
            arrays["topk_indices"] = topk
            arrays["num_layers"] = np.int64(num_layers)
            arrays["index_topk"] = np.int64(index_topk)
        else:
            # Geometry unknown here; store flat + whatever we know for later.
            arrays["indexer_topk_flat"] = flat.astype(np.int32)
            if num_layers:
                arrays["num_layers"] = np.int64(num_layers)
            if index_topk:
                arrays["index_topk"] = np.int64(index_topk)

        np.savez(path, **arrays)
        self._count += 1
        return True

    def summary(self) -> str:
        return (
            f"IndexerTraceSink: wrote {self._count} traces, "
            f"skipped {self._skipped} (no indexer_topk) -> {self.out_dir}"
        )
