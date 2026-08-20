"""GPU tests for the HiSparse swap-in kernel's FAST_LEN/NEWEST_SLOT split.

Two guarantees:
1. Baseline equivalence: with the default parameters (fast_len ==
   newest_slot == hot_buffer_size) the kernel output and every piece of
   mutated state are bitwise identical to an explicit-parameter call —
   i.e. the uniform configuration is unchanged by the template split.
2. Per-layer variant: a smaller hot_buffer_size with fast_len below it
   and newest_slot at the physical (padded) position compiles, runs, and
   returns valid device locations for every selected token.

Requires CUDA; skipped otherwise. Synthetic tensors only — no model.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from sglang.kernels.ops.kvcache.hisparse import load_cache_to_device_buffer_mla

ITEM = 64  # bytes per token record
MAX_REQS = 4
HOST_PER_REQ = 4096


def _make_state(hot: int, padded: int, seq_lens, top_k: int, seed: int = 0):
    """Consistent synthetic swap state for `len(seq_lens)` requests."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    device = "cuda"
    num_reqs = len(seq_lens)

    host_cache = torch.randint(
        0, 256, (MAX_REQS * HOST_PER_REQ, ITEM), dtype=torch.uint8,
        generator=g,
    ).to(device)
    device_buffer = torch.zeros(
        (MAX_REQS * padded, ITEM), dtype=torch.uint8, device=device
    )

    host_cache_locs = torch.full(
        (MAX_REQS, HOST_PER_REQ), -1, dtype=torch.int64, device=device
    )
    device_buffer_locs = torch.full(
        (MAX_REQS, padded), -1, dtype=torch.int32, device=device
    )
    device_buffer_tokens = torch.full(
        (MAX_REQS, padded), -1, dtype=torch.int32, device=device
    )
    lru_slots = (
        torch.arange(hot, dtype=torch.int16, device=device)
        .view(1, -1)
        .repeat(MAX_REQS, 1)
        .contiguous()
    )

    top_k_tokens = torch.full(
        (num_reqs, top_k), -1, dtype=torch.int32, device=device
    )
    for rid, seq in enumerate(seq_lens):
        host_cache_locs[rid, :seq] = (
            rid * HOST_PER_REQ + torch.arange(seq, device=device)
        )
        device_buffer_locs[rid] = rid * padded + torch.arange(
            padded, dtype=torch.int32, device=device
        )
        # tokens 0..hot-1 resident in slots 0..hot-1 (post-prefill claims)
        device_buffer_tokens[rid, :hot] = torch.arange(
            hot, dtype=torch.int32, device=device
        )
        # populate the resident device records so hits return real bytes
        device_buffer[rid * padded : rid * padded + min(hot, seq)] = host_cache[
            rid * HOST_PER_REQ : rid * HOST_PER_REQ + min(hot, seq)
        ]
        # selection: newest token, some resident hits, some misses
        picks = torch.randperm(seq - 1, generator=g)[: top_k - 1].to(
            torch.int32
        )
        top_k_tokens[rid, : top_k - 1] = picks.to(device)
        top_k_tokens[rid, top_k - 1] = seq - 1

    return {
        "top_k_tokens": top_k_tokens,
        "device_buffer_tokens": device_buffer_tokens,
        "host_cache_locs": host_cache_locs,
        "device_buffer_locs": device_buffer_locs,
        "host_cache": host_cache,
        "device_buffer": device_buffer,
        "top_k_device_locs": torch.full(
            (num_reqs, top_k), -1, dtype=torch.int32, device=device
        ),
        "req_pool_indices": torch.arange(
            num_reqs, dtype=torch.int64, device=device
        ),
        "seq_lens": torch.tensor(seq_lens, dtype=torch.int64, device=device),
        "lru_slots": lru_slots,
    }


def _clone(state):
    return {k: v.clone() for k, v in state.items()}


def _run(state, *, hot, top_k, **kwargs):
    load_cache_to_device_buffer_mla(
        **state,
        item_size_bytes=ITEM,
        num_top_k=top_k,
        hot_buffer_size=hot,
        page_size=1,
        block_size=256,
        **kwargs,
    )
    return state


def test_default_params_bitwise_identical():
    hot, top_k = 512, 256
    padded = hot + 1
    seq_lens = [300, hot + 40]  # one fast-path, one long-path request

    base = _run(
        _make_state(hot, padded, seq_lens, top_k), hot=hot, top_k=top_k
    )
    expl = _run(
        _make_state(hot, padded, seq_lens, top_k),
        hot=hot,
        top_k=top_k,
        fast_len=hot,
        newest_slot=hot,
    )
    for key in base:
        assert torch.equal(base[key], expl[key]), key


def test_per_layer_size_variant_runs_and_is_valid():
    physical = 512  # physical buffer of the max layer
    padded = physical + 1
    hot, top_k = 384, 256  # this layer's effective size
    fast_len = 320  # request-admission threshold (min over layers)
    seq_lens = [300, 450]  # one fast-path (<=320), one long-path request

    state = _make_state(hot, padded, seq_lens, top_k)
    # Long path binds the newest token at the physical newest slot. In the
    # real system the decode step writes the newest token's KV on-device;
    # stage both its claim and its bytes here.
    for rid, seq in enumerate(seq_lens):
        if seq > fast_len:
            state["device_buffer_tokens"][rid, physical] = seq - 1
            state["device_buffer"][rid * padded + physical] = state[
                "host_cache"
            ][rid * HOST_PER_REQ + seq - 1]
    _run(
        state,
        hot=hot,
        top_k=top_k,
        fast_len=fast_len,
        newest_slot=physical,
    )
    torch.cuda.synchronize()

    locs = state["top_k_device_locs"]
    tokens = state["top_k_tokens"]
    assert torch.all(locs[tokens >= 0] >= 0), "valid tokens must resolve"
    for rid in range(len(seq_lens)):
        row = locs[rid][tokens[rid] >= 0]
        assert torch.all(row // padded == rid), "locs stay in request range"
    # long-path request: every miss got copied from host — resolved bytes
    # must match the host bytes of the selected tokens
    rid = 1
    for j in range(top_k):
        tok = int(tokens[rid, j])
        loc = int(locs[rid, j])
        host_loc = rid * HOST_PER_REQ + tok
        assert torch.equal(
            state["device_buffer"][loc], state["host_cache"][host_loc]
        ), f"token {tok} bytes mismatch at loc {loc}"
