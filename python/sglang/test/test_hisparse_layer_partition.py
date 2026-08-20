"""Unit tests for the HiSparse per-layer DP partition helpers.

Pure-python (no torch/CUDA): these cover the profile resolution and size
computation that gate what the swap-in kernel receives per layer.
"""

import json

import pytest

from sglang.srt.mem_cache.sparsity.layer_partition import (
    DSV4_FLASH_SWE_PROFILE,
    compute_layer_buffer_sizes,
    load_layer_profile,
    map_profile_to_layers,
    resolve_layer_buffer_sizes,
)


def test_load_named_profile():
    profile = load_layer_profile("dsv4_flash_swe")
    assert profile == DSV4_FLASH_SWE_PROFILE
    assert min(profile.values()) > 0


def test_load_json_string_and_mapping():
    spec = {"2": 1.0, "10": 2.0}
    assert load_layer_profile(spec) == {2: 1.0, 10: 2.0}
    assert load_layer_profile(json.dumps(spec)) == {2: 1.0, 10: 2.0}


def test_load_rejects_bad_profiles():
    with pytest.raises(ValueError):
        load_layer_profile({})
    with pytest.raises(ValueError):
        load_layer_profile({2: 0.0})


def test_map_exact_layer_count_uses_depth_order():
    profile = {4: 2.0, 2: 1.0, 40: 3.0}
    assert map_profile_to_layers(profile, 3) == [1.0, 2.0, 3.0]


def test_map_by_nearest_relative_depth():
    # profile at depths 0.0 and 1.0; 4 layers -> first two shallow,
    # last two deep
    profile = {0: 1.0, 10: 3.0}
    weights = map_profile_to_layers(profile, 4)
    assert weights == [1.0, 1.0, 3.0, 3.0]


def test_sizes_scale_quantize_and_floor():
    sizes = compute_layer_buffer_sizes(
        [1.0, 0.5, 0.05], device_buffer_size=4096, floor=2048, quantum=256
    )
    # max-weight layer gets the physical buffer; mid quantized down;
    # tiny weight floored at top-k
    assert sizes == [4096, 2048, 2048]

    sizes = compute_layer_buffer_sizes(
        [1.0, 0.9], device_buffer_size=4096, floor=1024, quantum=256
    )
    assert sizes[0] == 4096
    assert sizes[1] == (int(4096 * 0.9) // 256) * 256 == 3584


def test_sizes_never_exceed_physical_or_violate_floor():
    sizes = compute_layer_buffer_sizes(
        [3.0, 2.0, 1.0, 0.1], device_buffer_size=4096, floor=2048
    )
    assert all(2048 <= b <= 4096 for b in sizes)
    assert max(sizes) == 4096  # largest layer pinned to physical size


def test_floor_larger_than_buffer_rejected():
    with pytest.raises(ValueError):
        compute_layer_buffer_sizes([1.0], device_buffer_size=1024, floor=2048)


def test_resolve_none_passthrough():
    assert (
        resolve_layer_buffer_sizes(None, 12, 4096, floor=2048) is None
    )


def test_resolve_dsv4_flash_end_to_end():
    sizes = resolve_layer_buffer_sizes(
        "dsv4_flash_swe", 12, device_buffer_size=4096, floor=2048
    )
    assert len(sizes) == 12
    assert all(2048 <= b <= 4096 for b in sizes)
    # profile max (layer 40, 0.255) -> physical size; profile min
    # (layer 42, 0.095) -> floored
    assert max(sizes) == 4096
    assert min(sizes) == 2048
    # mapping preserved depth order: layer 42 (deepest) is smallest
    assert sizes[-1] == 2048


def test_resolve_extends_to_more_layers_by_depth():
    sizes = resolve_layer_buffer_sizes(
        "dsv4_flash_swe", 43, device_buffer_size=4096, floor=2048
    )
    assert len(sizes) == 43
    assert all(2048 <= b <= 4096 for b in sizes)
