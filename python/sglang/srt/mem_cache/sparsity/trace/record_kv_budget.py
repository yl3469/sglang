"""Stage 2: record free KV-cache memory / capacity from a running server.

The DP total budget is grounded in real hardware headroom rather than an
arbitrary ratio. This module queries a live sglang server and dumps the KV
budget inputs to ``kv_budget.json``:

* ``GET /server_info`` -> per-DP ``internal_states[i].memory_usage.token_capacity``
  (== that DP rank's ``max_total_num_tokens``), plus ``kvcache`` bytes and
  ``startup_available_gpu_memory_gb`` when present.
* ``GET /metrics`` (Prometheus text) -> gauges
  ``sglang:max_total_num_tokens``, ``sglang:startup_available_gpu_memory_gb``,
  ``sglang:kv_available_tokens`` (scraped best-effort).

Downstream, :func:`run_per_layer_dp.kv_budget_to_ratio` turns
``per_dp_token_capacity`` into a mean per-layer capacity ratio.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional

__all__ = [
    "parse_server_info",
    "parse_metrics_text",
    "record_kv_budget",
]


def _extract_internal_states(info: dict) -> List[dict]:
    """Return the list of per-DP internal_states from a /server_info payload.

    PD-disaggregated servers nest under ``decode``; otherwise the states are at
    the top level.
    """
    if "decode" in info and info["decode"]:
        info = info["decode"][0]
    states = info.get("internal_states")
    if isinstance(states, list):
        return states
    return []


def parse_server_info(info: dict) -> Dict[str, object]:
    """Pull KV-budget fields out of a /server_info JSON payload."""
    states = _extract_internal_states(info)
    token_caps: List[int] = []
    kvcache_bytes: List[float] = []
    startup_gpu_gb: List[float] = []
    max_total: List[int] = []
    for st in states:
        mem = st.get("memory_usage") or {}
        if isinstance(mem.get("token_capacity"), (int, float)):
            token_caps.append(int(mem["token_capacity"]))
        if isinstance(mem.get("kvcache"), (int, float)):
            kvcache_bytes.append(float(mem["kvcache"]))
        if isinstance(st.get("max_total_num_tokens"), (int, float)):
            max_total.append(int(st["max_total_num_tokens"]))
        if isinstance(st.get("startup_available_gpu_memory_gb"), (int, float)):
            startup_gpu_gb.append(float(st["startup_available_gpu_memory_gb"]))

    out: Dict[str, object] = {}
    if token_caps:
        out["per_dp_token_capacity"] = token_caps
        # The min across DP ranks is the safe shared budget.
        out["token_capacity"] = min(token_caps)
    if max_total:
        out["max_total_num_tokens"] = min(max_total)
    if kvcache_bytes:
        out["kvcache_gb_per_dp"] = kvcache_bytes
    if startup_gpu_gb:
        out["startup_available_gpu_memory_gb"] = min(startup_gpu_gb)
    out["num_dp"] = len(states)
    return out


_METRIC_KEYS = (
    "sglang:max_total_num_tokens",
    "sglang:startup_available_gpu_memory_gb",
    "sglang:kv_available_tokens",
    "sglang:available_gpu_memory_gb",
)


def parse_metrics_text(text: str) -> Dict[str, float]:
    """Scrape the KV-related gauges from Prometheus /metrics text.

    Lines look like ``sglang:max_total_num_tokens{label="v"} 12345.0``. When a
    gauge is reported per-DP (multiple label sets), the MIN is kept as the safe
    shared value.
    """
    out: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for key in _METRIC_KEYS:
            if line.startswith(key):
                m = re.search(r"\s([0-9eE.+-]+)\s*$", line)
                if not m:
                    continue
                try:
                    val = float(m.group(1))
                except ValueError:
                    continue
                short = key.split(":", 1)[1]
                if short in out:
                    out[short] = min(out[short], val)
                else:
                    out[short] = val
    return out


def record_kv_budget(
    base_url: str,
    out_path: str,
    timeout: float = 10.0,
) -> Dict[str, object]:
    """Query /server_info + /metrics and write kv_budget.json. Returns the dict."""
    import requests

    budget: Dict[str, object] = {"base_url": base_url}

    headers = {}
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
    if api_key:
        headers["Authorization"] = (
            f"Bearer {api_key}" if os.environ.get("OPENAI_API_KEY") else api_key
        )

    try:
        r = requests.get(base_url + "/server_info", headers=headers, timeout=timeout)
        r.raise_for_status()
        budget.update(parse_server_info(r.json()))
    except Exception as exc:
        budget["server_info_error"] = str(exc)

    try:
        r = requests.get(base_url + "/metrics", headers=headers, timeout=timeout)
        if r.status_code == 200:
            budget["metrics"] = parse_metrics_text(r.text)
    except Exception as exc:
        budget["metrics_error"] = str(exc)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(budget, f, indent=2)
    return budget


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--out", default="kv_budget.json")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)

    budget = record_kv_budget(args.base_url, args.out, timeout=args.timeout)
    print(f"Wrote {args.out}")
    cap = budget.get("token_capacity")
    if cap is not None:
        print(
            f"  per-DP token_capacity: {budget.get('per_dp_token_capacity')} "
            f"(min {cap}), num_dp={budget.get('num_dp')}"
        )
    else:
        print(f"  WARNING: no token_capacity captured; budget={budget}")


if __name__ == "__main__":
    main()
