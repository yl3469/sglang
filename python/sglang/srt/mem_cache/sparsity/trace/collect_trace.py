"""Real DSv4-Flash inference -> HiSparse indexer-topk trace -> per-layer DP
capacity allocation, entirely via launch_server + the serving benchmark.

Pipeline (all host-side except the served model itself):
  0. prep SWE-bench -> custom JSONL (skipped if the file already exists;
     needs HF network, so pre-generate on a login node when offline)
  1. launch_server (--enable-hisparse --enable-return-indexer-topk)
  2. record_kv_budget -> kv_budget.json (DP total-budget input)
  3. smoke 1 request (confirm indexer_topk is non-empty)
  4. serving benchmark over K prompts, --indexer-trace-out -> req_*.npz
  5. replay -> curves -> run_per_layer_dp -> dp_allocations.csv

    python -m sglang.srt.mem_cache.sparsity.trace.collect_trace [--flags]
"""

import argparse
from pathlib import Path

from . import exp_common as ec


def collect(
    args: argparse.Namespace,
    dataset_jsonl: str,
    results_dir: Path,
    dp_extra_args: list,
) -> None:
    """Launch + smoke + trace collection + offline DP allocation."""
    trace_dir = results_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    server = ec.Server(
        model_path=args.model_path,
        port=args.port,
        tp=args.tp,
        dp=args.dp,
        max_total_tokens=args.max_total_tokens,
        max_running_requests=args.max_running_requests,
        log_path=results_dir / "server.log",
        extra_flags=["--enable-hisparse", "--enable-return-indexer-topk"],
    )
    server.launch()
    try:
        if not server.wait_ready():
            raise SystemExit(f"server not ready; see {server.log_path}")
        ec.log("server ready")

        ec.log("Stage 2: recording KV budget")
        ec.run_module(
            f"{ec.TRACE_PKG}.record_kv_budget",
            "--base-url",
            server.base_url,
            "--out",
            str(results_dir / "kv_budget.json"),
        )

        trace_args = [
            "--extra-request-body",
            '{"return_indexer_topk": true}',
            "--indexer-trace-out",
            str(trace_dir),
            "--indexer-num-layers",
            str(args.num_indexer_layers),
            "--indexer-topk",
            str(args.index_topk),
        ]

        ec.log("smoke: 1 request, confirm indexer_topk present")
        rc = ec.run_bench(
            server.base_url,
            dataset_jsonl,
            1,
            1,
            output_len=64,
            warmup_requests=0,
            extra_args=trace_args,
        )
        if rc != 0 or not list(trace_dir.glob("req_*.npz")):
            raise SystemExit(
                "no indexer trace captured on smoke; "
                "check --enable-return-indexer-topk"
            )
        ec.log(f"smoke OK: {len(list(trace_dir.glob('req_*.npz')))} trace(s)")

        ec.log(f"Stage 1: collecting traces over {args.num_prompts} prompts")
        ec.run_bench(
            server.base_url,
            dataset_jsonl,
            args.num_prompts,
            1,
            output_file=results_dir / "bench_serving.jsonl",
            output_len=128,
            warmup_requests=0,
            extra_args=trace_args,
        )
    finally:
        server.teardown()

    ec.log("Stage 3+4: per-layer DP allocation")
    rc = ec.run_module(
        f"{ec.TRACE_PKG}.run_per_layer_dp",
        str(trace_dir),
        "--policy",
        "lru",
        "--kv-budget",
        str(results_dir / "kv_budget.json"),
        *dp_extra_args,
        "--out-dir",
        str(results_dir / "per_layer_budget_dp"),
    )
    if rc != 0:
        print(f"DP stage returned nonzero (traces still under {trace_dir})")
    ec.log(f"DONE; artifacts in {results_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    ec.add_common_args(p)
    p.set_defaults(
        num_prompts=ec.env("NUM_PROMPTS", 16),
        # --enable-return-indexer-topk allocates a PINNED host buffer sized
        # (max_total_num_tokens+page) x layers x topk x 4B; keep the pool modest.
        max_running_requests=ec.env("MAX_RUNNING_REQUESTS", 8),
    )
    p.add_argument("--port", type=int, default=ec.env("PORT", 30000))
    p.add_argument("--index-topk", type=int, default=ec.env("INDEX_TOPK", 512))
    # DSv4-Flash: 43 layers, compress_ratio 4 -> C4A indexer layers on even ids 2..42
    p.add_argument(
        "--num-indexer-layers", type=int, default=ec.env("NUM_INDEXER_LAYERS", 21)
    )
    p.add_argument(
        "--results-dir",
        default=ec.env("RESULTS_DIR", str(ec.RESULTS_ROOT / ec.job_tag())),
    )
    args = p.parse_args()

    ec.setup_environment(args.hf_cache)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    swe_jsonl = ec.env("SWE_JSONL", str(results_dir / "swe_bench.jsonl"))
    if not Path(swe_jsonl).is_file() or Path(swe_jsonl).stat().st_size == 0:
        ec.log("Stage 0: preparing SWE-bench prompts")
        import os

        os.environ["HF_HUB_OFFLINE"] = "0"
        rc = ec.run_module(
            f"{ec.TRACE_PKG}.prep_swe_bench",
            "--num-prompts",
            str(args.num_prompts),
            "--min-tokens",
            "20000",
            "--max-tokens",
            "120000",
            "--tokenizer-path",
            args.model_path,
            "--out",
            swe_jsonl,
        )
        os.environ["HF_HUB_OFFLINE"] = "1"
        if rc != 0:
            raise SystemExit(
                f"SWE-bench prep failed; provide {swe_jsonl} manually and rerun"
            )

    collect(
        args,
        swe_jsonl,
        results_dir,
        # DSv4-Flash C4A indexer layers sit on even model ids 2..42.
        dp_extra_args=[
            "--compress-ratio",
            "4",
            "--layer-offset",
            "2",
            "--layer-stride",
            "2",
        ],
    )


if __name__ == "__main__":
    main()
