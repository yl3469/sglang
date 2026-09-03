"""A/B serving benchmark: HiSparse UNIFORM per-layer buffer (baseline) vs the
DP per-layer allocation (treatment), at MATCHED total device memory.

For each arm, launch the HiSparse server with that per-layer buffer config and
sweep concurrency with an identical fixed output length, then aggregate all
JSONL summaries into one summary.csv. Run directly on a GPU node:

    python -m sglang.srt.mem_cache.sparsity.trace.ab_sweep [--flags]

All knobs are also overridable via the same env vars the old sbatch wrapper
used (TP, DP, TOP_K, CONCURRENCY_LIST, NUM_PROMPTS, OUTPUT_LEN, DP_CSV, ...).
"""

import argparse
from pathlib import Path

from . import exp_common as ec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    ec.add_common_args(p)
    p.add_argument("--top-k", type=int, default=ec.env("TOP_K", 512))
    p.add_argument("--port-uniform", type=int, default=ec.env("PORT_UNIFORM", 31000))
    p.add_argument("--port-dp", type=int, default=ec.env("PORT_DP", 33000))
    p.add_argument(
        "--concurrency-list", default=ec.env("CONCURRENCY_LIST", "5 20 50 100 200")
    )
    p.add_argument(
        "--dp-csv",
        default=ec.env(
            "DP_CSV",
            str(ec.RESULTS_ROOT / "3191572/per_layer_budget_dp/dp_allocations.csv"),
        ),
    )
    p.add_argument("--dp-total-row", type=float, default=ec.env("DP_TOTAL_ROW", 0.3))
    p.add_argument("--b-mean", type=int, default=ec.env("B_MEAN", 1024))
    p.add_argument("--page-size", type=int, default=ec.env("PAGE_SIZE_SLOTS", 64))
    p.add_argument(
        "--out-dir",
        default=ec.env("OUT_DIR", str(ec.RESULTS_ROOT / f"ab_{ec.job_tag()}")),
    )
    args = p.parse_args()

    ec.setup_environment(args.hf_cache)
    ec.require_file(args.swe_jsonl, "dataset")
    ec.require_file(args.dp_csv, "dp csv")

    out_dir = Path(args.out_dir)
    cfgs = ec.build_buffer_configs(
        args.dp_csv,
        out_dir / "buffer_configs",
        args.dp_total_row,
        args.b_mean,
        args.top_k,
        args.page_size,
    )
    conc = ec.parse_concurrency(args.concurrency_list)

    # Baseline first (uniform), then treatment (DP). Distinct base ports so
    # DP-attention rpc_port ranges never collide across arms.
    for arm, port in (("uniform", args.port_uniform), ("dp", args.port_dp)):
        ec.sweep_arm(
            arm,
            out_dir,
            port,
            args,
            conc,
            server_flags=[
                "--enable-hisparse",
                "--hisparse-config",
                ec.hisparse_config(args.top_k, str(cfgs[arm])),
            ],
            output_len=args.output_len,
        )

    ec.log(f"aggregating results -> {out_dir}/summary.csv")
    ec.run_module(
        f"{ec.TRACE_PKG}.aggregate_ab",
        "--results-dir",
        str(out_dir),
        "--out",
        str(out_dir / "summary.csv"),
    )
    ec.log("DONE")


if __name__ == "__main__":
    main()
