"""High-concurrency point across ALL FOUR arms in one run, so they share a node
and are maximally comparable. HiSparse's host offload buys memory headroom;
that headroom should pay off under load, not at c=1.

    dense   = no HiSparse (native compressed+C4Indexer, full KV on GPU)
    uniform = HiSparse + uniform per-layer buffer
    dp      = HiSparse + DP per-layer buffer (same total memory)
    dp_mtp  = HiSparse + DP per-layer buffer + EAGLE/NEXTN MTP

    python -m sglang.srt.mem_cache.sparsity.trace.highconc_sweep [--flags]
"""

import argparse
from pathlib import Path

from . import exp_common as ec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    ec.add_common_args(p)
    p.set_defaults(num_prompts=ec.env("NUM_PROMPTS", 300))
    p.add_argument("--top-k", type=int, default=ec.env("TOP_K", 512))
    p.add_argument("--port-dense", type=int, default=ec.env("PORT_DENSE", 31000))
    p.add_argument("--port-uniform", type=int, default=ec.env("PORT_UNIFORM", 33000))
    p.add_argument("--port-dp", type=int, default=ec.env("PORT_DP", 35000))
    p.add_argument("--port-mtp", type=int, default=ec.env("PORT_MTP", 37000))
    p.add_argument("--concurrency-list", default=ec.env("CONCURRENCY_LIST", "128"))
    p.add_argument(
        "--arms",
        default=ec.env("ARMS", "dense uniform dp dp_mtp"),
        help="Space-separated subset of: dense uniform dp dp_mtp",
    )
    p.add_argument("--spec-steps", type=int, default=ec.env("SPEC_STEPS", 3))
    p.add_argument("--spec-topk", type=int, default=ec.env("SPEC_TOPK", 1))
    p.add_argument(
        "--spec-draft-tokens", type=int, default=ec.env("SPEC_DRAFT_TOKENS", 4)
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
        default=ec.env("OUT_DIR", str(ec.RESULTS_ROOT / f"hc_{ec.job_tag()}")),
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
    hs_uniform = [
        "--enable-hisparse",
        "--hisparse-config",
        ec.hisparse_config(args.top_k, str(cfgs["uniform"])),
    ]
    hs_dp = [
        "--enable-hisparse",
        "--hisparse-config",
        ec.hisparse_config(args.top_k, str(cfgs["dp"])),
    ]
    mtp = [
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        str(args.spec_steps),
        "--speculative-eagle-topk",
        str(args.spec_topk),
        "--speculative-num-draft-tokens",
        str(args.spec_draft_tokens),
    ]
    arm_specs = {
        "dense": (args.port_dense, []),
        "uniform": (args.port_uniform, hs_uniform),
        "dp": (args.port_dp, hs_dp),
        "dp_mtp": (args.port_mtp, hs_dp + mtp),
    }

    for arm in args.arms.split():
        if arm not in arm_specs:
            print(f"unknown arm {arm!r}, skipping", flush=True)
            continue
        port, flags = arm_specs[arm]
        ec.sweep_arm(
            arm,
            out_dir,
            port,
            args,
            conc,
            server_flags=flags,
            output_len=args.output_len,
        )
    ec.log("DONE")


if __name__ == "__main__":
    main()
