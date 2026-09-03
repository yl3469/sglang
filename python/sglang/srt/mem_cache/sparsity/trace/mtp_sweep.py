"""HiSparse alone vs HiSparse + EAGLE/NEXTN MTP, both with the DP per-layer
buffer config, at low concurrency.

Known issue: HiSparse + MTP currently crashes in _build_hisparse_decode_batch
(topk_p=None) -- this sweep exists to reproduce/track that and to measure the
combination once fixed.

    python -m sglang.srt.mem_cache.sparsity.trace.mtp_sweep [--flags]
"""

import argparse
from pathlib import Path

from . import exp_common as ec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    ec.add_common_args(p)
    p.set_defaults(num_prompts=ec.env("NUM_PROMPTS", 100))
    p.add_argument("--top-k", type=int, default=ec.env("TOP_K", 512))
    p.add_argument("--port-hs", type=int, default=ec.env("PORT_HS", 31000))
    p.add_argument("--port-mtp", type=int, default=ec.env("PORT_MTP", 33000))
    p.add_argument("--concurrency-list", default=ec.env("CONCURRENCY_LIST", "1 2 4 5"))
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
        default=ec.env("OUT_DIR", str(ec.RESULTS_ROOT / f"mtp_{ec.job_tag()}")),
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

    # Baseline (HiSparse only), then treatment (HiSparse + MTP). Distinct ports.
    ec.sweep_arm(
        "hisparse",
        out_dir,
        args.port_hs,
        args,
        conc,
        server_flags=hs_dp,
        output_len=args.output_len,
    )
    ec.sweep_arm(
        "hisparse_mtp",
        out_dir,
        args.port_mtp,
        args,
        conc,
        server_flags=hs_dp + mtp,
        output_len=args.output_len,
    )
    ec.log("DONE")


if __name__ == "__main__":
    main()
