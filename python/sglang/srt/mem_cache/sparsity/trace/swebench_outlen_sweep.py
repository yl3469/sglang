"""Realistic-output variant of the A/B sweep: instead of a flat 256-token
output, each request's output length FOLLOWS THE SWE-BENCH GOLD-PATCH LENGTH.

How: the custom dataset loader derives per-request output_len from the
assistant turn's token count WHEN --sharegpt-output-len is omitted; the
swe_bench_300.jsonl stores the gold patch as the assistant turn, so we simply
omit the output length (--output-len left unset). ignore_eos stays on =>
each request emits exactly its gold-patch token count: deterministic and
identical per-request across arms.

Arms: dense (no HiSparse), uniform, dp. (HiSparse+MTP is excluded: it crashes
in _build_hisparse_decode_batch, topk_p=None.)

    python -m sglang.srt.mem_cache.sparsity.trace.swebench_outlen_sweep [--flags]
"""

import argparse
from pathlib import Path

from . import exp_common as ec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # OUTPUT_LEN default None => follow the dataset (gold-patch) length.
    ec.add_common_args(p, output_len_default=None)
    p.set_defaults(num_prompts=ec.env("NUM_PROMPTS", 100))
    p.add_argument("--top-k", type=int, default=ec.env("TOP_K", 512))
    p.add_argument("--port-dense", type=int, default=ec.env("PORT_DENSE", 31000))
    p.add_argument("--port-uniform", type=int, default=ec.env("PORT_UNIFORM", 33000))
    p.add_argument("--port-dp", type=int, default=ec.env("PORT_DP", 35000))
    p.add_argument("--concurrency-list", default=ec.env("CONCURRENCY_LIST", "1 2 4 5"))
    p.add_argument(
        "--arms",
        default=ec.env("ARMS", "dense uniform dp"),
        help="Space-separated subset of: dense uniform dp",
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
        default=ec.env("OUT_DIR", str(ec.RESULTS_ROOT / f"swelen_{ec.job_tag()}")),
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
    arm_specs = {
        "dense": (args.port_dense, []),
        "uniform": (
            args.port_uniform,
            [
                "--enable-hisparse",
                "--hisparse-config",
                ec.hisparse_config(args.top_k, str(cfgs["uniform"])),
            ],
        ),
        "dp": (
            args.port_dp,
            [
                "--enable-hisparse",
                "--hisparse-config",
                ec.hisparse_config(args.top_k, str(cfgs["dp"])),
            ],
        ),
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
