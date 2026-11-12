"""DENSE baseline for the HiSparse A/B study.

Launch the SAME server with the SAME flags as ab_sweep, but WITHOUT
--enable-hisparse (and no per-layer buffer config). The model still runs its
native compressed + C4Indexer sparse attention -- HiSparse OFF only means the
full compressed KV stays on GPU (no host offload). This isolates the
cost/benefit of the host-offload mechanism itself. Output lands under
results/dense_<tag>/dense/bench_c*.jsonl so the aggregate/report tooling can
pick it up as a third arm.

    python -m sglang.srt.mem_cache.sparsity.trace.dense_baseline [--flags]
"""

import argparse
from pathlib import Path

from . import exp_common as ec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    ec.add_common_args(p)
    p.add_argument("--port", type=int, default=ec.env("PORT", 35000))
    p.add_argument("--concurrency-list", default=ec.env("CONCURRENCY_LIST", "1 2 4 5"))
    p.add_argument(
        "--out-dir",
        default=ec.env("OUT_DIR", str(ec.RESULTS_ROOT / f"dense_{ec.job_tag()}")),
    )
    args = p.parse_args()

    ec.setup_environment(args.hf_cache)
    ec.require_file(args.swe_jsonl, "dataset")

    # NOTE: without HiSparse the full compressed KV stays on GPU. If the server
    # OOMs at --max-total-tokens 262144, lower it and note the reduced budget.
    ec.sweep_arm(
        "dense",
        Path(args.out_dir),
        args.port,
        args,
        ec.parse_concurrency(args.concurrency_list),
        output_len=args.output_len,
    )
    ec.log("DONE")


if __name__ == "__main__":
    main()
