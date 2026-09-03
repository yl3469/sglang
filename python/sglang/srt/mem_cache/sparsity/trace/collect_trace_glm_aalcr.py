"""GLM variant of collect_trace: capture HiSparse indexer-topk traces from
GLM-5.2-NVFP4 over the AA-LCR long-context dataset and run the per-layer DP
allocation on them.

The AA-LCR JSONL must be pre-generated (prep_aa_lcr.py on a login node with HF
network); this script refuses to start without it.

    python -m sglang.srt.mem_cache.sparsity.trace.collect_trace_glm_aalcr [--flags]
"""

import argparse
from pathlib import Path

from . import exp_common as ec
from .collect_trace import collect

DEFAULT_GLM_MODEL = (
    "/scratch/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/"
    "yueyingl/models/GLM-5.2-NVFP4"
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    ec.add_common_args(p)
    p.set_defaults(
        model_path=ec.env("MODEL_PATH", DEFAULT_GLM_MODEL),
        num_prompts=ec.env("NUM_PROMPTS", 16),
        max_running_requests=ec.env("MAX_RUNNING_REQUESTS", 8),
    )
    p.add_argument("--port", type=int, default=ec.env("PORT", 30000))
    p.add_argument("--index-topk", type=int, default=ec.env("INDEX_TOPK", 2048))
    p.add_argument(
        "--num-indexer-layers", type=int, default=ec.env("NUM_INDEXER_LAYERS", 21)
    )
    p.add_argument(
        "--aalcr-jsonl",
        default=ec.env(
            "AALCR_JSONL", str(ec.RESULTS_ROOT / "_shared/aa_lcr_100.jsonl")
        ),
    )
    p.add_argument(
        "--results-dir",
        default=ec.env(
            "RESULTS_DIR", str(ec.RESULTS_ROOT / f"glm_aalcr_{ec.job_tag()}")
        ),
    )
    args = p.parse_args()

    ec.setup_environment(args.hf_cache)
    ec.require_file(
        args.aalcr_jsonl, "AA-LCR dataset (run prep_aa_lcr.py on a login node)"
    )
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # GLM traces use the default trace-order layer labeling (no compress-ratio
    # / layer-offset remap as on DSv4-Flash).
    collect(args, args.aalcr_jsonl, results_dir, dp_extra_args=[])


if __name__ == "__main__":
    main()
