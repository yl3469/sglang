"""HiSparse A/B (dense/uniform/dp) benched by aiperf Weka
trace replay of the semianalysisai/cc-traces-weka-062126 corpus (Claude Code
agent traces; prompts reconstructed from hash_ids so prefix-cache reuse
structure is preserved).

Differences vs the swebench sweep:
* bench client is aiperf (>= 0.13; upstream ships the
  ``semianalysis_cc_traces_weka_062126`` alias and the
  ``inferencex-agentx-mvp`` scenario), not sglang.benchmark.serving. Default
  binary: ``<SGLANG_DIR>/.venv-aiperf/bin/aiperf`` (override with AIPERF_BIN);
* radix cache stays DISABLED on every arm: HiSparse hard-requires
  --disable-radix-cache (arg_groups/hisparse_hook.py validate_hisparse), so a
  matched A/B cannot enable it. The dataset's prefix-reuse structure is still
  replayed on the wire; the server just cannot exploit it. FINDING, not a
  choice.
* one aiperf run per arm (scenario inferencex-agentx-mvp steady-state replay)
  instead of a concurrency sweep.

Modes:
  smoke: 1 arm (uniform by default), 3 traces, fixed-schedule replay of the
         first 2 minutes of each trace, no scenario. Proves the pipeline.
  full:  arms dense/uniform/dp, scenario inferencex-agentx-mvp, FULL corpus
         (393 traces; NUM_TRACES=0 omits --num-dataset-entries, which the
         AgentX tutorial reserves for smoke tests), --benchmark-duration 1800
         (the scenario default; 900 is the enforced minimum), fixed seed, same
         traces for every arm. Everything else the scenario locks (streaming,
         ignore_eos, first-turn-prefix cache bust, 10 s system idle-gap cap,
         no per-trace delay caps) is auto-filled by aiperf.

    .venv/bin/python -m sglang.srt.mem_cache.sparsity.trace.run_cctraces_aiperf --mode smoke
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from . import exp_common as ec

DEFAULT_AIPERF = str(ec.SGLANG_DIR / ".venv-aiperf/bin/aiperf")
DATASET_ALIAS = "semianalysis_cc_traces_weka_062126"
# Cold full-corpus reconstruction can exceed aiperf's 300 s configuration
# timeout (agentx-mvp tutorial, "Configuration times out"); the tutorial's
# recommended ceiling for a cold run.
AIPERF_CONFIGURE_TIMEOUT_S = "1800"


def run_aiperf(args, arm_dir: Path, base_url: str) -> int:
    cmd = [
        args.aiperf_bin,
        "profile",
        "--url",
        base_url,
        "--model",
        args.model_path,
        "--tokenizer",
        args.model_path,
        "--tokenizer-trust-remote-code",
        "--endpoint-type",
        "chat",
        "--streaming",
        "--use-server-token-count",
        "--public-dataset",
        DATASET_ALIAS,
    ]
    if args.num_traces > 0:
        # Smoke only: the AgentX tutorial says a reduced corpus is "never for
        # runs you intend to compare"; full mode replays all 393 traces.
        cmd += ["--num-dataset-entries", str(args.num_traces)]
    cmd += [
        "--max-context-length",
        str(args.max_context_length),
        "--concurrency",
        str(args.concurrency),
        "--random-seed",
        str(args.seed),
        "--artifact-dir",
        str(arm_dir / "aiperf"),
        "--ui",
        "simple",
    ]
    if args.mode == "full":
        cmd += [
            "--scenario",
            "inferencex-agentx-mvp",
            "--benchmark-duration",
            str(args.benchmark_duration),
        ]
    else:  # smoke: short window, no scenario (--fixed-schedule needs a file)
        cmd += [
            "--trace-idle-gap-cap-seconds",
            "10",
            "--benchmark-duration",
            str(args.benchmark_duration),
            "--benchmark-grace-period",
            "120",
        ]
    env = dict(os.environ)
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("AIPERF_DATASET_CONFIGURATION_TIMEOUT", AIPERF_CONFIGURE_TIMEOUT_S)
    env.setdefault(
        "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT", AIPERF_CONFIGURE_TIMEOUT_S
    )
    # datasets-lib cache: the corpus was pre-downloaded on a login node with
    # cache_dir=<hf_cache root>, which is NOT the default $HF_HOME/datasets.
    env["HF_DATASETS_CACHE"] = args.hf_cache
    log_file = arm_dir / "aiperf.log"
    print(f"+ {shlex.join(cmd)}", flush=True)
    with open(log_file, "wb") as f:
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)


def bench_arm(arm: str, out_dir: Path, port: int, args, server_flags) -> bool:
    arm_dir = out_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    server = ec.Server(
        model_path=args.model_path,
        port=port,
        tp=args.tp,
        dp=args.dp,
        max_total_tokens=args.max_total_tokens,
        max_running_requests=args.max_running_requests,
        log_path=arm_dir / "server.log",
        extra_flags=server_flags,
    )
    ec.log(f"ARM={arm} port={port} flags={list(server_flags)}")
    server.launch()
    try:
        if not server.wait_ready():
            print(f"[{arm}] server not ready in time", flush=True)
            return False
        ec.log(f"[{arm}] server ready")
        server.save_resolved_buffer_sizes(arm_dir / "resolved_buffer_sizes.txt")
        server.save_server_info(arm_dir / "server_info.json")
        rc = run_aiperf(args, arm_dir, server.base_url)
        ec.log(f"[{arm}] aiperf rc={rc}")
        return rc == 0
    finally:
        server.teardown()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    ec.add_common_args(p, output_len_default=None)
    # Model: GLM-5.2-NVFP4 (the shared-cache DeepSeek-V4-Flash snapshot was
    # wiped on Aug 28; blobs gone). Topology mirrors the proven glm_aalcr run
    # (TP2/DP2, dp attention). GLM KV is ~66.8 KB/token (78 DSA layers) with
    # only ~51 GB post-weights headroom per GPU, so the pool is 393216 tokens
    # (~26 GB/rank); traces whose context exceeds --max-context-length are
    # dropped client-side by aiperf and counted in the report.
    p.set_defaults(
        model_path=ec.env(
            "MODEL_PATH",
            "/scratch/fsw/portfolios/coreai/projects/coreai_horizon_dilations/"
            "users/yueyingl/models/GLM-5.2-NVFP4",
        ),
        tp=ec.env("TP", 2),
        dp=ec.env("DP", 2),
        max_total_tokens=ec.env("MAX_TOTAL_TOKENS", 393216),
        max_running_requests=ec.env("MAX_RUNNING_REQUESTS", 32),
    )
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--arms", default=ec.env("ARMS", ""))
    p.add_argument(
        "--max-context-length", type=int, default=ec.env("MAX_CONTEXT_LENGTH", 380000)
    )
    # GLM index_topk is 2048 (vs 512 on DSv4); b_mean keeps the same 2x-top_k
    # ratio the DSv4 configs used (1024/512 -> 4096/2048).
    p.add_argument("--top-k", type=int, default=ec.env("TOP_K", 2048))
    p.add_argument("--port-dense", type=int, default=ec.env("PORT_DENSE", 31000))
    p.add_argument("--port-uniform", type=int, default=ec.env("PORT_UNIFORM", 33000))
    p.add_argument("--port-dp", type=int, default=ec.env("PORT_DP", 35000))
    p.add_argument("--num-traces", type=int, default=ec.env("NUM_TRACES", 0))
    p.add_argument("--concurrency", type=int, default=ec.env("CONCURRENCY", 0))
    p.add_argument(
        "--benchmark-duration", type=int, default=ec.env("BENCH_DURATION", 0)
    )
    p.add_argument("--seed", type=int, default=ec.env("SEED", 20260908))
    p.add_argument(
        "--smoke-end-offset-ms", type=int, default=ec.env("SMOKE_END_OFFSET_MS", 120000)
    )
    p.add_argument("--aiperf-bin", default=ec.env("AIPERF_BIN", DEFAULT_AIPERF))
    p.add_argument(
        "--dp-csv",
        default=ec.env(
            "DP_CSV",
            str(ec.RESULTS_ROOT / "3191572/per_layer_budget_dp/dp_allocations.csv"),
        ),
    )
    p.add_argument("--dp-total-row", type=float, default=ec.env("DP_TOTAL_ROW", 0.3))
    p.add_argument("--b-mean", type=int, default=ec.env("B_MEAN", 4096))
    p.add_argument("--page-size", type=int, default=ec.env("PAGE_SIZE_SLOTS", 64))
    p.add_argument(
        "--out-dir",
        default=ec.env("OUT_DIR", str(ec.RESULTS_ROOT / f"cctraces_{ec.job_tag()}")),
    )
    args = p.parse_args()

    # Mode-dependent defaults for the 0-sentinel knobs.
    if args.mode == "smoke":
        args.arms = args.arms or "uniform"
        args.num_traces = args.num_traces or 5
        args.concurrency = args.concurrency or 3
        args.benchmark_duration = args.benchmark_duration or 300
    else:
        args.arms = args.arms or "dense uniform dp"
        # num_traces stays 0 -> full corpus (AgentX methodology).
        args.concurrency = args.concurrency or 8
        args.benchmark_duration = args.benchmark_duration or 1800

    ec.setup_environment(args.hf_cache)
    ec.require_file(args.dp_csv, "dp csv")
    ec.require_file(args.aiperf_bin, "aiperf binary")

    out_dir = Path(args.out_dir)
    cfgs = ec.build_buffer_configs(
        args.dp_csv,
        out_dir / "buffer_configs",
        args.dp_total_row,
        args.b_mean,
        args.top_k,
        args.page_size,
    )
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

    ok = True
    for arm in args.arms.split():
        if arm not in arm_specs:
            print(f"unknown arm {arm!r}, skipping", flush=True)
            continue
        port, flags = arm_specs[arm]
        ok = bench_arm(arm, out_dir, port, args, flags) and ok
    ec.log(f"DONE ok={ok}")
    if args.mode == "smoke" and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
