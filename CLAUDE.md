# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of SGLang (`git@github.com:yl3469/sglang.git`) used for **HiSparse** research: hierarchical sparse
attention with host-offloaded KV for DeepSeek Sparse Attention (DSA) models (DeepSeek V3.2/V4, GLM-5).
The `hisparse-dp` branch adds per-layer device-buffer budgets chosen by an offline dynamic program plus
the trace/A/B experiment tooling under `python/sglang/srt/mem_cache/sparsity/trace/`.

Rules in `.claude/rules/*.md` are auto-loaded (msgspec.Struct not dataclass, no defensive getattr,
out-of-place ScheduleBatch mutation, ForwardBatch.init_new purity, unit-test admission criteria, general
style). `.claude/rules/modify-component-must-read.md` lists skills that MUST be read before touching
speculative decoding, `Scheduler`/`TokenizerManager`/`ModelRunner` `__init__`, `SGLANG_*` env vars, or the
scripted runtime.

## Environment and commands

Install (uv; Python 3.12; the venv must live at `<repo>/.venv` because the experiment tooling resolves
the bundled cu13 toolkit from `.venv/lib/python3.12/site-packages/nvidia/cu13`):

```bash
uv venv .venv -p 3.12
uv pip install -p .venv/bin/python -e "python[dev]" --index-strategy unsafe-best-match --prerelease allow
```

Lint/format (pre-commit: black, isort profile=black, ruff F401/F821/UP037, codespell, clang-format):

```bash
pre-commit run --files <changed files>       # or: pre-commit run --all-files
```

Tests (see `test/README.md` and the `write-sglang-test` skill):

```bash
python3 test/registered/core/test_srt_endpoint.py                          # one file
python3 test/registered/core/test_srt_endpoint.py TestSRTEndpoint.test_x   # one method
pytest test/registered/unit/ -v                                            # no-GPU unit tests (test/pytest.ini)
pytest test/registered/unit/mem_cache/test_per_layer_budget_dp.py -v       # HiSparse DP allocator tests
python3 test/run_suite.py --hw cuda --suite base-b-test-1-gpu-small        # a CI suite
```

Every CI test file registers itself at module level with literal args, e.g.
`register_cuda_ci(est_time=80, stage="base-b", runner_config="1-gpu-small")`, and ends with a plain
`unittest.main()` / `pytest.main([__file__])` (the runner appends `-f`).

Serve and benchmark:

```bash
python -m sglang.launch_server --model-path <path> --tp-size 4 --trust-remote-code --port 30000
python -m sglang.benchmark.serving --backend sglang --base-url http://127.0.0.1:30000 \
    --dataset-name custom --dataset-path <jsonl> --num-prompts 200 --max-concurrency 8
```

HiSparse is enabled with `--enable-hisparse --hisparse-config '{"top_k": 512, "device_buffer_sizes_path": "<json>"}'`
and hard-requires `--disable-radix-cache` (`python/sglang/srt/arg_groups/hisparse_hook.py`).

### Cluster notes (NVIDIA Slurm cluster, aarch64 GB300)

- The login/VS Code node has **no GPU**; all serving/benchmark work runs under Slurm. Login and compute
  nodes are both **aarch64**, so a venv built on the login node runs on compute nodes. Compute nodes
  (`nvl72d*`) have **4x GB300** (Blackwell, ~280 GB each), 144 cores, ~920 GB RAM. Partitions: `batch`
  4h, `batch_long` 7d. Account `coreai_horizon_dilations`. The QOS rejects `--gpus-per-node` < 4
  (`QOSMinGRES`); always request the whole node.
- `sbatch -A coreai_horizon_dilations -p batch_long --nodes=1 --gpus-per-node=4 --time=08:00:00 <script>`
  works directly; the `slurm-broker` MCP tools are an alternative. Compute nodes have fast internet
  (a 167 GB HF checkpoint downloads in ~1 min), so `hf download` belongs inside the job.
- `$HOME` is a small NFS share; the repo, venvs, HF cache and results live on
  `/scratch/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/yueyingl/` (`hf_cache/` is the
  `HF_HOME`; set `UV_CACHE_DIR` there too). Use absolute paths in shell commands (the shell cwd can drift).
- No Rust toolchain on the cluster: build the editable install with `SGLANG_BUILD_RUST_EXTS=none`
  (skips the gRPC/multimodal/Rust-server extensions, which the HTTP serving path does not need).
- Running any `trace/*.py` module directly (outside `collect_trace`/`ab_sweep`) needs `CUDA_HOME`
  pointed at the bundled toolkit, or `deep_gemm` asserts on import. `.slurm_logs/common_env.sh`
  (untracked) exports the same env `exp_common.setup_environment` applies.

### Running the HiSparse A/B study (dense vs uniform vs DP)

Three arms, all on one 4-GPU node with `TP=4 DP=4` (DP-attention), served from
`DeepSeek-V4-Flash-0731` over the shared `results/_shared/swe_bench_300.jsonl` prompt set
(`prep_swe_bench.py --num-prompts 300 --min-tokens 20000 --max-tokens 120000`):

| arm | HiSparse | per-layer buffers | runner |
|---|---|---|---|
| `dense` | off (full compressed KV on GPU) | n/a | `dense_baseline.py` (`OUT_DIR=results/dense_<job>`) |
| `uniform` | on | flat, same total memory as `dp` | `ab_sweep.py` (`OUT_DIR=results/ab_<job>`) |
| `dp` | on | `run_per_layer_dp.py` allocation | same `ab_sweep.py` invocation |

Order: (1) `collect_trace.py` (`RESULTS_DIR=results/<job>`; ~9 min/prompt at 16 prompts, since
returning indexer top-k dominates) writes `results/<job>/per_layer_budget_dp/dp_allocations.csv`;
(2) `DP_CSV=<that csv> ab_sweep.py` runs `uniform` then `dp`; (3) `dense_baseline.py` is independent
and can run on another node in parallel; (4) `make_report.py --results-root results --out report.html`
merges every `ab_*`/`dense_*` folder. Model load takes ~7-15 min per server. Ready-made job scripts
live in `.slurm_logs/{hisparse_ab_pipeline,dense_job,ab_job}.sh` (untracked).

Gotchas seen on 2026-09-11 (jobs 319518 trace, 319722 dense, 321917 A/B): (a) the returned indexer
top-k covers every prefill token, so requests with prompts above ~45k tokens exceed aiohttp's 2 GiB
SSE line limit and are dropped from the trace (6 of 16 lost); (b) check `dp_allocations.csv` before
sweeping: at the default `DP_TOTAL_ROW=0.3` the DP allocation was flat (identical to uniform), so the
sweep used `DP_TOTAL_ROW=0.15`; (c) `run_per_layer_dp` is pure Python and took ~50 min on 10 traces.
(d) aiperf (`run_cctraces_aiperf.py`) treats ANY non-empty `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` as
offline and then rejects local tokenizer paths; export them EMPTY, not "0". (e) GLM-5.2-NVFP4 autotunes
FlashInfer kernels for ~25 min after weight load; set `READY_ATTEMPTS` (x15 s, default 120) above 200.
(f) Full-prompt indexer traces are 639 KB per prompt token on GLM (78 layers x 2048 top-k x 4 B). Use
`--indexer-topk-decode-only` (server) / `--decode-only-trace` (`collect_trace*.py`, env `DECODE_ONLY_TRACE=1`)
so a trace is output_len x layers x top-k; and shrink the trace server's pool (`MAX_TOTAL_TOKENS=131072` for
GLM) because the pinned capture buffer is pool_tokens x layers x top-k x 4 B. Then pass no `--kv-budget` row
(or refit offline) since the tiny pool makes that row violate the per-layer floors. (g) Decode-only curves sit on
a ~6% compulsory-miss floor; DP gains are a few percent of misses at total_ratio 0.1, none above 0.2. (h) One
dp arm (wider max per-layer buffer, ratio 0.3) stalled at concurrency 200 with the HiSparse device pool fully
used and nothing evictable until the 300 s scheduler watchdog killed it (`results/ab_334070/dp/server.log`).
(i) GLM: the canonical BF16 `zai-org/GLM-5.2` (1.5 TB) does not fit one 4x GB300 node; `zai-org/GLM-5.2-FP8`
(756 GB) does, with ~87 GB/GPU left, so use `MAX_TOTAL_TOKENS=393216` for the arms. FP8 weight load takes
~14 min; keep `READY_ATTEMPTS>=400`. Jobs: `.slurm_logs/glm_fp8_{aalcr,cctraces}_job.sh` (+ `glm_fp8_env.sh`).
(j) Per-layer buffers are NOT physically memory-matched: `alloc_device_buffer` hands every request
`device_buffer_size` (= max B_l) pool tokens, and a pool token spans all layers, so a dp config with max 1.5x
mean uses 1.5x the device KV of uniform. Physical width must be <= 32768 (int16 LRU slots). (k) 2026-09-14
fixes: the non-uniform reserved-slot write is one scatter (was 78 launches/step, 4-6% ITL tax), and the
swap-in copy moved out of the block-per-request select kernel into `copy_misses_kernel` (swap-in 4.2 -> 2.0
ms/step; HiSparse AA-LCR throughput +7..+23%). Profile a decode window with `.slurm_logs/profile_arm.py` and
summarize with `analyze_profile.py`; before/after via a HEAD `git worktree` + `PYTHONPATH`.
(l) Per-layer swap-in hit/miss counters: `--hisparse-config '{..., "swap_in_stats": true}'` (off by default; zero
measured cost either way). Read them from `/server_info` -> `internal_states[i].hisparse_swap_in_stats`
(`/get_internal_state` does not exist) or the decode log's "swap-in hit rate"; runners dump
`swap_stats_c<conc>.json` when `SWAP_IN_STATS=1`; `.slurm_logs/swap_stats_report.py` tabulates per layer.
(m) The DP works only when fit budget == served budget: `make_buffer_sizes --total-ratio auto` (needs the
`dp_meta.json` sidecar run_per_layer_dp now writes) picks the row for the served B_MEAN or falls back to uniform.
Measured GLM-FP8 AA-LCR @10240 slots: uniform 6.3% misses vs dp 5.6% (-12%), worst layer 0.158 -> 0.096.

## Architecture

`python/sglang/srt/` is the runtime ("SRT"); everything below is under it unless noted.

**Process layout and request flow.** `entrypoints/http_server.py` (FastAPI, `launch_server()`) and
`entrypoints/engine.py` (`_launch_subprocesses`) spawn one `Scheduler` process per TP/DP rank
(`managers/scheduler.py`), a detokenizer (`managers/detokenizer_manager.py`) and, for `dp_size > 1`, a
`data_parallel_controller.py`. `managers/tokenizer_manager.py` tokenizes and ships requests over ZMQ
`ipc://` sockets (`managers/scheduler_components/ipc_channels.py`; ports in `PortArgs`). The scheduler
forms batches (`schedule_batch.py`, `schedule_policy.py`; helpers in `scheduler_components/`) and calls
`managers/tp_worker.py` -> `model_executor/model_runner.py` (setup split into
`model_runner_components/`, execution in `model_executor/runner/` eager vs CUDA-graph runners) ->
`models/<arch>.py` forward -> logits/sampling -> back through ZMQ to detokenizer -> tokenizer manager ->
HTTP/SSE. `ForwardBatch` (`model_executor/forward_batch_info.py`) is the per-forward view of a
`ScheduleBatch`.

**Configuration.** `server_args.py` is one large `ServerArgs` whose `A[type, ...]` annotations are
turned into argparse by `arg_groups/arg_utils.py`. Feature "hooks" in `arg_groups/` (`hisparse_hook.py`,
`deepseek_v4_hook.py`, `speculative_hook.py`, `pd_disaggregation_hook.py`) resolve/validate feature
flags; `arg_groups/overrides.py` holds per-architecture model overrides. Process-static state goes through
`runtime_context.py` (`RuntimeContext`: parallel info, pristine server args, read-only config namespace
bags, test-only flag overrides) and is guarded by ratchet tests in `test/registered/unit/` (read the
`sglang-runtime-context` skill before adding module-level state). Env vars are declared only in
`environ.py`.

**KV cache and memory (`mem_cache/`).** Radix prefix caches (`radix_cache.py` and variants), allocators
(`allocator/`, incl. `hisparse.py`), device pools (`memory_pool.py`) and host pools (`memory_pool_host.py`,
`pool_host/`). HiCache = GPU<->host<->storage tiering: `hiradix_cache.py`, `managers/cache_controller.py`,
`storage/` backends. HiSparse lives in `mem_cache/sparsity/`: `core/sparse_coordinator.py` selects sparse
KV per layer, `factory.py` parses `--hisparse-config` (incl. per-layer `device_buffer_sizes[_path]`),
`algorithms/` (DSA, Quest), `backend/backend_adaptor.py`; the scheduler-side driver is
`managers/hisparse_coordinator.py` (staging admission, async host->device buffer loads, token stats), with
the device buffer pool in `mem_cache/hisparse_memory_pool.py`.

**Per-layer budget DP and experiment tooling (`mem_cache/sparsity/`).** `per_layer_budget_dp.py` is an
exact min-plus-convolution DP that splits a layer-sum buffer budget across layers from measured per-layer
miss curves; `per_layer_budget_replay.py` replays logged indexer top-k streams through LRU/Belady to
produce those curves. `trace/` is offline, host-side tooling (never on the serving path) forming a
pipeline: `prep_swe_bench.py` / `prep_aa_lcr.py` (datasets -> `custom` JSONL) -> `collect_trace.py`
(launch server with `--enable-hisparse --enable-return-indexer-topk`, `record_kv_budget.py`, benchmark
with `--indexer-trace-out` -> `req_*.npz`) -> `trace_to_curves.py` -> `run_per_layer_dp.py`
(`dp_allocations.csv`) -> `make_buffer_sizes.py` (matched `uniform` vs `dp` buffer JSONs, same total
memory) -> A/B runners `ab_sweep.py`, `highconc_sweep.py`, `mtp_sweep.py`, `dense_baseline.py`,
`run_cctraces_aiperf.py` (arms `dense`/`uniform`/`dp`, aiperf Weka replay of the cc-traces corpus) ->
`aggregate_ab.py` / `make_report.py` / `make_slide.py`. `exp_common.py` holds the shared server
lifecycle (own process group, readiness wait, teardown), env fixes (`setup_environment`), and env-var
overridable knobs (`TP`, `DP`, `MODEL_PATH`, `HF_CACHE`, `DP_CSV`, ...). Results default to
`trace/results/` (untracked).

**Other layers.** Attention backends: `layers/attention/` registered via `attention_registry.py` and
instantiated in `model_runner_components/attention_backend_setup.py`. Models: `models/` with
`models/registry.py` auto-import; loading in `model_loader/`. Speculative decoding: `speculative/`.
Kernels: `python/sglang/kernels/` (`jit/` build infra, per-op `ops/<group>/_jit_*.py`; `aot/` is the
separate `sglang-kernel` wheel). Scripted runtime: `sglang.test.scripted_runtime` drives the real
scheduler loop in-process for tests. Frontend DSL: `python/sglang/lang/`. Diffusion:
`python/sglang/multimodal_gen/`. Rust: `sgl-model-gateway/`, `rust/`, `experimental/sgl-router/`.
Docs: `docs_new/` (Mintlify cookbook).
