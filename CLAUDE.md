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
python -m sglang.launch_server --model-path <path> --tp-size 8 --trust-remote-code --port 30000
python -m sglang.benchmark.serving --backend sglang --base-url http://127.0.0.1:30000 \
    --dataset-name custom --dataset-path <jsonl> --num-prompts 200 --max-concurrency 8
```

HiSparse is enabled with `--enable-hisparse --hisparse-config '{"top_k": 512, "device_buffer_sizes_path": "<json>"}'`
and hard-requires `--disable-radix-cache` (`python/sglang/srt/arg_groups/hisparse_hook.py`).

### Cluster notes (NVIDIA cw-dfw Slurm cluster)

- The login/VS Code node has **no GPU and no CUDA toolkit**; all serving/benchmark work runs under Slurm.
  Nodes are 8x H100 80GB (`batch` 4h, `batch_short` 2h, `batch_long` 8h, `interactive` 4h),
  account `coreai_horizon_dilations`. Blackwell-only features (NVFP4 GEMMs) do not run here.
- `sbatch -A coreai_horizon_dilations -p batch --nodes=1 --gpus-per-node=8 --time=04:00:00 --wrap=...`
  works directly; the `slurm-broker` MCP tools are an alternative.
- `$HOME` is a 10 GB NFS share; the repo, venvs, HF caches and results live on Lustre
  (`/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/yueyingl/`, 50 TB quota).
  Lustre metadata is slow: `git` and `find` over the tree can take minutes; prefer targeted paths, and
  always use absolute paths in shell commands (the shell cwd can drift).

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
