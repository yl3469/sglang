#!/bin/bash
#SBATCH --job-name=hisparse_trace
#SBATCH --partition=batch
#SBATCH --account=coreai_horizon_dilations
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#
# Stage 6: real DSv4-Flash inference -> HiSparse indexer-topk trace -> per-layer
# DP capacity allocation, entirely via launch_server + bench_serving.
#
# Pipeline (all host-side except the served model itself):
#   1. launch_server  (DSv4-Flash, --enable-hisparse, DP attention,
#                       --enable-return-indexer-topk, --enable-metrics)
#   2. record_kv_budget   -> kv_budget.json   (DP total-budget input)
#   3. smoke 1 request     (confirm indexer_topk is non-empty)
#   4. bench_serving over K SWE-bench prompts, --indexer-trace-out -> req_*.npz
#   5. trace_to_curves (implicit) + run_per_layer_dp -> dp_allocations.csv
#
# Override any VAR below via `sbatch --export=ALL,VAR=... collect_trace.sbatch`.

set -o pipefail

SGLANG_DIR="/scratch/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/yueyingl/ai-horizon/avo-related/avo/sglang"
PY="$SGLANG_DIR/.venv/bin/python"
TRACE_PKG="sglang.srt.mem_cache.sparsity.trace"

MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/9e165c30e2704aec5d9d593cce3eebd58bbef1cb}"
HF_CACHE="${HF_CACHE:-/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/hf_cache}"

HOST="127.0.0.1"
PORT="${PORT:-30000}"
BASE_URL="http://${HOST}:${PORT}"

TP="${TP:-4}"
DP="${DP:-4}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
INDEX_TOPK="${INDEX_TOPK:-512}"
# DSv4-Flash: 43 layers, compress_ratio 4 -> C4A indexer layers on even ids 2..42
NUM_INDEXER_LAYERS="${NUM_INDEXER_LAYERS:-21}"
# Cap the KV memory pool. --enable-return-indexer-topk allocates a PINNED host
# buffer sized (max_total_num_tokens+page_size) x num_indexer_layers x index_topk
# x 4B; at the auto-sized max_total_num_tokens this OOM'd on top of the 4x
# ~102 GB hisparse host pools (job 3188782 died in init_indexer_capturer with
# CUDA out of memory right after weight load). We only serve a few long-context
# SWE-bench prompts at --max-concurrency 1, so a modest pool is plenty and keeps
# the capture buffer bounded. 262144 tokens x 21 x 512 x 4B ~= 11 GB pinned/rank.
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-262144}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"

RESULTS_DIR="${RESULTS_DIR:-$SGLANG_DIR/python/sglang/srt/mem_cache/sparsity/trace/results/${SLURM_JOB_ID:-local}}"
TRACE_DIR="$RESULTS_DIR/traces"
# SWE_JSONL may be pre-generated on a login node (which has HF network) and
# passed in via --export=ALL,SWE_JSONL=... so the compute node need not fetch it.
SWE_JSONL="${SWE_JSONL:-$RESULTS_DIR/swe_bench.jsonl}"

mkdir -p "$RESULTS_DIR" "$TRACE_DIR"
cd "$SGLANG_DIR" || exit 1

export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# CUDA toolkit: this venv ships a pip-installed toolkit under nvidia/cu13 and
# there is no system CUDA. deep_gemm's find_cuda_home() asserts on a missing
# CUDA_HOME during import, so point it at the bundled toolkit (verified to
# resolve the import). Falls back gracefully if the layout changes.
if [ -z "$CUDA_HOME" ]; then
  _CU13="$SGLANG_DIR/.venv/lib/python3.12/site-packages/nvidia/cu13"
  if [ -x "$_CU13/bin/nvcc" ]; then
    export CUDA_HOME="$_CU13"
    export PATH="$_CU13/bin:$PATH"
    # The bundled toolkit keeps libs in lib/ (no lib64), and ships only a
    # versioned libcudart.so.13. Expose the lib dir to the runtime/JIT linker
    # and (a startup step creates) an unversioned libcudart.so symlink so any
    # nvcc-CLI JIT that does -lcudart can link.
    export LD_LIBRARY_PATH="$_CU13/lib:$LD_LIBRARY_PATH"
    export LIBRARY_PATH="$_CU13/lib:$LIBRARY_PATH"
    [ -e "$_CU13/lib/libcudart.so" ] || \
      ln -sf libcudart.so.13 "$_CU13/lib/libcudart.so" 2>/dev/null
    echo "=== CUDA_HOME=$CUDA_HOME ==="
  else
    echo "WARNING: bundled cu13 toolkit not found at $_CU13; CUDA_HOME unset"
  fi
fi

# DeepGEMM / all runtime-JIT NVCC compiles: the bundled toolkit has nvcc 13.3
# but cuda-runtime headers 13.0, so cccl's compiler/header compatibility guard
# ("CUDA compiler and CUDA toolkit headers are incompatible") fires and killed
# the DSv4 scheduler during the fp8 weight layout transform. nvcc 13.3 vs cudart
# 13.0 is a minor, ABI-compatible skew, so disable that specific check for every
# nvcc invocation via NVCC_APPEND_FLAGS (honored by nvcc itself, independent of
# subprocess env propagation). VERIFIED locally: without this flag the exact
# cuda_toolkit.h:41 #error reproduces; with it the compile proceeds cleanly.
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+$NVCC_APPEND_FLAGS }-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
# Belt-and-suspenders: also ask DeepGEMM to prefer its NVRTC path (bundled
# libnvrtc is 13.0, matching the runtime headers). DeepGEMM parses this as a
# boolean literal ("true"/"false"), not 0/1.
export DG_JIT_USE_NVRTC="${DG_JIT_USE_NVRTC:-true}"

# DSv4-Flash routed experts are fp4-packed (config.json: expert_dtype=fp4,
# quant_method=fp8). With moe_runner_backend=auto and this env UNSET, the MoE
# falls through to the plain Triton fp8 kernel, which asserts "Hidden size
# mismatch" on the fp4-packed weight shape during CUDA-graph capture (job
# 3189341 died there). Setting SGLANG_DSV4_FP4_DEQUANT=1 selects the intended
# path: dequantize fp4 experts to fp8 and run Fp8MoEMethod (requires backend
# =auto, which we use). See fp8.py:378 and model_loader/loader.py:257.
export SGLANG_DSV4_FP4_DEQUANT="${SGLANG_DSV4_FP4_DEQUANT:-1}"

echo "=== node: $(hostname)  start: $(date) ==="
nvidia-smi || { echo "no GPUs visible"; exit 1; }

# ---------------------------------------------------------------------------
# Stage 0: SWE-bench -> custom JSONL (skip if already present).
# ---------------------------------------------------------------------------
if [ ! -s "$SWE_JSONL" ]; then
  echo "=== Stage 0: preparing SWE-bench prompts ==="
  # Requires network/HF cache; if offline and this fails, drop in your own JSONL.
  HF_HUB_OFFLINE=0 "$PY" -m ${TRACE_PKG}.prep_swe_bench \
    --num-prompts "$NUM_PROMPTS" \
    --min-tokens 20000 --max-tokens 120000 \
    --tokenizer-path "$MODEL_PATH" \
    --out "$SWE_JSONL" || {
      echo "SWE-bench prep failed; provide $SWE_JSONL manually and resubmit"
      exit 1
    }
fi

# ---------------------------------------------------------------------------
# Launch server.
# ---------------------------------------------------------------------------
echo "=== launching server (TP=$TP DP=$DP) ==="
"$PY" -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --trust-remote-code \
  --tp-size "$TP" \
  --dp-size "$DP" \
  --enable-dp-attention \
  --enable-hisparse \
  --disable-radix-cache \
  --disable-custom-all-reduce \
  --max-total-tokens "$MAX_TOTAL_TOKENS" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --enable-return-indexer-topk \
  --enable-metrics \
  --host "$HOST" --port "$PORT" \
  > "$RESULTS_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'echo "stopping server $SERVER_PID"; kill $SERVER_PID 2>/dev/null' EXIT

echo "=== waiting for server readiness ==="
READY=0
for i in $(seq 1 120); do
  if curl -sf "${BASE_URL}/health_generate" >/dev/null 2>&1 \
     || curl -sf "${BASE_URL}/get_model_info" >/dev/null 2>&1; then
    READY=1; break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "server process died; see $RESULTS_DIR/server.log"; exit 1
  fi
  sleep 15
done
[ "$READY" = "1" ] || { echo "server not ready in time"; exit 1; }
echo "=== server ready ($(date)) ==="

# ---------------------------------------------------------------------------
# Stage 2: record KV budget (DP total-budget input).
# ---------------------------------------------------------------------------
echo "=== Stage 2: recording KV budget ==="
"$PY" -m ${TRACE_PKG}.record_kv_budget \
  --base-url "$BASE_URL" \
  --out "$RESULTS_DIR/kv_budget.json"

# ---------------------------------------------------------------------------
# Stage 3-1 smoke: a single request must return a non-empty indexer_topk.
# ---------------------------------------------------------------------------
echo "=== smoke: 1 request, confirm indexer_topk present ==="
"$PY" -m sglang.benchmark.serving \
  --backend sglang \
  --base-url "$BASE_URL" \
  --dataset-name custom --dataset-path "$SWE_JSONL" \
  --num-prompts 1 --max-concurrency 1 --request-rate inf \
  --extra-request-body '{"return_indexer_topk": true}' \
  --indexer-trace-out "$TRACE_DIR" \
  --indexer-num-layers "$NUM_INDEXER_LAYERS" \
  --indexer-topk "$INDEX_TOPK" \
  --sharegpt-output-len 64 \
  || { echo "smoke request failed"; exit 1; }

if ! ls "$TRACE_DIR"/req_*.npz >/dev/null 2>&1; then
  echo "no indexer trace captured on smoke; check --enable-return-indexer-topk"
  exit 1
fi
echo "=== smoke OK: $(ls "$TRACE_DIR"/req_*.npz | wc -l) trace(s) ==="

# ---------------------------------------------------------------------------
# Stage 1: full bench_serving collection over K SWE-bench prompts.
# ---------------------------------------------------------------------------
echo "=== Stage 1: collecting traces over $NUM_PROMPTS prompts ==="
"$PY" -m sglang.benchmark.serving \
  --backend sglang \
  --base-url "$BASE_URL" \
  --dataset-name custom --dataset-path "$SWE_JSONL" \
  --num-prompts "$NUM_PROMPTS" --max-concurrency 1 --request-rate inf \
  --extra-request-body '{"return_indexer_topk": true}' \
  --indexer-trace-out "$TRACE_DIR" \
  --indexer-num-layers "$NUM_INDEXER_LAYERS" \
  --indexer-topk "$INDEX_TOPK" \
  --sharegpt-output-len 128 \
  --output-file "$RESULTS_DIR/bench_serving.jsonl"

# Server no longer needed for the offline stages.
kill $SERVER_PID 2>/dev/null; trap - EXIT

# ---------------------------------------------------------------------------
# Stage 3+4: replay -> curves -> DP allocation CSV (LRU + Belady gap).
# DSv4-Flash C4A indexer layers sit on even model ids 2..42.
# ---------------------------------------------------------------------------
echo "=== Stage 3+4: per-layer DP allocation ==="
"$PY" -m ${TRACE_PKG}.run_per_layer_dp \
  "$TRACE_DIR" \
  --policy lru \
  --kv-budget "$RESULTS_DIR/kv_budget.json" \
  --compress-ratio 4 \
  --layer-offset 2 --layer-stride 2 \
  --out-dir "$RESULTS_DIR/per_layer_budget_dp"

echo "=== DONE ($(date)) ==="
echo "Artifacts in: $RESULTS_DIR"
ls -R "$RESULTS_DIR"
