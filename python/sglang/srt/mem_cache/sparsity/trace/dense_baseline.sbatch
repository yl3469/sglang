#!/bin/bash
#SBATCH --job-name=hisparse_dense
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
# DENSE baseline for the HiSparse A/B study: launch the SAME DSv4-Flash server
# with the SAME flags as ab_sweep.sbatch, but WITHOUT --enable-hisparse (and no
# per-layer buffer config). The model still runs its native compressed +
# C4Indexer (index_topk) sparse attention -- HiSparse OFF only means the full
# compressed KV stays on GPU (pool_configurator c4_shrink_factor=1, no host
# offload). This isolates the cost/benefit of the host-offload mechanism itself:
#
#   dense  = no host offload (full compressed KV on GPU)
#   uniform= HiSparse on, uniform per-layer device buffer  (ab_sweep.sbatch)
#   dp     = HiSparse on, DP per-layer device buffer        (ab_sweep.sbatch)
#
# Same dataset, concurrency sweep, fixed output length -> directly comparable.
# Output lands under results/dense_<jobid>/dense/bench_c*.jsonl so the existing
# aggregate/report tooling can pick it up as a third arm.
#
# Override VARs via `sbatch --export=ALL,VAR=... dense_baseline.sbatch`.

set -o pipefail

SGLANG_DIR="/scratch/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/yueyingl/ai-horizon/avo-related/avo/sglang"
PY="$SGLANG_DIR/.venv/bin/python"
TRACE_PKG="sglang.srt.mem_cache.sparsity.trace"

MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/9e165c30e2704aec5d9d593cce3eebd58bbef1cb}"
HF_CACHE="${HF_CACHE:-/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/hf_cache}"

HOST="127.0.0.1"
PORT="${PORT:-35000}"
BASE_URL="http://${HOST}:${PORT}"

TP="${TP:-4}"
DP="${DP:-4}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-262144}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-256}"

# Low-concurrency focus (1-5) so points pair up 1:1 with the A/B low-load arm.
CONCURRENCY_LIST="${CONCURRENCY_LIST:-1 2 4 5}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"

RESULTS_ROOT="$SGLANG_DIR/python/sglang/srt/mem_cache/sparsity/trace/results"
SWE_JSONL="${SWE_JSONL:-$RESULTS_ROOT/_shared/swe_bench_300.jsonl}"

OUT_DIR="${OUT_DIR:-$RESULTS_ROOT/dense_${SLURM_JOB_ID:-local}}"
ARM_DIR="$OUT_DIR/dense"
mkdir -p "$ARM_DIR"
cd "$SGLANG_DIR" || exit 1

# ---- environment fixes (identical to the working A/B run) ------------------
export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
_CU13="$SGLANG_DIR/.venv/lib/python3.12/site-packages/nvidia/cu13"
if [ -z "$CUDA_HOME" ] && [ -x "$_CU13/bin/nvcc" ]; then
  export CUDA_HOME="$_CU13"
  export PATH="$_CU13/bin:$PATH"
  export LD_LIBRARY_PATH="$_CU13/lib:$LD_LIBRARY_PATH"
  export LIBRARY_PATH="$_CU13/lib:$LIBRARY_PATH"
  [ -e "$_CU13/lib/libcudart.so" ] || ln -sf libcudart.so.13 "$_CU13/lib/libcudart.so" 2>/dev/null
fi
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+$NVCC_APPEND_FLAGS }-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export DG_JIT_USE_NVRTC="${DG_JIT_USE_NVRTC:-true}"
export SGLANG_DSV4_FP4_DEQUANT="${SGLANG_DSV4_FP4_DEQUANT:-1}"

echo "=== node $(hostname) start $(date) ==="
nvidia-smi || { echo "no GPUs"; exit 1; }
[ -s "$SWE_JSONL" ] || { echo "missing dataset $SWE_JSONL"; exit 1; }

# ---------------------------------------------------------------------------
# Launch the DENSE server: identical to the A/B arms EXCEPT no --enable-hisparse
# and no --hisparse-config. --disable-radix-cache / --disable-custom-all-reduce
# are kept so the only difference vs the HiSparse arms is the host-offload.
# setsid gives the server its own process group for clean teardown.
# NOTE: without HiSparse the full compressed KV stays on GPU (c4_shrink_factor
# =1). If the server OOMs at MAX_TOTAL_TOKENS=262144, lower it via
# --export=ALL,MAX_TOTAL_TOKENS=... and note the reduced token budget.
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "=== DENSE (no HiSparse)  port=$PORT  ($(date)) ==="
echo "======================================================================"
setsid "$PY" -m sglang.launch_server \
  --model-path "$MODEL_PATH" --trust-remote-code \
  --tp-size "$TP" --dp-size "$DP" --enable-dp-attention \
  --disable-radix-cache --disable-custom-all-reduce \
  --max-total-tokens "$MAX_TOTAL_TOKENS" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --enable-metrics \
  --host "$HOST" --port "$PORT" \
  > "$ARM_DIR/server.log" 2>&1 &
SERVER_PID=$!

echo "=== [dense] waiting for readiness (pid $SERVER_PID) ==="
READY=0
for i in $(seq 1 120); do
  if curl -sf "${BASE_URL}/get_model_info" >/dev/null 2>&1; then READY=1; break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[dense] server died; see $ARM_DIR/server.log"; exit 1
  fi
  sleep 15
done
[ "$READY" = "1" ] || { echo "[dense] not ready in time"; kill -- -$SERVER_PID 2>/dev/null; exit 1; }
echo "=== [dense] server ready ($(date)) ==="

# Record the resolved token capacity for provenance (dense keeps full KV on GPU).
curl -sf "${BASE_URL}/get_server_info" > "$ARM_DIR/server_info.json" 2>/dev/null || true

for c in $CONCURRENCY_LIST; do
  echo "=== [dense] bench concurrency=$c num_prompts=$NUM_PROMPTS out_len=$OUTPUT_LEN ($(date)) ==="
  "$PY" -m sglang.benchmark.serving \
    --backend sglang --base-url "$BASE_URL" \
    --dataset-name custom --dataset-path "$SWE_JSONL" \
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$c" --request-rate inf \
    --sharegpt-output-len "$OUTPUT_LEN" \
    --warmup-requests 2 \
    --output-file "$ARM_DIR/bench_c${c}.jsonl" \
    > "$ARM_DIR/bench_c${c}.log" 2>&1 \
    || echo "[dense] bench c=$c returned nonzero (see log)"
done

echo "=== [dense] stopping server (full process-group teardown) ==="
kill -- -$SERVER_PID 2>/dev/null
sleep 5
kill -9 -- -$SERVER_PID 2>/dev/null
pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null

echo "=== DONE ($(date)) ==="
ls -R "$OUT_DIR"
