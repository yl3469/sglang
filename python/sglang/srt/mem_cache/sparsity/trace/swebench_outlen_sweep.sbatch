#!/bin/bash
#SBATCH --job-name=hisparse_swelen
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
# Realistic-output variant of the A/B sweep: instead of a flat 256-token output,
# each request's output length FOLLOWS THE SWE-BENCH GOLD-PATCH LENGTH.
#
# How: the custom dataset loader derives per-request output_len from the
# assistant turn's token count WHEN --sharegpt-output-len is omitted
# (benchmark/datasets/custom.py:125-127). Our swe_bench_300.jsonl already stores
# the gold patch as the assistant turn, so we simply DROP --sharegpt-output-len.
# ignore_eos stays on (default) => each request emits exactly its gold-patch
# token count: the SWE-bench length distribution, deterministic, and identical
# per-request across arms (so the A/B stays clean; only the length PROFILE
# changes from flat-256 to realistic).
#
# Arms: dense (no HiSparse), uniform, dp. (HiSparse+MTP is excluded: it crashes
# in _build_hisparse_decode_batch -> stash, topk_p=None.)
#
# Override VARs via `sbatch --export=ALL,VAR=... swebench_outlen_sweep.sbatch`.

set -o pipefail

SGLANG_DIR="/scratch/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/yueyingl/ai-horizon/avo-related/avo/sglang"
PY="$SGLANG_DIR/.venv/bin/python"
TRACE_PKG="sglang.srt.mem_cache.sparsity.trace"

MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/9e165c30e2704aec5d9d593cce3eebd58bbef1cb}"
HF_CACHE="${HF_CACHE:-/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/hf_cache}"

HOST="127.0.0.1"
PORT_DENSE="${PORT_DENSE:-31000}"
PORT_UNIFORM="${PORT_UNIFORM:-33000}"
PORT_DP="${PORT_DP:-35000}"

TP="${TP:-4}"
DP="${DP:-4}"
TOP_K="${TOP_K:-512}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-262144}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-256}"

# Low-concurrency focus; 100 prompts keeps the variable-output tail bounded in 2h.
CONCURRENCY_LIST="${CONCURRENCY_LIST:-1 2 4 5}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
# OUTPUT_LEN empty => follow the dataset (gold-patch) length per request.
# Set to a positive int to force a flat output length instead.
OUTPUT_LEN="${OUTPUT_LEN:-}"

# Which arms to run.
ARMS="${ARMS:-dense uniform dp}"

RESULTS_ROOT="$SGLANG_DIR/python/sglang/srt/mem_cache/sparsity/trace/results"
DP_CSV="${DP_CSV:-$RESULTS_ROOT/3191572/per_layer_budget_dp/dp_allocations.csv}"
SWE_JSONL="${SWE_JSONL:-$RESULTS_ROOT/_shared/swe_bench_300.jsonl}"
DP_TOTAL_ROW="${DP_TOTAL_ROW:-0.3}"
B_MEAN="${B_MEAN:-1024}"
PAGE_SIZE_SLOTS="${PAGE_SIZE_SLOTS:-64}"

OUT_DIR="${OUT_DIR:-$RESULTS_ROOT/swelen_${SLURM_JOB_ID:-local}}"
CFG_DIR="$OUT_DIR/buffer_configs"
mkdir -p "$OUT_DIR" "$CFG_DIR"
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
echo "=== output length: $([ -z "$OUTPUT_LEN" ] && echo 'FOLLOW SWE-bench gold-patch (variable)' || echo "flat $OUTPUT_LEN") ==="
nvidia-smi || { echo "no GPUs"; exit 1; }
[ -s "$SWE_JSONL" ] || { echo "missing dataset $SWE_JSONL"; exit 1; }
[ -s "$DP_CSV" ]    || { echo "missing dp csv $DP_CSV"; exit 1; }

# ---- build matched uniform & DP per-layer buffer configs -------------------
echo "=== building buffer configs from $DP_CSV (row $DP_TOTAL_ROW, B_mean $B_MEAN) ==="
"$PY" -m ${TRACE_PKG}.make_buffer_sizes \
  --dp-csv "$DP_CSV" --total-ratio "$DP_TOTAL_ROW" \
  --b-mean "$B_MEAN" --top-k "$TOP_K" --page-size "$PAGE_SIZE_SLOTS" \
  --out-dir "$CFG_DIR" || { echo "make_buffer_sizes failed"; exit 1; }
UNIFORM_CFG="$CFG_DIR/buffer_sizes_uniform.json"
DP_CFG="$CFG_DIR/buffer_sizes_dp.json"

# ---------------------------------------------------------------------------
# run_arm <arm_name> <base_port> <extra launch flags...>
# ---------------------------------------------------------------------------
run_arm() {
  local ARM="$1" PORT="$2"; shift 2
  local FLAGS=("$@")
  local BASE_URL="http://${HOST}:${PORT}"
  echo "======================================================================"
  echo "=== ARM=$ARM  port=$PORT  flags=[${FLAGS[*]}]  ($(date)) ==="
  echo "======================================================================"
  local ARM_DIR="$OUT_DIR/$ARM"
  mkdir -p "$ARM_DIR"

  setsid "$PY" -m sglang.launch_server \
    --model-path "$MODEL_PATH" --trust-remote-code \
    --tp-size "$TP" --dp-size "$DP" --enable-dp-attention \
    --disable-radix-cache --disable-custom-all-reduce \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --enable-metrics \
    "${FLAGS[@]}" \
    --host "$HOST" --port "$PORT" \
    > "$ARM_DIR/server.log" 2>&1 &
  local SERVER_PID=$!

  echo "=== [$ARM] waiting for readiness (pid $SERVER_PID) ==="
  local READY=0 i
  for i in $(seq 1 160); do
    if curl -sf "${BASE_URL}/get_model_info" >/dev/null 2>&1; then READY=1; break; fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
      echo "[$ARM] server died; see $ARM_DIR/server.log"; return 1
    fi
    sleep 15
  done
  if [ "$READY" != "1" ]; then
    echo "[$ARM] not ready in time"; kill -- -$SERVER_PID 2>/dev/null; return 1
  fi
  echo "=== [$ARM] server ready ($(date)) ==="

  grep -i "HiSparse per-layer device buffer sizes" "$ARM_DIR/server.log" | tail -1 \
    > "$ARM_DIR/resolved_buffer_sizes.txt" 2>/dev/null || true
  curl -sf "${BASE_URL}/get_server_info" > "$ARM_DIR/server_info.json" 2>/dev/null || true

  local c
  for c in $CONCURRENCY_LIST; do
    # Build the output-length arg: omit for dataset-following, else flat.
    local OUTLEN_ARGS=()
    if [ -n "$OUTPUT_LEN" ]; then
      OUTLEN_ARGS=(--sharegpt-output-len "$OUTPUT_LEN")
    fi
    echo "=== [$ARM] bench c=$c prompts=$NUM_PROMPTS out=$([ -z "$OUTPUT_LEN" ] && echo dataset || echo "$OUTPUT_LEN") ($(date)) ==="
    "$PY" -m sglang.benchmark.serving \
      --backend sglang --base-url "$BASE_URL" \
      --dataset-name custom --dataset-path "$SWE_JSONL" \
      --num-prompts "$NUM_PROMPTS" --max-concurrency "$c" --request-rate inf \
      "${OUTLEN_ARGS[@]}" \
      --warmup-requests 2 \
      --output-file "$ARM_DIR/bench_c${c}.jsonl" \
      > "$ARM_DIR/bench_c${c}.log" 2>&1 \
      || echo "[$ARM] bench c=$c returned nonzero (see log)"
  done

  echo "=== [$ARM] stopping server (full process-group teardown) ==="
  kill -- -$SERVER_PID 2>/dev/null
  sleep 5
  kill -9 -- -$SERVER_PID 2>/dev/null
  pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
  for i in $(seq 1 24); do
    curl -sf "http://${HOST}:${PORT}/get_model_info" >/dev/null 2>&1 || break
    sleep 5
  done
  sleep 10
  return 0
}

HS_UNIFORM=$(printf '{"top_k":%d,"device_buffer_sizes_path":"%s"}' "$TOP_K" "$UNIFORM_CFG")
HS_DP=$(printf '{"top_k":%d,"device_buffer_sizes_path":"%s"}' "$TOP_K" "$DP_CFG")

for arm in $ARMS; do
  case "$arm" in
    dense)   run_arm dense   "$PORT_DENSE" ;;
    uniform) run_arm uniform "$PORT_UNIFORM" --enable-hisparse --hisparse-config "$HS_UNIFORM" ;;
    dp)      run_arm dp      "$PORT_DP" --enable-hisparse --hisparse-config "$HS_DP" ;;
    *)       echo "unknown arm '$arm', skipping" ;;
  esac
done

echo "=== DONE ($(date)) ==="
ls -R "$OUT_DIR"
