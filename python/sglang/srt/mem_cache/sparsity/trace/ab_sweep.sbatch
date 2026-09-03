#!/bin/bash
#SBATCH --job-name=hisparse_ab
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
# A/B serving benchmark: HiSparse with a UNIFORM per-layer device buffer
# (baseline) vs. the DP per-layer allocation (treatment), at MATCHED total
# device memory. For each arm we launch the DSv4-Flash HiSparse server with that
# per-layer buffer config and sweep concurrency, forcing an identical fixed
# output length so both arms and all loads are directly comparable.
#
# Outputs one bench_serving JSONL per (arm, concurrency) plus an aggregated
# summary.csv (req throughput, token throughput, P50/P90/P99 TTFT/TPOT/E2E).
#
# Override VARs via `sbatch --export=ALL,VAR=... ab_sweep.sbatch`.

set -o pipefail

SGLANG_DIR="/scratch/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/yueyingl/ai-horizon/avo-related/avo/sglang"
PY="$SGLANG_DIR/.venv/bin/python"
TRACE_PKG="sglang.srt.mem_cache.sparsity.trace"

MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/9e165c30e2704aec5d9d593cce3eebd58bbef1cb}"
HF_CACHE="${HF_CACHE:-/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/hf_cache}"

HOST="127.0.0.1"
# Each arm uses a DISTINCT base port so the DP-attention rpc_port ranges
# (derived from the base port) never collide across arms even if a previous
# server's subprocesses are slow to release their sockets.
PORT_UNIFORM="${PORT_UNIFORM:-31000}"
PORT_DP="${PORT_DP:-33000}"

TP="${TP:-4}"
DP="${DP:-4}"
TOP_K="${TOP_K:-512}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-262144}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-256}"

# Load sweep + fixed output length (identical work per request across arms).
CONCURRENCY_LIST="${CONCURRENCY_LIST:-5 20 50 100 200}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"

# Inputs from the trace run: the DP allocation CSV + the (short) SWE dataset.
RESULTS_ROOT="$SGLANG_DIR/python/sglang/srt/mem_cache/sparsity/trace/results"
DP_CSV="${DP_CSV:-$RESULTS_ROOT/3191572/per_layer_budget_dp/dp_allocations.csv}"
SWE_JSONL="${SWE_JSONL:-$RESULTS_ROOT/_shared/swe_bench_300.jsonl}"
DP_TOTAL_ROW="${DP_TOTAL_ROW:-0.3}"     # which dp_allocations row to use
B_MEAN="${B_MEAN:-1024}"                 # uniform per-layer buffer (compressed slots)
PAGE_SIZE_SLOTS="${PAGE_SIZE_SLOTS:-64}"

OUT_DIR="${OUT_DIR:-$RESULTS_ROOT/ab_${SLURM_JOB_ID:-local}}"
CFG_DIR="$OUT_DIR/buffer_configs"
mkdir -p "$OUT_DIR" "$CFG_DIR"
cd "$SGLANG_DIR" || exit 1

# ---- environment fixes (identical to the working trace run) ----------------
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
# run_arm <arm_name> <buffer_sizes_json> <base_port>
#   launches the server (on its own base port so rpc_port ranges never collide
#   across arms), sweeps concurrency, writes one JSONL per load, then fully
#   tears down the server + all its subprocesses before returning.
# ---------------------------------------------------------------------------
run_arm() {
  local ARM="$1" CFG="$2" PORT="$3"
  local BASE_URL="http://${HOST}:${PORT}"
  echo "======================================================================"
  echo "=== ARM=$ARM  port=$PORT  buffer_config=$CFG  ($(date)) ==="
  echo "======================================================================"
  local ARM_DIR="$OUT_DIR/$ARM"
  mkdir -p "$ARM_DIR"

  local HS_CFG
  HS_CFG=$(printf '{"top_k":%d,"device_buffer_sizes_path":"%s"}' "$TOP_K" "$CFG")

  setsid "$PY" -m sglang.launch_server \
    --model-path "$MODEL_PATH" --trust-remote-code \
    --tp-size "$TP" --dp-size "$DP" --enable-dp-attention \
    --enable-hisparse --hisparse-config "$HS_CFG" \
    --disable-radix-cache --disable-custom-all-reduce \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --enable-metrics \
    --host "$HOST" --port "$PORT" \
    > "$ARM_DIR/server.log" 2>&1 &
  local SERVER_PID=$!

  echo "=== [$ARM] waiting for readiness (pid $SERVER_PID) ==="
  local READY=0 i
  for i in $(seq 1 120); do
    if curl -sf "${BASE_URL}/get_model_info" >/dev/null 2>&1; then READY=1; break; fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
      echo "[$ARM] server died; see $ARM_DIR/server.log"; return 1
    fi
    sleep 15
  done
  if [ "$READY" != "1" ]; then
    echo "[$ARM] not ready in time"
    kill -- -$SERVER_PID 2>/dev/null
    return 1
  fi
  echo "=== [$ARM] server ready ($(date)) ==="

  # Record the per-layer config the server actually resolved (from its log).
  grep -i "HiSparse per-layer device buffer sizes" "$ARM_DIR/server.log" | tail -1 \
    > "$ARM_DIR/resolved_buffer_sizes.txt" 2>/dev/null || true

  local c
  for c in $CONCURRENCY_LIST; do
    echo "=== [$ARM] bench concurrency=$c num_prompts=$NUM_PROMPTS out_len=$OUTPUT_LEN ($(date)) ==="
    "$PY" -m sglang.benchmark.serving \
      --backend sglang --base-url "$BASE_URL" \
      --dataset-name custom --dataset-path "$SWE_JSONL" \
      --num-prompts "$NUM_PROMPTS" --max-concurrency "$c" --request-rate inf \
      --sharegpt-output-len "$OUTPUT_LEN" \
      --warmup-requests 2 \
      --output-file "$ARM_DIR/bench_c${c}.jsonl" \
      > "$ARM_DIR/bench_c${c}.log" 2>&1 \
      || echo "[$ARM] bench c=$c returned nonzero (see log)"
  done

  echo "=== [$ARM] stopping server (full process-group teardown) ==="
  # Kill the whole process group (setsid gave the server its own group) so all
  # DP/TP scheduler subprocesses and their sockets are released.
  kill -- -$SERVER_PID 2>/dev/null
  sleep 5
  kill -9 -- -$SERVER_PID 2>/dev/null
  # Belt-and-suspenders: reap any stray launch_server for this arm's port.
  pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
  # Wait until the base port is actually free before returning.
  for i in $(seq 1 24); do
    if ! curl -sf "http://${HOST}:${PORT}/get_model_info" >/dev/null 2>&1; then
      break
    fi
    sleep 5
  done
  sleep 10
  return 0
}

# Baseline first (uniform), then treatment (DP). Same server args except the
# per-layer buffer config -> a clean A/B at matched total memory. Distinct
# ports guarantee no rpc_port collision between the two servers.
run_arm uniform "$UNIFORM_CFG" "$PORT_UNIFORM"
run_arm dp      "$DP_CFG"      "$PORT_DP"

# ---- aggregate all JSONL summaries into one CSV ----------------------------
echo "=== aggregating results -> $OUT_DIR/summary.csv ==="
"$PY" -m ${TRACE_PKG}.aggregate_ab --results-dir "$OUT_DIR" --out "$OUT_DIR/summary.csv" \
  || echo "aggregation failed (raw JSONL still available under $OUT_DIR)"

echo "=== DONE ($(date)) ==="
ls -R "$OUT_DIR"
