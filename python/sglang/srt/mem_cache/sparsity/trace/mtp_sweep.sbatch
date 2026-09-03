#!/bin/bash
#SBATCH --job-name=hisparse_mtp
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
# HiSparse x MTP composability check. HiSparse composes with MTP (verified in
# code): no validation blocks the combo, the pool configurator sizes for BOTH
# the HiSparse host->device shrink and the speculative draft inflation, and the
# HiSparse "1-token-per-decode-step" coordinator logic is bypassed under spec
# (spec verify runs as TARGET_VERIFY, an extend mode, not decode). We use the
# EAGLE/NEXTN MTP head (auto-selected from the DSv4-Flash checkpoint, no draft
# path needed) because it PRESERVES --enable-dp-attention (DSpark would require
# DP-attention off).
#
# Two arms, SAME HiSparse DP per-layer buffer, MATCHED memory; MTP is the only
# difference:
#   hisparse    = HiSparse + DP per-layer buffer                 (baseline)
#   hisparse_mtp= HiSparse + DP per-layer buffer + EAGLE/NEXTN MTP (treatment)
#
# Low-concurrency focus (MTP helps most when the batch is small / latency-bound
# and there is decode headroom). Same dataset + fixed output length as the A/B
# sweep so points are directly comparable.
#
# Override VARs via `sbatch --export=ALL,VAR=... mtp_sweep.sbatch`.

set -o pipefail

SGLANG_DIR="/scratch/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/yueyingl/ai-horizon/avo-related/avo/sglang"
PY="$SGLANG_DIR/.venv/bin/python"
TRACE_PKG="sglang.srt.mem_cache.sparsity.trace"

MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/9e165c30e2704aec5d9d593cce3eebd58bbef1cb}"
HF_CACHE="${HF_CACHE:-/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/hf_cache}"

HOST="127.0.0.1"
# Distinct base ports per arm so DP-attention rpc_port ranges never collide.
PORT_HS="${PORT_HS:-31000}"
PORT_MTP="${PORT_MTP:-33000}"

TP="${TP:-4}"
DP="${DP:-4}"
TOP_K="${TOP_K:-512}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-262144}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-256}"

# EAGLE/NEXTN MTP knobs (no draft-model-path: DeepseekV4ForCausalLMNextN is
# auto-selected from the same checkpoint). eagle-topk 1 = linear NEXTN chain.
SPEC_STEPS="${SPEC_STEPS:-3}"
SPEC_TOPK="${SPEC_TOPK:-1}"
SPEC_DRAFT_TOKENS="${SPEC_DRAFT_TOKENS:-4}"

# Low-concurrency focus (1,2,4,5) per the ask; MTP payoff concentrates here.
CONCURRENCY_LIST="${CONCURRENCY_LIST:-1 2 4 5}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"

RESULTS_ROOT="$SGLANG_DIR/python/sglang/srt/mem_cache/sparsity/trace/results"
DP_CSV="${DP_CSV:-$RESULTS_ROOT/3191572/per_layer_budget_dp/dp_allocations.csv}"
SWE_JSONL="${SWE_JSONL:-$RESULTS_ROOT/_shared/swe_bench_300.jsonl}"
DP_TOTAL_ROW="${DP_TOTAL_ROW:-0.3}"
B_MEAN="${B_MEAN:-1024}"
PAGE_SIZE_SLOTS="${PAGE_SIZE_SLOTS:-64}"

OUT_DIR="${OUT_DIR:-$RESULTS_ROOT/mtp_${SLURM_JOB_ID:-local}}"
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
nvidia-smi || { echo "no GPUs"; exit 1; }
[ -s "$SWE_JSONL" ] || { echo "missing dataset $SWE_JSONL"; exit 1; }
[ -s "$DP_CSV" ]    || { echo "missing dp csv $DP_CSV"; exit 1; }

# ---- build the DP per-layer buffer config (both arms share it) -------------
echo "=== building DP buffer config from $DP_CSV (row $DP_TOTAL_ROW, B_mean $B_MEAN) ==="
"$PY" -m ${TRACE_PKG}.make_buffer_sizes \
  --dp-csv "$DP_CSV" --total-ratio "$DP_TOTAL_ROW" \
  --b-mean "$B_MEAN" --top-k "$TOP_K" --page-size "$PAGE_SIZE_SLOTS" \
  --out-dir "$CFG_DIR" || { echo "make_buffer_sizes failed"; exit 1; }
DP_CFG="$CFG_DIR/buffer_sizes_dp.json"

# ---------------------------------------------------------------------------
# run_arm <arm_name> <base_port> <extra spec flags...>
#   Launches a HiSparse server (DP per-layer buffer) with the given extra
#   speculative flags, sweeps concurrency, then fully tears down.
# ---------------------------------------------------------------------------
run_arm() {
  local ARM="$1" PORT="$2"; shift 2
  local SPEC_FLAGS=("$@")
  local BASE_URL="http://${HOST}:${PORT}"
  echo "======================================================================"
  echo "=== ARM=$ARM  port=$PORT  spec_flags=[${SPEC_FLAGS[*]}]  ($(date)) ==="
  echo "======================================================================"
  local ARM_DIR="$OUT_DIR/$ARM"
  mkdir -p "$ARM_DIR"

  local HS_CFG
  HS_CFG=$(printf '{"top_k":%d,"device_buffer_sizes_path":"%s"}' "$TOP_K" "$DP_CFG")

  setsid "$PY" -m sglang.launch_server \
    --model-path "$MODEL_PATH" --trust-remote-code \
    --tp-size "$TP" --dp-size "$DP" --enable-dp-attention \
    --enable-hisparse --hisparse-config "$HS_CFG" \
    --disable-radix-cache --disable-custom-all-reduce \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --enable-metrics \
    "${SPEC_FLAGS[@]}" \
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

# Baseline (HiSparse only), then treatment (HiSparse + MTP). Distinct ports.
run_arm hisparse     "$PORT_HS"
run_arm hisparse_mtp "$PORT_MTP" \
  --speculative-algorithm EAGLE \
  --speculative-num-steps "$SPEC_STEPS" \
  --speculative-eagle-topk "$SPEC_TOPK" \
  --speculative-num-draft-tokens "$SPEC_DRAFT_TOKENS"

echo "=== DONE ($(date)) ==="
ls -R "$OUT_DIR"
