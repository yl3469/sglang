#!/bin/bash
#SBATCH --job-name=glm_aalcr_trace
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
# GLM-5.2 + AA-LCR trace collection -> per-layer DP allocation.
# Mirrors collect_trace.sbatch (DSv4-Flash + SWE-bench) but retargeted:
#   * MODEL  = GLM-5.2 (GlmMoeDsaForCausalLM, model_type glm_moe_dsa) -- a DSA
#              model, so --enable-hisparse is valid (hisparse_hook.py:96 names GLM-5).
#   * DATA   = AA-LCR (71k-114k input tokens) -- HiSparse's true target regime.
#
# GLM-5.2 structure (from config.json): 78 layers, ALL deepseek_sparse_attention;
# indexer_types has 21 "full" layers at positions 0,1,2,6,10,...,74 and 57
# "shared". index_topk=2048 (4x DSv4's 512). Because the full-indexer layers are
# NOT a clean offset+stride, the DP CSV layer mapping is left to inspect from the
# real captured trace: the smoke step below prints the actual per-request trace
# shape (n_indexer_layers x topk), and Stage 3+4 is run with NUM_INDEXER_LAYERS
# taken from that. Set NUM_INDEXER_LAYERS explicitly to override.
#
# Override VARs via `sbatch --export=ALL,VAR=... collect_trace_glm_aalcr.sbatch`.

set -o pipefail

SGLANG_DIR="/scratch/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/yueyingl/ai-horizon/avo-related/avo/sglang"
PY="$SGLANG_DIR/.venv/bin/python"
TRACE_PKG="sglang.srt.mem_cache.sparsity.trace"

# GLM-5.2 NVFP4 (native NVIDIA quant, fits 4xGB300; smaller than FP8).
MODEL_PATH="${MODEL_PATH:-/scratch/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/yueyingl/models/GLM-5.2-NVFP4}"
HF_CACHE="${HF_CACHE:-/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/hf_cache}"

HOST="127.0.0.1"
PORT="${PORT:-30000}"
BASE_URL="http://${HOST}:${PORT}"

# DP=2 for trace collection. RATIONALE across three OOM/fail iterations:
#  * The indexer capturer's PINNED host buffer is max_total_tokens *
#    num_hidden_layers * index_topk * 4B, allocated PER DP RANK. For GLM (DSA ->
#    num_indexer_layers = ALL 78 layers, model_config.py:236; index_topk=2048)
#    that is 131072*78*2048*4B ~= 84 GB PER RANK.
#  * DP=4 (job 3222826) -> 4 x 84 = 335 GB capturer + HiCache + 465 GB weights
#    OOM'd the 900 GB node when the first 84k-token prompt filled the capturer.
#  * DP=1/TP=4 (job 3224331) FAILED HARD: IndexerTopkCapturer asserts
#    attn_tp_size == 1 (indexer_topk.py:30) -> the capturer REQUIRES DP attention,
#    so DP>1 is mandatory whenever --enable-return-indexer-topk is on.
#  * DP=2 is the sweet spot: 2 x 84 = 168 GB capturer + 2 x ~27 GB HiCache
#    (~54 GB) + 465 GB weights ~= 690 GB < 900 GB. TP=2 within each DP group.
#    AA-LCR prompts (71-114k tok) still need max_total_tokens >= 131072 to fit
#    the longest prompt, so that stays.
TP="${TP:-2}"
DP="${DP:-2}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
INDEX_TOPK="${INDEX_TOPK:-2048}"
# 21 "full" indexer layers in GLM-5.2 (see header). Overridable; the smoke step
# validates against the real captured trace shape.
NUM_INDEXER_LAYERS="${NUM_INDEXER_LAYERS:-21}"
# AA-LCR prompts run up to ~114k input tokens, so the pool must exceed that to
# serve the longest prompt at concurrency 1 -> 131072 is the floor.
#
# HOST-RAM OOM FIX (job 3217394 died here, 900GB node): the
# --enable-return-indexer-topk capturer allocates a PINNED host buffer of
# max_total_tokens * NUM_INDEXER_LAYERS * INDEX_TOPK * 4B PER RANK. At 262144 x
# 21 x 2048 x 4B that is ~45 GB/rank x 4 DP = ~180 GB, stacked on the ~107 GB
# HiCache host pool (26.83 GB x 4) and the 465 GB NVFP4 weights transiting host
# during load -> >750 GB -> OOM. GLM's INDEX_TOPK=2048 makes the capturer 4x
# DSv4's (topk 512). Halving max_total_tokens to 131072 halves BOTH the capturer
# (~90 GB) and HiCache (~54 GB) while still covering the 114k-token max prompt.
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-131072}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"

RESULTS_DIR="${RESULTS_DIR:-$SGLANG_DIR/python/sglang/srt/mem_cache/sparsity/trace/results/glm_aalcr_${SLURM_JOB_ID:-local}}"
TRACE_DIR="$RESULTS_DIR/traces"
# Pre-generated on a login node (has HF network) by prep_aa_lcr.py.
AALCR_JSONL="${AALCR_JSONL:-$SGLANG_DIR/python/sglang/srt/mem_cache/sparsity/trace/results/_shared/aa_lcr_100.jsonl}"

mkdir -p "$RESULTS_DIR" "$TRACE_DIR"
cd "$SGLANG_DIR" || exit 1

export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# CUDA toolkit (bundled cu13; same as the working DSv4 run).
if [ -z "$CUDA_HOME" ]; then
  _CU13="$SGLANG_DIR/.venv/lib/python3.12/site-packages/nvidia/cu13"
  if [ -x "$_CU13/bin/nvcc" ]; then
    export CUDA_HOME="$_CU13"
    export PATH="$_CU13/bin:$PATH"
    export LD_LIBRARY_PATH="$_CU13/lib:$LD_LIBRARY_PATH"
    export LIBRARY_PATH="$_CU13/lib:$LIBRARY_PATH"
    [ -e "$_CU13/lib/libcudart.so" ] || ln -sf libcudart.so.13 "$_CU13/lib/libcudart.so" 2>/dev/null
    echo "=== CUDA_HOME=$CUDA_HOME ==="
  fi
fi
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+$NVCC_APPEND_FLAGS }-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export DG_JIT_USE_NVRTC="${DG_JIT_USE_NVRTC:-true}"
# NOTE: SGLANG_DSV4_FP4_DEQUANT is DSv4-specific; GLM uses its own NVFP4 path, so
# it is intentionally NOT set here. Add model-specific env only if startup asks.

echo "=== node $(hostname) start $(date) ==="
echo "=== MODEL=$MODEL_PATH ==="
nvidia-smi || { echo "no GPUs"; exit 1; }

[ -s "$AALCR_JSONL" ] || { echo "missing AA-LCR dataset $AALCR_JSONL (run prep_aa_lcr.py on a login node)"; exit 1; }
[ -d "$MODEL_PATH" ] || { echo "model dir not found: $MODEL_PATH"; exit 1; }

# ---------------------------------------------------------------------------
# Launch server (GLM-5.2 DSA + HiSparse).
# --enable-dp-attention only makes sense (and only replicates the capturer) when
# DP>1; for the default DP=1 trace-collection config it is omitted.
# ---------------------------------------------------------------------------
DP_ATTN_FLAG=()
if [ "$DP" -gt 1 ]; then DP_ATTN_FLAG=(--enable-dp-attention); fi
echo "=== launching server (TP=$TP DP=$DP) ==="
"$PY" -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --trust-remote-code \
  --tp-size "$TP" \
  --dp-size "$DP" \
  "${DP_ATTN_FLAG[@]}" \
  --weight-loader-drop-cache-after-load \
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

echo "=== waiting for readiness ==="
READY=0
for i in $(seq 1 160); do
  if curl -sf "${BASE_URL}/get_model_info" >/dev/null 2>&1; then READY=1; break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "server died; see $RESULTS_DIR/server.log"; exit 1
  fi
  sleep 15
done
[ "$READY" = "1" ] || { echo "server not ready in time"; exit 1; }
echo "=== server ready ($(date)) ==="

# Stage 2: KV budget.
echo "=== Stage 2: recording KV budget ==="
"$PY" -m ${TRACE_PKG}.record_kv_budget \
  --base-url "$BASE_URL" --out "$RESULTS_DIR/kv_budget.json" || true

# Stage 3-1 smoke: one request must return a non-empty indexer_topk. Print the
# captured trace shape so we can confirm NUM_INDEXER_LAYERS for GLM.
echo "=== smoke: 1 request, confirm indexer_topk present ==="
"$PY" -m sglang.benchmark.serving \
  --backend sglang --base-url "$BASE_URL" \
  --dataset-name custom --dataset-path "$AALCR_JSONL" \
  --num-prompts 1 --max-concurrency 1 --request-rate inf \
  --extra-request-body '{"return_indexer_topk": true}' \
  --indexer-trace-out "$TRACE_DIR" \
  --indexer-num-layers "$NUM_INDEXER_LAYERS" \
  --indexer-topk "$INDEX_TOPK" \
  --sharegpt-output-len 64 \
  || { echo "smoke request failed"; exit 1; }

if ! ls "$TRACE_DIR"/req_*.npz >/dev/null 2>&1; then
  echo "no indexer trace captured on smoke; check --enable-return-indexer-topk"; exit 1
fi
echo "=== captured trace shape ==="
"$PY" - "$TRACE_DIR" <<'PYEOF'
import glob, os, sys
import numpy as np
d = sys.argv[1]
f = sorted(glob.glob(os.path.join(d, "req_*.npz")))[0]
z = np.load(f)
for k in z.files:
    print(f"  {k}: shape={z[k].shape} dtype={z[k].dtype}")
PYEOF

# Stage 1: full collection over K AA-LCR prompts.
echo "=== Stage 1: collecting traces over $NUM_PROMPTS prompts ==="
"$PY" -m sglang.benchmark.serving \
  --backend sglang --base-url "$BASE_URL" \
  --dataset-name custom --dataset-path "$AALCR_JSONL" \
  --num-prompts "$NUM_PROMPTS" --max-concurrency 1 --request-rate inf \
  --extra-request-body '{"return_indexer_topk": true}' \
  --indexer-trace-out "$TRACE_DIR" \
  --indexer-num-layers "$NUM_INDEXER_LAYERS" \
  --indexer-topk "$INDEX_TOPK" \
  --sharegpt-output-len 128 \
  --output-file "$RESULTS_DIR/bench_serving.jsonl"

kill $SERVER_PID 2>/dev/null; trap - EXIT

# Stage 3+4: replay -> curves -> DP allocation CSV.
# GLM index_topk=2048; compress_ratio for the DSA c4 path is model-specific.
# --layer-offset/--layer-stride are used only to label CSV columns with model
# layer ids; GLM's full-indexer layers are non-uniform (0,1,2,6,10,...) so we
# leave the default trace-order labeling (0..N-1) unless overridden.
echo "=== Stage 3+4: per-layer DP allocation ==="
"$PY" -m ${TRACE_PKG}.run_per_layer_dp \
  "$TRACE_DIR" \
  --policy lru \
  --kv-budget "$RESULTS_DIR/kv_budget.json" \
  --out-dir "$RESULTS_DIR/per_layer_budget_dp" \
  || echo "DP stage returned nonzero (traces still saved under $TRACE_DIR)"

echo "=== DONE ($(date)) ==="
ls -R "$RESULTS_DIR"
