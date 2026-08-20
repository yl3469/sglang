#!/bin/bash
# bench_serving sweeps against a running server.
# Usage: run_bench.sh <host> <arm-tag> [capture|sweep|long]
set -eo pipefail
TARGET_HOST=${1:?host}
TAG=${2:?arm tag}
MODE=${3:-sweep}
RUN=/home/yueyingl/hisparse_dp_run
MODEL=/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sglang
export PYTHONUNBUFFERED=1
mkdir -p "$RUN/bench"

common=(--backend sglang --host "$TARGET_HOST" --port 30000
        --model "$MODEL" --dataset-name random
        --random-input-len 24000 --random-output-len 256
        --random-range-ratio 0.8 --seed 42)

case $MODE in
  capture)
    # long-context, few prompts: enough decode steps to fill the capture cap
    python -m sglang.bench_serving "${common[@]}" \
      --num-prompts 8 --request-rate inf --max-concurrency 4 \
      --output-file "$RUN/bench/capture_drive.jsonl"
    ;;
  sweep)
    for rate in 0.5 1 2 4 8; do
      python -m sglang.bench_serving "${common[@]}" \
        --num-prompts 32 --request-rate "$rate" \
        --output-file "$RUN/bench/${TAG}_rate${rate}.jsonl"
    done
    ;;
  long)
    # 64k-token contexts (~16k compressed slots vs 3072-slot buffers):
    # the regime where host swap-in is actually on the critical path
    long_common=(--backend sglang --host "$TARGET_HOST" --port 30000
                 --model "$MODEL" --dataset-name random
                 --random-input-len 64000 --random-output-len 256
                 --random-range-ratio 0.9 --seed 42)
    for conc in 4 8 16; do
      python -m sglang.bench_serving "${long_common[@]}" \
        --num-prompts 16 --request-rate inf --max-concurrency "$conc" \
        --output-file "$RUN/bench/${TAG}_conc${conc}.jsonl"
    done
    ;;
esac
