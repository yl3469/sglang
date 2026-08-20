#!/bin/bash
# Launch DSv4-Flash-0731 with sglang HiSparse on the allocated GPU node.
# Usage: launch_server.sh <mode> where mode = capture | baseline | dp
set -eo pipefail
MODE=${1:?mode: capture|baseline|dp}
RUN=/home/yueyingl/hisparse_dp_run
MODEL=/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sglang
export CUDA_HOME=$CONDA_PREFIX
export PYTHONUNBUFFERED=1
PORT=30000

case $MODE in
  capture)
    HISPARSE_CFG=$(python -c "import json;print(json.dumps({
      'device_buffer_size': 4096, 'host_to_device_ratio': 5,
      'selection_capture': {'path': '$RUN/cap', 'max_steps_per_layer': 768},
      'memory_report_path': '$RUN/memreport'}))")
    EXTRA=(--disable-cuda-graph)
    ;;
  baseline)
    # uniform at the DP mean (equal effective LRU capacity vs dp arm)
    UNIFORM=$(python -c "import json;print(json.load(open('$RUN/dp.json'))['uniform_mean'])")
    HISPARSE_CFG=$(python -c "import json;print(json.dumps({
      'device_buffer_size': $UNIFORM, 'host_to_device_ratio': 5}))")
    EXTRA=()
    ;;
  dp)
    SERVE_DBS=$(python -c "import json;print(json.load(open('$RUN/dp.json'))['serve_device_buffer_size'])")
    HISPARSE_CFG=$(python -c "import json;print(json.dumps({
      'device_buffer_size': $SERVE_DBS, 'host_to_device_ratio': 5,
      'layer_buffer_profile': '@$RUN/dp.profile.json'}))")
    EXTRA=()
    ;;
esac

exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name dsv4-flash \
  --trust-remote-code \
  --tp 4 \
  --port $PORT \
  --page-size 64 \
  --max-running-requests 32 \
  --mem-fraction-static 0.85 \
  --disable-radix-cache \
  --kv-cache-dtype fp8_e4m3 \
  --moe-runner-backend deep_gemm \
  --disable-custom-all-reduce \
  --enable-hisparse \
  --hisparse-config "$HISPARSE_CFG" \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 64}' \
  "${EXTRA[@]}"
