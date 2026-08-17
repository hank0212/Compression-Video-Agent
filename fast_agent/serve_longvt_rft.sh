#!/usr/bin/env bash
# LongVT-RFT (Qwen2.5-VL-7B fine-tune) served the same way our arm servers are, so the
# ONLY differences from run2 are the weights and the base model -- not the harness.
#
# Qwen2.5-VL needs the repo's tool-calling chat template; the stock one has no tool block
# (LongVT/examples/eval/run_eval.sh branches on exactly this).
#
# NOTE what this model does NOT get: Qwen2.5-VL emits no per-frame `<t seconds>` text
# markers (Qwen3-VL does). LongVT-RFT therefore has no textual time anchor at all -- its
# localization is a learned frame-index -> time mapping. That is the skill RFT trains, and
# it is the thing this run exists to measure.
set -euo pipefail
GPU=${1:-4}; PORT=${2:-8033}
CKPT=${CKPT:-/local1/cfyang/LongVT-RFT}
FRAMES=${FRAMES:-64}
PY=/local1/cfyang/miniconda3/envs/vllm/bin
LOG=/local1/cfyang/hanklin/outputs/lvbench_agent
TEMPLATE=/home/cfyang/hanklin/LongVT/examples/eval/tool_call_qwen2_5_vl.jinja

echo "[serve] LongVT-RFT gpu=$GPU port=$PORT frames=$FRAMES"
CUDA_VISIBLE_DEVICES=$GPU nohup "$PY/vllm" serve "$CKPT" \
  --served-model-name qwen3vl \
  --chat-template "$TEMPLATE" \
  --tool-call-parser hermes --enable-auto-tool-choice \
  --trust-remote-code \
  --port "$PORT" \
  --max-model-len 40960 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.88 \
  --limit-mm-per-prompt '{"image":768,"video":2}' \
  --mm-processor-kwargs '{"max_pixels":50176,"min_pixels":3136}' \
  --media-io-kwargs "{\"video\":{\"num_frames\":$FRAMES}}" \
  --allowed-local-media-path /local1/cfyang \
  --mm-processor-cache-gb 0 \
  >> "$LOG/server_longvt_rft.log" 2>&1 &
echo "pid $! -> $LOG/server_longvt_rft.log"
