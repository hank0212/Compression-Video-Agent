#!/bin/bash
# Serve LongVT-RFT (Qwen2.5-VL-7B fine-tune) for the interleaved tool-calling demo.
# Reuses the box's existing `vllm` env (vLLM 0.16). Runs on ONE GPU slice.
set -e

GPU=${1:-5}
PORT=${2:-8000}
CKPT=${CKPT:-/local1/cfyang/LongVT-RFT}
REPO=/home/cfyang/hanklin/LongVT
TEMPLATE=$REPO/examples/eval/tool_call_qwen2_5_vl.jinja

# RESEARCH.md footgun: a stale VIRTUAL_ENV hijacks `conda run -n vllm` -> wrong env.
unset VIRTUAL_ENV
export CUDA_VISIBLE_DEVICES=$GPU
export no_proxy=localhost,127.0.0.1
export VLLM_LOGGING_LEVEL=INFO

# GPU5 has ~23GB held by leftover qwen35 procs -> only ~24.5GB free.
# vLLM needs (util*total) FREE at startup; 0.48*47GiB=22.7GiB fits.
echo "[serve] GPU=$GPU PORT=$PORT CKPT=$CKPT"
exec conda run -n vllm --no-capture-output vllm serve "$CKPT" \
  --served-model-name LongVT-RFT \
  --tool-call-parser hermes \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --chat-template "$TEMPLATE" \
  --port "$PORT" \
  --gpu-memory-utilization 0.48 \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 600}' \
  --max-num-seqs 2
