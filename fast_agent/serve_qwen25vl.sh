#!/usr/bin/env bash
# Qwen2.5-VL-7B-Instruct, plain -- the L0 harness check ONLY.
#
#   ./serve_qwen25vl.sh [GPU] [PORT]      # default GPU 3, :8033
#
# WHY THIS SERVER EXISTS
# ----------------------
# Every other arm in this chapter is internally comparable but has no published
# counterpart, so a wrong harness would look like a result. L0 is the one arm that
# does have a published number to hit: the LongVT paper's Table 2 reports
# Qwen2.5-VL-7B at **30.7** on LVBench under sparse (64-frame) sampling, scored with
# upstream lmms-eval's `lvbench` task. Reproducing that validates the plumbing the
# video arms also use -- LVBench loader, option parsing, frame sampling, extractor.
#
# Pair it with:
#   run_plain.py --dataset lvbench --prompt-style lmms_eval --frames 64
# which forces T=0, max_tokens=16 and PNG frames to match lmms-eval's own defaults.
#
# SIZING: 64 images at 224^2. Qwen2.5-VL is patch 14 / merge 2 -> 28x28 px per token,
# so 50176/784 = 64 tok per frame -> 4,096 visual tokens. 16384 ctx is ~4x headroom.
# No tool-calling flags: this arm has no tools, and enabling the hermes parser would
# be a gratuitous difference from the task it is reproducing.
set -euo pipefail
PY=/local1/cfyang/miniconda3/envs/vllm/bin
# NOTE: /local1/cfyang/models--Qwen--Qwen2.5-VL-7B-Instruct/ (the path in CLAUDE.md) is a
# STUB -- its snapshot dir holds config.json and nothing else, so vLLM starts and then dies
# in the tokenizer with "expected str, bytes or os.PathLike object, not NoneType". The real
# 16 GB checkout lives in the HF hub cache below.
MODEL=/local1/cfyang/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5
LOG=/local1/cfyang/hanklin/outputs/lvbench_plain
mkdir -p "$LOG"

GPU=${1:-3}
PORT=${2:-8033}
# Shared box: GPU 3 carries ~4 GB of someone else's work. 0.75 of 49 GB leaves them
# room and is still ~2x what 7B weights + this KV cache need. Check nvidia-smi first
# (CLAUDE.md), and never SIGKILL anything holding a CUDA context.
GPU_UTIL=${GPU_UTIL:-0.75}

echo "qwen2.5-vl-7b gpu=$GPU port=$PORT util=$GPU_UTIL"
CUDA_VISIBLE_DEVICES=$GPU nohup "$PY/vllm" serve "$MODEL" \
  --port "$PORT" \
  --served-model-name qwen25vl \
  --max-model-len 16384 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --limit-mm-per-prompt '{"image":128}' \
  --mm-processor-kwargs '{"max_pixels":50176,"min_pixels":3136}' \
  --mm-processor-cache-gb 0 \
  >> "$LOG/server_qwen25vl.log" 2>&1 &
echo "pid $! -> $LOG/server_qwen25vl.log"
