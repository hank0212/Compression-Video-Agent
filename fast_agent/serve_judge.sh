#!/usr/bin/env bash
# Single-GPU vLLM server for the case_notes reader VLM (Qwen3-VL-8B).
#
# Judge payloads are small (<=8 montage PNGs + a few k text tokens), so 32k ctx
# on ONE GPU is plenty — unlike serve_qwen3vl_baseline.sh's DP3/131k setup.
# Serves the LOCAL snapshot path (no hub lookup; /local1 is 97% full).
#
# Usage: ./serve_judge.sh [GPU] [PORT]
#   GPU   single CUDA device index (default 5)
#   PORT  default 8010 (repo convention; judge_client reads $OPENAI_BASE_URL)
#   GPU_UTIL=0.55 ./serve_judge.sh 5   # shrink footprint on a shared GPU
#
# Shared-box etiquette: check `nvidia-smi` for a GPU with >=25 GB free BEFORE
# launching; never kill other users' processes (see CLAUDE.md GPU rules).
set -euo pipefail

GPU=${1:-5}
PORT=${2:-8010}
MODEL=${MODEL:-/local1/cfyang/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b}
GPU_UTIL=${GPU_UTIL:-0.55}

unset VIRTUAL_ENV   # stale venv hijacks conda envs (see serve_qwen3vl_baseline.sh)
export CUDA_VISIBLE_DEVICES=$GPU
export no_proxy=localhost,127.0.0.1
export VLLM_LOGGING_LEVEL=INFO

# Absolute binary: `conda run -n vllm vllm` can silently resolve to dvd_tool's
# vllm 0.16 via PATH (same footgun documented in serve_qwen3vl_baseline.sh).
VLLM_BIN=/local1/cfyang/miniconda3/envs/vllm/bin/vllm

exec "$VLLM_BIN" serve "$MODEL" \
  --served-model-name qwen3-vl-8b-judge \
  --port "$PORT" \
  --trust-remote-code \
  --max-model-len 32768 \
  --limit-mm-per-prompt.image 16 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-seqs 4
