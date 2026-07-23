#!/usr/bin/env bash
# Serve zero-shot Qwen3-VL-8B-Instruct as the model-under-test for the LongVT
# tool-calling eval harness (lmms-eval async_openai + MCP crop_video / compress_video).
#
# Qwen3-VL note: NO --chat-template — the model ships a native tool-call
# template (run_eval.sh only injects tool_call_qwen2_5_vl.jinja when
# IS_QWEN3_VL=False). Keep hermes parser + auto tool choice.
#
# Frames arrive as individual base64 image_urls (protocol.to_qwen3_vl_openai_messages),
# not video_url: 768 global frames + <=128/crop round => raise the per-prompt
# image limit and context length accordingly.
#
# FLASHVID_COMPRESS=1 additionally enables the vLLM plugin (_archive/flashvid_plugin)
# that swaps EVS token selection for FlashVID DySeg+ADTS on the VIDEO-modality path
# only (--video-pruning-rate). crop_video (images) is completely unaffected — this
# is the mechanism that lets one server carry both bandwidths, selected per tool call.
# See longvt_compression/FLASHVID_TOOL_PLAN.md.
#
# Usage: ./serve_qwen3vl_baseline.sh [GPUS] [PORT]
#   GPUS  comma list for CUDA_VISIBLE_DEVICES (default 3,4 -> DP=2)
#   PORT  default 8010
#   FLASHVID_COMPRESS=1 FLASHVID_RETENTION=0.3 ./serve_qwen3vl_baseline.sh ...
set -euo pipefail

GPUS=${1:-3,4}
PORT=${2:-8010}
MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
DP=$(awk -F, '{print NF}' <<<"$GPUS")

unset VIRTUAL_ENV   # stale venv hijacks `conda run` (see serve_longvt.sh)
export CUDA_VISIBLE_DEVICES=$GPUS
export no_proxy=localhost,127.0.0.1
export VLLM_LOGGING_LEVEL=INFO

# `conda run -n vllm vllm serve` is NOT safe: the login profile puts the
# dvd_tool env first on PATH, and conda run's own subprocess resolution
# follows it, silently launching dvd_tool's vllm 0.16.0 instead of this env's
# 0.19.0 (confirmed via /proc/<pid>/exe — same footgun as the eval-side
# accelerate launch). flashvid_vllm is only installed into THIS env, so that
# swap silently no-ops the compression plugin. Use the absolute binary.
VLLM_BIN=/local1/cfyang/miniconda3/envs/vllm/bin/vllm

FLASHVID_ARGS=()
if [ "${FLASHVID_COMPRESS:-0}" = "1" ]; then
  PLUGIN_DIR=/home/cfyang/hanklin/_archive/flashvid_plugin
  touch "$PLUGIN_DIR/FLASHVID_ON"
  export FLASHVID_SENTINEL="$PLUGIN_DIR/FLASHVID_ON"
  export FLASHVID_TSM=${FLASHVID_TSM:-attn_div_v2}
  RETENTION=${FLASHVID_RETENTION:-0.3}
  # video_pruning_rate is the FRACTION DROPPED (EVS convention) -> 1 - retention.
  PRUNE_RATE=$(awk -v r="$RETENTION" 'BEGIN{print 1-r}')
  FLASHVID_ARGS=(--video-pruning-rate "$PRUNE_RATE" --limit-mm-per-prompt.video 4)
  echo "FlashVID compression ON: retention=$RETENTION (video-pruning-rate=$PRUNE_RATE), sentinel=$PLUGIN_DIR/FLASHVID_ON"
fi

exec "$VLLM_BIN" serve "$MODEL" \
  --data-parallel-size "$DP" \
  --port "$PORT" \
  --tool-call-parser hermes \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --max-model-len 131072 \
  --limit-mm-per-prompt.image 2048 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 6 \
  "${FLASHVID_ARGS[@]}"
