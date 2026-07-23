#!/usr/bin/env bash
# Qwen3-VL-8B-Instruct tool-calling baseline + compression tool: same harness as
# run_qwen3vl_baseline.sh, but the agent has TWO tools — crop_video (full detail,
# images) and compress_video (wide overview, video_url -> engine-side FlashVID
# pruning when the server is started with FLASHVID_COMPRESS=1). See
# longvt_compression/FLASHVID_TOOL_PLAN.md / FLASHVID_INTEGRATION.md.
#
# Prereqs:
#   - server up WITH compression:
#       FLASHVID_COMPRESS=1 ./serve_qwen3vl_baseline.sh 3,4 8010
#   - same video symlinks / conda envs as run_qwen3vl_baseline.sh
#
# Usage: ./run_qwen3vl_compress.sh [TASK] [NPROC] [extra lmms_eval args...]
set -euo pipefail

TASK=${1:-videomme_long_reward_tool}
NPROC=${2:-4}
shift 2 2>/dev/null || shift $# || true

REPO=/home/cfyang/hanklin/longvt_compression
cd "$REPO"

export HF_HOME=/local1/cfyang/.cache/huggingface
export OPENAI_API_BASE="http://localhost:8010/v1"
export OPENAI_BASE_URL="http://localhost:8010/v1"
export OPENAI_MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
export OPENAI_API_KEY="EMPTY"
export USE_LLM_JUDGE=True
export DECORD_EOF_RETRY_MAX=409600
export no_proxy=localhost,127.0.0.1
export CUDA_VISIBLE_DEVICES=""
export LMMS_EVAL_USE_CACHE=True
export LMMS_EVAL_HOME=${LMMS_EVAL_HOME:-/local1/cfyang/hanklin/outputs/lmms_eval_home}

# ONLY difference from run_qwen3vl_baseline.sh: the FlashVID-aware MCP server
# (crop_video + compress_video) and video_tool_names telling the client which
# tool's result to attach as video_url instead of images.
MCP_PATH="./examples/video_tools/mcp_server_flashvid.py"
MODEL_VERSION="Qwen/Qwen3-VL-8B-Instruct"
MAX_FRAME_NUM=768

FLASHVID_BIN=/local1/cfyang/miniconda3/envs/flashvid/bin
export PATH="$FLASHVID_BIN:$PATH"
unset VIRTUAL_ENV
exec "$FLASHVID_BIN/accelerate" launch \
    --num_processes="$NPROC" --main_process_port 12398 -m lmms_eval \
    --model async_openai \
    --model_args "model_version=$MODEL_VERSION,mcp_server_path=$MCP_PATH,fps=1,max_frames=$MAX_FRAME_NUM,max_pixels=50176,base_url=$OPENAI_API_BASE,api_key=$OPENAI_API_KEY,num_cpus=1,timeout=12000,is_qwen3_vl=True,retain_tool_history=True,max_tool_rounds=8,force_final_answer=True,max_total_images=1900,video_tool_names=compress_video" \
    --tasks "$TASK" \
    --batch_size 1 \
    --output_path ./eval_logs \
    --log_samples \
    --include_path ./lmms_eval_tasks \
    "$@"
