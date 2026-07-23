#!/usr/bin/env bash
# Zero-shot Qwen3-VL-8B-Instruct tool-calling baseline through the LongVT eval
# harness (lmms-eval async_openai + MCP crop_video). Adapted from run_eval.sh.
#
# Prereqs:
#   - model-under-test server up:  ./serve_qwen3vl_baseline.sh   (GPUs 3,4 -> :8010)
#   - videos symlinked at $HF_HOME/videomme/data (900 long-split mp4s)
#   - conda env `flashvid` = editable lmms-eval 0.5.0 (+ mcp, math-verify)
#
# Judge: same Qwen3-VL-8B server on :8010 (fires only when exact-match fails).
# NOTE: 8222 serves LongVT-RFT (stale CLAUDE.md), NOT a usable judge.
# OPENAI_API_BASE = model under test; OPENAI_BASE_URL = judge. Different vars!
#
# Usage: ./run_qwen3vl_baseline.sh [TASK] [NPROC] [extra lmms_eval args...]
#   TASK   default videomme_long_reward_tool
#   NPROC  accelerate processes (default 4); each runs 1 request at a time
#   e.g. smoke:  ./run_qwen3vl_baseline.sh videomme_long_reward_tool 1 --limit 8
set -euo pipefail

TASK=${1:-videomme_long_reward_tool}
NPROC=${2:-4}
shift 2 2>/dev/null || shift $# || true

REPO=/home/cfyang/hanklin/longvt_compression
cd "$REPO"

export HF_HOME=/local1/cfyang/.cache/huggingface
export OPENAI_API_BASE="http://localhost:8010/v1"    # model under test
export OPENAI_BASE_URL="http://localhost:8010/v1"    # LLM judge (same server)
export OPENAI_MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
export OPENAI_API_KEY="EMPTY"
export USE_LLM_JUDGE=True
export DECORD_EOF_RETRY_MAX=409600
export no_proxy=localhost,127.0.0.1
# API-only client: keep accelerate ranks off the GPUs (they otherwise open
# NCCL contexts on busy cards owned by other users).
export CUDA_VISIBLE_DEVICES=""
# Persist every finished sample to disk immediately (jsonl per rank under
# $LMMS_EVAL_HOME/eval_cache/...). Makes runs stop/resume-able and lets
# scratchpad partial scorers compute running accuracy mid-run.
# NOTE: resume requires the SAME process count (cache files are keyed
# task_rank{r}_world_size{w}).
export LMMS_EVAL_USE_CACHE=True
export LMMS_EVAL_HOME=${LMMS_EVAL_HOME:-/local1/cfyang/hanklin/outputs/lmms_eval_home}

MCP_PATH="./examples/video_tools/mcp_server.py"
MODEL_VERSION="Qwen/Qwen3-VL-8B-Instruct"
MAX_FRAME_NUM=768

# Hard-prepend the flashvid env: the login profile puts another env (dvd_tool)
# first on PATH, which beats `conda run` — and lmms-eval's MCPClient spawns a
# bare `python` for mcp_server.py, so PATH must resolve BOTH `accelerate` and
# `python` to the flashvid env.
FLASHVID_BIN=/local1/cfyang/miniconda3/envs/flashvid/bin
export PATH="$FLASHVID_BIN:$PATH"
unset VIRTUAL_ENV
exec "$FLASHVID_BIN/accelerate" launch \
    --num_processes="$NPROC" --main_process_port 12399 -m lmms_eval \
    --model async_openai \
    --model_args "model_version=$MODEL_VERSION,mcp_server_path=$MCP_PATH,fps=1,max_frames=$MAX_FRAME_NUM,max_pixels=50176,base_url=$OPENAI_API_BASE,api_key=$OPENAI_API_KEY,num_cpus=1,timeout=12000,is_qwen3_vl=True,retain_tool_history=True,max_tool_rounds=8,force_final_answer=True,max_total_images=1900" \
    --tasks "$TASK" \
    --batch_size 1 \
    --output_path ./eval_logs \
    --log_samples \
    --include_path ./lmms_eval_tasks \
    "$@"
