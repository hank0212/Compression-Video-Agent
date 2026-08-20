#!/usr/bin/env bash
# The frames x retention grid, seed 0, LVBench 1,549. Frozen 2026-08-19.
#
#     run_grid.sh <gpu> <arm> [arm ...]
#     run_grid.sh 3 u32 v32 v64          # one chain per GPU, run them in parallel
#
#                    no compression        VidCom2 25%
#     32 frames           u32                  v32
#     64 frames           u64                  v64      (+ v64_50 at 50%)
#    128 frames          u128                 v128
#
# Every arm renders at the source's NATIVE resolution (704x1280, 880 tokens per grid
# step for the 87 of 103 videos whose source allows it), because size.longest_edge is set
# past the point where the source becomes the binding constraint. Resolution is therefore
# fixed BY SATURATION, and the only variables are frame count and retention.
#
# Arms are independent, so each GPU runs its own chain. Ports are unique per ARM, not per
# GPU, so two chains can never collide.
#
# ABORTS rather than continues on: failed preflight, a server that never answers, or a run
# that finishes with transport-error rows. The last is not paranoia -- an earlier fill
# wrote 88 APIConnectionError rows that scored as wrong and moved an arm 6.4 pp before
# they were caught.
set -u
set -o pipefail   # else `preflight | grep` reports GREP's status and a failed check passes
LOG=/local1/cfyang/hanklin/outputs/lvbench_agent
PF=$LOG/preflight
AF=/home/cfyang/hanklin/longvt_compression/fast_agent
PY=/local1/cfyang/miniconda3/envs/vllm/bin/python
mkdir -p "$PF"

GPU=${1:?usage: run_grid.sh <gpu> <arm> [arm ...]}
shift

# arm -> port frames retention tag needgib.  needgib is weights 16.4 + KV (context x
# 144 KiB/token) + vision-tower and encoder-cache head-room; the chain refuses to start an
# arm the free memory cannot hold, instead of OOMing an hour in.
spec () {
  case "$1" in
    u32)    echo "8061 32  1.00 u32_native       28" ;;
    v32)    echo "8062 32  0.25 v32_native_r025  28" ;;
    u64)    echo "8063 64  1.00 u64_native       32" ;;
    v64)    echo "8064 64  0.25 v64_native_r025  32" ;;
    v64_50) echo "8065 64  0.50 v64_native_r050  32" ;;
    u128)   echo "8066 128 1.00 u128_native      38" ;;
    v128)   echo "8067 128 0.25 v128_native_r025 38" ;;
    *) return 1 ;;
  esac
}

run_arm () {
  local ARM=$1 PORT FRAMES RET TAG NEEDGIB
  read -r PORT FRAMES RET TAG NEEDGIB < <(spec "$ARM") || { echo "unknown arm $ARM"; return 1; }
  local DIR="$LOG/lvbench_${TAG}_seed0"
  echo
  echo "=================================================================="
  echo "[gpu$GPU] $TAG  ($ARM, $FRAMES frames, retention $RET)  $(date +%H:%M)"
  echo "=================================================================="

  if [ -f "$DIR/results.jsonl" ] && [ "$(wc -l < "$DIR/results.jsonl")" -ge 1549 ]; then
    echo "[gpu$GPU] already complete at n=$(wc -l < "$DIR/results.jsonl"); skipping"
    return 0
  fi

  # ---- 1. geometry, over all 103 videos, before anything is served --------------
  if ! PYTHONPATH=/home/cfyang/hanklin CUDA_VISIBLE_DEVICES="" $PY \
        -m longvt_compression.fast_agent.preflight_arm \
        --frames "$FRAMES" --retention "$RET" --workers 6 \
        --out "$PF/${TAG}.json" 2>&1 | grep -vi warning; then
    echo "[gpu$GPU] ABORT: preflight failed for $TAG"; return 1
  fi

  # ---- 2. server ----------------------------------------------------------------
  cd "$AF"
  # Size the request from what is ACTUALLY free: vLLM refuses to start when the requested
  # fraction exceeds free memory, and neighbours on this shared box come and go.
  local FREE TOT UTIL AVAIL
  read -r TOT FREE < <(nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits -i "$GPU" | tr ',' ' ')
  UTIL=$(awk -v f="$FREE" -v t="$TOT" 'BEGIN{u=f/t-0.02; if(u>0.88)u=0.88; printf "%.2f", u}')
  AVAIL=$(awk -v u="$UTIL" -v t="$TOT" 'BEGIN{printf "%.1f", u*t/1024}')
  if awk -v a="$AVAIL" -v n="$NEEDGIB" 'BEGIN{exit !(a < n)}'; then
    echo "[gpu$GPU] ABORT: $TAG needs ~${NEEDGIB} GiB, only ${AVAIL} GiB free on GPU $GPU"
    return 1
  fi
  echo "[gpu$GPU] gpu-memory-utilization $UTIL (${AVAIL} GiB of ${TOT} MiB card; arm needs ~${NEEDGIB})"
  local SPID
  SPID=$(GPU_FORCE=$GPU GPUUTIL=$UTIL bash serve_arm.sh "$ARM" | sed -n 's/^pid \([0-9]*\).*/\1/p')
  local up=0 i
  for i in $(seq 1 80); do
    if curl -s -m 3 "localhost:$PORT/v1/models" 2>/dev/null | grep -q qwen3vl; then up=1; break; fi
    sleep 15
  done
  if [ "$up" != 1 ]; then
    echo "[gpu$GPU] ABORT: server for $TAG never answered (see $LOG/server_arm_$ARM.log)"
    [ -n "${SPID:-}" ] && kill "$SPID" 2>/dev/null
    sleep 60; return 1
  fi
  echo "[gpu$GPU] server up after $((i*15))s (pid $SPID)"
  # `|| true`: an UNCOMPRESSED arm has no pruning line to find, and grep returning 1
  # under `set -e` would kill the chain right after the server came up.
  grep -a "vidcom2_vllm\] ACTIVE\|video_pruning_rate" "$LOG/server_arm_$ARM.log" | tail -2 || true

  # ---- 3. the run ----------------------------------------------------------------
  cd /home/cfyang/hanklin
  PYTHONPATH=/home/cfyang/hanklin $PY \
    -m longvt_compression.fast_agent.run_agent \
    --dataset lvbench --skim-mode video --frames "$FRAMES" --tools \
    --proxy-dir "$LOG/skim_proxies_hires_$FRAMES" --require-proxy \
    --num 1549 --seed 0 --tag "$TAG" \
    --base-url "http://localhost:$PORT/v1" --workers 8 2>&1 | grep -E "^\[agent\]|SUMMARY"

  # ---- 4. integrity ---------------------------------------------------------------
  local n e u
  n=$(wc -l < "$DIR/results.jsonl")
  e=$(grep -c '"error"' "$DIR/results.jsonl" || true)
  u=$($PY -c "import json;print(len({json.loads(l)['question_id'] for l in open('$DIR/results.jsonl')}))")
  echo "[gpu$GPU] $TAG -> n=$n unique=$u error_rows=$e"
  # SIGTERM only. SIGKILL on a CUDA process leaks the memory on this shared box.
  [ -n "${SPID:-}" ] && kill "$SPID" 2>/dev/null
  sleep 90
  if [ "$e" -ne 0 ]; then
    echo "[gpu$GPU] ABORT: $TAG finished with $e transport-error rows"
    return 1
  fi
}

set -e
for a in "$@"; do run_arm "$a"; done
echo "[gpu$GPU] CHAIN COMPLETE ($*) $(date +%H:%M)"
