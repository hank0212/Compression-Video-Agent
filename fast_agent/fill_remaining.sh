#!/usr/bin/env bash
# Append the missing 1,049 questions to the two 128-frame arms still at n=500.
# Hard health gate: if the server is not answering, ABORT rather than let run_agent
# write transport-error rows -- 88 such rows on the previous attempt scored as wrong
# and biased the arm down 6.4 pp before they were stripped.
set -u
LOG=/local1/cfyang/hanklin/outputs/lvbench_agent
AF=/home/cfyang/hanklin/longvt_compression/fast_agent

run_arm () {   # arm port gpuutil tag
  local ARM=$1 PORT=$2 UTIL=$3 TAG=$4
  echo "[fill] ===== $TAG (arm $ARM, port $PORT, util $UTIL)  $(date +%H:%M)"
  cd "$AF"
  SPID=$(GPUUTIL=$UTIL bash serve_arm.sh "$ARM" | sed -n 's/^pid \([0-9]*\).*/\1/p')
  local up=0
  for i in $(seq 1 80); do
    if curl -s -m 3 "localhost:$PORT/v1/models" 2>/dev/null | grep -q qwen3vl; then up=1; break; fi
    sleep 15
  done
  if [ "$up" != 1 ]; then
    echo "[fill] ABORT: server for $TAG never came up; NOT running the agent"
    [ -n "${SPID:-}" ] && kill "$SPID" 2>/dev/null
    sleep 60; return 1
  fi
  echo "[fill] server up after $((i*15))s"
  cd /home/cfyang/hanklin
  PYTHONPATH=/home/cfyang/hanklin /local1/cfyang/miniconda3/envs/vllm/bin/python \
    -m longvt_compression.fast_agent.run_agent \
    --dataset lvbench --skim-mode video --frames 128 --tools \
    --proxy-dir "$LOG/skim_proxies_hires_128" --require-proxy \
    --num 1549 --seed 0 --tag "$TAG" \
    --base-url "http://localhost:$PORT/v1" --workers 8 2>&1 | grep -E "^\[agent\]|SUMMARY"
  local n=$(wc -l < "$LOG/lvbench_${TAG}_seed0/results.jsonl")
  local e=$(grep -c '"error"' "$LOG/lvbench_${TAG}_seed0/results.jsonl")
  echo "[fill] $TAG -> n=$n  error_rows=$e"
  # SIGTERM only; never SIGKILL a CUDA process (driver leaks the memory)
  [ -n "${SPID:-}" ] && kill "$SPID" 2>/dev/null
  sleep 90
}

run_arm p 8047 0.70 pD_128f_evs_050
run_arm o 8046 0.70 pC2_128f_vidcom2_050_FIXED
echo "[fill] DONE $(date +%H:%M)"
