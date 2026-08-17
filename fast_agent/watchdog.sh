#!/usr/bin/env bash
# Restart a vLLM server that has WEDGED: /v1/models still answers 200 while
# /v1/chat/completions never returns and the engine reports Running: 0.
#
# Seen 2026-08-12 on both arm servers, minutes after their clients were killed
# mid-request -- the API process survives, the request path does not. A liveness probe
# on /v1/models would have called both servers healthy the whole time, so the probe here
# is a real completion.
#
#   nohup bash watchdog.sh > watchdog.log 2>&1 &
#
# GPU SAFETY: SIGTERM only, never SIGKILL. vLLM spawns VLLM::EngineCore as a separate
# process; killing only the parent leaves that child holding ~48 GB (observed). So after
# terminating the API server, terminate any EngineCore still alive, then WAIT for
# nvidia-smi to show the memory back before relaunching -- starting a new server against
# a GPU whose memory has not been released just fails differently.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=/local1/cfyang/hanklin/outputs/lvbench_agent
PROBE_TIMEOUT=45
INTERVAL=240

declare -A fails=([8031]=0 [8033]=0)
declare -A arm=([8031]=d [8033]=rft)
declare -A gpu=([8031]=7 [8033]=4)

healthy() {   # a real completion, not /v1/models
  curl -s -m "$PROBE_TIMEOUT" "localhost:$1/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3vl","messages":[{"role":"user","content":"hi"}],"max_tokens":2}' \
    -o /dev/null -w '%{http_code}' 2>/dev/null | grep -q 200
}

restart() {
  local port=$1 g=${gpu[$1]} a=${arm[$1]}
  echo "[watchdog] $(date +%H:%M:%S) restarting :$port (gpu $g)"
  local pid
  pid=$(ss -lptn "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "$pid" ] && kill "$pid" 2>/dev/null
  sleep 20
  # orphaned engine cores on this GPU
  for ec in $(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    ps -o cmd= -p "$ec" 2>/dev/null | grep -q EngineCore && { echo "[watchdog] SIGTERM EngineCore $ec"; kill "$ec" 2>/dev/null; }
  done
  for _ in $(seq 1 30); do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "${used:-99999}" -lt 5000 ] && break
    sleep 10
  done
  echo "[watchdog] gpu $g now at ${used:-?} MiB; relaunching"
  if [ "$a" = rft ]; then bash "$HERE/serve_longvt_rft.sh" "$g" "$port"; else bash "$HERE/serve_arm.sh" "$a"; fi
  for _ in $(seq 1 60); do healthy "$port" && break; sleep 15; done
  echo "[watchdog] :$port back up: $(healthy "$port" && echo yes || echo NO)"
  fails[$port]=0
}

while true; do
  for port in 8031 8033; do
    if healthy "$port"; then
      fails[$port]=0
    else
      fails[$port]=$(( ${fails[$port]} + 1 ))
      echo "[watchdog] $(date +%H:%M:%S) :$port probe failed (${fails[$port]}/2)"
      [ "${fails[$port]}" -ge 2 ] && restart "$port"
    fi
  done
  sleep "$INTERVAL"
done
