#!/bin/bash
# Overnight paired run, maximizing GPU use: both arms cover the SAME question set.
# Phase 1: baseline sharded across GPU 3/4/7 (cheap ~6s/sample -> ~30min for full split).
# Phase 2: compress sharded across GPU 3/4/7 (the slow, interesting arm).
# Resumable: re-running skips finished qids. Self-contained; run with nohup &.
set -u
NUM=${1:-0}                       # 0 = full long split (~876 present questions)
TAG=${2:-run1}
GPUS=(3 4 7)
LOG=/data2/cfyang/tmp/claude-22022/-home-cfyang-hanklin/5cb42490-8567-4620-866b-9220122f1ad9/scratchpad
cd /home/cfyang/hanklin/longvt_compression

numarg=""; [ "$NUM" != "0" ] && numarg="--num $NUM"

run_arm() {  # arm -> 3 shards across the 3 GPUs, wait for all
  local arm=$1
  for i in 0 1 2; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} PYTHONPATH= nohup conda run --no-capture-output -n flashvid \
      python -u -m fast_agent.run_eval --arm $arm $numarg --shard $i --num-shards 3 --tag $TAG \
      > $LOG/${TAG}_${arm}_g${GPUS[$i]}.log 2>&1 &
    echo "  launched $arm shard $i on GPU${GPUS[$i]} (pid $!)"
  done
  wait
  echo "[$arm DONE] $(date)"
}

echo "[START $(date)] NUM=${NUM:-full} TAG=$TAG"
run_arm baseline
run_arm compress
echo "[ALL DONE $(date)] results in /local1/cfyang/hanklin/outputs/fast_agent/runs/{baseline,compress}_${TAG}/"
