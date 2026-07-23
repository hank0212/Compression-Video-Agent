#!/bin/bash
# Concurrent paired run: baseline + compress at the SAME TIME so compress can be
# watched/iterated live. GPU split (no contention):
#   baseline -> GPU3 (solo, ~90min for full split)
#   compress -> GPU4 (shard 0/2) + GPU7 (shard 1/2)
# Each process is independent & resumable (skips finished qids). Same 882 Qs,
# video-sharded, so the two arms stay paired by question_id.
set -u
TAG=${1:-run1}
LOG=/data2/cfyang/tmp/claude-22022/-home-cfyang-hanklin/5cb42490-8567-4620-866b-9220122f1ad9/scratchpad
cd /home/cfyang/hanklin/longvt_compression

launch() {  # gpu arm shard nshards
  CUDA_VISIBLE_DEVICES=$1 PYTHONPATH= nohup conda run --no-capture-output -n flashvid \
    python -u -m fast_agent.run_eval --arm $2 --shard $3 --num-shards $4 --tag $TAG \
    > $LOG/${TAG}_$2_g$1.log 2>&1 &
  echo "  $2 shard $3/$4 on GPU$1 -> $LOG/${TAG}_$2_g$1.log"
}

echo "[concurrent start $(date)] tag=$TAG"
launch 3 baseline 0 1
launch 4 compress 0 2
launch 7 compress 1 2
echo "all launched. live acc: watch runs/{baseline,compress}_${TAG}/summary.json"
