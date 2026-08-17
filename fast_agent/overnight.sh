#!/usr/bin/env bash
# Overnight queue. Each entry is retried because run_agent is resumable: a re-invocation
# skips the question_ids already in results.jsonl, so a crashed run continues rather than
# restarting. A run is "done" only when results.jsonl has all 1549 rows.
PY=/local1/cfyang/miniconda3/envs/vllm/bin/python
O=/local1/cfyang/hanklin/outputs/lvbench_agent
N=1549
cd /home/cfyang/hanklin

rows() { wc -l < "$O/lvbench_$1_seed0/results.jsonl" 2>/dev/null || echo 0; }

# A question that died on a transport error is written as pred=None, i.e. scored WRONG,
# and the resume logic would then skip it forever. Drop those rows before each attempt so
# the retry actually re-runs them.
drop_errors() {
  local f="$O/lvbench_$1_seed0/results.jsonl"
  [ -f "$f" ] || return 0
  $PY - "$f" <<'PY'
import json,sys
p=sys.argv[1]
R=[json.loads(l) for l in open(p) if l.strip()]
# Also DEDUPE. The resume set is built once at startup, so two runner processes that
# ever overlap on one directory both append the same questions -- observed 131 rows
# for 88 questions after a lane was relaunched before its predecessor had exited.
# Last write wins.
seen={}
for r in R: seen[r['question_id']]=r
k=[r for r in seen.values() if 'error' not in r]
if len(k)!=len(R):
    open(p,'w').write("".join(json.dumps(r)+"\n" for r in k))
    print(f"[queue] cleaned {len(R)-len(k)} rows (errors + duplicates) from {p}")
PY
}

wait_for() {   # wait_for <tag>  -- block until an already-running run completes
  echo "[queue] waiting on $1 ($(rows $1)/$N)"
  while [ "$(rows $1)" -lt "$N" ]; do sleep 60; done
  echo "[queue] $1 complete"
}

healthy() {    # a real completion -- /v1/models answers 200 even on a wedged server
  curl -s -m 45 "localhost:$1/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"qwen3vl","messages":[{"role":"user","content":"hi"}],"max_tokens":2}' \
    -o /dev/null -w '%{http_code}' 2>/dev/null | grep -q 200
}

launch() {     # launch <tag> <port> <extra args...>   (NOVR overrides the question count)
  local tag=$1 port=$2; shift 2
  local N=${NOVR:-$N}
  for attempt in 1 2 3 4 5 6; do
    drop_errors "$tag"
    [ "$(rows $tag)" -ge "$N" ] && { echo "[queue] $tag already complete"; return 0; }
    # never start an attempt against a server that is down or wedged -- the watchdog
    # brings it back, and burning an attempt against a dead port wastes a retry slot
    until healthy "$port"; do echo "[queue] $tag waiting for :$port"; sleep 60; done
    echo "[queue] $tag attempt $attempt ($(rows $tag)/$N done)"
    $PY -m longvt_compression.fast_agent.run_agent \
      --tools crop_video --skim-mode video --frames 64 --num $N --seed 0 --tag "$tag" \
      --require-proxy --base-url "http://localhost:$port/v1" --workers 8 "$@" \
      >> "$O/${tag}_seed0.log" 2>&1
    sleep 20
  done
  [ "$(rows $tag)" -ge "$N" ] || echo "[queue] !! $tag STILL INCOMPLETE ($(rows $tag)/$N)"
}

case "$1" in
  a) launch run2 8030
     launch run4 8030 --crop-source oracle ;;
  c) until curl -s -m 2 localhost:8031/v1/models >/dev/null; do sleep 15; done
     launch ctrl 8031 --crop-source ctrl ;;
  # run5: the uncompressed twin of run3. run2 and run3 are matched on visual tokens
  # (1,280 each), so run3 buys its 4x frames by spending 1/4 the detail per frame -- and
  # loses 6 pp. That leaves two readings: 4x frames is worthless, or 4x frames helps but
  # compression's damage more than cancels it. run5 sees the same 256 frames at FULL
  # detail (5,120 visual tokens, no pruning) and separates them.
  # n=400 not 1549: run5 is a DIAGNOSTIC, and at 5,120 visual tokens resent every
  # round (with the mm processor cache off) it runs at ~40 s/question -- 16 h for the
  # full set. data.load_dataset(n=400, seed=0) is an exact PREFIX of the n=1549 ordering,
  # so run2/run3/run4 restricted to those same 400 qids are directly comparable.
  d) NOVR=400 launch run5 8031 --frames 256 --num 400 ;;
  *) echo "usage: overnight.sh a|c" >&2; exit 1 ;;
esac
echo "[queue] lane $1 finished at $(date)"
