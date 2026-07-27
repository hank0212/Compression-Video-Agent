"""VideoMME-long eval runner (batch=1; shard across GPUs with --shard/--num-shards).

Each run writes to config.RUN_ROOT/<arm>_<tag>/:
  results.jsonl   one line per sample (question_id, pred, gold, correct, tool_calls, traj_path)
  summary.json    live accuracy overall + per-task-type (rewritten each sample, all shards merged)
  traj/<qid>.json full replayable trajectory (thinking, actions, montage paths)
  media/<qid>/    frame montages (initial skim + each tool call)

  CUDA_VISIBLE_DEVICES=3 python -m fast_agent.run_eval --arm compress --n 200 --shard 0 --num-shards 3

Resumable: re-running skips question_ids already in results.jsonl.
"""

import argparse
import glob
import hashlib
import json
import os
import time
from collections import defaultdict

from tqdm import tqdm

from . import config, data, oracle
from .agent_loop import run_sample
from .model import Engine

# Oracle arms (LVBench only): forced GT-region evidence, then answer. The arm name maps
# to the forced-evidence mode; oracle_crop is the primary "is perception enough given
# perfect localization" instrument, the other two isolate the compression form.
ORACLE_ARMS = {"oracle_crop": "crop", "oracle_compress": "compress",
               "oracle_both": "both", "oracle_ctrl": "ctrl"}

ARM_TOOLS = {
    "baseline": (),
    "crop": ("crop_video",),
    "compress": ("crop_video", "compress_video"),   # "crop + compression" (autonomous)
    "oracle_crop": ("crop_video",),                 # forced GT-region crop (PRIMARY)
    "oracle_compress": ("compress_video",),         # forced GT-region compress
    "oracle_both": ("compress_video", "crop_video"),# forced GT-region compress + crop
    # Wrong-location control: same tool, same width, same token budget as oracle_crop,
    # but the crop is placed AWAY from the evidence. oracle_crop - oracle_ctrl isolates
    # "the right pixels" from "more pixels" (uniform frame count alone moved VideoMME
    # +13.3pp, so an uncontrolled oracle gain is not attributable to localization).
    "oracle_ctrl": ("crop_video",),
}


# Manifest fields that change model behavior. Two runs may share results ONLY if all of
# these match -- otherwise a "full run" would silently mix samples generated under
# different prompts/compressors/budgets, which is exactly the kind of quiet corruption
# that invalidates a paired comparison.
REUSE_KEYS = ("dataset", "arm", "tools", "compressor", "skim_timestamps",
              "unify_time_format", "fixed_retention", "model_snapshot",
              "initial_frames", "max_rounds", "max_new_tokens")


def _read_jsonl(path: str) -> list:
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def gather_reusable(dataset: str, arm: str, manifest: dict, self_dir: str,
                    tags: list | None = None) -> dict:
    """Collect already-computed results for this (dataset, arm) from OTHER run dirs whose
    manifest is behaviourally identical. Returns {question_id: row}. Lets a later, larger
    run skip everything a pilot already answered instead of recomputing it."""
    found, sources = {}, []
    for d in sorted(glob.glob(os.path.join(config.RUN_ROOT, f"{dataset}_{arm}_*"))):
        if os.path.abspath(d) == os.path.abspath(self_dir):
            continue
        tag = os.path.basename(d)[len(f"{dataset}_{arm}_"):]
        if tags and tag not in tags:
            continue
        other = {}
        mp = os.path.join(d, "run_manifest.json")
        if os.path.exists(mp):
            try:
                other = json.load(open(mp))
            except json.JSONDecodeError:
                pass
        mismatch = [k for k in REUSE_KEYS if other.get(k) != manifest.get(k)]
        if mismatch:
            print(f"[reuse] skip {os.path.basename(d)} (differs on: {', '.join(mismatch)})")
            continue
        rows = [r for r in _read_jsonl(os.path.join(d, "results.jsonl"))
                if r.get("question_id") is not None and not r.get("error")]
        new = 0
        for r in rows:
            qid = str(r["question_id"])
            if qid not in found:
                r = dict(r); r["reused_from"] = os.path.basename(d)
                found[qid] = r
                new += 1
        if new:
            sources.append(f"{os.path.basename(d)}(+{new})")
    if sources:
        print(f"[reuse] importing {len(found)} results from: {', '.join(sources)}")
    return found


def write_summary(run_dir: str, arm: str):
    by_task, total = defaultdict(lambda: [0, 0]), [0, 0]
    tool_hist = defaultdict(int)
    strict_ok = fin_used = 0
    for p in glob.glob(os.path.join(run_dir, "results.jsonl")):
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    # A partial trailing line (process killed mid-write on a multi-hour
                    # run) must not kill aggregation -- and must not block resume.
                    continue
                ok = bool(r.get("correct"))
                by_task[r["task_type"]][0] += ok
                by_task[r["task_type"]][1] += 1
                total[0] += ok
                total[1] += 1
                strict_ok += bool(r.get("correct_strict"))
                fin_used += bool(r.get("finalizer_used"))
                for c in r.get("tool_calls", []):
                    tool_hist[c["name"]] += 1
    n = max(total[1], 1)
    summary = {
        "arm": arm,
        "n": total[1],
        "accuracy": round(100 * total[0] / n, 2),          # lenient (headline)
        "accuracy_strict": round(100 * strict_ok / n, 2),  # pre-finalizer, LongVT-faithful
        "finalizer_rate": round(100 * fin_used / n, 1),    # % of samples that needed the finalizer
        "by_task": {t: {"correct": c, "n": n_t, "acc": round(100 * c / max(n_t, 1), 1)}
                    for t, (c, n_t) in sorted(by_task.items())},
        "tool_calls": dict(tool_hist),
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARM_TOOLS))
    ap.add_argument("--dataset", default="videomme", choices=["videomme", "lvbench"],
                    help="benchmark to eval (default videomme; existing launch scripts unaffected)")
    ap.add_argument("--num", type=int, default=None, help="total samples before sharding")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--tag", default=time.strftime("%m%d"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true",
                    help="print the model's raw generation each round")
    ap.add_argument("--reuse", nargs="?", const="auto", default=None,
                    metavar="TAGS",
                    help="skip questions already answered by other runs of the same "
                         "dataset+arm whose config matches: bare --reuse takes any "
                         "compatible run; --reuse tagA,tagB restricts to those tags")
    args = ap.parse_args()

    run_dir = os.path.join(config.RUN_ROOT, f"{args.dataset}_{args.arm}_{args.tag}")
    os.makedirs(run_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.jsonl")

    # A small immutable manifest makes a run auditable without duplicating
    # model weights or raw video.  Per-sample reasoning remains in traj/*.json.
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.exists(manifest_path):
        with open(manifest_path, "w") as mf:
            json.dump({"schema_version": 2, "arm": args.arm, "tag": args.tag,
                       "dataset": args.dataset,
                       "num": args.num, "seed": args.seed,
                       "shard": args.shard, "num_shards": args.num_shards,
                       "tools": list(ARM_TOOLS[args.arm]),
                       "compressor": config.COMPRESSOR,
                       # prompt-shaping flags change model behavior -> must be recorded
                       # or two runs with the same tag are not comparable
                       "skim_timestamps": config.SKIM_TIMESTAMPS,
                       "unify_time_format": data.UNIFY_TIME_FORMAT,
                       "fixed_retention": config.FIXED_RETENTION,
                       "model_snapshot": config.MODEL_SNAPSHOT,
                       "initial_frames": config.INITIAL_FRAMES,
                       "max_rounds": config.MAX_ROUNDS,
                       "max_new_tokens": config.MAX_NEW_TOKENS}, mf, indent=1)

    rows = data.load_dataset(args.dataset, n=args.num, seed=args.seed)
    if args.arm in ORACLE_ARMS:
        if args.dataset != "lvbench":
            ap.error(f"--arm {args.arm} requires --dataset lvbench (needs per-question evidence timestamps)")
        n_before = len(rows)
        rows = [r for r in rows if r.get("evidence")]
        if len(rows) < n_before:
            print(f"[eval] {args.arm}: skipped {n_before - len(rows)} rows without usable evidence")
    # Shard by VIDEO (not question index) so all questions of one video land on the
    # same worker — avoids concurrent AV1-proxy transcodes racing on the same file.
    def _shard_of(vid: str) -> int:
        return int(hashlib.md5(vid.encode()).hexdigest(), 16) % args.num_shards
    rows = [r for r in rows if _shard_of(r["videoID"]) == args.shard]

    done = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for l in f:
                if not l.strip():
                    continue
                try:                       # tolerate a truncated trailing line
                    done.add(json.loads(l)["question_id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    # Cross-run reuse: import results this (dataset, arm) already produced under another
    # tag, provided the manifest matches on every behaviour-affecting key. Scaling --num
    # is safe because the seeded draw is a PREFIX (n=60 is the first 60 of n=200), so a
    # later full run recomputes only the genuinely new questions.
    if args.reuse:
        manifest = json.load(open(manifest_path))
        tags = None if args.reuse == "auto" else [t.strip() for t in args.reuse.split(",")]
        pool = gather_reusable(args.dataset, args.arm, manifest, run_dir, tags)
        imported = 0
        if pool:
            with open(results_path, "a") as rf:
                for r in rows:
                    qid = str(r["question_id"])
                    if qid in done or qid not in pool:
                        continue
                    rf.write(json.dumps(pool[qid]) + "\n")
                    done.add(qid)
                    imported += 1
            if imported:
                write_summary(run_dir, args.arm)
        print(f"[reuse] adopted {imported} previously-computed results into {os.path.basename(run_dir)}")

    rows = [r for r in rows if str(r["question_id"]) not in done]
    print(f"[eval] arm={args.arm} shard={args.shard}/{args.num_shards} "
          f"todo={len(rows)} (skipped {len(done)}) -> {run_dir}")

    engine = Engine()
    t0 = time.time()
    # ascii + mininterval keeps the nohup logfile readable (non-tty tqdm otherwise
    # prints one bar line per refresh). tqdm.write lines stay the durable record.
    bar = tqdm(rows, desc=f"{args.arm}/{args.tag}", ascii=True, mininterval=30,
               dynamic_ncols=True)
    with open(results_path, "a") as f:
        for i, row in enumerate(bar):
            t = time.time()
            try:
                if args.arm in ORACLE_ARMS:
                    r = oracle.run_oracle_sample(engine, row, record_dir=run_dir,
                                                 mode=ORACLE_ARMS[args.arm],
                                                 verbose=args.verbose)
                else:
                    r = run_sample(engine, row, tool_names=ARM_TOOLS[args.arm],
                                   record_dir=run_dir, verbose=args.verbose)
            except Exception as e:  # per-sample isolation
                import traceback
                traceback.print_exc()
                r = {"question_id": str(row["question_id"]), "task_type": row["task_type"],
                     "pred": None, "pred_strict": None, "pred_lenient": None,
                     "finalizer_used": False, "gold": row["answer"], "correct": False,
                     "correct_strict": False, "tool_calls": [], "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(r) + "\n")
            f.flush()
            s = write_summary(run_dir, args.arm)  # live accuracy after every sample
            bar.set_postfix(acc=f"{s['accuracy']}%", strict=f"{s['accuracy_strict']}%",
                            n=s["n"], avg=f"{(time.time()-t0)/(i+1):.0f}s")
            tqdm.write(
                f"[{i+1}/{len(rows)}] {row['question_id']} pred={r.get('pred')} "
                f"gold={r['gold']} ok={r.get('correct')} rounds={r.get('rounds','-')} "
                f"{time.time()-t:.0f}s | acc {s['accuracy']}% (strict {s['accuracy_strict']}%) "
                f"n={s['n']} fin={s['finalizer_rate']}%")
    bar.close()

    s = write_summary(run_dir, args.arm)
    print(f"\n[SUMMARY {args.arm}/{args.tag}] lenient {s['accuracy']}% | "
          f"strict {s['accuracy_strict']}% | finalizer {s['finalizer_rate']}% over n={s['n']}")
    for t, d in s["by_task"].items():
        print(f"  {t:<28} {d['correct']:>3}/{d['n']:<3} = {d['acc']}%")
    print(f"  tool calls: {s['tool_calls']}")


if __name__ == "__main__":
    main()
