"""Single-shot lossless A/B/C on VideoMME-long (NO agent loop, NO tools).

Tests whether FlashVID compression is "lossless" at the agent's operating point.
For each sample the SAME MCQ is answered three ways from a FIXED visual view;
the only thing that differs between arms is the visual token stream (identical
prompt text), so accuracy gaps are attributable to the view alone.

  skim64   : 64 uniform frames as images  (the agent's actual sparse initial view)
  comp640  : 640 uniform frames -> FlashVID @ retention 0.14 (video modality)
  full640  : 640 uniform frames uncompressed, retention 1.0  (the fidelity ceiling)

Two questions, one paired run:
  comp640 vs full640  -> does compression DISCARD answer-bearing info? (losslessness)
  comp640 vs skim64   -> does 10x coverage at ~matched budget beat the sparse skim?

  CUDA_VISIBLE_DEVICES=0 python -m fast_agent.lossless_singleshot --num 30

Writes results.jsonl (one row/sample, all three preds + token counts) + summary.json
to config.OUTPUT_DIR/lossless_singleshot_<tag>/. Resumable.
"""

import argparse
import json
import os
import time
from collections import defaultdict

import torch
from tqdm import tqdm

from . import config, data, tools
from .agent_loop import assemble, decode
from .model import Engine

# no-tool answer instruction (single-shot; no tool mention)
ANSWER_INSTR = (
    "Think briefly inside <think></think> tags, then give your final answer as "
    "<answer>X</answer> where X is one of the option letters A, B, C, or D."
)


def _view_text(dur: float) -> str:
    # Deliberately frame-count-agnostic so all three arms share IDENTICAL text:
    # only the visual token stream differs between arms.
    return (f"This video is {dur:.0f} seconds long. The frames above are uniformly "
            f"sampled across the whole video, from 0s to {dur:.0f}s.")


@torch.inference_mode()
def answer_from_clip(engine: Engine, clip, question: str, dur: float) -> tuple[str, str]:
    """Single-shot: assemble [clip, prompt], greedy decode, extract letter.
    One bounded finalizer nudge if no letter surfaced. Returns (pred, raw_text)."""
    prompt = f"{question}\n\n{_view_text(dur)}\n\n{ANSWER_INSTR}"
    messages = [{"role": "user", "parts": [clip, prompt]}]
    a = assemble(engine, messages, tools=None)
    text = decode(engine, a)
    pred = data.extract_answer(text)
    if pred is None:  # identical finalizer across arms
        messages.append({"role": "assistant", "parts": [text]})
        messages.append({"role": "user", "parts": [
            "Reply with <answer>X</answer> where X is one of A, B, C, or D."]})
        a = assemble(engine, messages, tools=None)
        text2 = decode(engine, a)
        pred = data.extract_answer(text2)
        text = text + "\n[FINALIZER]\n" + text2
    return pred, text


def run_arm_skim64(engine, row, dur):
    pils = tools.initial_frames(row["video_path"])   # config.INITIAL_FRAMES = 64
    clip = engine.encode_images(pils, meta={"role": "skim64"})
    pred, _ = answer_from_clip(engine, clip, data.format_question(row), dur)
    return pred, {"n_frames": len(pils), "tokens": int(clip.n_tokens)}


def run_arm_video(engine, row, dur, retention: float, max_frames: int = 640):
    # config.FLOOR_ENGAGE_FRAMES is 0 (set in main) so retention == FIXED exactly.
    config.COMPRESS_MAX_FRAMES = max_frames
    config.FIXED_RETENTION = retention
    vt, times = tools.compress_tensor(row["video_path"], None, None)
    clip = engine.encode_video_compressed(vt, target_tokens=1, frame_times=times,
                                          meta={"role": f"video_r{retention}"})
    pred, _ = answer_from_clip(engine, clip, data.format_question(row), dur)
    m = clip.meta
    return pred, {"n_frames": int(vt.shape[0]), "base_tokens": m["base_tokens"],
                  "kept_tokens": m["kept_tokens"], "retention": round(m["retention"], 3),
                  "tokens": int(clip.n_tokens)}


ARMS = {
    "skim64":  lambda e, r, d: run_arm_skim64(e, r, d),
    "comp640": lambda e, r, d: run_arm_video(e, r, d, retention=0.14, max_frames=640),
    "full640": lambda e, r, d: run_arm_video(e, r, d, retention=1.0, max_frames=640),
    # CONTROL: same video path + ~matched token budget as comp640, but frames are
    # UNIFORMLY subsampled (retention 1.0 on ~104 frames -> ~2340 tokens) instead of
    # FlashVID-selected. comp640 vs unif_matched isolates FlashVID's SELECTION from
    # the budget cut: tie -> selection adds nothing (not broken); comp loses -> our
    # selection picks worse tokens than uniform (impl/method problem).
    "unif_matched": lambda e, r, d: run_arm_video(e, r, d, retention=1.0, max_frames=104),
}


def summarize(run_dir, arms):
    rows = []
    with open(os.path.join(run_dir, "results.jsonl")) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    n = len(rows)
    out = {"n": n, "arms": {}, "by_task": {}}
    for arm in arms:
        ok = sum(r["preds"].get(arm) == r["gold"] for r in rows)
        out["arms"][arm] = {"correct": ok, "acc": round(100 * ok / max(n, 1), 1)}
    # pairwise agreement with the ceiling
    if "comp640" in arms and "full640" in arms:
        agree = sum(r["preds"].get("comp640") == r["preds"].get("full640") for r in rows)
        out["comp_vs_full_agreement"] = round(100 * agree / max(n, 1), 1)
    by_task = defaultdict(lambda: {a: [0, 0] for a in arms})
    for r in rows:
        for a in arms:
            by_task[r["task_type"]][a][0] += r["preds"].get(a) == r["gold"]
            by_task[r["task_type"]][a][1] += 1
    out["by_task"] = {t: {a: {"correct": c[0], "n": c[1]} for a, c in d.items()}
                      for t, d in sorted(by_task.items())}
    # median token counts per arm
    tok = {a: sorted(r["views"][a]["tokens"] for r in rows if a in r["views"]) for a in arms}
    out["median_tokens"] = {a: (v[len(v) // 2] if v else None) for a, v in tok.items()}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="n30")
    ap.add_argument("--arms", default="skim64,comp640,full640")
    args = ap.parse_args()
    arms = args.arms.split(",")

    # Force the video arms to sample exactly 640 frames at the exact retention
    # (floor never engages -> retention == FIXED_RETENTION), independent of the
    # agent-tuned config defaults (2128 frames / 0.1).
    config.COMPRESS_MAX_FRAMES = 640
    config.FLOOR_ENGAGE_FRAMES = 0

    run_dir = os.path.join(config.OUTPUT_DIR, f"lossless_singleshot_{args.tag}")
    os.makedirs(run_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.jsonl")
    with open(os.path.join(run_dir, "run_manifest.json"), "w") as f:
        json.dump({"kind": "lossless_singleshot", "arms": arms, "num": args.num,
                   "seed": args.seed, "compress_max_frames": 640,
                   "comp_retention": 0.14, "full_retention": 1.0,
                   "initial_frames": config.INITIAL_FRAMES,
                   "model_snapshot": config.MODEL_SNAPSHOT}, f, indent=1)

    done = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            done = {json.loads(l)["question_id"] for l in f if l.strip()}

    rows = data.load_long_split(n=args.num, seed=args.seed)
    rows = [r for r in rows if str(r["question_id"]) not in done]
    print(f"[lossless] arms={arms} todo={len(rows)} (skipped {len(done)}) -> {run_dir}")

    engine = Engine()
    t0 = time.time()
    with open(results_path, "a") as f:
        for i, row in enumerate(tqdm(rows, ascii=True, mininterval=20)):
            dur = data.video_duration(row["video_path"])
            preds, views, timings = {}, {}, {}
            for arm in arms:
                ta = time.time()
                try:
                    p, v = ARMS[arm](engine, row, dur)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    p, v = None, {"error": f"{type(e).__name__}: {e}"}
                preds[arm], views[arm] = p, v
                timings[arm] = round(time.time() - ta, 1)
            rec = {"question_id": str(row["question_id"]), "task_type": row["task_type"],
                   "gold": row["answer"], "duration": round(dur, 1),
                   "preds": preds, "views": views, "seconds": timings}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            s = summarize(run_dir, arms)
            acc_str = " | ".join(f"{a} {s['arms'][a]['acc']}%" for a in arms)
            tqdm.write(f"[{i+1}/{len(rows)}] {row['question_id']} "
                       + " ".join(f"{a}={preds[a]}" for a in arms)
                       + f" gold={row['answer']}  ::  {acc_str}"
                       + (f" | comp==full {s.get('comp_vs_full_agreement')}%"
                          if 'comp_vs_full_agreement' in s else "")
                       + f"  ({(time.time()-t0)/(i+1):.0f}s/sample)")

    s = summarize(run_dir, arms)
    print("\n===== SUMMARY (single-shot, paired) =====")
    print(f" n={s['n']}")
    for a in arms:
        print(f"  {a:<9} {s['arms'][a]['correct']:>3}/{s['n']:<3} = {s['arms'][a]['acc']}%"
              f"   (median {s['median_tokens'][a]} tokens)")
    if "comp_vs_full_agreement" in s:
        print(f"  comp640 == full640 on {s['comp_vs_full_agreement']}% of samples")


if __name__ == "__main__":
    main()
