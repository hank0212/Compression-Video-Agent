"""Behavioral probes: re-run the AGENT MODEL under stripped-down contexts to turn
judge opinions into measurements (companion to evidence_judge.py).

  CUDA_VISIBLE_DEVICES=4 python -m fast_agent.probes \
      --tools-run compress_run2 --base-run baseline_run1 \
      [--probes p1,p2] [--limit N] [--p2-bw-sample 50]

P1 — evidence-only replay (all fixed+broke flips): rebuild exactly the evidence the
  tools arm gathered (initial skim + every successful tool result, re-decoded /
  re-compressed deterministically with the same code path and config), presented as
  ONE user turn with none of the agent's multi-round reasoning history, forced to
  answer. Agent-wrong + replay-right => the evidence was sufficient; the failure was
  multi-round integration/reasoning drift, not the evidence.

P2 — compressed-only readout (compress-involved flips + a seeded sample of
  compress-involved both_wrong): context = question + ONLY the compressed tokens of
  the exact span(s) the agent compressed. Right => compression preserved the
  answer-bearing signal; wrong => the compression (or the chosen span) lost it.

Determinism note: compress calls reproduce the RECORDED retention exactly by passing
target_tokens = round(retention * base_tokens) — encode_video_compressed then derives
max(FIXED_RETENTION, target/base) = the recorded value on floor-engaged short spans,
and ignores target on long spans (fixed retention). Reproduced kept_tokens is stored
next to the recorded value as a per-sample determinism check.

Writes RUN_ROOT/probe_<toolsrun>_vs_<baserun>/probes.jsonl (resumable by (qid, probe)).
"""

import argparse
import json
import os
import random
import time

import torch

from . import config, data, tools, viz
from .agent_loop import assemble, decode
from .model import Engine


# ---------------------------------------------------------------------------
def successful_calls(traj: dict) -> list[dict]:
    out = []
    for r in traj["rounds"]:
        tr = r.get("tool_result")
        if tr:
            out.append(tr)
    return out


def _answer_once(engine: Engine, messages: list) -> tuple[str | None, str, int]:
    """Assemble + greedy decode; one letter-demand retry (mirrors run_sample)."""
    a = assemble(engine, messages)
    ctx = int(a.embeds.shape[1])
    text = decode(engine, a)
    pred = data.extract_answer(text)
    if pred is None:
        messages = messages + [
            {"role": "assistant", "parts": [text]},
            {"role": "user", "parts": [
                "Your answer must be one of the option letters. "
                "Reply with <answer>X</answer> where X is A, B, C or D."]},
        ]
        a = assemble(engine, messages)
        text2 = decode(engine, a)
        pred = data.extract_answer(text2)
        text = text + "\n---retry---\n" + text2
    return pred, text, ctx


class ClipCache:
    """Per-qid cache so P1 and P2 on the same sample encode each span once."""

    def __init__(self, engine: Engine, video_path: str):
        self.engine, self.path = engine, video_path
        self._cache: dict = {}

    def skim(self):
        if "skim" not in self._cache:
            pils = tools.initial_frames(self.path)
            self._cache["skim"] = (self.engine.encode_images(pils), len(pils))
        return self._cache["skim"]

    def crop(self, s: float, e: float):
        key = ("crop", round(s, 2), round(e, 2))
        if key not in self._cache:
            frames = tools.crop_frames(self.path, s, e)
            self._cache[key] = (self.engine.encode_images(frames), len(frames))
        return self._cache[key]

    def compress(self, s: float, e: float, recorded: dict):
        key = ("compress", round(s, 2), round(e, 2))
        if key not in self._cache:
            vt, times = tools.compress_tensor(self.path, s, e)
            target = max(1, round(recorded["retention"] * recorded["base_tokens"]))
            clip = self.engine.encode_video_compressed(vt, target, times)
            self._cache[key] = clip
        return self._cache[key]


def p1_messages(traj: dict, cache: ClipCache) -> tuple[list, list]:
    """Question + skim + every successful tool result, one user turn, no history."""
    dur = traj["duration"]
    clip0, n0 = cache.skim()
    parts = [clip0,
             data.format_question({"question": traj["question"],
                                   "options": traj["options"]})
             + "\n\n" + config.initial_view_text(dur, n0)]
    repro = []
    for tr in successful_calls(traj):
        s, e = float(tr["span"][0]), float(tr["span"][1])
        if tr["tool"] == "crop_video":
            clip, n = cache.crop(s, e)
            parts += [f"\n\nA crop_video call returned {n} full-detail frames "
                      f"covering {s:.0f}s-{e:.0f}s (1 fps):\n", clip]
        else:
            clip = cache.compress(s, e, tr)
            repro.append({"span": [s, e], "kept_recorded": tr["kept_tokens"],
                          "kept_reproduced": clip.meta["kept_tokens"]})
            parts += [f"\n\nA compress_video call returned a compressed overview of "
                      f"{s:.0f}s-{e:.0f}s ({clip.meta['kept_tokens']} tokens):\n", clip]
    parts += ["\n\nBased on ALL the frames and overviews above, answer the question. "
              + config.ANSWER_INSTR]
    return [{"role": "user", "parts": parts}], repro


def p2_messages(traj: dict, cache: ClipCache) -> tuple[list, list]:
    """Question + ONLY the compressed clip(s) the agent produced. No skim, no crops."""
    dur = traj["duration"]
    parts = [data.format_question({"question": traj["question"],
                                   "options": traj["options"]})
             + f"\n\nThis video is {dur:.0f} seconds long. You are shown ONLY "
             "compressed overviews of the span(s) below — no other frames."]
    repro = []
    for tr in successful_calls(traj):
        if tr["tool"] != "compress_video":
            continue
        s, e = float(tr["span"][0]), float(tr["span"][1])
        clip = cache.compress(s, e, tr)
        repro.append({"span": [s, e], "kept_recorded": tr["kept_tokens"],
                      "kept_reproduced": clip.meta["kept_tokens"]})
        parts += [f"\n\nCompressed overview of {s:.0f}s-{e:.0f}s "
                  f"({clip.meta['kept_tokens']} tokens):\n", clip]
    parts += ["\n\n" + config.ANSWER_INSTR]
    return [{"role": "user", "parts": parts}], repro


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools-run", required=True)
    ap.add_argument("--base-run", required=True)
    ap.add_argument("--probes", default="p1,p2")
    ap.add_argument("--limit", type=int, default=None, help="max samples per probe")
    ap.add_argument("--p2-bw-sample", type=int, default=50,
                    help="seeded sample size of compress-involved both_wrong for P2")
    ap.add_argument("--qids", default=None, help="comma-separated qid filter (smoke)")
    ap.add_argument("--skip-qids", default=None,
                    help="comma-separated qids to exclude (e.g. known-corrupted source video)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    def _run_dir(name):
        return name if os.path.isabs(name) else os.path.join(config.RUN_ROOT, name)

    run_tools = viz.load_run(_run_dir(args.tools_run))
    run_base = viz.load_run(_run_dir(args.base_run))
    tools_by = {str(t["question_id"]): t for t in run_tools}
    cats = viz.pair_runs(run_tools, run_base)

    def has_compress(qid):
        return any(tr["tool"] == "compress_video"
                   for tr in successful_calls(tools_by[qid]))

    flips = cats["fixed"] + cats["broke"]
    p1_items = [("p1", it["qid"], it) for it in flips]
    p2_flips = [it for it in flips if has_compress(it["qid"])]
    bw_comp = [it for it in cats["both_wrong"] if has_compress(it["qid"])]
    rng = random.Random(0)
    rng.shuffle(bw_comp)
    p2_items = [("p2", it["qid"], it)
                for it in p2_flips + bw_comp[: args.p2_bw_sample]]

    todo = []
    if "p1" in args.probes:
        todo += p1_items[: args.limit] if args.limit else p1_items
    if "p2" in args.probes:
        todo += p2_items[: args.limit] if args.limit else p2_items
    if args.qids:
        keep = set(args.qids.split(","))
        todo = [x for x in todo if x[1] in keep]
    if args.skip_qids:
        skip = set(args.skip_qids.split(","))
        todo = [x for x in todo if x[1] not in skip]

    probe_dir = os.path.join(
        config.RUN_ROOT,
        f"probe_{os.path.basename(args.tools_run)}_vs_{os.path.basename(args.base_run)}")
    os.makedirs(probe_dir, exist_ok=True)
    out_path = os.path.join(probe_dir, "probes.jsonl")
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            if line.strip():
                r = json.loads(line)
                done.add((str(r["qid"]), r["probe"]))
    todo = [x for x in todo if (x[1], x[0]) not in done]
    # group by qid so P1/P2 share one ClipCache (one decode+encode per span)
    todo.sort(key=lambda x: (x[1], x[0]))
    print(f"[probes] {len(todo)} to run (skipped {len(done)} done) -> {out_path}",
          flush=True)

    engine = Engine(device=args.device)
    t0 = time.time()
    cache, cache_qid = None, None
    for i, (probe, qid, it) in enumerate(todo):
        traj = tools_by[qid]
        if qid != cache_qid:
            cache, cache_qid = ClipCache(engine, traj["video_path"]), qid
        t1 = time.time()
        row = {"qid": qid, "probe": probe, "bucket": next(
                   b for b, v in cats.items() if any(x["qid"] == qid for x in v)),
               "task": traj["task_type"], "gold": traj["gold"],
               "agent_pred": traj["pred"]}
        try:
            msgs, repro = (p1_messages if probe == "p1" else p2_messages)(traj, cache)
            pred, text, ctx = _answer_once(engine, msgs)
            row.update({"pred": pred, "correct": pred == traj["gold"],
                        "context_tokens": ctx, "compress_repro": repro,
                        "text_tail": text[-400:]})
        except Exception as e:
            import traceback
            traceback.print_exc()
            row.update({"pred": None, "correct": False,
                        "error": f"{type(e).__name__}: {e}"})
        finally:
            torch.cuda.empty_cache()
        row["seconds"] = round(time.time() - t1, 1)
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{i + 1}/{len(todo)}] {probe} {qid:<8} ({row['bucket']}) "
              f"agent={row['agent_pred']} probe={row.get('pred')} gold={row['gold']} "
              f"{row['seconds']}s (avg {(time.time() - t0) / (i + 1):.0f}s)", flush=True)

    n_ok = n = 0
    for line in open(out_path):
        if line.strip():
            r = json.loads(line)
            n += 1
            n_ok += bool(r.get("correct"))
    print(f"[probes done] {n} rows, {n_ok} correct -> {out_path}")


if __name__ == "__main__":
    main()
