"""Plain single-pass eval against a vLLM OpenAI endpoint. NO tools, NO agent loop.

  python -m longvt_compression.fast_agent.run_plain --dataset lvbench --num 3

This is the backbone the agentic runs are built on top of, so it deliberately
reuses fast_agent's own pieces rather than reimplementing them:

  data.load_lvbench / format_question   identical question rendering
  tools.initial_frames_with_timestamps  identical 64-frame skim
  config.initial_view_text              identical "frame i is at ~t seconds" legend
  config.ANSWER_INSTR                   identical answer instruction
  data.extract_answer                   identical scoring (refusal-aware)

Only the model call differs: one HTTP request instead of the in-process HF
`assemble`+`decode`. So when tools are switched on later, the prompt, the frames
and the scoring are already the same and the two runs are directly comparable.

Writes per run:
  results.jsonl   one line per question (pred, gold, correct, usage, traj_path)
  run_manifest.json  everything that changes behaviour (incl. temperature+seed)
  traj/<qid>.json full raw response text -- the failure-mode record
Resumable: re-running skips question_ids already present in results.jsonl.
"""

import argparse
import base64
import io
import json
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from . import config, data, longvt_scoring, tools

DEFAULT_BASE_URL = os.environ.get("FA_BASE_URL", "http://localhost:8010/v1")
DEFAULT_MODEL = os.environ.get("FA_MODEL", "qwen3vl")


def _b64_image(pil, quality: int = 90, png: bool = False) -> str:
    buf = io.BytesIO()
    if png:
        pil.save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    pil.convert("RGB").save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# Official-style LVBench instruction: frames + question + options, answer with a
# bare letter. No <think>, no duration/timestamp legend, no tool references --
# those all exist for the agentic arm and would be dangling text in a no-tool run.
OFFICIAL_INSTR = (
    "Please select the best answer from the options above and directly provide "
    "the letter representing your choice without giving any explanation."
)


# Upstream lmms-eval `lvbench.yaml` post_prompt, verbatim -- note there is NO trailing
# period (utils.py's own default has one, but the yaml override does not, and the yaml
# wins). This is the prompt behind the paper's Qwen2.5-VL-7B LVBench numbers.
LMMS_EVAL_POST_PROMPT = "\nAnswer the question with the option letter"


def _lmms_eval_prompt(row: dict) -> str:
    """`lvbench_doc_to_text` reproduced: stem + "(A) text" options + post_prompt.

    Our loader normalizes LVBench's native `(A) text` into VideoMME's `A. text`
    (data._split_lvbench_question) so format_question renders uniformly; undo that
    here, because upstream renders `f"({letter}) {opt}"` and prompt text is not a
    detail we get to normalize when the point is matching a published number.
    """
    opts = []
    for o in row["options"]:
        letter, text = o[0], o[2:].strip() if len(o) > 2 else ""
        opts.append(f"({letter}) {text}")
    return row["question"] + "\n" + "\n".join(opts) + LMMS_EVAL_POST_PROMPT


def build_messages(row: dict, pils: list, skim_times: list, dur: float,
                   style: str = "official") -> list:
    """Text turn for one question.

    style="official":   LVBench question + options + bare-letter instruction. Nothing
        else -- this isolates vanilla model performance on the 64-frame input.
    style="lmms_eval":  upstream lmms-eval's `lvbench` task prompt, verbatim. Use this
        (with T=0, max_tokens=16, PNG frames) to reproduce a published number; every
        other style is ours and is only internally comparable.
    style="longvt":     the agentic prompt (duration + per-frame timestamp legend +
        tool instructions). Kept because the tool arm NEEDS the timestamp legend to
        aim a crop -- without it, measured GT-evidence coverage was 0.00 on 6/6.
    """
    if style == "lmms_eval":
        prompt = _lmms_eval_prompt(row)
    elif style == "official":
        # Raw LVBench question: no clock->seconds annotation either, since that
        # rewrite ("17:16 (= 1036 s)") exists only so tool calls get seconds.
        prompt = row["question"] + "\n" + "\n".join(row["options"]) + "\n" + OFFICIAL_INSTR
    else:
        prompt = (
            data.format_question(row)
            + "\n\n" + config.initial_view_text(dur, len(pils), skim_times)
            + "\n\n" + config.ANSWER_INSTR
        )
    # lmms-eval encodes frames as PNG by default (LMMS_IMAGE_ENCODE_FORMAT); match it
    # for the reproduction style rather than silently re-compressing at JPEG q90.
    png = style == "lmms_eval"
    content = [{"type": "image_url", "image_url": {"url": _b64_image(p, png=png)}}
               for p in pils]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def answer_one(client, model: str, row: dict, args) -> dict:
    """Decode frames, one chat call, score. Returns the results.jsonl row."""
    t0 = time.time()
    dur = data.video_duration(row["video_path"])
    pils, skim_times = tools.initial_frames_with_timestamps(row["video_path"])
    t_decode = time.time() - t0

    messages = build_messages(row, pils, skim_times, dur, style=args.prompt_style)
    t1 = time.time()
    # presence_penalty: Qwen3-VL's own recommended value for the small instruct
    # models (8B/4B/2B) is 2.0. Without it Qwen3-VL-8B degenerates into token-level
    # repetition loops ("frame 458, frame 459, frame 461, ...") that burn the whole
    # max_tokens budget and emit no <answer> at all -- measured at 7-10% of LVBench
    # questions. Raising max_tokens does NOT fix that (legit answers need p99 ~1700
    # of the 2048 cap; loops just run longer), the penalty does.
    extra = {"repetition_penalty": args.repetition_penalty}
    if args.top_k > 0:
        extra["top_k"] = args.top_k
    resp = client.chat.completions.create(
        model=model, messages=messages,
        temperature=args.temperature, top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        max_tokens=args.max_tokens, seed=args.seed,
        extra_body=extra,
    )
    t_gen = time.time() - t1

    text = resp.choices[0].message.content or ""
    pred = data.extract_answer(text)
    # Second, independent scorer: upstream lmms-eval's own extractor. Both columns are
    # written so a gap between them is visible instead of being absorbed into the
    # headline -- LongVT's table scores its vanilla row and its tool rows with
    # DIFFERENT scorers, so scorer effects are a real confound in this literature.
    pred_lmms = longvt_scoring.extract_mcq_answer(text, choices=["A", "B", "C", "D"]) or None
    gold = row["answer"]
    usage = resp.usage
    return {
        "question_id": str(row["question_id"]), "task_type": row["task_type"],
        "videoID": row["videoID"], "duration": dur,
        "pred": pred, "gold": gold, "correct": pred == gold,
        "pred_lmms": pred_lmms, "correct_lmms": pred_lmms == gold,
        "n_frames": len(pils),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "finish_reason": resp.choices[0].finish_reason,
        "decode_seconds": round(t_decode, 2), "gen_seconds": round(t_gen, 2),
        "evidence": row.get("evidence"),
        "localization_trivial": row.get("localization_trivial"),
        "_text": text,   # popped into traj/, never written to results.jsonl
    }


def write_summary(run_dir: str) -> dict:
    by_task, total = defaultdict(lambda: [0, 0]), [0, 0]
    none_pred = 0
    lmms_ok, lmms_seen, scorer_disagree = 0, 0, 0
    with open(os.path.join(run_dir, "results.jsonl")) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ok = bool(r.get("correct"))
            by_task[r.get("task_type", "?")][0] += ok
            by_task[r.get("task_type", "?")][1] += 1
            total[0] += ok
            total[1] += 1
            none_pred += r.get("pred") is None
            if "correct_lmms" in r:
                lmms_seen += 1
                lmms_ok += bool(r["correct_lmms"])
                scorer_disagree += bool(r["correct_lmms"]) != ok
    n = max(total[1], 1)
    s = {"n": total[1], "accuracy": round(100 * total[0] / n, 2),
         "no_answer": none_pred,
         "by_task": {t: {"correct": c, "n": m, "acc": round(100 * c / max(m, 1), 1)}
                     for t, (c, m) in sorted(by_task.items())}}
    if lmms_seen:
        # Reported alongside, never instead of: a large disagreement means our
        # extractor is doing something the official one is not, and that has to be
        # diagnosed rather than picked between.
        s["accuracy_lmms_eval"] = round(100 * lmms_ok / max(lmms_seen, 1), 2)
        s["scorer_disagree"] = scorer_disagree
        s["scorer_disagree_pct"] = round(100 * scorer_disagree / max(lmms_seen, 1), 2)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lvbench", choices=["lvbench", "videomme"])
    ap.add_argument("--num", type=int, default=None, help="questions (default: all)")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed AND data-draw seed")
    ap.add_argument("--tag", default=time.strftime("%m%d"))
    # None = "take this style's default"; an explicit flag always wins (see below).
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--presence-penalty", dest="presence_penalty", type=float, default=0.0)
    ap.add_argument("--frequency-penalty", dest="frequency_penalty", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", dest="repetition_penalty", type=float, default=1.0)
    ap.add_argument("--top-k", dest="top_k", type=int, default=20)
    ap.add_argument("--prompt-style", dest="prompt_style", default="official",
                    choices=["official", "lmms_eval", "longvt"],
                    help="official = question+options+bare-letter instruction (this "
                         "baseline); lmms_eval = upstream lmms-eval lvbench task, "
                         "verbatim (reproduces a published number; forces T=0, "
                         "max_tokens=16, PNG frames unless overridden); "
                         "longvt = agentic prompt with timestamp legend")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent questions; vLLM batches them server-side")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out-root", default="/local1/cfyang/hanklin/outputs/lvbench_plain")
    args = ap.parse_args()

    # Per-style sampling defaults. lmms_eval reproduces a published number, so it
    # inherits lmms-eval's own defaults: temperature 0 (async_openai's default -- the
    # lvbench task yaml sets none) and max_new_tokens 16 (the yaml DOES set that).
    # Any explicitly passed flag still wins, so a deviation is always deliberate.
    _STYLE_DEFAULTS = {"lmms_eval": {"temperature": 0.0, "max_tokens": 16}}
    style_defaults = _STYLE_DEFAULTS.get(args.prompt_style, {})
    if args.temperature is None:
        args.temperature = style_defaults.get("temperature", 0.7)
    if args.max_tokens is None:
        args.max_tokens = style_defaults.get("max_tokens", 32)

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    run_dir = os.path.join(args.out_root, f"{args.dataset}_plain_{args.tag}_seed{args.seed}")
    os.makedirs(os.path.join(run_dir, "traj"), exist_ok=True)
    results_path = os.path.join(run_dir, "results.jsonl")

    manifest = {"dataset": args.dataset, "num": args.num, "seed": args.seed,
                "temperature": args.temperature, "top_p": args.top_p,
                "presence_penalty": args.presence_penalty,
                "frequency_penalty": args.frequency_penalty,
                "repetition_penalty": args.repetition_penalty,
                "top_k": args.top_k, "prompt_style": args.prompt_style,
                "image_format": "PNG" if args.prompt_style == "lmms_eval" else "JPEG90",
                "max_tokens": args.max_tokens, "model": args.model,
                "initial_frames": config.INITIAL_FRAMES, "max_pixels": config.MAX_PIXELS,
                "skim_timestamps": config.SKIM_TIMESTAMPS,
                "unify_time_format": data.UNIFY_TIME_FORMAT,
                "tools": [], "arm": "plain",
                "launched": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(run_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    rows = data.load_dataset(args.dataset, n=args.num, seed=args.seed)

    done = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                try:
                    done.add(str(json.loads(line)["question_id"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    todo = [r for r in rows if str(r["question_id"]) not in done]
    print(f"[plain] {args.dataset} n={len(rows)} todo={len(todo)} (skip {len(done)}) "
          f"style={args.prompt_style} T={args.temperature} top_p={args.top_p} "
          f"top_k={args.top_k} pp={args.presence_penalty} fp={args.frequency_penalty} "
          f"rp={args.repetition_penalty} max_tok={args.max_tokens} seed={args.seed}\n"
          f"        -> {run_dir}", flush=True)

    lock = threading.Lock()
    t0, n_done = time.time(), [0]

    def work(row):
        try:
            r = answer_one(client, args.model, row, args)
        except Exception as e:
            r = {"question_id": str(row["question_id"]), "task_type": row.get("task_type"),
                 "pred": None, "gold": row["answer"], "correct": False,
                 "error": f"{type(e).__name__}: {e}", "_text": ""}
        text = r.pop("_text", "")
        qid = r["question_id"]
        tp = os.path.join(run_dir, "traj", f"{qid}.json")
        with open(tp, "w") as tf:
            json.dump({"question_id": qid, "question": row["question"],
                       "options": row["options"], "gold": row["answer"],
                       "pred": r.get("pred"), "response": text,
                       "evidence": row.get("evidence"),
                       "time_reference": row.get("time_reference")}, tf, indent=1)
        r["traj_path"] = tp
        with lock:
            with open(results_path, "a") as rf:
                rf.write(json.dumps(r) + "\n")
            n_done[0] += 1
            s = write_summary(run_dir)
            el = time.time() - t0
            print(f"[{n_done[0]}/{len(todo)}] {qid} pred={r.get('pred')} gold={r['gold']} "
                  f"{'OK ' if r.get('correct') else 'X  '}| acc {s['accuracy']}% n={s['n']} "
                  f"| {el / max(n_done[0], 1):.1f}s/q"
                  + (f" | ERR {r['error'][:60]}" if r.get("error") else ""), flush=True)
        return r

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))

    s = write_summary(run_dir)
    print(f"\n[SUMMARY] accuracy {s['accuracy']}% over n={s['n']} "
          f"| no-answer {s['no_answer']} | {time.time() - t0:.0f}s total")
    for t, d in s["by_task"].items():
        print(f"  {t:<28} {d['correct']:>3}/{d['n']:<3} = {d['acc']}%")


if __name__ == "__main__":
    main()
