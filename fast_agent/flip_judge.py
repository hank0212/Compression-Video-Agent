"""Judge-VLM triage of a paired run: classify every flip (and both_wrong) so
human inspection starts from an aggregated failure taxonomy instead of 481 raw
trajectories.

  python -m fast_agent.flip_judge --tools-run compress_run1 --base-run baseline_run1 \
      [--buckets fixed,broke,both_wrong] [--limit N] [--dry-run]

Judge = any OpenAI-compatible VLM server (serve_judge.sh starts a single-GPU
Qwen3-VL-8B). Endpoint from $OPENAI_BASE_URL (default http://localhost:8010/v1),
model auto-discovered from /v1/models (override $FA_JUDGE_MODEL).

Each sample's dossier = question/options/gold + both arms' answers, the initial
skim montage, and per tool round the action line + truncated thinking + the
montage of frames that call returned. The judge sees the GOLD answer and both
arms side by side (the agent had neither) — that asymmetry is what makes an
8B self-judge usable for TRIAGE. Treat categories as triage, not ground truth;
the report lists exemplar qids per category for human replay via viz.show.

Writes RUN_ROOT/judge_<toolsrun>_vs_<baserun>/:
  judgments.jsonl  one line per sample (resumable: rerun skips existing qids)
  report.md/.json  category x bucket / x tool-mix / x task cross-tabs + exemplars
"""

import argparse
import base64
import json
import os
import re
import time
from collections import Counter, defaultdict

import requests

from . import config, viz

DEFAULT_ENDPOINT = os.environ.get("OPENAI_BASE_URL", "http://localhost:8010/v1")

# One category must be chosen per sample; grounded in the 2026-06-12 archive
# audit (never-gathered / tool-hallucinated / over-trust).
CATEGORIES = {
    "broke": {
        "bad_span": "the tool calls never covered the answer-bearing part of the video, and the wrong answer traces to that miss",
        "over_exploration": "the initial skim already supported the correct answer (baseline proves it); the extra tool context diluted or distracted the model away from it",
        "misread_tool_frames": "the answer-bearing content IS visible in the frames a tool returned, but the model drew the wrong conclusion from it",
        "format_or_extraction": "the model's reasoning reached the right content but the final letter was mangled (no/garbled <answer>, letter mismatch with its own conclusion)",
        "annotation_or_ambiguous": "the question/options/gold look ambiguous or wrong; neither answer is clearly better",
    },
    "fixed": {
        "localization_gain": "a tool located the answer-bearing moment that the sparse skim missed, and the answer follows from that localization",
        "detail_gain": "the skim covered the right moment but at too coarse a view; full-detail crop frames revealed the deciding detail",
        "coverage_gain": "the answer needed aggregating content across a long span; the compressed overview supplied that coverage",
        "unclear_lucky": "the tools arm's reasoning does not actually support its (correct) answer better than baseline's; the flip looks incidental",
    },
    "both_wrong": {
        "evidence_never_gathered": "neither the skim nor any tool call ever covered the answer-bearing part of the video",
        "evidence_seen_but_misread": "the answer-bearing content is visible in the skim or tool frames of at least one arm, but both misread it",
        "skim_insufficient_and_tools_failed": "the skim could not answer it and the tools arm tried the right kind of exploration but failed (bad spans, budget exhausted, compression too lossy)",
        "annotation_or_ambiguous": "the question/options/gold look ambiguous or wrong; the 'wrong' answers are defensible",
    },
}

BUCKET_BLURB = {
    "broke": "The BASELINE arm answered CORRECTLY; the TOOLS arm answered WRONG. Diagnose what the tool-using trajectory did that lost a question the plain skim got right.",
    "fixed": "The BASELINE arm answered WRONG; the TOOLS arm answered CORRECTLY. Diagnose what the tools actually contributed.",
    "both_wrong": "BOTH arms answered WRONG. Diagnose the shared failure.",
}


# ---------------------------------------------------------------------------
# dossier

def _trunc(s: str, head: int = 700, tail: int = 500) -> str:
    s = (s or "").strip()
    if len(s) <= head + tail + 20:
        return s
    return s[:head] + f"\n …[{len(s) - head - tail} chars truncated]… \n" + s[-tail:]


def _image_part(path: str | None) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def _arm_rounds_text(traj: dict, arm: str) -> list:
    """Interleaved [text, image?, text, image?, ...] parts for one arm's rounds."""
    parts = [{"type": "text", "text": f"\n--- {arm} arm trajectory ---"}]
    for r in traj["rounds"]:
        a = r.get("action") or {}
        if a.get("kind") == "tool_call":
            act = f"{a['name']}({a['start']:.0f}s-{a['end']:.0f}s)"
            if a.get("error"):
                act += f" -> REJECTED: {a['error']}"
        else:
            act = a.get("kind", "?")
        txt = (f"\nRound {r['round']} [{act}]\nmodel output:\n"
               f"{_trunc(r['thinking'])}")
        tr = r.get("tool_result")
        if tr:
            cap = f"{tr['tool']} returned {tr.get('n_frames', '?')} frames covering {tr['span'][0]:.0f}s-{tr['span'][1]:.0f}s"
            if tr.get("kept_tokens"):
                cap += (f", compressed {tr['base_tokens']}->{tr['kept_tokens']} tokens "
                        f"(retention {tr['retention']})")
            txt += f"\n{cap}. Montage of those frames:"
            parts.append({"type": "text", "text": txt})
            img = _image_part(tr.get("montage"))
            if img:
                parts.append(img)
        else:
            parts.append({"type": "text", "text": txt})
    return parts


def build_dossier(bucket: str, tools_t: dict, base_t: dict) -> list:
    """OpenAI-style multimodal content list for one judge call."""
    opts = "\n".join(tools_t["options"])
    verdict = lambda t: "CORRECT" if t["correct"] else "WRONG"
    cats = "\n".join(f"- {k}: {v}" for k, v in CATEGORIES[bucket].items())
    header = (
        "You are auditing a video-QA agent experiment. Two arms of the SAME model "
        "answered the same multiple-choice question about the same video:\n"
        "- BASELINE arm: saw only a 64-frame uniform skim of the whole video.\n"
        "- TOOLS arm: saw the same skim, plus up to 5 tool rounds — crop_video "
        "(<=128 full-detail frames of a chosen span) and/or compress_video (a "
        "compressed-token overview of a long span).\n\n"
        f"{BUCKET_BLURB[bucket]}\n\n"
        f"=== SAMPLE {tools_t['question_id']} ({tools_t['task_type']}) ===\n"
        f"Video duration: {tools_t['duration']:.0f}s\n"
        f"QUESTION: {tools_t['question']}\n{opts}\n"
        f"GOLD ANSWER: {tools_t['gold']}\n"
        f"BASELINE answered: {base_t['pred']} ({verdict(base_t)})\n"
        f"TOOLS answered: {tools_t['pred']} ({verdict(tools_t)})\n\n"
        "IMPORTANT viewing caveat: every montage below is a subsampled, low-res "
        "thumbnail grid (<=48 thumbs at 160px) of what the agent actually consumed "
        "(agent saw up to 128 frames at 224px, or compressed tokens). Small "
        "text/objects may be legible to the agent but not to you — be conservative "
        "about claiming something is ABSENT from what the agent saw.\n\n"
        "Initial 64-frame uniform skim (both arms saw these frames):"
    )
    parts = [{"type": "text", "text": header}]
    img = _image_part(tools_t["initial"].get("montage"))
    if img:
        parts.append(img)
    parts += _arm_rounds_text(base_t, "BASELINE")
    parts += _arm_rounds_text(tools_t, "TOOLS")
    parts.append({"type": "text", "text": (
        f"\nClassify this sample into exactly ONE category:\n{cats}\n\n"
        "Reply with ONLY a JSON object, no other text:\n"
        "{\n"
        '  "category": "<one key from the list above>",\n'
        '  "evidence_span_estimate_s": [start, end] or null,  // where in the video the answer evidence most likely lives\n'
        '  "evidence_visible_in_tool_frames": true/false/null, // null if no tool frames or cannot tell\n'
        '  "from_compress": true/false,  // was a compress_video result central to the failure/gain\n'
        '  "rationale": "<2-4 sentences: what the trajectories show, tied to the category>",\n'
        '  "confidence": "low"/"medium"/"high"\n'
        "}"
    )})
    return parts


# ---------------------------------------------------------------------------
# judge client

def discover_model(endpoint: str) -> str:
    m = os.environ.get("FA_JUDGE_MODEL")
    if m:
        return m
    r = requests.get(f"{endpoint}/models", timeout=10)
    r.raise_for_status()
    return r.json()["data"][0]["id"]

def _chat(endpoint: str, model: str, messages: list, max_tokens: int = 600) -> str:
    r = requests.post(f"{endpoint}/chat/completions", json={
        "model": model, "messages": messages,
        "temperature": 0.0, "max_tokens": max_tokens,
    }, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def _parse_judgment(text: str, bucket: str) -> dict | None:
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if d.get("category") not in CATEGORIES[bucket]:
        return None
    return d


def judge_sample(endpoint: str, model: str, content: list, bucket: str) -> tuple[dict | None, str]:
    """One judged sample; on unparseable output, one retry demanding bare JSON."""
    messages = [{"role": "user", "content": content}]
    reply = _chat(endpoint, model, messages)
    d = _parse_judgment(reply, bucket)
    if d is None:
        messages += [
            {"role": "assistant", "content": reply},
            {"role": "user", "content":
                "Reply again with ONLY the JSON object in the requested schema. "
                f"\"category\" must be one of: {', '.join(CATEGORIES[bucket])}."},
        ]
        reply = _chat(endpoint, model, messages)
        d = _parse_judgment(reply, bucket)
    return d, reply


# ---------------------------------------------------------------------------
# report

def _tool_mix(used: list[str]) -> str:
    if any(u == "compress_video" for u in used):
        return "compress" + ("+crop" if "crop_video" in used else "")
    return "crop" if used else "none"


def write_report(judge_dir: str) -> dict:
    js = list(viz.judgments(judge_dir).values())
    ok = [j for j in js if j.get("category")]
    rep = {"n": len(js), "parsed": len(ok),
           "by_bucket": {}, "by_tool_mix": {}, "by_task": {}, "exemplars": {}}
    md = [f"# Judge report — {os.path.basename(judge_dir)}",
          f"{len(js)} judged, {len(js) - len(ok)} parse failures\n"]
    for bucket in ("broke", "fixed", "both_wrong"):
        sub = [j for j in ok if j["bucket"] == bucket]
        if not sub:
            continue
        cat = Counter(j["category"] for j in sub)
        rep["by_bucket"][bucket] = dict(cat)
        md.append(f"\n## {bucket} (n={len(sub)})\n")
        md.append("| category | n | % | compress-involved |")
        md.append("|---|---|---|---|")
        for c, n in cat.most_common():
            comp = sum(1 for j in sub if j["category"] == c
                       and (j.get("from_compress") or "compress" in _tool_mix(j["used"])))
            md.append(f"| {c} | {n} | {100 * n / len(sub):.0f}% | {comp} |")
        mix = Counter((_tool_mix(j["used"]), j["category"]) for j in sub)
        rep["by_tool_mix"][bucket] = {f"{m}/{c}": n for (m, c), n in mix.items()}
        task = Counter((j["task"], j["category"]) for j in sub)
        rep["by_task"][bucket] = {f"{t}/{c}": n for (t, c), n in task.items()}
        md.append(f"\n**by task type ({bucket}):**\n")
        by_t = defaultdict(Counter)
        for (t, c), n in task.items():
            by_t[t][c] += n
        for t, cc in sorted(by_t.items(), key=lambda kv: -sum(kv[1].values())):
            md.append(f"- {t}: " + ", ".join(f"{c} {n}" for c, n in cc.most_common()))
        md.append(f"\n**exemplars ({bucket})** — replay with `viz.show(run, qid=...)`:\n")
        rep["exemplars"][bucket] = {}
        for c, _ in cat.most_common():
            ex = [j for j in sub if j["category"] == c][:5]
            rep["exemplars"][bucket][c] = [j["qid"] for j in ex]
            md.append(f"- **{c}**:")
            for j in ex:
                first = (j.get("rationale") or "").split(". ")[0]
                md.append(f"    - `{j['qid']}` ({j['task']}) — {first}")
    text = "\n".join(md) + "\n"
    with open(os.path.join(judge_dir, "report.md"), "w") as f:
        f.write(text)
    with open(os.path.join(judge_dir, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    return rep


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools-run", required=True, help="run dir name under RUN_ROOT (or abs path)")
    ap.add_argument("--base-run", required=True)
    ap.add_argument("--buckets", default="fixed,broke,both_wrong")
    ap.add_argument("--limit", type=int, default=None, help="max samples per bucket")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--dry-run", action="store_true",
                    help="build dossiers only; write a text preview, no judge calls")
    args = ap.parse_args()

    def _run_dir(name):
        return name if os.path.isabs(name) else os.path.join(config.RUN_ROOT, name)

    run_tools = viz.load_run(_run_dir(args.tools_run))
    run_base = viz.load_run(_run_dir(args.base_run))
    tools_by, base_by = ({t["question_id"]: t for t in r} for r in (run_tools, run_base))
    cats = viz.pair_runs(run_tools, run_base)
    print("paired buckets:", {k: len(v) for k, v in cats.items()})

    judge_dir = os.path.join(
        config.RUN_ROOT,
        f"judge_{os.path.basename(args.tools_run)}_vs_{os.path.basename(args.base_run)}")
    os.makedirs(judge_dir, exist_ok=True)
    out_path = os.path.join(judge_dir, "judgments.jsonl")
    done = set(viz.judgments(judge_dir)) if os.path.exists(out_path) else set()

    todo = []
    for bucket in args.buckets.split(","):
        items = cats[bucket][: args.limit] if args.limit else cats[bucket]
        todo += [(bucket, it) for it in items if it["qid"] not in done]
    print(f"[judge] {len(todo)} to judge (skipped {len(done)} done) -> {judge_dir}")

    if args.dry_run:
        if todo:
            bucket, it = todo[0]
            parts = build_dossier(bucket, tools_by[it["qid"]], base_by[it["qid"]])
            n_img = sum(p["type"] == "image_url" for p in parts)
            n_chr = sum(len(p["text"]) for p in parts if p["type"] == "text")
            preview = "\n".join(p["text"] if p["type"] == "text" else "[IMAGE]"
                                for p in parts)
            pv = os.path.join(judge_dir, "dossier_preview.txt")
            with open(pv, "w") as f:
                f.write(preview)
            print(f"[dry-run] first dossier ({bucket}/{it['qid']}): "
                  f"{n_img} images, {n_chr} text chars -> {pv}")
        return

    model = discover_model(args.endpoint)
    print(f"[judge] endpoint={args.endpoint} model={model}")
    t0 = time.time()
    with open(out_path, "a") as f:
        for i, (bucket, it) in enumerate(todo):
            t = time.time()
            row = {"qid": it["qid"], "bucket": bucket, "task": it["task"],
                   "gold": it["gold"], "base_pred": it["base"],
                   "tools_pred": it["tools"], "used": it["used"]}
            try:
                parts = build_dossier(bucket, tools_by[it["qid"]], base_by[it["qid"]])
                d, raw = judge_sample(args.endpoint, model, parts, bucket)
                if d is None:
                    row.update({"category": None, "raw": raw})
                else:
                    row.update(d)
            except Exception as e:  # per-sample isolation
                import traceback
                traceback.print_exc()
                row.update({"category": None, "error": f"{type(e).__name__}: {e}"})
            row["judge_seconds"] = round(time.time() - t, 1)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(todo)}] {bucket:<10} {it['qid']:<8} "
                  f"-> {row.get('category')} ({row.get('confidence', '-')}) "
                  f"{row['judge_seconds']}s (avg {(time.time() - t0) / (i + 1):.0f}s)")

    rep = write_report(judge_dir)
    print(f"\n[report] {judge_dir}/report.md")
    for bucket, cc in rep["by_bucket"].items():
        print(f"  {bucket}: " + ", ".join(f"{c} {n}" for c, n in
                                          sorted(cc.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
