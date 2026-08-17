"""Read the flip cases of a paired run and write a visual notebook.

Two experiments, one instrument:

  # 1. compression tool: does giving the agent crop+compress help?
  python -m fast_agent.case_notes --exp videomme

  # 2. oracle: does forcing the crop onto the GT evidence span help?
  python -m fast_agent.case_notes --exp lvbench

For every question where the two arms disagree (one right, one wrong) this sends
the WHOLE trajectory -- the agent's own reasoning, every tool call, and the
montage of frames each call returned -- to a local Qwen3-VL and asks it to
summarise what the agent did and why the answer changed.  It never asks Qwen to
answer the question; the point is to read behaviour, not to re-grade.

Output: RUN_ROOT/notes_<exp>/{notes.jsonl, CASES_<exp>.ipynb}.  notes.jsonl is
resumable -- rerunning skips qids already done, so a crashed run just continues.

Needs a judge server:  bash fast_agent/serve_judge.sh 0 8010
"""

import argparse
import base64
import glob
import io
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import nbformat as nbf
from PIL import Image

from . import config, data
from .judge_client import DEFAULT_ENDPOINT, _chat, _image_part, _trunc, discover_model

# ---------------------------------------------------------------- experiments

EXPERIMENTS = {
    # exp: (method arm, comparison arm, method label, comparison label, blurb)
    "videomme": (
        "compress_run2", "baseline_run1", "tools", "baseline",
        "VideoMME-long, 882 paired questions. The `tools` arm may call "
        "`crop_video` (<=128 full-detail frames of a chosen window) and "
        "`compress_video` (a wide span squeezed by FlashVID into a matched token "
        "budget). The `baseline` arm answers from a 64-frame skim with no tools. "
        "Net effect is a null -- 67 fixed, 75 broke -- so these flips are the "
        "interesting minority, not the typical question.",
    ),
    "lvbench": (
        "lvbench_oracle_crop_cot", "lvbench_crop_full", "oracle", "autonomous",
        "LVBench, 140 paired questions. Both arms get the SAME tool "
        "(`crop_video`) and the same budget -- the only difference is who picks "
        "the window. The `autonomous` arm chooses its own; the `oracle` arm is "
        "forced by the harness onto the ground-truth evidence span. So a flip "
        "isolates localisation from perception.",
    ),
}

# --------------------------------------------------------------- trajectories


def load_run(run: str) -> dict:
    out = {}
    for p in sorted(glob.glob(os.path.join(config.RUN_ROOT, run, "traj", "*.json"))):
        d = json.load(open(p))
        out[str(d["question_id"])] = d
    return out


def rescore(t: dict) -> str | None:
    """Re-derive the prediction with the CURRENT parser.

    The stored `pred`/`correct` fields are stale on a handful of samples (a
    parser bug fixed after these runs), so never trust them.
    """
    return data.extract_answer("\n".join((r.get("thinking") or "") for r in t["rounds"]))


def calls(t: dict) -> list[dict]:
    """Successful tool calls, in order, with their result metadata."""
    out = []
    for r in t["rounds"]:
        a = r.get("action") or {}
        if a.get("kind") != "tool_call" or a.get("error"):
            continue
        tr = r.get("tool_result") or {}
        out.append({"round": r.get("round"), "name": a.get("name"),
                    "start": a.get("start"), "end": a.get("end"),
                    "forced": bool(a.get("oracle_forced")),
                    "n_frames": tr.get("n_frames"), "montage": tr.get("montage"),
                    "kept_tokens": tr.get("kept_tokens"),
                    "base_tokens": tr.get("base_tokens"),
                    "retention": tr.get("retention")})
    return out


def _clean(s: str | None) -> str:
    """Model text with the tool-call JSON blob collapsed -- the action line
    already says what was called, and the raw blob crowds out the reasoning."""
    s = (s or "").strip()
    s = re.sub(r"<tool_call>\s*\{.*?\}\s*</tool_call>", "[tool call]", s, flags=re.S)
    return s.strip()


def call_line(c: dict) -> str:
    span = f"{c['start']:.0f}s-{c['end']:.0f}s" if c["start"] is not None else "?"
    bits = [f"{c['name']}({span})"]
    if c["forced"]:
        bits.append("FORCED by harness at the ground-truth span")
    if c.get("n_frames"):
        bits.append(f"{c['n_frames']} frames")
    if c.get("retention") is not None:
        bits.append(f"{c['base_tokens']:,}->{c['kept_tokens']:,} tokens "
                    f"(retention {c['retention']:.2f})")
    return " | ".join(bits)


def replay(t: dict, label: str) -> str:
    """Chronological text replay of one arm: reasoning + what it called."""
    lines = [f"--- {label} arm, {len(t['rounds'])} rounds ---"]
    for r in t["rounds"]:
        a = r.get("action") or {}
        kind = a.get("kind", "?")
        if kind == "tool_call" and not a.get("error"):
            tr = r.get("tool_result") or {}
            head = call_line({"name": a.get("name"), "start": a.get("start"),
                              "end": a.get("end"), "forced": bool(a.get("oracle_forced")),
                              "n_frames": tr.get("n_frames"),
                              "kept_tokens": tr.get("kept_tokens"),
                              "base_tokens": tr.get("base_tokens"),
                              "retention": tr.get("retention")})
        elif kind == "tool_call":
            head = f"{a.get('name')} REJECTED: {a.get('error')}"
        else:
            head = kind
        think = _clean(r.get("thinking"))
        lines.append(f"[round {r.get('round')}] -> {head}\n{_trunc(think, 700, 400)}")
    return "\n\n".join(lines)


# --------------------------------------------------------------------- dossier

SYSTEM = (
    "You are helping a researcher read the behaviour of a video agent. The agent was "
    "given a multiple-choice question about a long video and, in some arms, tools to "
    "look at it: crop_video returns up to 128 full-detail frames of a short window, "
    "compress_video returns a wide span squeezed into a small token budget (it keeps "
    "the gist and loses fine detail like small text and exact counts).\n\n"
    "You will see the agent's own transcript from two arms that disagreed, plus the "
    "frames each tool call returned. Do NOT answer the question yourself and do NOT "
    "re-grade the agent. Explain what it did and why the answer changed between the "
    "two arms, grounding what you say in the agent's own words and in what you can "
    "actually see in the frames.\n\n"
    "Every montage is a grid of at most 48 low-resolution thumbnails, read "
    "left-to-right then top-to-bottom, uniformly spanning the window. The agent saw "
    "more frames at higher resolution than you do, so small text and exact counts may "
    "be legible to it and not to you. If you cannot tell, say so plainly -- that is a "
    "useful answer, not a failure."
)

SCHEMA = (
    "Reply with ONLY a JSON object:\n"
    '{\n'
    '  "summary": "<3-5 sentences: what the agent did, in order -- what it noticed in '
    'the first view, what it decided to look at and why, what came back, how it '
    'answered>",\n'
    '  "why": "<2-4 sentences: why this question flipped between the two arms. Be '
    'concrete about the mechanism, and quote the agent where it helps>",\n'
    '  "saw_the_evidence": "yes" | "no" | "cannot tell",\n'
    '  "keep_as_example": true | false,\n'
    '  "keep_reason": "<one line: if this is a clean example of behaviour worth '
    'training toward or away from, say which and why; else empty string>"\n'
    "}"
)


def dossier(exp: str, m: dict, b: dict, bucket: str) -> list:
    """Message content for one flip case: text + the frames each arm actually saw."""
    _, _, mlab, blab, _ = EXPERIMENTS[exp]
    gold = m["gold"]
    mp, bp = rescore(m), rescore(b)

    verdict = (f"The {mlab} arm answered {mp} (CORRECT) and the {blab} arm answered "
               f"{bp} (WRONG). Explain what the {mlab} arm did that won the question."
               if bucket == "fixed" else
               f"The {mlab} arm answered {mp} (WRONG) and the {blab} arm answered "
               f"{bp} (CORRECT). Explain what the {mlab} arm did that lost a question "
               f"the {blab} arm already had right.")

    head = [f"QUESTION {m['question_id']}  (task: {m.get('task_type')})",
            f"Video is {m.get('duration', 0):.0f}s long.", "",
            m["question"], ""]
    head += [f"  {o}" for o in m["options"]]
    head += ["", f"Ground truth: {gold}", "", verdict, ""]
    if m.get("oracle"):
        o = m["oracle"]
        head.append(f"NOTE: the {mlab} arm's tool call was forced by the harness onto the "
                    f"ground-truth evidence span {o.get('evidence')} (time_reference "
                    f"{o.get('time_reference')!r}). That call is a harness action, not a "
                    f"choice the model made -- so judge only how it USED what it was given.")

    parts: list = [{"type": "text", "text": "\n".join(head)}]

    init = (m.get("initial") or {}).get("montage")
    if init:
        parts.append({"type": "text", "text":
                      f"\nThe first thing both arms saw: a {m['initial'].get('n_frames')}-frame "
                      f"uniform skim of the whole video."})
        p = _image_part(init)
        if p:
            parts.append(p)

    for arm_t, lab in ((b, blab), (m, mlab)):
        parts.append({"type": "text", "text": "\n" + replay(arm_t, lab)})
        for c in calls(arm_t):
            if not c["montage"]:
                continue
            parts.append({"type": "text", "text":
                          f"\nFrames returned to the {lab} arm by {call_line(c)}:"})
            p = _image_part(c["montage"])
            if p:
                parts.append(p)

    parts.append({"type": "text", "text": "\n" + SCHEMA})
    return parts


_JSON = re.compile(r"\{.*\}", re.S)


def ask(endpoint: str, model: str, parts: list) -> dict:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": parts}]
    reply = _chat(endpoint, model, msgs, max_tokens=900)
    m = _JSON.search(reply or "")
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    msgs += [{"role": "assistant", "content": reply},
             {"role": "user", "content": "Reply again with ONLY the JSON object."}]
    reply = _chat(endpoint, model, msgs, max_tokens=900)
    m = _JSON.search(reply or "")
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"summary": "", "why": "", "error": "unparseable", "raw": (reply or "")[:800]}


# -------------------------------------------------------------------- notebook

def _thumb(path: str | None, width: int = 760, quality: int = 72) -> str:
    """Montage -> inline JPEG data URI, downscaled so the notebook stays openable.

    The originals are 1280px PNGs at up to 1.2 MB each; at ~190 flip cases that
    would be a multi-hundred-MB notebook.
    """
    if not path or not os.path.exists(path):
        return "<i style='color:#888'>[no frames on disk]</i>"
    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (f'<img src="data:image/jpeg;base64,{b64}" style="max-width:100%;'
            'border:1px solid #444;border-radius:4px;margin:4px 0"/>')


def _rounds_md(t: dict, lab: str) -> list[str]:
    md = [f"**{lab} arm** — {len(t['rounds'])} round(s)\n"]
    for r in t["rounds"]:
        a = r.get("action") or {}
        kind = a.get("kind", "?")
        c = None
        if kind == "tool_call" and not a.get("error"):
            tr = r.get("tool_result") or {}
            c = {"name": a.get("name"), "start": a.get("start"), "end": a.get("end"),
                 "forced": bool(a.get("oracle_forced")), "n_frames": tr.get("n_frames"),
                 "kept_tokens": tr.get("kept_tokens"), "base_tokens": tr.get("base_tokens"),
                 "retention": tr.get("retention"), "montage": tr.get("montage")}
            head = "`" + call_line(c) + "`"
        elif kind == "tool_call":
            head = f"`{a.get('name')}` **rejected** — {a.get('error')}"
        else:
            head = f"*{kind}*"
        md.append(f"> **round {r.get('round')}** → {head}")
        think = _clean(r.get("thinking"))
        if think:
            md.append("```\n" + _trunc(think, 1100, 500) + "\n```")
        else:
            md.append("*(no text this round)*")
        if c and c.get("montage"):
            md.append(_thumb(c["montage"]))
    return md


def build_notebook(exp: str, notes: dict, method: dict, base: dict, out_path: str):
    mrun, brun, mlab, blab, blurb = EXPERIMENTS[exp]
    cells = []

    fixed = [q for q, n in notes.items() if n["bucket"] == "fixed"]
    broke = [q for q, n in notes.items() if n["bucket"] == "broke"]
    keep = [q for q, n in notes.items() if n.get("keep_as_example")]

    cells.append(nbf.v4.new_markdown_cell(
        f"# {exp} — flip cases\n\n"
        f"{blurb}\n\n"
        f"**Arms.** `{mlab}` = `{mrun}` · `{blab}` = `{brun}`\n\n"
        f"**Flips read here:** {len(fixed)} fixed (the {blab} arm was wrong, the {mlab} "
        f"arm right) and {len(broke)} broke (the reverse). "
        f"{len(keep)} are flagged as candidate training examples.\n\n"
        f"Every case below shows the question, both arms' answers, a plain-language "
        f"summary of what the agent did, then the full trajectory in order — the "
        f"agent's own words, every tool call, the window it chose, and the frames that "
        f"came back. The summaries were written by a local Qwen3-VL reading the same "
        f"trajectories and frames you see; it was never asked to answer the questions. "
        f"Predictions are re-derived with the current parser, not read from disk.\n\n"
        f"> Montages are grids of ≤48 thumbnails spanning the window left-to-right, "
        f"top-to-bottom. They are downscaled here — the agent saw sharper frames."))

    # Conclusions are written by hand after reading the notes, not generated.
    syn = os.path.join(os.path.dirname(out_path), "synthesis.md")
    if os.path.exists(syn):
        cells.append(nbf.v4.new_markdown_cell(open(syn).read()))

    idx = ["| qid | task | flip | " + blab + " | " + mlab + " | gold | saw evidence? | keep |",
           "|---|---|---|---|---|---|---|---|"]
    for q, n in notes.items():
        idx.append(f"| {q} | {n['task_type']} | {n['bucket']} | {n['base_pred']} | "
                   f"{n['method_pred']} | {n['gold']} | {n.get('saw_the_evidence','')} | "
                   f"{'yes' if n.get('keep_as_example') else ''} |")
    cells.append(nbf.v4.new_markdown_cell("## Index\n\n" + "\n".join(idx)))

    for bucket, qids, title in (("fixed", fixed, "Fixed"), ("broke", broke, "Broke")):
        if not qids:
            continue
        cells.append(nbf.v4.new_markdown_cell(
            f"---\n# {title} — {len(qids)} cases\n\n"
            + (f"The `{mlab}` arm got these right and the `{blab}` arm got them wrong."
               if bucket == "fixed" else
               f"The `{blab}` arm already had these right and the `{mlab}` arm lost them.")))
        for q in qids:
            n = notes[q]
            m, b = method[q], base[q]
            opts = "\n".join(
                f"- {o}" + ("  ← **gold**" if o.strip().startswith(n["gold"] + ".") else "")
                for o in m["options"])
            md = [f"### `{q}` · {n['task_type']} · **{title.upper()}**", "",
                  f"**{m['question']}**", "", opts, "",
                  f"| | {blab} | {mlab} | gold |", "|---|---|---|---|",
                  f"| answer | **{n['base_pred']}** {'✅' if n['base_pred']==n['gold'] else '❌'} "
                  f"| **{n['method_pred']}** {'✅' if n['method_pred']==n['gold'] else '❌'} "
                  f"| {n['gold']} |", "",
                  f"*video {m.get('duration',0):.0f}s · {m.get('videoID','')}*", ""]
            if m.get("oracle"):
                o = m["oracle"]
                md.append(f"*forced span `{o.get('region')}` from ground-truth evidence "
                          f"`{o.get('evidence')}` (`{o.get('time_reference')}`)*\n")
            if n.get("summary"):
                md += [f"**What happened.** {n['summary']}", ""]
            if n.get("why"):
                verb = "was fixed" if bucket == "fixed" else "broke"
                md += [f"**Why it {verb}.** {n['why']}", ""]
            if n.get("keep_as_example"):
                md += [f"> 🔖 **Training example** — {n.get('keep_reason','')}", ""]
            if n.get("error"):
                md += [f"> ⚠️ summary unavailable ({n['error']})", ""]
            init = (m.get("initial") or {}).get("montage")
            if init:
                md += ["**Opening view** — "
                       f"{m['initial'].get('n_frames')}-frame skim of the whole video, "
                       "seen by both arms", _thumb(init)]
            md += ["", "#### Trajectory"]
            md += _rounds_md(b, blab)
            md += _rounds_md(m, mlab)
            cells.append(nbf.v4.new_markdown_cell("\n".join(md)))

    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    with open(out_path, "w") as f:
        nbf.write(nb, f)
    return out_path


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=list(EXPERIMENTS))
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the flip counts and dossier sizes, call nothing")
    ap.add_argument("--notebook-only", action="store_true",
                    help="rebuild the notebook from notes.jsonl without re-reading")
    args = ap.parse_args()

    mrun, brun, mlab, blab, _ = EXPERIMENTS[args.exp]
    method, base = load_run(mrun), load_run(brun)
    common = sorted(set(method) & set(base), key=lambda q: (q.split("-")[0].zfill(6), q))

    flips = []
    for q in common:
        mc = rescore(method[q]) == method[q]["gold"]
        bc = rescore(base[q]) == base[q]["gold"]
        if mc and not bc:
            flips.append((q, "fixed"))
        elif bc and not mc:
            flips.append((q, "broke"))
    if args.limit:
        flips = flips[: args.limit]

    out_dir = os.path.join(config.RUN_ROOT, f"notes_{args.exp}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "notes.jsonl")
    nb_path = os.path.join(out_dir, f"CASES_{args.exp}.ipynb")

    print(f"[{args.exp}] {mrun} vs {brun}: {len(common)} paired, "
          f"{sum(1 for _,k in flips if k=='fixed')} fixed / "
          f"{sum(1 for _,k in flips if k=='broke')} broke")

    if args.dry_run:
        for q, bucket in flips[:3]:
            parts = dossier(args.exp, method[q], base[q], bucket)
            nim = sum(1 for p in parts if p["type"] == "image_url")
            nch = sum(len(p["text"]) for p in parts if p["type"] == "text")
            print(f"  {q} [{bucket}] images={nim} text_chars={nch}")
        return

    done = {}
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                r = json.loads(line)
                done[str(r["qid"])] = r
            except json.JSONDecodeError:
                continue

    todo = [(q, k) for q, k in flips if q not in done]
    if todo and not args.notebook_only:
        model = discover_model(args.endpoint)
        print(f"[{args.exp}] judging {len(todo)} cases with {model} ({len(done)} cached)")
        lock = threading.Lock()
        n = [0]

        def work(item):
            q, bucket = item
            m, b = method[q], base[q]
            row = {"qid": q, "bucket": bucket, "task_type": m.get("task_type"),
                   "gold": m["gold"], "method_pred": rescore(m), "base_pred": rescore(b),
                   "method_run": mrun, "base_run": brun}
            try:
                row.update(ask(args.endpoint, model, dossier(args.exp, m, b, bucket)))
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
            with lock:
                with open(out_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                n[0] += 1
                print(f"  [{n[0]}/{len(todo)}] {q} {bucket}", flush=True)
            return row

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, todo))

        for line in open(out_path):
            try:
                r = json.loads(line)
                done[str(r["qid"])] = r
            except json.JSONDecodeError:
                continue

    notes = {q: done[q] for q, _ in flips if q in done}
    build_notebook(args.exp, notes, method, base, nb_path)
    size = os.path.getsize(nb_path) / 1e6
    print(f"[{args.exp}] wrote {nb_path} ({len(notes)} cases, {size:.1f} MB)")


if __name__ == "__main__":
    main()
