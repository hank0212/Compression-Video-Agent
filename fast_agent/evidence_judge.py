"""Two-stage evidence judge for a paired run: blind evidence probe, THEN attribution.

Fixes the flip_judge.py design flaws (see plan i-finished-the-baseline-mighty-volcano):
hindsight bias (judge saw GOLD before assessing evidence), one forced category
conflating span choice / compression fidelity / integration, no calibration bucket,
and no quantitative localization measure.

  python -m fast_agent.evidence_judge --tools-run compress_run2 --base-run baseline_run1 \
      [--buckets fixed,broke,both_wrong,control] [--control-n 60] [--limit N] \
      [--workers 4] [--dry-run] [--report-only]

Protocol per sample:
  Stage A (blind, 2 calls, NO gold / NO verdicts / NO agent thinking):
    A-base : question + options + initial skim montage        -> judge answers the MCQ
    A-tools: same + every tool round's returned montage        -> judge answers again
    Each also rates the evidence (decisive/partial/absent) and estimates the
    evidence span in seconds. A blind reader answering correctly from the tools
    context is EVIDENCE the gathered evidence suffices; failing is evidence the
    span/coverage was wrong — independent of what the agent concluded.
  Stage B (attribution, 1 call, flips + both_wrong only): gold + both arms' answers
    + truncated agent thinking + the stage-A verdicts revealed; asks for a category
    (same taxonomy as flip_judge for comparability), the pivotal tool call, and
    whether the compress output was actually used in the final reasoning.
  control bucket = stratified both_right sample, stage A only: calibrates how well
    the judge reads montages at all (its blind accuracy ceiling).

Writes RUN_ROOT/ejudge_<toolsrun>_vs_<baserun>/:
  judgments.jsonl  one line per sample, written only when complete (resumable)
  manifest.json    args + model + timestamp
  report.md/.json  cross-instrument report (merges probes.jsonl if present)
"""

import argparse
import json
import os
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from . import config, viz
from .flip_judge import (
    CATEGORIES,
    DEFAULT_ENDPOINT,
    _chat,
    _image_part,
    _trunc,
    discover_model,
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
LETTERS = ("A", "B", "C", "D")
EVIDENCE_LEVELS = ("decisive", "partial", "absent")


# ---------------------------------------------------------------------------
# trajectory -> facts

def successful_calls(traj: dict) -> list[dict]:
    """Successful tool calls of a trajectory with span + compression facts."""
    out = []
    for r in traj["rounds"]:
        tr = r.get("tool_result")
        if tr:
            out.append({"round": r["round"], "tool": tr["tool"],
                        "span": [float(tr["span"][0]), float(tr["span"][1])],
                        "n_frames": tr.get("n_frames"),
                        "kept_tokens": tr.get("kept_tokens"),
                        "base_tokens": tr.get("base_tokens"),
                        "retention": tr.get("retention"),
                        "montage": tr.get("montage")})
    return out


def _montage_caption(kind: str, span: tuple[float, float], n_frames, extra: str = "") -> str:
    """Honest caption for a montage grid: what it shows and how thumbs map to time."""
    a, b = span
    return (f"[{kind}] montage grid of <=48 low-res thumbnails, reading order "
            f"left-to-right then top-to-bottom, uniformly spanning {a:.0f}s-{b:.0f}s"
            + (f" ({n_frames} frames)" if n_frames else "") + f". {extra}".rstrip())


# ---------------------------------------------------------------------------
# Stage A — blind evidence probe

STAGE_A_SCHEMA = (
    'Reply with ONLY a JSON object, no other text:\n'
    "{\n"
    '  "answer": "A"/"B"/"C"/"D",   // your best answer from the frames shown; you MUST pick one even if unsure\n'
    '  "evidence": "decisive"/"partial"/"absent",  // how well the frames shown actually support that answer\n'
    '  "evidence_span_s": [start, end] or null,    // where in the VIDEO (seconds) the deciding evidence most likely lives\n'
    '  "why": "<1-2 sentences: what in the frames drives your answer>"\n'
    "}"
)


def stage_a_dossier(t: dict, arm: str) -> list:
    """Blind dossier: question + options + one arm's visual context. No gold, no
    verdicts, no agent reasoning — the judge must answer the MCQ itself."""
    opts = "\n".join(t["options"])
    dur = t["duration"]
    header = (
        "You are answering a multiple-choice question about a long video, using ONLY "
        "the frame montages below (subsampled low-res thumbnails — small text/objects "
        "may be hard to read; still commit to your best answer).\n\n"
        f"Video duration: {dur:.0f}s\n"
        f"QUESTION: {t['question']}\n{opts}\n\n"
    )
    parts = [{"type": "text", "text": header + _montage_caption(
        "initial skim", (0.0, dur), t["initial"]["n_frames"],
        "This is a sparse uniform sample of the whole video.")}]
    img = _image_part(t["initial"].get("montage"))
    if img:
        parts.append(img)
    if arm == "tools":
        for c in successful_calls(t):
            extra = ""
            if c["tool"] == "compress_video":
                extra = ("NOTE: this is a uniform preview of the span that was "
                         "compressed (the downstream model received "
                         f"{c['kept_tokens']} compressed tokens from {c['base_tokens']}, "
                         "not these exact thumbnails).")
            parts.append({"type": "text", "text": "\n" + _montage_caption(
                f"tool result {c['round'] + 1}: {c['tool']}",
                tuple(c["span"]), c.get("n_frames"), extra)})
            img = _image_part(c.get("montage"))
            if img:
                parts.append(img)
    parts.append({"type": "text", "text": "\n" + STAGE_A_SCHEMA})
    return parts


def parse_stage_a(text: str) -> dict | None:
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    ans = str(d.get("answer", "")).strip().upper()[:1]
    if ans not in LETTERS:
        return None
    ev = str(d.get("evidence", "")).strip().lower()
    span = d.get("evidence_span_s")
    if not (isinstance(span, (list, tuple)) and len(span) == 2
            and all(isinstance(x, (int, float)) for x in span)):
        span = None
    return {"answer": ans,
            "evidence": ev if ev in EVIDENCE_LEVELS else None,
            "evidence_span_s": list(span) if span else None,
            "why": str(d.get("why", ""))[:600]}


# ---------------------------------------------------------------------------
# Stage B — attribution with everything revealed

STAGE_B_SCHEMA = (
    'Reply with ONLY a JSON object, no other text:\n'
    "{\n"
    '  "category": "<one key from the list above>",\n'
    '  "pivotal_call": <1-based index of the tool call most responsible for the outcome, or null>,\n'
    '  "compress_used": "used"/"ignored"/"na",  // did the TOOLS arm\'s final reasoning actually draw on the compress_video output ("na" if compress never fired)\n'
    '  "rationale": "<2-4 sentences tied to the category>",\n'
    '  "confidence": "low"/"medium"/"high"\n'
    "}"
)

BUCKET_BLURB = {
    "broke": "The BASELINE arm answered CORRECTLY; the TOOLS arm answered WRONG. Diagnose what the tool-using trajectory did that lost a question the plain skim got right.",
    "fixed": "The BASELINE arm answered WRONG; the TOOLS arm answered CORRECTLY. Diagnose what the tools actually contributed.",
    "both_wrong": "BOTH arms answered WRONG. Diagnose the shared failure.",
}


def _round_lines(traj: dict) -> str:
    """Compact text-only replay of the tools arm's rounds (thinking truncated)."""
    lines = []
    for r in traj["rounds"]:
        a = r.get("action") or {}
        if a.get("kind") == "tool_call":
            act = f"{a['name']}({a['start']:.0f}s-{a['end']:.0f}s)"
            if a.get("error"):
                act += f" -> REJECTED: {a['error']}"
        else:
            act = a.get("kind", "?")
        lines.append(f"Round {r['round']} [{act}]\n{_trunc(r['thinking'], 500, 400)}")
    return "\n\n".join(lines)


def stage_b_dossier(bucket: str, tools_t: dict, base_t: dict,
                    a_base: dict | None, a_tools: dict | None) -> list:
    opts = "\n".join(tools_t["options"])
    verdict = lambda t: "CORRECT" if t["correct"] else "WRONG"
    cats = "\n".join(f"- {k}: {v}" for k, v in CATEGORIES[bucket].items())
    blind = ""
    if a_base and a_tools:
        blind = (
            "\nAn independent blind reader (who saw only the frames, not the gold "
            "answer) was asked the same question:\n"
            f"- from the skim alone it answered {a_base['answer']} "
            f"(evidence: {a_base['evidence']})\n"
            f"- from the skim + all tool results it answered {a_tools['answer']} "
            f"(evidence: {a_tools['evidence']}, "
            f"estimated evidence span: {a_tools['evidence_span_s']})\n")
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
        f"TOOLS answered: {tools_t['pred']} ({verdict(tools_t)})\n"
        f"{blind}\n"
        "TOOLS arm rounds (montages were already shown to the blind reader; "
        "thinking is the agent's own reasoning, truncated):\n\n"
        f"{_round_lines(tools_t)}\n\n"
        f"Classify this sample into exactly ONE category:\n{cats}\n\n"
        + STAGE_B_SCHEMA
    )
    return [{"type": "text", "text": header}]


def parse_stage_b(text: str, bucket: str) -> dict | None:
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if d.get("category") not in CATEGORIES[bucket]:
        return None
    pc = d.get("pivotal_call")
    return {"category": d["category"],
            "pivotal_call": pc if isinstance(pc, int) else None,
            "compress_used": str(d.get("compress_used", "na")).lower(),
            "rationale": str(d.get("rationale", ""))[:800],
            "confidence": str(d.get("confidence", ""))[:10]}


# ---------------------------------------------------------------------------
# judge one sample (up to 3 calls, each with one bare-JSON retry)

def _ask(endpoint: str, model: str, content: list, parse, retry_hint: str,
         max_tokens: int = 600):
    messages = [{"role": "user", "content": content}]
    reply = _chat(endpoint, model, messages, max_tokens=max_tokens)
    d = parse(reply)
    if d is None:
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": retry_hint}]
        reply = _chat(endpoint, model, messages, max_tokens=max_tokens)
        d = parse(reply)
    return d, reply


def judge_sample(endpoint: str, model: str, bucket: str,
                 tools_t: dict, base_t: dict) -> dict:
    row = {}
    a_retry = "Reply again with ONLY the JSON object in the requested schema. \"answer\" must be A, B, C or D."
    t0 = time.time()
    a_base, raw = _ask(endpoint, model, stage_a_dossier(base_t, "base"),
                       parse_stage_a, a_retry)
    row["stageA_base"] = a_base
    if a_base is None:
        row["stageA_base_raw"] = raw
    a_tools, raw = _ask(endpoint, model, stage_a_dossier(tools_t, "tools"),
                        parse_stage_a, a_retry)
    row["stageA_tools"] = a_tools
    if a_tools is None:
        row["stageA_tools_raw"] = raw
    if bucket in CATEGORIES:  # flips + both_wrong get attribution; control does not
        b_retry = ("Reply again with ONLY the JSON object in the requested schema. "
                   f"\"category\" must be one of: {', '.join(CATEGORIES[bucket])}.")
        b, raw = _ask(endpoint, model,
                      stage_b_dossier(bucket, tools_t, base_t, a_base, a_tools),
                      lambda s: parse_stage_b(s, bucket), b_retry)
        row["stageB"] = b
        if b is None:
            row["stageB_raw"] = raw
    row["judge_seconds"] = round(time.time() - t0, 1)
    return row


# ---------------------------------------------------------------------------
# derived chain locus (objective crosses; judge text never decides these)

def span_overlap(calls: list[dict], ev_span) -> float:
    """Longest fractional overlap between any tool span and the judged evidence span."""
    if not ev_span or not calls:
        return 0.0
    a, b = float(ev_span[0]), float(ev_span[1])
    if b <= a:
        b = a + 1.0
    best = 0.0
    for c in calls:
        s, e = c["span"]
        inter = max(0.0, min(b, e) - max(a, s))
        best = max(best, inter / (b - a))
    return round(best, 3)


def derive_locus(row: dict, probes: dict) -> str:
    """Locate the failure/gain on the chain: span choice -> compression fidelity ->
    integration. Uses ONLY objective bits: agent outcomes, blind-judge answers,
    probe answers. Returns a locus label per bucket."""
    gold = row["gold"]
    ab = (row.get("stageA_base") or {}).get("answer")
    at = (row.get("stageA_tools") or {}).get("answer")
    p1 = probes.get((row["qid"], "p1"), {}).get("pred")
    p2 = probes.get((row["qid"], "p2"), {}).get("pred")
    bucket = row["bucket"]

    if bucket == "broke":
        if at == gold:  # blind reader answers from the gathered evidence
            if p1 == gold:
                return "integration:reasoning_drift"      # evidence+context fine, multi-round reasoning lost it
            if p1 is not None:
                return "integration:context_readout"      # even a fresh single-turn read fails
            return "integration:unverified"
        if p2 == gold:
            return "evidence:span_ok_compression_ok_judge_missed"  # rare: model reads tokens better than judge reads thumbs
        if row["overlap_tools"] < 0.25:
            return "evidence:span_miss"
        return "evidence:insufficient_in_span"
    if bucket == "fixed":
        if ab == gold:
            return "gain:not_evidence_driven"             # blind reader answers from skim alone
        if at == gold or p1 == gold:
            return "gain:genuine_evidence"                # SFT candidate
        return "gain:unverified"
    if bucket == "both_wrong":
        # single blind hit is chance-contaminated here (blind-tools acc ~= chance on
        # this bucket); demand behavioral proof or TWO independent blind reads on gold
        if p1 == gold or p2 == gold or (at == gold and ab == gold):
            return "headroom:readout"                     # evidence gathered & sufficient, both arms misread
        if row["overlap_tools"] < 0.25:
            return "headroom:never_gathered"              # localization headroom (RL target)
        return "headroom:hard_or_ambiguous"
    return "control"


# ---------------------------------------------------------------------------
# report

def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def _mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant pairs (binomial)."""
    import math
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, round(p, 4))


def load_probes(probe_dir: str) -> dict:
    """{(qid, probe): row} from probes.jsonl if present."""
    out = {}
    p = os.path.join(probe_dir, "probes.jsonl")
    if not os.path.exists(p):
        return out
    for line in open(p):
        if line.strip():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[(str(r["qid"]), r["probe"])] = r
    return out


def write_report(judge_dir: str, probe_dir: str) -> dict:
    rows = list(viz.judgments(judge_dir).values())
    probes = load_probes(probe_dir)
    for r in rows:
        r["locus"] = derive_locus(r, probes)
    flips = [r for r in rows if r["bucket"] in ("fixed", "broke")]
    bw = [r for r in rows if r["bucket"] == "both_wrong"]
    ctrl = [r for r in rows if r["bucket"] == "control"]
    n_fixed = sum(r["bucket"] == "fixed" for r in rows)
    n_broke = sum(r["bucket"] == "broke" for r in rows)

    def mix(r):
        return "compress" if "compress_video" in r["used"] else ("crop" if r["used"] else "none")

    def blind_acc(rs, key):
        ok = [r for r in rs if (r.get(key) or {}).get("answer")]
        k = sum((r[key]["answer"] == r["gold"]) for r in ok)
        return k, len(ok)

    md = [f"# Evidence-chain report — {os.path.basename(judge_dir)}",
          f"{len(rows)} judged ({n_fixed} fixed / {n_broke} broke / {len(bw)} both_wrong / {len(ctrl)} control), "
          f"probes merged: {len(probes)}\n"]

    # -- headline: net effect + McNemar
    p_mc = _mcnemar(n_broke, n_fixed)
    md += ["## Headline",
           f"- Net tools effect: **{n_fixed - n_broke:+d}** (fixed {n_fixed} vs broke {n_broke}), "
           f"exact McNemar p = **{p_mc}** — "
           + ("the net effect is not distinguishable from churn." if p_mc > 0.05
              else "the net effect is statistically real."),
           ]
    # compress vs crop flip split
    cc = Counter((r["bucket"], mix(r)) for r in flips)
    net_comp = cc[("fixed", "compress")] - cc[("broke", "compress")]
    net_crop = cc[("fixed", "crop")] - cc[("broke", "crop")]
    md += [f"- Flip split by tool mix: compress-involved net **{net_comp:+d}** "
           f"({cc[('fixed', 'compress')]} fixed / {cc[('broke', 'compress')]} broke, "
           f"McNemar p={_mcnemar(cc[('broke', 'compress')], cc[('fixed', 'compress')])}); "
           f"crop-only net **{net_crop:+d}** "
           f"({cc[('fixed', 'crop')]} fixed / {cc[('broke', 'crop')]} broke, "
           f"p={_mcnemar(cc[('broke', 'crop')], cc[('fixed', 'crop')])})."]

    # -- chain locus tables
    md.append("\n## Where on the chain (span -> compression -> integration)\n")
    for bucket in ("broke", "fixed", "both_wrong"):
        sub = [r for r in rows if r["bucket"] == bucket]
        if not sub:
            continue
        loc = Counter(r["locus"] for r in sub)
        md.append(f"**{bucket}** (n={len(sub)})\n")
        md.append("| locus | n | % | 95% CI | compress-involved |")
        md.append("|---|---|---|---|---|")
        for l, n in loc.most_common():
            comp = sum(1 for r in sub if r["locus"] == l and mix(r) == "compress")
            lo, hi = _wilson(n, len(sub))
            md.append(f"| {l} | {n} | {100 * n / len(sub):.0f}% | {lo}-{hi}% | {comp} |")
        md.append("")

    # -- localization quantification
    md.append("## Localization (tool spans vs blind-judged evidence span)\n")
    md.append("| bucket | n with ev-span | median overlap | overlap>=0.5 | overlap<0.25 |")
    md.append("|---|---|---|---|---|")
    for bucket in ("broke", "fixed", "both_wrong"):
        sub = [r for r in rows if r["bucket"] == bucket and r.get("overlap_tools") is not None
               and (r.get("stageA_tools") or {}).get("evidence_span_s")]
        if not sub:
            continue
        ov = sorted(r["overlap_tools"] for r in sub)
        med = ov[len(ov) // 2]
        hi = sum(o >= 0.5 for o in ov)
        lo = sum(o < 0.25 for o in ov)
        md.append(f"| {bucket} | {len(sub)} | {med:.2f} | {hi} ({100*hi/len(sub):.0f}%) | {lo} ({100*lo/len(sub):.0f}%) |")

    # -- calibration
    md.append("\n## Calibration (how much to trust the blind judge)\n")
    for name, rs in (("control (both_right)", ctrl), ("all flips", flips), ("both_wrong", bw)):
        kb, nb = blind_acc(rs, "stageA_base")
        kt, nt = blind_acc(rs, "stageA_tools")
        if nb:
            md.append(f"- {name}: blind-from-skim {kb}/{nb} ({100*kb/nb:.0f}%), "
                      f"blind-from-tools-context {kt}/{nt} ({100*kt/nt:.0f}%)")
    ag = [r for r in rows if (r.get("stageA_base") or {}).get("answer")]
    same = sum((r["stageA_base"]["answer"] == r["base_pred"]) for r in ag)
    if ag:
        md.append(f"- blind-skim answer agrees with the baseline ARM's answer on "
                  f"{same}/{len(ag)} ({100*same/len(ag):.0f}%) (same frames, different context)")
    md.append("- CAVEAT: the judge reads 48-thumb montages, not the agent's full-res "
              "context — its control accuracy is the ceiling of what 'blind reader "
              "answers correctly' can mean; blind accuracy at/below chance (25%) on a "
              "bucket means judge answers there carry little signal on their own, which "
              "is why both_wrong readout requires probes or double-blind agreement.")

    # -- probe crosses (only where probes ran)
    if probes:
        md.append("\n## Probe outcomes (agent model, behavioral)\n")
        for pk, label in (("p1", "P1 evidence-only replay"), ("p2", "P2 compressed-only readout")):
            sub = [(r, probes[(r["qid"], pk)]) for r in rows if (r["qid"], pk) in probes]
            if not sub:
                continue
            k = sum(p["correct"] for _, p in sub)
            md.append(f"**{label}** — {k}/{len(sub)} correct ({100*k/len(sub):.0f}%)")
            per = Counter((r["bucket"], bool(p["correct"])) for r, p in sub)
            md.append("| bucket | probe correct | probe wrong |")
            md.append("|---|---|---|")
            for bucket in ("broke", "fixed", "both_wrong"):
                if per[(bucket, True)] + per[(bucket, False)]:
                    md.append(f"| {bucket} | {per[(bucket, True)]} | {per[(bucket, False)]} |")
            md.append("")

    # -- stage B secondary annotation
    md.append("\n## Stage-B categories (judge attribution, secondary)\n")
    for bucket in ("broke", "fixed", "both_wrong"):
        sub = [r for r in rows if r["bucket"] == bucket and (r.get("stageB") or {}).get("category")]
        if not sub:
            continue
        cat = Counter(r["stageB"]["category"] for r in sub)
        md.append(f"- **{bucket}** (n={len(sub)}): "
                  + ", ".join(f"{c} {n}" for c, n in cat.most_common()))
    cu = Counter((r.get("stageB") or {}).get("compress_used") for r in rows
                 if "compress_video" in r["used"] and r.get("stageB"))
    if cu:
        md.append(f"- compress output in final reasoning (per stage B): {dict(cu)}")

    # -- training implications
    sft = sorted(r["qid"] for r in rows if r["locus"] == "gain:genuine_evidence")
    ro = sorted(r["qid"] for r in rows if r["locus"].startswith("integration:")
                or r["locus"] == "headroom:readout")
    ng = sorted(r["qid"] for r in rows if r["locus"] == "headroom:never_gathered")
    md += ["\n## Training implications",
           f"- **SFT candidates (verified genuine evidence gains)**: {len(sft)} qids: {', '.join(sft)}",
           f"- **Read-out failures** (evidence sufficient, answer wrong — reward/training must touch evidence weighing): {len(ro)}",
           f"- **Localization headroom** (evidence never gathered — the routing-policy target): {len(ng)}"]

    # -- exemplars per locus
    md.append("\n## Exemplars per locus — replay with viz.show_judged(...)\n")
    by_loc = defaultdict(list)
    for r in rows:
        if r["bucket"] != "control":
            by_loc[r["locus"]].append(r)
    for l, rs in sorted(by_loc.items(), key=lambda kv: -len(kv[1])):
        md.append(f"- **{l}** (n={len(rs)}):")
        for r in rs[:4]:
            why = ((r.get("stageB") or {}).get("rationale")
                   or (r.get("stageA_tools") or {}).get("why") or "")
            md.append(f"    - `{r['qid']}` ({r['task']}) — {why.split('. ')[0]}")

    rep = {"n": len(rows),
           "buckets": dict(Counter(r["bucket"] for r in rows)),
           "locus": {b: dict(Counter(r["locus"] for r in rows if r["bucket"] == b))
                     for b in ("broke", "fixed", "both_wrong")},
           "mcnemar_all": p_mc, "sft_candidates": sft,
           "readout_failures": ro, "never_gathered": ng}
    with open(os.path.join(judge_dir, "report.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(judge_dir, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    return rep


# ---------------------------------------------------------------------------

def control_sample(both_right: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Stratified-by-task sample of both_right for judge calibration."""
    by_task = defaultdict(list)
    for it in both_right:
        by_task[it["task"]].append(it)
    rng = random.Random(seed)
    total = len(both_right)
    picked = []
    for task, items in sorted(by_task.items()):
        k = max(1, round(n * len(items) / total))
        rng.shuffle(items)
        picked += items[:k]
    rng.shuffle(picked)
    return picked[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools-run", required=True)
    ap.add_argument("--base-run", required=True)
    ap.add_argument("--buckets", default="fixed,broke,both_wrong,control")
    ap.add_argument("--control-n", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None, help="max samples per bucket")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    def _run_dir(name):
        return name if os.path.isabs(name) else os.path.join(config.RUN_ROOT, name)

    judge_dir = os.path.join(
        config.RUN_ROOT,
        f"ejudge_{os.path.basename(args.tools_run)}_vs_{os.path.basename(args.base_run)}")
    probe_dir = os.path.join(
        config.RUN_ROOT,
        f"probe_{os.path.basename(args.tools_run)}_vs_{os.path.basename(args.base_run)}")
    if args.report_only:
        rep = write_report(judge_dir, probe_dir)
        print(f"[report] {judge_dir}/report.md")
        print(json.dumps(rep["locus"], indent=1))
        return

    run_tools = viz.load_run(_run_dir(args.tools_run))
    run_base = viz.load_run(_run_dir(args.base_run))
    tools_by = {str(t["question_id"]): t for t in run_tools}
    base_by = {str(t["question_id"]): t for t in run_base}
    cats = viz.pair_runs(run_tools, run_base)
    cats["control"] = control_sample(cats.pop("both_right"), args.control_n)
    print("buckets:", {k: len(v) for k, v in cats.items()})

    os.makedirs(judge_dir, exist_ok=True)
    done = set(viz.judgments(judge_dir))
    todo = []
    for bucket in args.buckets.split(","):
        items = cats[bucket][: args.limit] if args.limit else cats[bucket]
        todo += [(bucket, it) for it in items if it["qid"] not in done]
    print(f"[ejudge] {len(todo)} to judge (skipped {len(done)} done) -> {judge_dir}")

    if args.dry_run:
        for bucket in ("broke", "control"):
            it = next((it for b, it in todo if b == bucket), None)
            if not it:
                continue
            t, b_ = tools_by[it["qid"]], base_by[it["qid"]]
            for name, parts in (("A_base", stage_a_dossier(b_, "base")),
                                ("A_tools", stage_a_dossier(t, "tools")),
                                ("B", stage_b_dossier(bucket, t, b_, None, None)
                                 if bucket in CATEGORIES else [])):
                if not parts:
                    continue
                n_img = sum(p["type"] == "image_url" for p in parts)
                txt = "\n".join(p["text"] if p["type"] == "text" else "[IMAGE]" for p in parts)
                pv = os.path.join(judge_dir, f"preview_{bucket}_{name}.txt")
                with open(pv, "w") as f:
                    f.write(txt)
                print(f"[dry-run] {bucket}/{it['qid']} stage {name}: {n_img} images, "
                      f"{len(txt)} chars -> {pv}")
        return

    model = discover_model(args.endpoint)
    print(f"[ejudge] endpoint={args.endpoint} model={model} workers={args.workers}")
    with open(os.path.join(judge_dir, "manifest.json"), "w") as f:
        json.dump({"tools_run": args.tools_run, "base_run": args.base_run,
                   "buckets": args.buckets, "control_n": args.control_n,
                   "model": model, "endpoint": args.endpoint,
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)

    out_path = os.path.join(judge_dir, "judgments.jsonl")
    lock = threading.Lock()
    t0 = time.time()
    n_done = [0]

    def work(item):
        bucket, it = item
        t, b = tools_by[it["qid"]], base_by[it["qid"]]
        row = {"qid": it["qid"], "bucket": bucket, "task": it["task"],
               "gold": it["gold"], "base_pred": it["base"], "tools_pred": it["tools"],
               "used": it["used"], "duration": t["duration"],
               "tool_calls": [{k: c[k] for k in ("tool", "span", "kept_tokens",
                                                 "base_tokens", "retention")}
                              for c in successful_calls(t)]}
        try:
            row.update(judge_sample(args.endpoint, model, bucket, t, b))
            ev = (row.get("stageA_tools") or {}).get("evidence_span_s")
            row["overlap_tools"] = span_overlap(successful_calls(t), ev)
        except Exception as e:
            import traceback
            traceback.print_exc()
            row.update({"stageA_base": None, "stageA_tools": None, "stageB": None,
                        "overlap_tools": None, "error": f"{type(e).__name__}: {e}"})
        with lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            n_done[0] += 1
            i = n_done[0]
            ab = (row.get("stageA_base") or {}).get("answer")
            at = (row.get("stageA_tools") or {}).get("answer")
            sb = (row.get("stageB") or {}).get("category")
            print(f"[{i}/{len(todo)}] {bucket:<10} {it['qid']:<8} gold={it['gold']} "
                  f"blind base/tools={ab}/{at} B={sb} "
                  f"{row.get('judge_seconds', '-')}s (avg {(time.time() - t0) / i:.1f}s)",
                  flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))

    rep = write_report(judge_dir, probe_dir)
    print(f"\n[report] {judge_dir}/report.md")
    print(json.dumps(rep["locus"], indent=1))


if __name__ == "__main__":
    main()
