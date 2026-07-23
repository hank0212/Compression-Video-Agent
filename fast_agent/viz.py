"""Notebook visualization of stored trajectories.

    from fast_agent import viz
    run = viz.load_run("/local1/cfyang/hanklin/outputs/fast_agent/runs/compress_0712")
    viz.summary(run)                 # accuracy table + tool-use histogram
    viz.show(run, 0)                 # replay sample 0 (or viz.show(run, qid="653-1"))
    viz.show_wrong(run)              # index of incorrect samples to inspect

Renders inline HTML: question + options (gold/pred marked), the initial skim
montage, then each round's thinking, action, and the frames that tool returned.
"""

import base64
import glob
import json
import os

try:  # notebook rendering only; pair_runs/judgments work without IPython (flip_judge CLI)
    from IPython.display import HTML, display
except ImportError:
    HTML = display = None


def load_run(run_dir: str) -> list[dict]:
    """Load all trajectory JSONs in a run dir, ordered by question_id.

    Missing ``traj`` is treated as an empty/incomplete run.  This makes the
    analysis helpers useful while a run is still being populated and keeps a
    missing trajectory from being confused with a model-level wrong answer.
    """
    paths = sorted(glob.glob(os.path.join(os.fspath(run_dir), "traj", "*.json")))
    out = []
    for p in paths:
        with open(p) as f:
            t = json.load(f)
        t["_path"] = p
        out.append(t)
    return out


def _img(path: str | None, width: int | None = None) -> str:
    if not path or not os.path.exists(path):
        return "<i style='color:#888'>[no frames]</i>" if path else ""
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    w = f'width="{width}"' if width else 'style="max-width:100%"'
    return (f'<img src="data:image/png;base64,{b64}" {w} '
            'style="border:1px solid #333;border-radius:4px;margin:2px 0"/>')


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def summary(run: list[dict]):
    """Accuracy overall + per task type, and a tool-use histogram."""
    from collections import defaultdict
    by = defaultdict(lambda: [0, 0])
    tools = defaultdict(int)
    for t in run:
        by[t["task_type"]][0] += bool(t["correct"])
        by[t["task_type"]][1] += 1
        for r in t["rounds"]:
            a = r.get("action") or {}
            if a.get("kind") == "tool_call" and not a.get("error"):
                tools[a["name"]] += 1
    n = len(run)
    c = sum(bool(t["correct"]) for t in run)
    rows = "".join(
        f"<tr><td>{_esc(k)}</td><td align=right>{v[0]}/{v[1]}</td>"
        f"<td align=right>{100*v[0]/max(v[1],1):.1f}%</td></tr>"
        for k, v in sorted(by.items()))
    th = " · ".join(f"{k}: {v}" for k, v in sorted(tools.items())) or "none"
    display(HTML(
        f"<h3>Accuracy {100*c/max(n,1):.1f}% ({c}/{n})</h3>"
        f"<b>tool calls:</b> {th}"
        f"<table style='border-collapse:collapse'>"
        f"<tr><th align=left>task</th><th>correct</th><th>acc</th></tr>{rows}</table>"))


def show_wrong(run: list[dict]) -> list[int]:
    idx = [i for i, t in enumerate(run) if not t["correct"]]
    print(f"{len(idx)} incorrect: {idx}")
    return idx


def _tools_used(t: dict) -> list[str]:
    """Names of successful (non-errored) tool calls in a trajectory, in order."""
    return [r["action"]["name"] for r in t["rounds"]
            if (r.get("action") or {}).get("kind") == "tool_call"
            and not r["action"].get("error")]


def _run_rows(run: list[dict] | str) -> list[dict]:
    return load_run(run) if isinstance(run, (str, os.PathLike)) else list(run)


def pair_runs(run_tools: list[dict] | str, run_base: list[dict] | str) -> dict:
    """Match two runs by question_id into the four paired buckets:
      fixed  = baseline WRONG -> tools RIGHT   (tools helped)
      broke  = baseline RIGHT -> tools WRONG   (tools hurt)
    Each item carries both preds and which tools were used (no printing —
    flips() wraps this for the notebook)."""
    run_tools, run_base = _run_rows(run_tools), _run_rows(run_base)
    base = {str(t["question_id"]): t for t in run_base}
    cats = {"fixed": [], "broke": [], "both_right": [], "both_wrong": []}
    for t in run_tools:
        b = base.get(str(t["question_id"]))
        if not b:
            continue
        key = ("both_right" if b["correct"] and t["correct"]
               else "both_wrong" if not b["correct"] and not t["correct"]
               else "fixed" if t["correct"] else "broke")
        cats[key].append({"qid": str(t["question_id"]), "task": t["task_type"],
                          "base": b["pred"], "tools": t["pred"], "gold": t["gold"],
                          "used": _tools_used(t)})
    return cats


def flips(run_tools: list[dict], run_base: list[dict], show_examples: int = 0) -> dict:
    """Paired baseline-vs-tools case study (see pair_runs), printed."""
    cats = pair_runs(run_tools, run_base)
    n = sum(len(v) for v in cats.values())
    net = len(cats["fixed"]) - len(cats["broke"])
    print(f"paired on {n} questions   NET tools effect: {net:+d}  (fixed {len(cats['fixed'])} - broke {len(cats['broke'])})")
    print(f"  both_right {len(cats['both_right'])}   both_wrong {len(cats['both_wrong'])}")
    for k in ("fixed", "broke"):
        print(f"\n[{k.upper()}]")
        for r in cats[k][:show_examples] if show_examples else cats[k]:
            print(f"  {r['qid']:<8} {r['task']:<24} base={r['base']} tools={r['tools']} "
                  f"gold={r['gold']}  used={r['used']}")
    return cats


def judgments(judge_dir: str) -> dict[str, dict]:
    """Load flip_judge output (judgments.jsonl) as {qid: judgment}. Pass the
    judge run dir, e.g. runs/judge_compress_run1_vs_baseline_run1."""
    out = {}
    p = (judge_dir if os.fspath(judge_dir).endswith(".jsonl")
         else os.path.join(judge_dir, "judgments.jsonl"))
    if not os.path.exists(p):
        return out
    with open(p) as f:
        for line in f:
            if line.strip():
                try:
                    j = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a truncated final line after a crash
                if j.get("qid") is not None:
                    out[str(j["qid"])] = j
    return out


def tool_routing(run: list[dict]) -> dict:
    """Per-task-type: how often the model reached for crop vs compress vs nothing.
    This is the routing-hypothesis test (does coverage get used on coverage tasks?)."""
    from collections import defaultdict
    by = defaultdict(lambda: {"n": 0, "crop": 0, "compress": 0, "none": 0, "acc": [0, 0]})
    for t in run:
        used = _tools_used(t)
        d = by[t["task_type"]]
        d["n"] += 1
        d["crop"] += "crop_video" in used
        d["compress"] += "compress_video" in used
        d["none"] += not used
        d["acc"][0] += bool(t["correct"]); d["acc"][1] += 1
    print(f"{'task':<26}{'n':>4}{'crop':>6}{'compress':>10}{'none':>6}{'acc%':>7}")
    for k, d in sorted(by.items()):
        print(f"{k:<26}{d['n']:>4}{d['crop']:>6}{d['compress']:>10}{d['none']:>6}"
              f"{100*d['acc'][0]/max(d['acc'][1],1):>6.0f}%")
    return dict(by)


def probe_rows(probe_dir: str) -> dict[tuple, dict]:
    """Load probes.jsonl as {(qid, probe): row}."""
    out = {}
    p = (probe_dir if os.fspath(probe_dir).endswith(".jsonl")
         else os.path.join(probe_dir, "probes.jsonl"))
    if not os.path.exists(p):
        return out
    with open(p) as f:
        for line in f:
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out[(str(r["qid"]), r["probe"])] = r
    return out


def show_judged(run_tools: list[dict], run_base: list[dict], qid: str,
                judge_row: dict | None = None, probes: dict | None = None):
    """Replay one sample with the three instruments' verdicts side by side:
    blind judge (stage A), attribution (stage B), and behavioral probes (P1/P2).
    `judge_row` is one evidence_judge judgments.jsonl row; `probes` the dict from
    viz.probe_rows()."""
    t = next((x for x in run_tools if str(x["question_id"]) == str(qid)), None)
    b = next((x for x in run_base if str(x["question_id"]) == str(qid)), None)
    if t is None or b is None:
        print("qid not found in both runs")
        return
    gold = t["gold"]

    def chip(label, ans, extra=""):
        if ans is None:
            return ""
        ok = str(ans).upper() == gold
        c = "#2a6" if ok else "#c33"
        return (f"<span style='display:inline-block;margin:2px 6px 2px 0;padding:2px 8px;"
                f"border:1px solid {c};border-radius:10px;color:{c}'>"
                f"{_esc(label)}: <b>{_esc(str(ans))}</b>{_esc(extra)}</span>")

    j = judge_row or {}
    a_b, a_t = j.get("stageA_base") or {}, j.get("stageA_tools") or {}
    sb = j.get("stageB") or {}
    p1 = (probes or {}).get((str(qid), "p1")) or {}
    p2 = (probes or {}).get((str(qid), "p2")) or {}
    panel = (
        f"<div style='margin:8px 0;padding:8px;border:1px solid #888;border-radius:6px'>"
        f"<b>instruments</b> (gold={gold}) &nbsp; "
        + chip("agent-base", b["pred"]) + chip("agent-tools", t["pred"])
        + chip("blind-skim", a_b.get("answer"), f" ({a_b.get('evidence', '?')})")
        + chip("blind-tools", a_t.get("answer"), f" ({a_t.get('evidence', '?')})")
        + chip("P1 replay", p1.get("pred")) + chip("P2 compressed-only", p2.get("pred"))
        + (f"<div style='margin-top:4px'><b>locus:</b> {_esc(j.get('locus', '?'))} · "
           f"<b>stage-B:</b> {_esc(sb.get('category', '-'))} "
           f"(compress {_esc(sb.get('compress_used', '-'))})</div>" if j else "")
        + (f"<div style='color:#888'>blind evidence span: {a_t.get('evidence_span_s')} · "
           f"tool-span overlap {j.get('overlap_tools')}</div>" if a_t else "")
        + (f"<div style='font-size:12px;margin-top:4px'>{_esc(sb.get('rationale', ''))}</div>"
           if sb else "")
        + "</div>")
    display(HTML(panel))
    show([t], qid=str(qid))


_ACT_COLOR = {"tool_call": "#2a6", "answer": "#26a", "repeat": "#a53",
              "no_action": "#a53", "tool_withheld": "#777", "answer_no_letter": "#a53"}


def show(run: list[dict], i: int = 0, qid: str | None = None):
    """Replay one sample end to end."""
    t = next((x for x in run if x["question_id"] == qid), None) if qid else run[i]
    if t is None:
        print("not found")
        return

    opts = "".join(
        f"<div style='padding:1px 0'>{'✅' if o[:1]==t['gold'] else ('🔵' if o[:1]==t['pred'] else '&nbsp;&nbsp;')} "
        f"{_esc(o)}</div>" for o in t["options"])
    verdict = ("<span style='color:#2a6'>CORRECT</span>" if t["correct"]
               else "<span style='color:#c33'>WRONG</span>")
    head = (
        f"<div style='font-family:sans-serif;max-width:920px'>"
        f"<h3>[{_esc(t['question_id'])}] {_esc(t['task_type'])} &nbsp; {verdict} "
        f"&nbsp; pred=<b>{t['pred']}</b> gold=<b>{t['gold']}</b></h3>"
        f"<div><b>video</b> {_esc(t['videoID'])} · {t['duration']:.0f}s · "
        f"tools: {', '.join(t['tools']) or 'none (baseline)'}</div>"
        f"<div style='margin:6px 0'><b>Q:</b> {_esc(t['question'])}</div>"
        f"<div style='margin:4px 0'>{opts}</div>"
        f"<div style='color:#888;margin-top:8px'>initial skim · {t['initial']['n_frames']} frames "
        f"(✅ gold · 🔵 model pick)</div>{_img(t['initial']['montage'])}<hr/>")

    body = []
    for r in t["rounds"]:
        a = r.get("action") or {}
        color = _ACT_COLOR.get(a.get("kind"), "#555")
        label = a.get("kind", "?")
        if a.get("kind") == "tool_call":
            label = (f"{a['name']}({a['start']:.0f}s–{a['end']:.0f}s)"
                     + (f"  ⚠ {a['error']}" if a.get("error") else ""))
        block = (
            f"<div style='margin:8px 0;padding:8px;border-left:3px solid {color};background:#0000000a'>"
            f"<b>Round {r['round']}</b> "
            f"<span style='color:#888'>· {r['context_tokens']} ctx tokens</span> "
            f"<span style='color:{color}'>· {_esc(label)}</span>"
            f"<pre style='white-space:pre-wrap;font-size:12px;margin:6px 0;"
            f"max-height:280px;overflow:auto;background:#00000010;padding:6px'>"
            f"{_esc(r['thinking'])}</pre>")
        tr = r.get("tool_result")
        if tr:
            cap = f"{tr['tool']} · {tr.get('n_frames','?')} frames · {tr['span'][0]:.0f}–{tr['span'][1]:.0f}s"
            if tr.get("kept_tokens"):
                cap += f" · {tr['base_tokens']}→{tr['kept_tokens']} tokens (ret {tr['retention']})"
            block += f"<div style='color:#888'>{cap}</div>{_img(tr['montage'])}"
        body.append(block + "</div>")

    display(HTML(head + "".join(body) + "</div>"))
