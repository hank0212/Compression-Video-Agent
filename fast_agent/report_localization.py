"""Localization scorecard (LVBench, n=183): does the autonomous arm crop the right TIME?

Answers one question the accuracy tables cannot: the oracle arm is handed the GT evidence
span, the autonomous arm has to find it. How close does it get?

Three complementary numbers, because no single one is honest here:
  - HIT   (recall): is the GT evidence fully inside the union of the arm's crops? "Did it
                    look in the right place at all" -- insensitive to how wide it looked.
  - IoU   (precision of the aim): best single crop call vs the GT span. Punishes the
                    "crop 500 seconds and hope" strategy that HIT rewards.
  - OFFSET: |center(best crop) - center(evidence)| as % of video duration. Degrades
                    gracefully where IoU saturates at 0 (a near-miss and a wild miss both
                    score IoU=0; offset separates them).

CAVEAT on raw IoU: LVBench evidence spans are tiny (median 14s, p25 4s, 30/183 zero-width)
while the crop tool returns up to 128 frames @1fps = a 128s window. A perfectly-aimed crop
on a 4s reference is IoU<=0.03 by tool geometry alone. So the GT span is floored to
config.ORACLE_CROP_MIN_WIDTH (16s, the SAME floor oracle.py applies) before the IoU -- which
makes oracle_crop the natural IoU~1.0 ceiling and every other arm readable against it.
Raw-GT IoU is reported alongside so the floor's effect is visible, not hidden.

Usage:  python -m longvt_compression.fast_agent.report_localization [--out DIR] [--fig]
"""

import argparse
import json
import os
import statistics as st

from . import config

OUT_ROOT = "/local1/cfyang/hanklin/outputs/fast_agent"
FEATURES = os.path.join(OUT_ROOT, "analysis", "features.json")
ARMS = ("baseline", "crop", "oracle_ctrl", "oracle_crop")
ARM_LABEL = {"baseline": "direct (no tools)",
             "crop": "autonomous crop",
             "oracle_ctrl": "oracle ctrl (wrong place)",
             "oracle_crop": "oracle crop (GT)"}
IOU_THRESH = (0.1, 0.3, 0.5)


# --------------------------------------------------------------------------- primitives
def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _floor_span(s: float, e: float, min_w: float) -> tuple[float, float]:
    """Widen a span about its centre to at least min_w (never clamped to the video here --
    the GT reference only ever feeds an IoU, and clamping would make the floor asymmetric
    for evidence at t=0)."""
    if e - s >= min_w:
        return (s, e)
    mid = 0.5 * (s + e)
    return (mid - min_w / 2, mid + min_w / 2)


def _union_len(spans, clip: tuple[float, float]) -> float:
    """Total length of `spans` intersected with `clip` (spans may overlap)."""
    segs = sorted((max(s, clip[0]), min(e, clip[1])) for s, e in spans)
    total, cur_s, cur_e = 0.0, None, None
    for s, e in segs:
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            total += (cur_e - cur_s) if cur_e is not None else 0.0
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def score_row(r: dict, arm: str, min_w: float) -> dict:
    """Per-question localization scores for one arm."""
    ev_raw = (r["ev_start"], r["ev_end"])
    ev = _floor_span(*ev_raw, min_w)
    spans = [tuple(s) for s in r[f"{arm}_spans"]]
    dur = r["duration"]

    best_iou = max((_iou(s, ev) for s in spans), default=0.0)
    best_iou_raw = max((_iou(s, ev_raw) for s in spans), default=0.0)
    # union-of-calls IoU: credits multi-call search, punishes total time inspected
    union_inter = _union_len(spans, ev)
    union_tot = _union_len(spans, (0.0, dur))
    union_iou = (union_inter / (union_tot + (ev[1] - ev[0]) - union_inter)
                 if union_tot > 0 else 0.0)
    cov = union_inter / (ev[1] - ev[0]) if ev[1] > ev[0] else 0.0

    # aim error from the crop whose centre lands closest to the evidence centre
    ev_c = 0.5 * (ev[0] + ev[1])
    offset = min((abs(0.5 * (s + e) - ev_c) for s, e in spans), default=None)

    return {"qid": r["qid"], "task_type": r["task_type"],
            "trivial": r["localization_trivial"],
            "correct": bool(r[f"{arm}_correct"]),
            "n_calls": r[f"{arm}_ncalls"],
            "hit": cov >= 1.0, "overlap": cov > 0.0, "cov": cov,
            "best_iou": best_iou, "best_iou_raw": best_iou_raw, "union_iou": union_iou,
            "offset_s": offset, "offset_pct": (offset / dur * 100) if offset else offset,
            "ev_width": ev_raw[1] - ev_raw[0], "duration": dur,
            "seen_s": union_tot, "seen_pct": union_tot / dur * 100}


def aggregate(rows: list[dict]) -> dict:
    """Arm-level summary. Questions with no tool call count as IoU 0 / HIT 0 (a refusal to
    search IS a localization failure) but are excluded from the offset median, which is
    only defined where a crop exists."""
    n = len(rows)
    if not n:
        return {}
    offs = [r["offset_pct"] for r in rows if r["offset_pct"] is not None]
    out = {"n": n,
           "acc": 100 * sum(r["correct"] for r in rows) / n,
           "hit": 100 * sum(r["hit"] for r in rows) / n,
           "overlap": 100 * sum(r["overlap"] for r in rows) / n,
           "iou_mean": st.mean(r["best_iou"] for r in rows),
           "iou_median": st.median(r["best_iou"] for r in rows),
           "iou_raw_mean": st.mean(r["best_iou_raw"] for r in rows),
           "union_iou_mean": st.mean(r["union_iou"] for r in rows),
           "offset_median_pct": st.median(offs) if offs else None,
           "n_no_call": sum(r["n_calls"] == 0 for r in rows),
           "seen_median_pct": st.median(r["seen_pct"] for r in rows),
           "calls_median": st.median(r["n_calls"] for r in rows)}
    for t in IOU_THRESH:
        out[f"r@{t}"] = 100 * sum(r["best_iou"] >= t for r in rows) / n
    return out


# --------------------------------------------------------------------------- tables
def _fmt(v, spec=".3f"):
    return "--" if v is None else format(v, spec)


def table_main(agg: dict) -> str:
    hdr = (f"| arm | n | acc % | HIT % | any-overlap % | IoU mean | IoU med | "
           + " | ".join(f"R@{t} %" for t in IOU_THRESH)
           + " | centre offset (% of video) | video seen % | calls |")
    sep = "|" + "---|" * (11 + len(IOU_THRESH) - 3 + 3)
    sep = "|" + "---|" * (hdr.count("|") - 1)
    lines = [hdr, sep]
    for a in ARMS:
        s = agg[a]
        lines.append(
            f"| {ARM_LABEL[a]} | {s['n']} | {s['acc']:.1f} | {s['hit']:.1f} | "
            f"{s['overlap']:.1f} | {s['iou_mean']:.3f} | {s['iou_median']:.3f} | "
            + " | ".join(f"{s[f'r@{t}']:.1f}" for t in IOU_THRESH)
            + f" | {_fmt(s['offset_median_pct'], '.1f')} | {s['seen_median_pct']:.1f} | "
              f"{s['calls_median']:.0f} |")
    return "\n".join(lines)


def table_strata(rows_by_arm: dict, key, order, label: str) -> str:
    lines = [f"| {label} | n | " + " | ".join(f"{ARM_LABEL[a]} IoU / HIT%" for a in
                                              ("crop", "oracle_crop")) + " | crop acc % |",
             "|---|---|---|---|---|"]
    for k in order:
        sub = {a: [r for r in rows_by_arm[a] if key(r) == k] for a in rows_by_arm}
        if not sub["crop"]:
            continue
        cells = []
        for a in ("crop", "oracle_crop"):
            s = aggregate(sub[a])
            cells.append(f"{s['iou_mean']:.3f} / {s['hit']:.0f}")
        c = aggregate(sub["crop"])
        lines.append(f"| {k} | {len(sub['crop'])} | " + " | ".join(cells)
                     + f" | {c['acc']:.1f} |")
    return "\n".join(lines)


def table_iou_vs_acc(rows: list[dict], oracle: dict) -> str:
    """Does aiming better actually pay off in accuracy? Buckets the autonomous arm by its
    own IoU and shows accuracy inside each bucket, with the oracle's accuracy on the SAME
    questions as the per-bucket ceiling."""
    buckets = [("IoU = 0 (missed)", lambda v: v == 0),
               ("0 < IoU < 0.1", lambda v: 0 < v < 0.1),
               ("0.1 <= IoU < 0.3", lambda v: 0.1 <= v < 0.3),
               ("0.3 <= IoU < 0.5", lambda v: 0.3 <= v < 0.5),
               ("IoU >= 0.5", lambda v: v >= 0.5)]
    lines = ["| autonomous crop IoU bucket | n | crop acc % | oracle acc % (same q) | gap pp |",
             "|---|---|---|---|---|"]
    for name, pred in buckets:
        sub = [r for r in rows if pred(r["best_iou"])]
        if not sub:
            continue
        qids = {r["qid"] for r in sub}
        osub = [r for r in oracle if r["qid"] in qids]
        ca = 100 * sum(r["correct"] for r in sub) / len(sub)
        oa = 100 * sum(r["correct"] for r in osub) / len(osub)
        lines.append(f"| {name} | {len(sub)} | {ca:.1f} | {oa:.1f} | {oa - ca:+.1f} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- figure
def make_figure(rows_by_arm: dict, path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    xs = [i / 100 for i in range(101)]

    ax = axes[0]
    for a in ARMS:
        v = sorted(r["best_iou"] for r in rows_by_arm[a])
        ys = [100 * sum(1 for x in v if x >= t) / len(v) for t in xs]
        ax.plot(xs, ys, label=ARM_LABEL[a], lw=2)
    ax.set_xlabel("temporal IoU threshold"); ax.set_ylabel("% of questions >= threshold")
    ax.set_title("Localization: R@IoU curve"); ax.grid(alpha=.3); ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist([r["best_iou"] for r in rows_by_arm["crop"]], bins=20, range=(0, 1),
            color="#4477aa")
    ax.set_xlabel("best-call IoU vs GT evidence"); ax.set_ylabel("questions")
    ax.set_title("Autonomous arm: where the aim lands"); ax.grid(alpha=.3)

    ax = axes[2]
    hit = [r for r in rows_by_arm["crop"] if r["hit"]]
    miss = [r for r in rows_by_arm["crop"] if not r["hit"]]
    for lbl, sub, c in (("evidence hit", hit, "#228833"), ("evidence missed", miss, "#cc3311")):
        ax.scatter([r["ev_width"] + .5 for r in sub], [r["best_iou"] for r in sub],
                   s=14, alpha=.6, label=f"{lbl} (n={len(sub)})", color=c)
    ax.set_xscale("log"); ax.set_xlabel("GT evidence width (s, +0.5 for log)")
    ax.set_ylabel("best-call IoU"); ax.set_title("IoU vs how narrow the target is")
    ax.grid(alpha=.3); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    return path


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=FEATURES)
    ap.add_argument("--out", default=os.path.join(OUT_ROOT, "analysis"))
    ap.add_argument("--min-width", type=float, default=config.ORACLE_CROP_MIN_WIDTH,
                    help="GT floor for the IoU denominator (default: oracle's own floor)")
    ap.add_argument("--fig", action="store_true", help="also write localization.png")
    args = ap.parse_args()

    F = json.load(open(args.features))
    rows_by_arm = {a: [score_row(r, a, args.min_width) for r in F] for a in ARMS}
    agg = {a: aggregate(rows_by_arm[a]) for a in ARMS}

    ev_w = sorted(r["ev_width"] for r in rows_by_arm["crop"])
    md = [
        "# Localization scorecard — LVBench, autonomous vs oracle",
        "",
        f"n = {len(F)} questions | GT evidence width: median {st.median(ev_w):.0f}s, "
        f"p25 {ev_w[len(ev_w)//4]:.0f}s, {sum(1 for w in ev_w if w == 0)} zero-width "
        f"| IoU computed against the GT span floored to {args.min_width:.0f}s "
        "(the same floor oracle.py applies, so oracle_crop ≈ 1.0 by construction).",
        "",
        "**HIT** = GT evidence fully inside the union of the arm's crops (recall). "
        "**IoU** = best single crop call vs GT (precision of the aim). "
        "**centre offset** = |centre(nearest crop) − centre(evidence)| as % of duration.",
        "",
        table_main(agg),
        "",
        "## Does better aim pay off?",
        "",
        table_iou_vs_acc(rows_by_arm["crop"], rows_by_arm["oracle_crop"]),
        "",
        "## By question type",
        "",
        table_strata(rows_by_arm, lambda r: r["task_type"],
                     sorted({r["task_type"] for r in rows_by_arm["crop"]}), "task type"),
        "",
        "## Localization-trivial (question states its own timestamp) vs not",
        "",
        table_strata(rows_by_arm, lambda r: r["trivial"], [True, False],
                     "states its own timestamp"),
        "",
        "## By how narrow the target is",
        "",
        table_strata(rows_by_arm,
                     lambda r: ("<=5s" if r["ev_width"] <= 5 else
                                "5-30s" if r["ev_width"] <= 30 else
                                "30-120s" if r["ev_width"] <= 120 else ">120s"),
                     ["<=5s", "5-30s", "30-120s", ">120s"], "GT evidence width"),
        "",
        f"Raw-GT IoU (no floor), autonomous arm: mean {agg['crop']['iou_raw_mean']:.3f} "
        f"— reported only to show the floor's effect; on a 4s reference the 128-frame crop "
        f"tool caps IoU at ~0.03, so raw IoU measures the tool's granularity, not the agent.",
        "",
    ]
    md = "\n".join(md)
    print(md)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "localization.md"), "w") as f:
        f.write(md)
    json.dump({"min_width": args.min_width, "summary": agg,
               "per_question": {a: rows_by_arm[a] for a in ARMS}},
              open(os.path.join(args.out, "localization.json"), "w"), indent=1)
    paths = [os.path.join(args.out, "localization.md"),
             os.path.join(args.out, "localization.json")]
    if args.fig:
        paths.append(make_figure(rows_by_arm, os.path.join(args.out, "localization.png")))
    print("\nwrote:\n  " + "\n  ".join(paths))


if __name__ == "__main__":
    main()
