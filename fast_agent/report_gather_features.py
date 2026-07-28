"""Stage 2: localization/behaviour features for the report (LVBench, n=183)."""
import json, statistics as st

O = "/local1/cfyang/hanklin/outputs/fast_agent"
F = json.load(open(f"{O}/analysis/features.json"))
out = {}

# sanity: do the feature rows reproduce the arm accuracies?
out["acc_check"] = {a: sum(r[f"{a}_correct"] for r in F) / len(F)
                    for a in ("baseline", "crop", "oracle_crop", "oracle_ctrl")}
out["n"] = len(F)

# --- coverage: did the autonomous arm's crops ever contain the evidence?
cov = [r["crop_cov"] for r in F]
out["coverage"] = {"mean": st.mean(cov),
                   "zero": sum(c == 0 for c in cov) / len(cov),
                   "full": sum(c >= 1.0 for c in cov) / len(cov)}
hit = [r for r in F if r["crop_cov"] >= 1.0]
miss = [r for r in F if r["crop_cov"] == 0.0]
out["hit_vs_miss"] = {
    "n_hit": len(hit), "n_miss": len(miss),
    "crop_acc_on_hit": sum(r["crop_correct"] for r in hit) / max(len(hit), 1),
    "oracle_acc_on_hit": sum(r["oracle_crop_correct"] for r in hit) / max(len(hit), 1),
    "crop_acc_on_miss": sum(r["crop_correct"] for r in miss) / max(len(miss), 1),
    "oracle_acc_on_miss": sum(r["oracle_crop_correct"] for r in miss) / max(len(miss), 1),
}


def corr(xs, ys):
    if len(xs) < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** .5
    return num / den if den else None


def centers(rows):
    xs, ys = [], []
    for r in rows:
        sp = r["crop_spans"]
        if not sp:
            continue
        xs.append(sum(sp[0]) / 2)
        ys.append((r["ev_start"] + r["ev_end"]) / 2)
    return xs, ys


out["aim"] = {"r_on_hit": corr(*centers(hit)), "r_on_miss": corr(*centers(miss))}

# --- search geometry: how wide are the crops vs the evidence?
w = [e - s for r in F for s, e in r["crop_spans"]]
out["spans"] = {"n_calls": len(w), "median_width": st.median(w) if w else None,
                "median_ev_width": st.median([r["ev_width"] for r in F]),
                "median_calls": st.median([r["crop_ncalls"] for r in F]),
                "median_duration": st.median([r["duration"] for r in F])}
# fraction of the video actually inspected
frac = []
for r in F:
    tot = sum(e - s for s, e in r["crop_spans"])
    frac.append(min(tot / r["duration"], 1.0))
out["spans"]["median_video_fraction_seen"] = st.median(frac)

# --- where does the evidence sit, and does that predict failure to find it?
buckets = {"first 10%": [], "10-33%": [], "33-66%": [], "last 33%": []}
for r in F:
    f = r["ev_center_frac"]
    k = ("first 10%" if f < .10 else "10-33%" if f < .33 else
         "33-66%" if f < .66 else "last 33%")
    buckets[k].append(r["crop_cov"] == 0.0)
out["zero_cov_by_position"] = {k: {"n": len(v), "zero_cov": sum(v) / len(v)}
                              for k, v in buckets.items() if v}

# --- refusals and degenerate behaviour
for a in ("baseline", "crop", "oracle_crop", "oracle_ctrl"):
    out.setdefault("behaviour", {})[a] = {
        "finalizer": sum(r[f"{a}_finalizer"] for r in F) / len(F),
        "pred_none": sum(r[f"{a}_pred"] is None for r in F) / len(F),
        "says_notfound": sum(r[f"{a}_says_notfound"] for r in F) / len(F),
        "whole_video": sum(r[f"{a}_whole_video"] for r in F) / len(F),
        "median_think_chars": st.median([r[f"{a}_think_chars"] for r in F]),
    }

# --- the ceiling: what fails even with perfect evidence?
oc_fail = [r for r in F if not r["oracle_crop_correct"]]
out["ceiling"] = {"n_oracle_fail": len(oc_fail),
                  "share_of_all": len(oc_fail) / len(F),
                  "also_fail_direct": sum(not r["baseline_correct"] for r in oc_fail) / len(oc_fail)}

# --- localization-trivial questions (the ones that state their own timestamp)
tv = [r for r in F if r["localization_trivial"]]
nt = [r for r in F if not r["localization_trivial"]]
out["trivial"] = {
    "n_trivial": len(tv), "n_nontrivial": len(nt),
    "crop_acc_trivial": sum(r["crop_correct"] for r in tv) / max(len(tv), 1),
    "crop_acc_nontrivial": sum(r["crop_correct"] for r in nt) / max(len(nt), 1),
    "cov_trivial": st.mean([r["crop_cov"] for r in tv]) if tv else None,
    "cov_nontrivial": st.mean([r["crop_cov"] for r in nt]) if nt else None,
}

json.dump(out, open(f"{O}/analysis/final_report_features.json", "w"), indent=1)
print(json.dumps(out, indent=1))
