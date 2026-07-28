"""Recompute every headline number for the final report straight from the run
artifacts, so nothing in the report is quoted from prose."""
import json, os, glob, math, random
from collections import Counter

R = "/local1/cfyang/hanklin/outputs/fast_agent/runs"
O = "/local1/cfyang/hanklin/outputs/fast_agent"
OUT = {}


def load(run):
    d = {}
    p = os.path.join(R, run, "results.jsonl")
    if not os.path.exists(p):
        return d
    for ln in open(p):
        try:
            j = json.loads(ln)
        except json.JSONDecodeError:
            continue
        qid = str(j.get("question_id"))
        d[qid] = j
    return d


def acc(d, keys=None):
    keys = list(d) if keys is None else list(keys)
    if not keys:
        return None
    return sum(1 for k in keys if d[k].get("correct")) / len(keys)


def mcnemar(a, b, keys):
    """exact two-sided McNemar on paired dicts a,b over keys."""
    n01 = sum(1 for k in keys if not a[k].get("correct") and b[k].get("correct"))
    n10 = sum(1 for k in keys if a[k].get("correct") and not b[k].get("correct"))
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return n01, n10, min(1.0, p)


def boot(a, b, keys, iters=10000, seed=0):
    rng = random.Random(seed)
    keys = list(keys)
    ds = []
    for _ in range(iters):
        s = [keys[rng.randrange(len(keys))] for _ in keys]
        ds.append(sum(b[k].get("correct", False) for k in s) / len(s)
                  - sum(a[k].get("correct", False) for k in s) / len(s))
    ds.sort()
    return ds[int(.025 * iters)], ds[int(.975 * iters)]


# ---------------------------------------------------------------- EXP 1: VideoMME
b1, c2 = load("baseline_run1"), load("compress_run2")
keys = sorted(set(b1) & set(c2))
n01, n10, p = mcnemar(b1, c2, keys)
lo, hi = boot(b1, c2, keys)
e1 = {"n_paired": len(keys), "baseline": acc(b1, keys), "compress": acc(c2, keys),
      "fixed": n01, "broke": n10, "p": p, "ci": [lo, hi]}

# tool adoption + call mix in the compress arm
ncalls = Counter(); ntool = 0
for k in keys:
    tc = c2[k].get("tool_calls") or []
    if tc:
        ntool += 1
    for c in tc:
        ncalls[c.get("name")] += 1
e1["tool_adoption"] = ntool / len(keys)
e1["calls"] = dict(ncalls)

# per-task deltas
tt = {}
for k in keys:
    t = b1[k].get("task_type", "?")
    tt.setdefault(t, [0, 0, 0])
    tt[t][0] += 1
    tt[t][1] += bool(b1[k].get("correct"))
    tt[t][2] += bool(c2[k].get("correct"))
e1["by_task"] = {t: {"n": v[0], "base": v[1] / v[0], "comp": v[2] / v[0],
                     "delta": (v[2] - v[1]) / v[0]} for t, v in tt.items() if v[0] >= 10}

# official FlashVID sweep (stock generate)
off = {}
for f in sorted(glob.glob(f"{O}/official_flashvid/*.jsonl")):
    rows = [json.loads(x) for x in open(f) if x.strip()]
    nm = os.path.basename(f)[:-6]
    ok = sum(1 for r in rows if r.get("pred") == r.get("gold"))
    off[nm] = {"n": len(rows), "acc": ok / max(len(rows), 1),
               "tok": rows[0].get("input_tokens") if rows else None,
               "preds": {r["qid"]: r.get("pred") for r in rows},
               "gold": {r["qid"]: r.get("gold") for r in rows}}
e1["official"] = off

# single-shot lossless (our harness)
ss = {}
for d in ("lossless_singleshot_n30", "lossless_singleshot_n30u"):
    rows = [json.loads(x) for x in open(f"{O}/{d}/results.jsonl") if x.strip()]
    for arm in rows[0]["preds"]:
        ok = sum(1 for r in rows if r["preds"].get(arm) == r["gold"])
        ss[arm] = {"n": len(rows), "acc": ok / len(rows),
                   "tok": rows[0]["views"][arm]["tokens"],
                   "n_frames": rows[0]["views"][arm]["n_frames"],
                   "preds": {r["question_id"]: r["preds"].get(arm) for r in rows},
                   "gold": {r["question_id"]: r["gold"] for r in rows}}
e1["singleshot"] = ss
OUT["exp1"] = e1

# ---------------------------------------------------------------- EXP 2: LVBench
arms = {"direct": "lvbench_baseline_full", "autonomous": "lvbench_crop_full",
        "oracle_crop": "lvbench_oracle_crop_full", "oracle_ctrl": "lvbench_oracle_ctrl_full"}
D = {k: load(v) for k, v in arms.items()}
common = sorted(set.intersection(*[set(v) for v in D.values()]))
# oracle_ctrl errors out where evidence spans the whole video -> no control placement
bad = [k for k in common if D["oracle_ctrl"][k].get("error")]
paired = [k for k in common if k not in bad]
e2 = {"n_common": len(common), "n_ctrl_impossible": len(bad), "n_paired": len(paired),
      "acc": {k: acc(v, paired) for k, v in D.items()},
      "refusal": {k: sum(1 for q in paired if v[q].get("pred") is None) / len(paired)
                  for k, v in D.items()}}
for a, b in [("direct", "autonomous"), ("direct", "oracle_crop"),
             ("oracle_ctrl", "oracle_crop"), ("direct", "oracle_ctrl")]:
    n01, n10, p = mcnemar(D[a], D[b], paired)
    lo, hi = boot(D[a], D[b], paired)
    e2.setdefault("contrasts", {})[f"{b}-{a}"] = {
        "delta": acc(D[b], paired) - acc(D[a], paired),
        "fixed": n01, "broke": n10, "p": p, "ci": [lo, hi]}

# CoT rerun
cot = load("lvbench_oracle_crop_cot")
ck = sorted(set(cot) & set(D["oracle_crop"]))
n01, n10, p = mcnemar(D["oracle_crop"], cot, ck)
e2["cot"] = {"n": len(ck), "no_cot": acc(D["oracle_crop"], ck), "cot": acc(cot, ck),
             "fixed": n01, "broke": n10, "p": p}

# parity
par = json.load(open(f"{O}/analysis/parity.json"))
pq = {r["qid"] for r in par["stock_img"]}
cust = {k: v for k, v in D["direct"].items() if k in pq}
e2["parity"] = {"n": len(pq),
                "custom": acc(cust), "custom_none": sum(1 for v in cust.values() if v.get("pred") is None)}
for arm in ("stock_img", "stock_vid"):
    rows = par[arm]
    e2["parity"][arm] = sum(1 for r in rows if r["pred"] == r["gold"]) / len(rows)
    e2["parity"][arm + "_none"] = sum(1 for r in rows if r["pred"] is None)

# coverage / localization features
F = json.load(open(f"{O}/analysis/features.json"))
fx = {str(r.get("qid", r.get("question_id"))): r for r in F} if isinstance(F, list) else F
e2["features_keys"] = (list(fx.values())[0].keys().__iter__().__length_hint__()
                       if fx else 0)
OUT["exp2"] = e2

# ---------------------------------------------------------------- qid128
q1 = json.load(open(f"{O}/analysis/qid128/summary.json"))
q2 = json.load(open(f"{O}/analysis/qid128_r30/summary.json"))
OUT["qid128"] = {
    "r01": {"kept": q1["grid"]["kept"], "base": q1["grid"]["base"],
            "conc": q1["evidence_share"] / q1["uniform_share"]},
    "ladder": {r: {"kept": q2["stats"][r]["kept"], "conc": q2["stats"][r]["concentration"],
                   "median": q2["stats"][r]["median_kept"],
                   "x_crop": q2["stats"][r]["kept"] / 6144}
               for r in q2["stats"]},
    "agent_r03": q2["agent"],
}
json.dump(OUT, open(f"{O}/analysis/final_report_numbers.json", "w"), indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str)[:6000])
