"""Paired comparison and stratification for LVBench skim arms.

    PYTHONPATH=/home/cfyang/hanklin python -m longvt_compression.fast_agent.analyze_pairs

Two rules this module enforces, both learned the hard way:

1. PAIR ON `question_id`, NEVER ON LINE ORDER. results.jsonl is written in completion
   order under N workers, so two runs of the same question set are not row-aligned.
   Pairing positionally once produced 351/289 flips where the truth was 157/95.

2. AN UNPAIRED "hit vs miss" SPLIT IS NOT CAUSAL. Questions with wide evidence spans are
   both easier to hit AND easier to answer, so `acc | hit` minus `acc | miss` inside one
   arm measures selection, not the value of hitting. Only the paired split -- same
   questions, coverage changed by the arm -- supports a causal reading.

Reported per comparison: delta, paired flips, continuity-corrected McNemar |z|, and a
bootstrap 95% CI on the paired delta (resampling question_ids, which is the unit of
independence).
"""

import json
import math
import os
import random

OUT = "/local1/cfyang/hanklin/outputs/lvbench_agent"


def load(tag):
    p = f"{OUT}/{tag}/results.jsonl"
    if not os.path.exists(p):
        return None
    return {r["question_id"]: r for r in (json.loads(l) for l in open(p))}


def acc(d, ks):
    return 100 * sum(d[q]["correct"] for q in ks) / len(ks) if ks else float("nan")


def mcnemar(d1, d2, ks):
    b = sum(1 for q in ks if d1[q]["correct"] and not d2[q]["correct"])
    c = sum(1 for q in ks if d2[q]["correct"] and not d1[q]["correct"])
    z = (abs(b - c) - 1) / math.sqrt(b + c) if b + c else 0.0
    return b, c, z


def boot_ci(d1, d2, ks, iters=4000, seed=0):
    """Percentile CI on Acc(d2)-Acc(d1), resampling question_ids with replacement."""
    rng = random.Random(seed)
    diff = [d2[q]["correct"] - d1[q]["correct"] for q in ks]
    n = len(diff)
    if n == 0:
        return float("nan"), float("nan")
    stats = []
    for _ in range(iters):
        s = sum(diff[rng.randrange(n)] for _ in range(n))
        stats.append(100 * s / n)
    stats.sort()
    return stats[int(.025 * iters)], stats[int(.975 * iters)]


def compare(name, d1, d2, ks, label1="A", label2="B"):
    b, c, z = mcnemar(d1, d2, ks)
    lo, hi = boot_ci(d1, d2, ks)
    print(f"  {name:38} n={len(ks):5d}  {label1} {acc(d1,ks):6.2f}%  {label2} {acc(d2,ks):6.2f}%"
          f"  delta {acc(d2,ks)-acc(d1,ks):+6.2f}  CI [{lo:+.2f},{hi:+.2f}]"
          f"  flips {b:4d}/{c:<4d} |z|={z:.2f}{'' if z<=1.96 else '  SIG'}")


def frames_inside(rec, N):
    """How many of N uniformly sampled frames land inside the GT evidence span."""
    dur = rec["duration"]
    ev = rec.get("evidence")
    if not ev or dur <= 0:
        return None
    t0, t1 = float(ev[0]), float(ev[1])
    return sum(1 for i in range(N) if t0 <= i * dur / N <= t1)


def main():
    arms = {
        "g 64f":  (load("lvbench_d4_g_64f_noprune_seed0"), 64),
        "j 256f": (load("lvbench_d5_j_256f_uniform_seed0"), 256),
        "l 256f 2x": (load("lvbench_d6_l_256f_2xbudget_seed0"), 256),
    }
    arms = {k: v for k, v in arms.items() if v[0]}
    gd, gn = arms["g 64f"]
    jd, jn = arms["j 256f"]
    K = [q for q in sorted(set(gd) & set(jd))
         if gd[q].get("evidence") and gd[q]["duration"] > 0]

    print("=" * 100)
    print("HEADLINE PAIRED COMPARISONS")
    print("=" * 100)
    compare("g 64f -> j 256f (1x budget)", gd, jd, K, "g", "j")
    if "l 256f 2x" in arms:
        ld = arms["l 256f 2x"][0]
        Kl = [q for q in K if q in ld]
        compare("j 256f 1x -> l 256f 2x", jd, ld, Kl, "j", "l")
        compare("g 64f 1x -> l 256f 2x", gd, ld, Kl, "g", "l")

    gi = {q: frames_inside(gd[q], gn) for q in K}
    ji = {q: frames_inside(jd[q], jn) for q in K}

    print()
    print("=" * 100)
    print("WHO BENEFITS FROM 64 -> 256 TEMPORAL DENSITY?  (paired, g vs j)")
    print("=" * 100)

    print("\n-- by evidence-span width")
    spans = {q: float(gd[q]["evidence"][1]) - float(gd[q]["evidence"][0]) for q in K}
    bins = [(0, 0, "zero-width"), (0.001, 8, "1-8 s"), (8, 20, "8-20 s"),
            (20, 60, "20-60 s"), (60, 1e9, ">60 s")]
    for lo, hi, nm in bins:
        ks = [q for q in K if (spans[q] == 0 if nm == "zero-width" else lo <= spans[q] < hi)]
        if len(ks) >= 25:
            compare(nm, gd, jd, ks, "g", "j")

    print("\n-- by change in frames landing inside the evidence")
    buckets = [(lambda a, b: b - a == 0, "no change"),
               (lambda a, b: 0 < b - a <= 5, "+1..5 frames"),
               (lambda a, b: 5 < b - a <= 20, "+6..20 frames"),
               (lambda a, b: b - a > 20, ">+20 frames")]
    for fn, nm in buckets:
        ks = [q for q in K if fn(gi[q], ji[q])]
        if len(ks) >= 25:
            compare(nm, gd, jd, ks, "g", "j")

    print("\n-- by hit status (PAIRED -- the causal split)")
    for nm, fn in (("both hit", lambda q: gi[q] > 0 and ji[q] > 0),
                   ("j hits, g misses", lambda q: gi[q] == 0 and ji[q] > 0),
                   ("neither hits", lambda q: gi[q] == 0 and ji[q] == 0)):
        ks = [q for q in K if fn(q)]
        if len(ks) >= 25:
            compare(nm, gd, jd, ks, "g", "j")

    print("\n-- by task type")
    types = {}
    for q in K:
        types.setdefault(gd[q].get("task_type") or "?", []).append(q)
    for t, ks in sorted(types.items(), key=lambda x: -len(x[1])):
        if len(ks) >= 25:
            compare(t, gd, jd, ks, "g", "j")


if __name__ == "__main__":
    main()
