"""Paired coverage-transition analysis: what does trading spatial fidelity for temporal
coverage buy, at a FIXED downstream visual-token budget?

    PYTHONPATH=/home/cfyang/hanklin python -m longvt_compression.fast_agent.coverage_transition

THE COMPARISON
    U32   32 uniform frames, retention 1.00  -> 14,080 visual tokens
    V128  128 uniform frames, retention 0.25 -> 14,080 visual tokens

Same downstream budget. Sampling is NESTED -- verified from the proxies' real packet
timestamps over all 103 videos, max offset 0.00 ms, because make_skim_proxy writes at
`fps = N / round(duration)` so t_i(32) = i*dur/32 = t_4i(128) exactly. There are therefore
only three coverage states per question and no `U32 hit -> V128 miss` cell.

WHY THE NAIVE TABLE IS NOT USED
    U32   hit 36.1%   acc|hit 44.62   acc|miss 37.11
    V128  hit 57.7%   acc|hit 41.55   acc|miss 40.06
`acc|hit` is computed on a DIFFERENT question subset in each arm. At 32 frames the
questions that happen to be hit skew toward wide evidence spans, which are easier on their
own merits; at 128 frames many narrow, harder questions become hits too. So 44.62 vs 41.55
is mostly a change in subset composition, not evidence becoming less useful. Every number
below is instead computed on a FIXED subset with both arms paired by `question_id`.

WHAT THE DELTA MEANS
    delta(U32 -> V128) = coverage benefit + compression cost
The intervention changes two things at once, so this is a fixed-budget NET effect, not the
causal effect of coverage. Separating the two needs U128 (128 frames, retention 1.00);
where that arm has finished a question, the decomposition is reported alongside.
"""

import json
import math
import os
import random
import statistics as st
from collections import Counter

OUT = "/local1/cfyang/hanklin/outputs/lvbench_agent"
SMALL_N = 30          # below this a per-cell delta is reported but flagged as unreliable


def load(tag):
    p = f"{OUT}/lvbench_{tag}_seed0/results.jsonl"
    if not os.path.exists(p):
        return None
    return {str(r["question_id"]): r for r in (json.loads(l) for l in open(p))}


def coverage(rec, n_frames):
    """Uniform samples at t_i = i*duration/N against the annotated evidence span.

    Returns (n_inside, nearest_distance_to_span_centre, best_cell_IoU). `n_frames` is the
    DECODED frame count, which is the set of instants whose pixels reach the model; the
    model then merges them in pairs (temporal_patch_size=2), so the timestamp markers it
    reads are at N/2 positions that are NOT nested. Containment is scored on the frames
    because that is what determines whether evidence pixels entered the model at all.
    """
    dur = rec["duration"]
    ev = rec.get("evidence")
    if not ev or dur <= 0:
        return None
    t0, t1 = float(ev[0]), float(ev[1])
    ctr = (t0 + t1) / 2
    ts = [i * dur / n_frames for i in range(n_frames)]
    inside = sum(1 for t in ts if t0 <= t <= t1)
    near = min(abs(t - ctr) for t in ts)
    return inside, near


def transitions(a, b, ks):
    """right->right, right->wrong (break), wrong->right (fix), wrong->wrong."""
    rr = sum(1 for q in ks if a[q]["correct"] and b[q]["correct"])
    br = sum(1 for q in ks if a[q]["correct"] and not b[q]["correct"])
    fx = sum(1 for q in ks if not a[q]["correct"] and b[q]["correct"])
    ww = sum(1 for q in ks if not a[q]["correct"] and not b[q]["correct"])
    return rr, br, fx, ww


def mcnemar(br, fx):
    n = br + fx
    if n == 0:
        return 0.0, 1.0
    z = (abs(fx - br) - 1) / math.sqrt(n)
    p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    return z, p


def boot_ci(a, b, ks, iters=4000, seed=0):
    if len(ks) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    d = []
    for _ in range(iters):
        s = [rng.choice(ks) for _ in ks]
        d.append(100 * (sum(1 for q in s if b[q]["correct"])
                        - sum(1 for q in s if a[q]["correct"])) / len(s))
    d.sort()
    return d[int(.025 * iters)], d[int(.975 * iters)]


def acc(d, ks):
    return 100 * sum(1 for q in ks if d[q]["correct"]) / len(ks) if ks else float("nan")


def report_pair(name, a, b, ks, an="A", bn="B", ci=True):
    rr, br, fx, ww = transitions(a, b, ks)
    z, p = mcnemar(br, fx)
    d = acc(b, ks) - acc(a, ks)
    lo, hi = boot_ci(a, b, ks) if ci else (float("nan"), float("nan"))
    flag = "  <- n<%d, UNRELIABLE" % SMALL_N if len(ks) < SMALL_N else ""
    print(f"  {name:30s} n={len(ks):4d}  {an} {acc(a,ks):5.2f}  {bn} {acc(b,ks):5.2f}  "
          f"d={d:+6.2f}  [{lo:+6.2f},{hi:+6.2f}]  fix {fx:3d}  break {br:3d}  "
          f"|z|={z:4.2f} p={p:.4f}{flag}")
    return dict(n=len(ks), a=acc(a, ks), b=acc(b, ks), d=d, fix=fx, brk=br, rr=rr, ww=ww,
                z=z, p=p, lo=lo, hi=hi)


def main():
    A = {k: load(v) for k, v in [
        ("U32", "u32_native"), ("V128", "v128_native_r025"),
        ("U64", "u64_native"), ("V64_50", "v64_native_r050"),
        ("U128", "u128_native")]}
    complete = {k: v for k, v in A.items() if v and len(v) >= 1549}
    ids = sorted(set.intersection(*[set(v) for v in complete.values()]))
    ref = A["U64"]
    ev = [q for q in ids if coverage(ref[q], 32)]
    C32 = {q: coverage(ref[q], 32) for q in ev}
    C128 = {q: coverage(ref[q], 128) for q in ev}

    print("=" * 104)
    print("COVERAGE-TRANSITION ANALYSIS   U32 (32f x 1.00) vs V128 (128f x 0.25)")
    print("both arms deliver 14,080 downstream visual tokens; paired strictly by question_id")
    print("=" * 104)

    viol = [q for q in ev if C32[q][0] > 0 and C128[q][0] == 0]
    print(f"\nnesting check: questions where U32 hits and V128 misses = {len(viol)}  (must be 0)")
    print(f"questions with a usable GT evidence span: {len(ev)} of {len(ids)}")

    cells = [
        ("both miss", [q for q in ev if C32[q][0] == 0 and C128[q][0] == 0]),
        ("miss -> hit", [q for q in ev if C32[q][0] == 0 and C128[q][0] > 0]),
        ("both hit", [q for q in ev if C32[q][0] > 0 and C128[q][0] > 0]),
    ]

    # ---- 1 + 2 -------------------------------------------------------------------
    print("\n[1+2] PAIRED COVERAGE-TRANSITION TABLE, with fix/break counts")
    print("      'both miss' means EXACT GT-WINDOW CONTAINMENT UNCHANGED, not coverage unchanged")
    res = {}
    for nm, ks in cells:
        res[nm] = report_pair(nm, A["U32"], A["V128"], ks, "U32", "V128")
    res["ALL"] = report_pair("ALL", A["U32"], A["V128"], ev, "U32", "V128")

    print("\n      right->right / wrong->wrong, to show how much of each cell never moved:")
    for nm, _ in cells:
        r = res[nm]
        print(f"        {nm:14s} rr {r['rr']:4d}  ww {r['ww']:4d}  fix {r['fix']:3d}  "
              f"break {r['brk']:3d}   churn {100*(r['fix']+r['brk'])/r['n']:.1f}%")

    # ---- 3 + 4 -------------------------------------------------------------------
    tts = [t for t, _ in Counter(ref[q]["task_type"] for q in ev).most_common()]
    for nm, ks in cells[1:]:
        print(f"\n[{3 if nm=='miss -> hit' else 4}] TASK-TYPE BREAKDOWN inside '{nm}'  (n={len(ks)})")
        if nm == "miss -> hit":
            print("      Q: when the extra frames genuinely find the annotated evidence, who benefits?")
        else:
            print("      Q: when both samplers already contain the evidence, who is hurt by the")
            print("         more aggressive compression? (coverage is identical here)")
        for t in tts:
            sel = [q for q in ks if ref[q]["task_type"] == t]
            if not sel:
                continue
            report_pair(t[:28], A["U32"], A["V128"], sel, "U32", "V128", ci=len(sel) >= SMALL_N)

    # ---- 5 -----------------------------------------------------------------------
    print("\n[5] HOW COARSE IS THE BINARY HIT LABEL? nearest-sample distance to the span centre")
    print(f"  {'cell':14s} {'n':>5s} {'U32 near':>9s} {'V128 near':>10s} {'improvement':>12s} "
          f"{'U32 inside':>11s} {'V128 inside':>12s}")
    for nm, ks in cells:
        d32 = [C32[q][1] for q in ks]
        d128 = [C128[q][1] for q in ks]
        imp = [C32[q][1] - C128[q][1] for q in ks]
        i32 = [C32[q][0] for q in ks]
        i128 = [C128[q][0] for q in ks]
        print(f"  {nm:14s} {len(ks):5d} {st.median(d32):8.1f}s {st.median(d128):9.1f}s "
              f"{st.median(imp):11.1f}s {st.mean(i32):11.2f} {st.mean(i128):12.2f}")
    print("  -> in 'both miss' V128 still moves much closer to the evidence; exact containment")
    print("     is unchanged there, but proximity is not.")

    miss = cells[0][1]
    med = st.median(C128[q][1] for q in miss)
    print(f"\n  splitting 'both miss' at V128's median nearest distance ({med:.1f}s):")
    report_pair(f"near half (<{med:.0f}s)", A["U32"], A["V128"],
                [q for q in miss if C128[q][1] < med], "U32", "V128")
    report_pair(f"far half (>={med:.0f}s)", A["U32"], A["V128"],
                [q for q in miss if C128[q][1] >= med], "U32", "V128")

    # ---- 6/7: decomposition where U128 has finished --------------------------------
    if A["U128"]:
        sub = set(A["U128"])
        print(f"\n[6] PARTIAL DECOMPOSITION using U128 (128f x 1.00), {len(sub)}/1549 finished")
        print("      U32 -> U128  isolates the coverage/frame-count benefit (no extra compression)")
        print("      U128 -> V128 isolates the compression cost at fixed frames")
        for nm, ks in cells:
            s = [q for q in ks if q in sub]
            if len(s) < SMALL_N:
                print(f"  {nm:14s} n={len(s)} too small"); continue
            print(f"  --- {nm} (n={len(s)})")
            report_pair("  U32 -> U128  (coverage)", A["U32"], A["U128"], s, "U32", "U128")
            report_pair("  U128 -> V128 (compression)", A["U128"], A["V128"], s, "U128", "V128")
            report_pair("  U32 -> V128  (net)", A["U32"], A["V128"], s, "U32", "V128")

    print("\n[7] CAVEATS")
    print("  - delta(U32->V128) = coverage benefit + compression cost. It is a FIXED-BUDGET NET")
    print("    effect, never the causal effect of temporal coverage on its own.")
    print(f"  - cells with n < {SMALL_N} are flagged UNRELIABLE and should not carry a claim.")
    print("  - the naive acc|hit vs acc|miss table is deliberately not reported: its two columns")
    print("    are different question subsets in each arm.")
    print("  - a matched-budget null control (V64_50 vs V128, delta -0.45pp overall, p=0.74) is")
    print("    available; any task-type effect claimed here should be checked against it.")


if __name__ == "__main__":
    main()
