"""Separate the benefit of more frames from the cost of compression, and ask which of the
answers that extra frames rescue actually survive compression.

    PYTHONPATH=/home/cfyang/hanklin python -m longvt_compression.fast_agent.temporal_rescue

THE THREE ARMS, all n=1,549 and all joined by `question_id`:
    U32   32 frames,  retention 1.00 -> 14,080 visual tokens
    U128  128 frames, retention 1.00 -> 56,320 visual tokens
    V128  128 frames, retention 0.25 -> 14,080 visual tokens

    U32  -> U128   more frames, no extra compression   = the temporal benefit
    U128 -> V128   identical frames, a quarter kept    = the compression cost
    U32  -> V128   both at once, same budget as U32    = the fixed-budget net effect

Sampling is nested: the proxies are written at `fps = N/round(duration)`, so
t_i(32) = i*dur/32 = t_4i(128) exactly (verified from real packet timestamps over all 103
videos, max offset 0.00 ms). U128 and V128 share their frame timestamps, so their hit
status is identical by construction and the coverage groups are well defined.

The unpaired `acc | hit` vs `acc | miss` split is NOT used anywhere here. At 32 frames the
questions that happen to be hit skew toward wide, easy evidence spans, so that split
measures subset composition as much as the value of containment. Every figure below is on
a fixed question subset with all three arms paired.
"""

import argparse
import json
import math
import os
import random
from collections import Counter

OUT = "/local1/cfyang/hanklin/outputs/lvbench_agent"
SMALL_N = 30
ARMS = {"U32": "u32_native", "U128": "u128_native", "V128": "v128_native_r025",
        "U64": "u64_native"}


def load(tag):
    p = f"{OUT}/lvbench_{tag}_seed0/results.jsonl"
    return {str(r["question_id"]): r for r in (json.loads(l) for l in open(p))}


def hit(rec, n_frames):
    """At least one sampled frame timestamp inside the GT evidence window."""
    dur, ev = rec["duration"], rec.get("evidence")
    if not ev or dur <= 0:
        return None
    t0, t1 = float(ev[0]), float(ev[1])
    return any(t0 <= i * dur / n_frames <= t1 for i in range(n_frames))


def acc(A, k, ks):
    return 100 * sum(1 for q in ks if A[k][q]["correct"]) / len(ks) if ks else float("nan")


def fixbreak(A, a, b, ks):
    fx = sum(1 for q in ks if not A[a][q]["correct"] and A[b][q]["correct"])
    br = sum(1 for q in ks if A[a][q]["correct"] and not A[b][q]["correct"])
    z = (abs(fx - br) - 1) / math.sqrt(fx + br) if fx + br else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))) if fx + br else 1.0
    return fx, br, z, p


def boot(A, a, b, ks, iters=4000, seed=0):
    if len(ks) < SMALL_N:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    d = []
    for _ in range(iters):
        s = [rng.choice(ks) for _ in ks]
        d.append(100 * (sum(1 for q in s if A[b][q]["correct"])
                        - sum(1 for q in s if A[a][q]["correct"])) / len(s))
    d.sort()
    return d[int(.025 * iters)], d[int(.975 * iters)]


def line(A, name, ks, note=""):
    lo1, hi1 = boot(A, "U32", "U128", ks)
    lo2, hi2 = boot(A, "U128", "V128", ks)
    f1, b1, z1, p1 = fixbreak(A, "U32", "U128", ks)
    f2, b2, z2, p2 = fixbreak(A, "U128", "V128", ks)
    small = "  <- n<%d" % SMALL_N if len(ks) < SMALL_N else ""
    print(f"  {name:26s} {len(ks):4d} {acc(A,'U32',ks):6.2f} {acc(A,'U128',ks):6.2f} "
          f"{acc(A,'V128',ks):6.2f} | {acc(A,'U128',ks)-acc(A,'U32',ks):+6.2f} "
          f"[{lo1:+5.1f},{hi1:+5.1f}] p={p1:.3f} {f1:3d}/{b1:<3d} | "
          f"{acc(A,'V128',ks)-acc(A,'U128',ks):+6.2f} [{lo2:+5.1f},{hi2:+5.1f}] p={p2:.3f} "
          f"{f2:3d}/{b2:<3d} | {acc(A,'V128',ks)-acc(A,'U32',ks):+6.2f}{small}{note}")


def header():
    print(f"  {'group':26s} {'n':>4s} {'U32':>6s} {'U128':>6s} {'V128':>6s} | "
          f"{'frame gain U128-U32':^38s} | {'compression V128-U128':^38s} | {'net':>6s}")
    print(f"  {'':26s} {'':>4s} {'':>6s} {'':>6s} {'':>6s} | "
          f"{'delta  95% CI          p   fix/brk':38s} | "
          f"{'delta  95% CI          p   fix/brk':38s} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=f"{OUT}/temporal_rescue_cases.json")
    a = ap.parse_args()

    A = {k: load(v) for k, v in ARMS.items()}
    ids = sorted(set.intersection(*[set(v) for v in A.values()]))
    ref = A["U64"]
    ev = [q for q in ids if hit(ref[q], 32) is not None]
    H32 = {q: hit(ref[q], 32) for q in ev}
    H128 = {q: hit(ref[q], 128) for q in ev}

    bad = [q for q in ev if H32[q] and not H128[q]]
    print(f"nesting check: U32 hit with 128-frame miss = {len(bad)} (must be 0)")
    print(f"questions with a usable GT evidence span: {len(ev)} of {len(ids)}\n")

    groups = [("both miss", [q for q in ev if not H32[q] and not H128[q]]),
              ("miss -> hit", [q for q in ev if not H32[q] and H128[q]]),
              ("both hit", [q for q in ev if H32[q] and H128[q]])]

    print("=" * 150)
    print("[1] COVERAGE-TRANSITION GROUPS  (U32 / U128 / V128, paired on question_id)")
    print("=" * 150)
    header()
    for nm, ks in groups:
        line(A, nm, ks)
    line(A, "ALL", ev)

    print("\n" + "=" * 150)
    print("[2] TASK TYPE inside 'miss -> hit'  -- where the extra 128-frame sampling newly")
    print("    enters the annotated evidence window")
    print("=" * 150)
    mh = groups[1][1]
    header()
    tts = [t for t, _ in Counter(ref[q]["task_type"] for q in mh).most_common()]
    for t in tts:
        line(A, t[:26], [q for q in mh if ref[q]["task_type"] == t])

    print("\n" + "=" * 150)
    print("[3] TEMPORAL-RESCUE SURVIVAL RATE")
    print("    rescue      = U32 wrong -> U128 right  (the extra frames fixed it)")
    print("    survival    = P(V128 correct | rescue) -- does the fix outlive compression?")
    print("=" * 150)

    def rescue(ks):
        return [q for q in ks if not A["U32"][q]["correct"] and A["U128"][q]["correct"]]

    allr = rescue(ev)
    kept = [q for q in allr if A["V128"][q]["correct"]]
    lost = [q for q in allr if not A["V128"][q]["correct"]]
    print(f"\n  temporal rescues overall      {len(allr):4d}")
    print(f"  preserved by V128             {len(kept):4d}")
    print(f"  destroyed by compression      {len(lost):4d}")
    print(f"  survival rate                 {100*len(kept)/len(allr):.1f}%")

    # a baseline to compare against: how often is U128 right overall preserved?
    u128r = [q for q in ev if A["U128"][q]["correct"]]
    base = 100 * sum(1 for q in u128r if A["V128"][q]["correct"]) / len(u128r)
    print(f"\n  for reference, P(V128 right | U128 right) over ALL questions: {base:.1f}%")
    surv = 100 * len(kept) / len(allr)
    print(f"  -> a rescued answer survives compression {surv:.1f}% of the time against "
          f"{base:.1f}% for U128's correct answers at large,")
    print(f"     so compression is {'MORE' if surv < base else 'LESS'} likely to destroy exactly "
          f"what the extra frames just fixed ({base-surv:+.1f} pp).")

    print(f"\n  {'subset':34s} {'rescues':>8s} {'kept':>6s} {'lost':>6s} {'survival':>9s}")
    for nm, ks in groups:
        r = rescue(ks)
        if not r:
            print(f"  {nm:34s} {0:8d}")
            continue
        k = sum(1 for q in r if A["V128"][q]["correct"])
        print(f"  {nm:34s} {len(r):8d} {k:6d} {len(r)-k:6d} {100*k/len(r):8.1f}%"
              + ("   <- few" if len(r) < SMALL_N else ""))
    print()
    for t in tts:
        r = rescue([q for q in mh if ref[q]["task_type"] == t])
        if not r:
            print(f"  {('miss->hit / ' + t)[:34]:34s} {0:8d}")
            continue
        k = sum(1 for q in r if A["V128"][q]["correct"])
        print(f"  {('miss->hit / ' + t)[:34]:34s} {len(r):8d} {k:6d} {len(r)-k:6d} "
              f"{100*k/len(r):8.1f}%" + ("   <- few" if len(r) < SMALL_N else ""))

    grp = {q: nm for nm, ks in groups for q in ks}
    export = {
        "definition": {
            "preserved": "U32 wrong -> U128 right -> V128 right",
            "destroyed": "U32 wrong -> U128 right -> V128 wrong",
        },
        "counts": {"rescues": len(allr), "preserved": len(kept), "destroyed": len(lost),
                   "survival_rate": round(100 * len(kept) / len(allr), 2)},
        "preserved": [{"question_id": q, "task_type": ref[q]["task_type"],
                       "coverage_group": grp[q], "videoID": ref[q]["videoID"],
                       "evidence": ref[q]["evidence"], "duration": ref[q]["duration"],
                       "gold": ref[q]["gold"], "U32": A["U32"][q]["pred"],
                       "U128": A["U128"][q]["pred"], "V128": A["V128"][q]["pred"]}
                      for q in kept],
        "destroyed": [{"question_id": q, "task_type": ref[q]["task_type"],
                       "coverage_group": grp[q], "videoID": ref[q]["videoID"],
                       "evidence": ref[q]["evidence"], "duration": ref[q]["duration"],
                       "gold": ref[q]["gold"], "U32": A["U32"][q]["pred"],
                       "U128": A["U128"][q]["pred"], "V128": A["V128"][q]["pred"]}
                      for q in lost],
    }
    with open(a.export, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\n  question ids exported -> {a.export}")
    print(f"    preserved (wrong->right->right) {len(kept)}")
    print(f"    destroyed (wrong->right->wrong) {len(lost)}")


if __name__ == "__main__":
    main()
