"""Coverage / IoU / flip analysis across the frames x compression grid.

    PYTHONPATH=/home/cfyang/hanklin python -m longvt_compression.fast_agent.analyze_grid

Eight arms: four frame counts (32/64/128/256) at a matched ~11.6k visual-token budget,
each with compression off, and 128/256 additionally with EVS and VidCom2 at a matched
FINAL budget.

Two things this has to keep straight:

  * The 128-frame arms cover 500 questions and the others 1,549. Every cross-frame
    number below is computed on the INTERSECTION, and the n is printed, so nothing is
    ever compared across different question sets.
  * Coverage is a property of the frame count alone -- where the samples land in time --
    so it is identical for compression on and off at the same frame count. What
    compression changes is how many tokens each of those samples carries.
"""
import math, os, json, random

OUT = "/local1/cfyang/hanklin/outputs/lvbench_agent"

ARMS = {
    ("uniform",  32): "lvbench_d3_e_nopruning_seed0",
    ("uniform",  64): "lvbench_d4_g_64f_noprune_seed0",
    ("uniform", 128): "lvbench_pA_128f_uni_1x_seed0",
    ("uniform", 256): "lvbench_d5_j_256f_uniform_seed0",
    ("EVS",     128): "lvbench_pD_128f_evs_050_seed0",
    ("EVS",     256): "lvbench_d5_k_256f_evs_seed0",
    ("VidCom2", 128): "lvbench_pC2_128f_vidcom2_050_FIXED_seed0",
    ("VidCom2", 256): "lvbench_pI2_256f_vidcom2_025_FIXED_seed0",
}
PREFIX_UNFIXED = set()   # all VidCom2 arms now run the post-2026-08-17 fixed plugin


def load(tag):
    p = f"{OUT}/{tag}/results.jsonl"
    return {r["question_id"]: r for r in (json.loads(l) for l in open(p))} if os.path.exists(p) else None


def acc(d, ks):
    return 100 * sum(d[q]["correct"] for q in ks) / len(ks) if ks else float("nan")


def mcnemar(d1, d2, ks):
    b = sum(1 for q in ks if d1[q]["correct"] and not d2[q]["correct"])
    c = sum(1 for q in ks if d2[q]["correct"] and not d1[q]["correct"])
    return b, c, ((abs(b - c) - 1) / math.sqrt(b + c) if b + c else 0.0)


def boot(d1, d2, ks, iters=4000, seed=0):
    rng = random.Random(seed)
    diff = [d2[q]["correct"] - d1[q]["correct"] for q in ks]
    n = len(diff)
    st = sorted(100 * sum(diff[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return st[int(.025 * iters)], st[int(.975 * iters)]


def cov(rec, N):
    """(frames inside evidence, nearest-sample distance, best cell IoU) for N uniform samples."""
    dur = rec["duration"]; ev = rec.get("evidence")
    if not ev or dur <= 0: return None
    t0, t1 = float(ev[0]), float(ev[1]); ctr = (t0 + t1) / 2
    delta = dur / N
    ts = [i * dur / N for i in range(N)]
    inside = sum(1 for t in ts if t0 <= t <= t1)
    near = min(abs(t - ctr) for t in ts)
    best = 0.0
    for t in ts:
        a0, a1 = t - delta / 2, t + delta / 2
        inter = max(0.0, min(a1, t1) - max(a0, t0))
        union = (a1 - a0) + (t1 - t0) - inter
        if union > 0: best = max(best, inter / union)
    return inside, near, best, delta


D = {k: load(v) for k, v in ARMS.items()}
missing = [k for k, v in D.items() if v is None]
if missing: print("MISSING:", missing)
D = {k: v for k, v in D.items() if v}

ALL = sorted(set.intersection(*[set(v) for v in D.values()]))
ALL = [q for q in ALL if D[("uniform", 32)][q].get("evidence") and D[("uniform", 32)][q]["duration"] > 0]
COVER = {N: {q: cov(D[("uniform", N)][q], N) for q in ALL} for N in (32, 64, 128, 256)}
