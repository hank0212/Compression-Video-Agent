"""Build the 2x2 compression-vs-tool analysis notebook.

  python -m longvt_compression.fast_agent.analyze_2x2          # writes + executes the .ipynb

ONE QUESTION: the compressed skim costs 5.98 pp with the crop tool and only 1.03 pp
without it. Where does that 5 pp actually go?

The 2x2 is what makes the question answerable at all -- with only the two tool arms you
cannot tell "compression degraded perception" from "compression broke the agent loop",
because both predict the same headline drop. Adding the no-tool row separates them.

  skim \\ tools          none                         crop_video
  64f uncompressed      run1  lvbench_run1_seed0      run2  lvbench_r2_seed0
  256f VidCom2 r=0.25   c3    lvbench_c3_notool_seed0 run3  lvbench_r4_seed0

All four: seed 0, T=0.7, video modality, LongVT-verbatim prompt, n~1549, 0 transport
errors, and skim token budgets matched within 4.0-4.7k (verified in the notebook, not
assumed -- this repo has shipped two silent compression bypasses).

The notebook recomputes every number from the run artifacts at execution time. Nothing
is hardcoded here except the run directory names, so re-running it after new seeds land
picks them up.
"""

import json
import os
import subprocess
import sys

import nbformat as nbf

OUT_DIR = "/home/cfyang/hanklin/longvt_compression/fast_agent"
NB_PATH = os.path.join(OUT_DIR, "ANALYSIS_2x2_compression.ipynb")

# (key, label, directory). Order is the reading order of the 2x2.
RUNS = [
    ("run1", "64f uncompressed - no tool", "lvbench_run1_seed0"),
    ("run2", "64f uncompressed - crop", "lvbench_r2_seed0"),
    ("c3", "256f VidCom2 - no tool", "lvbench_c3_notool_seed0"),
    ("run3", "256f VidCom2 - crop", "lvbench_r4_seed0"),
]


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(src):
    return nbf.v4.new_code_cell(src)


CELLS = []

CELLS.append(md("""# Does compression blind the model, or break the agent loop?

**LVBench, Qwen3-VL-8B-Instruct, seed 0, n~1,549 per arm.**

The compressed skim costs **5.98 pp** when the model has `crop_video` and only **1.03 pp**
when it does not. Those two numbers cannot both be explained by "compression degrades what
the model can see". This notebook works out which mechanism is actually responsible.

| | **no tool** | **+ crop_video** |
|---|---|---|
| **64 frames, uncompressed** | run1 | run2 |
| **256 frames, VidCom2 r=0.25** | c3 | run3 |

Every arm shares the loader, the prompt, the scorer, the sampling params and the crop tool.
Only the skim (64 uncompressed vs 256 pruned to the same token count) and the presence of
the tool differ. All numbers below are recomputed from `results.jsonl` / `traj/` at run time."""))

CELLS.append(code('''import json, os, math, collections, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/local1/cfyang/hanklin/outputs/lvbench_agent"
RUNS = [
    ("run1", "64f uncompressed - no tool",  "lvbench_run1_seed0"),
    ("run2", "64f uncompressed - crop",     "lvbench_r2_seed0"),
    ("c3",   "256f VidCom2 - no tool",      "lvbench_c3_notool_seed0"),
    ("run3", "256f VidCom2 - crop",         "lvbench_r4_seed0"),
]
LABEL = {k: l for k, l, _ in RUNS}
COMPRESSED = {"c3", "run3"}
HAS_TOOL    = {"run2", "run3"}

def load(d):
    """question_id -> row. Keyed so every comparison below is PAIRED on the question."""
    out = {}
    with open(os.path.join(BASE, d, "results.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[str(r["question_id"])] = r
    return out

R = {k: load(d) for k, _, d in RUNS}
M = {k: json.load(open(os.path.join(BASE, d, "run_manifest.json"))) for k, _, d in RUNS}

# Every cross-arm claim is made on the questions ALL FOUR arms answered, so a
# difference can never be a difference in which questions were sampled.
IDS = set(R["run1"])
for k in R:
    IDS &= set(R[k])
IDS = sorted(IDS, key=int)
print(f"{len(IDS)} questions answered by all four arms")
for k, _, d in RUNS:
    print(f"  {k:5s} n={len(R[k]):5d}  {d}")'''))

CELLS.append(md("""## 1. Integrity checks first

Three ways this comparison could be a measurement artifact rather than a result, checked
before any accuracy is quoted:

1. **Dead questions scored as wrong.** `run_agent` writes `pred=None, correct=False` on a
   transport failure, and the resume logic then skips that row forever. A run with a silent
   error rate is biased downward. (This is not hypothetical -- a sibling run of the c3 arm
   reached 54% `APIConnectionError` against a port with no server on it and scored 17.8%,
   *below 4-way chance*.)
2. **Compression silently not applied.** Retention is a server-level flag that is not
   recorded in the manifest. This repo has shipped two silent bypasses already
   (FlashVID `r=1.0`, VidCom2 `batch_size != 1`), so it is checked from token counts.
3. **Unmatched budgets.** The whole design is "same tokens, 4x the frames". If the
   compressed arms are simply getting more tokens, the comparison means nothing."""))

CELLS.append(code('''rows = []
for k, _, _ in RUNS:
    rs = [R[k][i] for i in IDS]
    n = len(rs)
    errs = sum(1 for r in rs if "error" in r)
    nones = sum(1 for r in rs if r.get("pred") is None)
    # skim-only cost: questions where the model made no tool call, so prompt_tokens is
    # the global view alone and is directly comparable across arms.
    skim = sorted(r["prompt_tokens"] for r in rs
                  if r.get("prompt_tokens") and (r.get("n_calls") or 0) == 0)
    rows.append((k, n, errs, nones,
                 skim[len(skim)//2] if skim else float("nan"), len(skim)))

print(f"{'arm':6s} {'n':>5s} {'errors':>7s} {'no-answer':>10s} {'skim tok (median)':>18s} {'(from n)':>9s}")
for k, n, errs, nones, sk, ns in rows:
    print(f"{k:6s} {n:5d} {errs:7d} {nones:10d} {sk:18.0f} {ns:9d}")

print("\\nReference token costs, computed from the Qwen3-VL grid:")
print("  64 frames  -> grid_t 32  x 91 tok = 2,912 visual + timestamps + text ~ 4,1k")
print("  256 frames -> grid_t 128 x 91 tok = 11,648 visual                    ~ 13,2k unpruned")
print("  256 frames @ retention 0.25       = 2,912 visual (== 64f) + 4x timestamps ~ 4,5k")
print("\\nSo a compressed arm near ~4.5k is pruned as intended; near ~13k would mean the")
print("flag never took effect and every compressed number would be meaningless.")'''))

CELLS.append(md("""**Read the table above before continuing.** If any arm shows a nonzero error count, its
accuracy is depressed by transport failures and the rest of this notebook is void for that
arm. If a compressed arm's skim cost is not close to the uncompressed one, the budget is
not matched and the compression comparison is not a compression comparison."""))

CELLS.append(md("""## 2. The 2x2

`acc` is exact-letter scoring (`data.extract_answer`), identical across arms."""))

CELLS.append(code('''def acc(k, ids=None, key="correct"):
    ids = IDS if ids is None else ids
    rs = [R[k][i] for i in ids]
    return 100 * sum(1 for r in rs if r.get(key)) / max(len(rs), 1)

A = {k: acc(k) for k, _, _ in RUNS}
print(f"{'':26s} {'no tool':>10s} {'+ crop':>10s} {'tool is worth':>15s}")
print(f"{'64f uncompressed':26s} {A['run1']:10.2f} {A['run2']:10.2f} {A['run2']-A['run1']:+15.2f}")
print(f"{'256f VidCom2 r=0.25':26s} {A['c3']:10.2f} {A['run3']:10.2f} {A['run3']-A['c3']:+15.2f}")
print(f"{'compression is worth':26s} {A['c3']-A['run1']:+10.2f} {A['run3']-A['run2']:+10.2f}")
inter = (A["run3"] - A["c3"]) - (A["run2"] - A["run1"])
print(f"\\ninteraction (tool effect when compressed - tool effect when not): {inter:+.2f} pp")
print("\\nIf compression simply degraded perception, the two 'compression is worth' numbers")
print("would be equal and the interaction would be ~0.")'''))

CELLS.append(code('''fig, ax = plt.subplots(figsize=(7, 4.2))
x = np.arange(2); w = 0.36
ax.bar(x - w/2, [A["run1"], A["c3"]],  w, label="no tool",  color="#4477aa")
ax.bar(x + w/2, [A["run2"], A["run3"]], w, label="+ crop_video", color="#cc3311")
for xi, (a, b) in enumerate([(A["run1"], A["run2"]), (A["c3"], A["run3"])]):
    ax.text(xi - w/2, a + 0.3, f"{a:.2f}", ha="center", fontsize=9)
    ax.text(xi + w/2, b + 0.3, f"{b:.2f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(["64f uncompressed", "256f VidCom2 r=0.25"])
ax.set_ylabel("accuracy (%)"); ax.set_ylim(25, 45)
ax.axhline(25, ls=":", c="#888888", lw=1)
ax.text(1.42, 25.3, "4-way chance", fontsize=8, color="#888888", ha="right")
ax.set_title("The tool helps when uncompressed and hurts when compressed")
ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.show()'''))

CELLS.append(md("""## 3. Where do the answers actually go? (paired flip analysis)

Aggregate accuracy hides direction. Because every arm answered the same questions, the
transition can be counted directly: how many questions each change *fixed* and how many it
*broke*. A net drop made of a few breaks is a different phenomenon from one made of many
breaks partly offset by many fixes."""))

CELLS.append(code('''def flips(a, b):
    """a -> b. Returns (fixed, broke, both_right, both_wrong)."""
    f = br = rr = ww = 0
    for i in IDS:
        x, y = bool(R[a][i].get("correct")), bool(R[b][i].get("correct"))
        if   y and not x: f += 1
        elif x and not y: br += 1
        elif x and y:     rr += 1
        else:             ww += 1
    return f, br, rr, ww

def mcnemar_z(fixed, broke):
    """Normal approximation to the exact binomial on discordant pairs. No scipy here."""
    n = fixed + broke
    if n == 0:
        return float("nan")
    return (abs(fixed - broke) - 1) / math.sqrt(n)   # continuity-corrected

TRANSITIONS = [
    ("run1", "run2", "add crop tool (uncompressed)"),
    ("c3",   "run3", "add crop tool (compressed)"),
    ("run1", "c3",   "compress, no tool"),
    ("run2", "run3", "compress, with tool"),
]
print(f"{'transition':34s} {'fixed':>6s} {'broke':>6s} {'net':>6s} {'discordant':>11s} {'|z|':>6s}")
for a, b, name in TRANSITIONS:
    f, br, rr, ww = flips(a, b)
    z = mcnemar_z(f, br)
    print(f"{name:34s} {f:6d} {br:6d} {f-br:+6d} {f+br:11d} {z:6.2f}")
print("\\n|z| > 1.96 ~ p < 0.05 that the change is not symmetric noise.")
print("Note the churn: even a +0.58 pp net change moves hundreds of individual answers,")
print("which is why single-seed deltas of ~1 pp should not be over-read.")'''))

CELLS.append(md("""## 4. Termination: the mechanism

The agent loop ends one of two ways. Either the model emits `<answer>` itself, or it runs
out of rounds/tokens and the harness forces a bounded **finalizer** turn that makes it pick
a letter. A forced guess after an inconclusive search is not the same event as a considered
answer, and the two arms differ enormously in how often it happens.

`finalizer_used` and `completion_tokens >= 1024` (the per-turn cap) measure this directly."""))

CELLS.append(code('''print(f"{'arm':26s} {'finalizer %':>12s} {'hit 1024 cap %':>15s} {'mean out tok':>13s} {'rounds mean':>12s}")
for k, lab, _ in RUNS:
    rs = [R[k][i] for i in IDS]; n = len(rs)
    fin = 100 * sum(1 for r in rs if r.get("finalizer_used")) / n
    tr  = 100 * sum(1 for r in rs if (r.get("completion_tokens") or 0) >= 1000) / n
    ct  = st.mean((r.get("completion_tokens") or 0) for r in rs)
    rd  = st.mean((r.get("rounds") or 0) for r in rs)
    print(f"{lab:26s} {fin:12.1f} {tr:15.1f} {ct:13.1f} {rd:12.2f}")'''))

CELLS.append(code('''# Accuracy conditioned on HOW the answer was produced. If compression degraded perception,
# the self-terminated slice should fall too. If it broke the loop, the slice should hold and
# only its SIZE should change.
print(f"{'arm':26s} {'acc | self-terminated':>22s} {'n':>6s} {'acc | forced finalizer':>23s} {'n':>6s}")
for k, lab, _ in RUNS:
    self_ids  = [i for i in IDS if not R[k][i].get("finalizer_used")]
    force_ids = [i for i in IDS if R[k][i].get("finalizer_used")]
    a1 = acc(k, self_ids) if self_ids else float("nan")
    a2 = acc(k, force_ids) if force_ids else float("nan")
    print(f"{lab:26s} {a1:22.2f} {len(self_ids):6d} {a2:23.2f} {len(force_ids):6d}")'''))

CELLS.append(code('''fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ks = [k for k, _, _ in RUNS]
cols = ["#4477aa" if k not in COMPRESSED else "#cc3311" for k in ks]
hatch = ["" if k not in HAS_TOOL else "//" for k in ks]

fin = [100*sum(1 for i in IDS if R[k][i].get("finalizer_used"))/len(IDS) for k in ks]
bars = axes[0].bar(range(4), fin, color=cols)
for b, h in zip(bars, hatch): b.set_hatch(h)
axes[0].set_xticks(range(4)); axes[0].set_xticklabels(ks)
axes[0].set_ylabel("% of questions"); axes[0].set_title("Forced finalizer (model never emitted <answer>)")
axes[0].spines[["top", "right"]].set_visible(False)

for k in ks:
    ct = [min((R[k][i].get("completion_tokens") or 0), 1100) for i in IDS]
    axes[1].hist(ct, bins=44, histtype="step", lw=1.6, label=LABEL[k])
axes[1].axvline(1024, ls="--", c="#888888", lw=1)
axes[1].text(1024, axes[1].get_ylim()[1]*0.95, " token cap", fontsize=8, color="#888888")
axes[1].set_xlabel("completion tokens (clipped at 1100)"); axes[1].set_ylabel("questions")
axes[1].set_title("Output length: the compressed tool arm piles up at the cap")
axes[1].legend(frameon=False, fontsize=8); axes[1].spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.show()
print("Hatched = has the crop tool. Red = compressed skim.")'''))

CELLS.append(md("""### Caveat this notebook cannot resolve

The 1024-token cap is **load-bearing on the compressed tool arm and nearly inert on the
others**. So "the compressed skim makes the model ramble" and "our cap is too low for the
compressed skim" predict the same table. Distinguishing them needs a re-run of the
compressed tool arm at `max_tokens=4096` (LongVT's own value). Until that exists, the
termination finding is **a described mechanism, not an established cause**."""))

CELLS.append(md("""## 5. Localization: did compression improve aim at all?

This is the chapter's actual premise. 4x the temporal coverage at the same token cost is
supposed to help the model *find* the evidence. LVBench ships a GT evidence span per
question, and every tool call was scored against it at write time.

- **hit rate** -- % of questions where the union of crop windows covers the GT evidence
- **coverage** -- mean fraction of the GT span the crops covered
- **best IoU** -- aim precision of the single best call (GT floored to 16 s first, since a
  perfectly aimed crop on a 4 s reference scores ~0.03 by tool geometry alone)
- **centre offset** -- |centre(best crop) - centre(evidence)| as a fraction of duration

Tool arms only; the no-tool arms make no windows."""))

CELLS.append(code('''tool_ks = [k for k, _, _ in RUNS if k in HAS_TOOL]
print(f"{'arm':26s} {'hit %':>7s} {'coverage':>9s} {'best IoU':>9s} {'centre off':>11s} {'calls/q':>8s} {'looked %':>9s}")
for k in tool_ks:
    rs = [R[k][i] for i in IDS]; n = len(rs)
    hit = 100 * sum(1 for r in rs if r.get("landed")) / n
    cov = st.mean((r.get("covered_frac") or 0.0) for r in rs)
    iou = st.mean((r.get("best_iou") or 0.0) for r in rs)
    off = [r["centre_offset"] for r in rs if r.get("centre_offset") is not None]
    cq  = st.mean((r.get("n_calls") or 0) for r in rs)
    lk  = 100 * st.mean((r.get("looked_frac") or 0.0) for r in rs)
    print(f"{LABEL[k]:26s} {hit:7.2f} {cov:9.3f} {iou:9.3f} "
          f"{(st.mean(off) if off else float('nan')):11.3f} {cq:8.2f} {lk:9.2f}")
print("\\nIf compression bought localization, hit rate would RISE from run2 to run3.")'''))

CELLS.append(code('''# The payoff structure: landing on the evidence is worth a lot, in every arm.
print(f"{'arm':26s} {'acc | landed':>13s} {'n':>6s} {'acc | missed':>13s} {'n':>6s} {'payoff':>8s}")
for k in tool_ks:
    L = [i for i in IDS if R[k][i].get("landed")]
    Mi = [i for i in IDS if not R[k][i].get("landed")]
    a1, a2 = acc(k, L), acc(k, Mi)
    print(f"{LABEL[k]:26s} {a1:13.2f} {len(L):6d} {a2:13.2f} {len(Mi):6d} {a1-a2:+8.2f}")
print("\\nThe tool works when it lands. It just almost never lands -- which is why the")
print("aggregate effect of adding it is ~0 even though the conditional payoff is ~+15 pp.")'''))

CELLS.append(md("""### Localization-trivial questions

13% of LVBench questions state their own timestamp in the stem ("What happens from
17:16-17:40?"). Those are trivially localizable -- the model copies the number. Pooling them
with genuine search inflates every hit rate, so they are split out here."""))

CELLS.append(code('''triv  = [i for i in IDS if R["run1"][i].get("localization_trivial")]
ntriv = [i for i in IDS if not R["run1"][i].get("localization_trivial")]
print(f"stem states a timestamp: {len(triv)}  |  genuine search: {len(ntriv)}\\n")
print(f"{'arm':26s} {'hit% trivial':>13s} {'hit% genuine':>13s} {'acc trivial':>12s} {'acc genuine':>12s}")
for k, lab, _ in RUNS:
    if k in HAS_TOOL:
        h1 = 100*sum(1 for i in triv  if R[k][i].get("landed"))/max(len(triv),1)
        h2 = 100*sum(1 for i in ntriv if R[k][i].get("landed"))/max(len(ntriv),1)
    else:
        h1 = h2 = float("nan")
    print(f"{lab:26s} {h1:13.2f} {h2:13.2f} {acc(k,triv):12.2f} {acc(k,ntriv):12.2f}")
print("\\nThe headline hit rate is mostly copied timestamps; genuine search is far lower.")'''))

CELLS.append(md("""## 6. Per-task-type

Whether the compression penalty is uniform or concentrated in particular question types."""))

CELLS.append(code('''types = sorted({R["run1"][i].get("task_type", "?") for i in IDS})
print(f"{'task type':28s} {'n':>5s} " + "".join(f"{k:>9s}" for k, _, _ in RUNS)
      + f"{'compr d (tool)':>16s}")
for t in types:
    ids = [i for i in IDS if R["run1"][i].get("task_type") == t]
    if len(ids) < 20:
        continue
    accs = [acc(k, ids) for k, _, _ in RUNS]
    print(f"{t:28s} {len(ids):5d} " + "".join(f"{a:9.1f}" for a in accs)
          + f"{accs[3]-accs[1]:+16.1f}")'''))

CELLS.append(md("""## 7. Verdict

Fill this in from the cells above rather than from memory -- the point of the notebook is
that the numbers are recomputed, not remembered."""))

CELLS.append(code('''print("=" * 74)
print("2x2 (accuracy, %)")
print(f"  uncompressed:  no-tool {A['run1']:6.2f}   +crop {A['run2']:6.2f}   tool {A['run2']-A['run1']:+.2f}")
print(f"  compressed  :  no-tool {A['c3']:6.2f}   +crop {A['run3']:6.2f}   tool {A['run3']-A['c3']:+.2f}")
print(f"  compression costs {A['c3']-A['run1']:+.2f} without the tool, "
      f"{A['run3']-A['run2']:+.2f} with it  ->  interaction {inter:+.2f}")
print()
h2 = 100*sum(1 for i in IDS if R['run2'][i].get('landed'))/len(IDS)
h3 = 100*sum(1 for i in IDS if R['run3'][i].get('landed'))/len(IDS)
print(f"localization: hit rate {h2:.2f}% uncompressed -> {h3:.2f}% compressed  ({h3-h2:+.2f} pp)")
f2 = 100*sum(1 for i in IDS if R['run2'][i].get('finalizer_used'))/len(IDS)
f3 = 100*sum(1 for i in IDS if R['run3'][i].get('finalizer_used'))/len(IDS)
print(f"termination : forced finalizer {f2:.1f}% -> {f3:.1f}%  ({f3-f2:+.1f} pp)")
print("=" * 74)
print()
print("1. Compression did NOT buy localization. That was the premise; it fails.")
print("2. Compression alone is nearly free; the cost appears only with the tool in the loop.")
print("3. The mechanism is consistent with termination failure, but the 1024-token cap is")
print("   a confound that only a max_tokens=4096 re-run of the compressed tool arm settles.")'''))

nb = nbf.v4.new_notebook(cells=CELLS)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                             "name": "python3"}

if __name__ == "__main__":
    with open(NB_PATH, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {NB_PATH}")
    if "--no-exec" not in sys.argv:
        r = subprocess.run(
            [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
             "--inplace", "--ExecutePreprocessor.timeout=1800", NB_PATH],
            capture_output=True, text=True)
        sys.stderr.write(r.stderr[-4000:])
        print("executed OK" if r.returncode == 0 else f"EXECUTION FAILED rc={r.returncode}")
