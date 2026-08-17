"""Faithfulness diagnostic for the LVBench skim arms.

    PYTHONPATH=/home/cfyang/hanklin python -m longvt_compression.fast_agent.diagnose_runs

WHAT THIS IS FOR
----------------
Every number in this chapter is a *difference between two arms*. A difference is
only meaningful if the arms differ in exactly the one thing we claim. This script
tries to falsify that, rather than assuming it: it re-derives comparability from
the artefacts on disk instead of trusting the manifest.

The checks are ordered by how badly a failure would corrupt the result:

 1. TIMELINE  -- the proxy must reproduce the SOURCE duration. Qwen3-VL derives every
    `<t seconds>` marker as frame_index / fps read off the container. A proxy whose
    duration drifts relabels the whole video, and nothing downstream would notice.
 2. PAIRING   -- arms must cover the same question_ids with the same gold/options.
    results.jsonl is written in completion order under N workers, so LINE ORDER IS
    NOT ALIGNED across runs; anything paired positionally is silently wrong. (This
    bit us once already: 351/289 flips vs the true 157/95.)
 3. BUDGET    -- prompt_tokens must match the intended visual-token budget. Qwen3-VL
    caps the whole video at size.longest_edge/(2*1024) = 12,288 tokens regardless of
    frame count, so "more frames" silently thins rather than adds.
 4. INTEGRITY -- transport errors, missing answers, duplicates. A dead arm looks like
    a bad arm: an earlier run wrote 223 APIConnectionErrors as pred=None/correct=False
    and scored 17.8%, below chance.
"""

import json
import os
import subprocess
from collections import Counter

OUT = "/local1/cfyang/hanklin/outputs/lvbench_agent"

# (label, run dir, proxy dir, frames, expected visual-token budget)
ARMS = [
    ("e 32f no-prune",  "lvbench_d3_e_nopruning_seed0",  "skim_proxies_hires",     32,  12288),
    ("f 32f VidCom2",   "lvbench_d3_f_vidcom2_seed0",    "skim_proxies_hires",     32,   3072),
    ("g 64f no-prune",  "lvbench_d4_g_64f_noprune_seed0","skim_proxies_hires_64",  64,  12288),
    ("i 256f VidCom2",  "lvbench_d4_i_256f_vidcom2_seed0","skim_proxies_hires_256",256, 12288),
]


def load(run):
    p = os.path.join(OUT, run, "results.jsonl")
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p)]


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json",
         "-show_entries", "stream=nb_frames,width,height:format=duration", path],
        capture_output=True, text=True)
    if out.returncode != 0:
        return None
    d = json.loads(out.stdout)
    st = d["streams"][0]
    return {"n": int(st.get("nb_frames") or 0), "w": int(st["width"]),
            "h": int(st["height"]), "dur": float(d["format"]["duration"])}


def check_timeline(arms):
    """The proxy's own duration must reproduce the source's, or every timestamp lies."""
    print("\n" + "=" * 78)
    print("1. TIMELINE FIDELITY  (proxy duration vs source duration)")
    print("=" * 78)
    for label, run, pdir, frames, _ in arms:
        rows = load(run)
        if rows is None:
            print(f"  {label:18} SKIP (no results)")
            continue
        # one row per video: duration comes from the dataset, not the proxy
        vids = {}
        for r in rows:
            vids.setdefault(r["videoID"], r["duration"])
        worst, bad, nframes = (0.0, None), [], Counter()
        for vid, src_dur in vids.items():
            pp = os.path.join(OUT, pdir, f"{vid}_n{frames}.mp4")
            if not os.path.exists(pp):
                bad.append((vid, "MISSING PROXY"))
                continue
            info = probe(pp)
            if info is None:
                bad.append((vid, "ffprobe failed"))
                continue
            nframes[info["n"]] += 1
            drift = abs(info["dur"] - src_dur) / max(src_dur, 1)
            if drift > worst[0]:
                worst = (drift, vid)
            if drift > 0.02:
                bad.append((vid, f"duration {info['dur']:.0f}s vs source {src_dur:.0f}s"))
        print(f"  {label:18} {len(vids):3d} videos | worst drift {worst[0]*100:.2f}% "
              f"({worst[1]}) | frame counts {dict(nframes)}")
        for v, why in bad[:5]:
            print(f'      BAD {v}: {why}')
        if not bad:
            print("      OK -- every proxy reproduces its source timeline within 2%")


def check_pairing(arms):
    """Arms must cover the same questions with the same labels."""
    print("\n" + "=" * 78)
    print("2. CROSS-ARM PAIRING  (same questions, same gold, same options)")
    print("=" * 78)
    loaded = [(l, load(r)) for l, r, *_ in arms]
    loaded = [(l, rows) for l, rows in loaded if rows]
    if len(loaded) < 2:
        print("  fewer than 2 arms present")
        return
    ref_label, ref = loaded[0]
    refm = {r["question_id"]: r for r in ref}
    print(f"  reference = {ref_label} ({len(ref)} rows, {len(refm)} unique qids)")
    if len(ref) != len(refm):
        print(f"      DUPLICATE question_ids in reference: {len(ref)-len(refm)}")
    for label, rows in loaded[1:]:
        m = {r["question_id"]: r for r in rows}
        dup = len(rows) - len(m)
        only_ref, only_arm = set(refm) - set(m), set(m) - set(refm)
        gold_mismatch = [q for q in set(refm) & set(m) if refm[q]["gold"] != m[q]["gold"]]
        vid_mismatch = [q for q in set(refm) & set(m) if refm[q]["videoID"] != m[q]["videoID"]]
        ev_mismatch = [q for q in set(refm) & set(m)
                       if refm[q].get("evidence") != m[q].get("evidence")]
        flag = "OK " if not (dup or only_ref or only_arm or gold_mismatch or vid_mismatch) else "BAD"
        print(f"  {flag} {label:18} n={len(rows):4d} dup={dup} "
              f"only_ref={len(only_ref)} only_arm={len(only_arm)} "
              f"gold_mismatch={len(gold_mismatch)} video_mismatch={len(vid_mismatch)} "
              f"evidence_mismatch={len(ev_mismatch)}")


def check_budget(arms):
    """prompt_tokens must land on the intended visual-token budget."""
    print("\n" + "=" * 78)
    print("3. TOKEN BUDGET  (measured prompt_tokens vs intended)")
    print("=" * 78)
    print(f"  {'arm':18} {'n':>5} {'median':>8} {'p10':>8} {'p90':>8} {'intended':>9} {'grid':>6} {'tok/grid':>9}")
    for label, run, _, frames, budget in arms:
        rows = load(run)
        if rows is None:
            print(f"  {label:18} SKIP")
            continue
        pt = sorted(r["prompt_tokens"] for r in rows if r.get("prompt_tokens"))
        if not pt:
            print(f"  {label:18} no prompt_tokens recorded")
            continue
        med = pt[len(pt) // 2]
        grid = frames // 2
        print(f"  {label:18} {len(pt):5d} {med:8,} {pt[len(pt)//10]:8,} "
              f"{pt[9*len(pt)//10]:8,} {budget:9,} {grid:6d} {med/grid:9.1f}")


def check_integrity(arms):
    """A dead arm looks like a bad arm. Count the ways a row can be junk."""
    print("\n" + "=" * 78)
    print("4. INTEGRITY  (errors, missing answers, extraction, termination)")
    print("=" * 78)
    print(f"  {'arm':18} {'n':>5} {'err':>5} {'pred=None':>10} {'trunc':>6} {'finalizer':>10} "
          f"{'strict!=loose':>14} {'acc':>7}")
    for label, run, *_ in arms:
        rows = load(run)
        if rows is None:
            print(f"  {label:18} SKIP")
            continue
        n = len(rows)
        err = sum("error" in r for r in rows)
        none = sum(r.get("pred") is None for r in rows)
        trunc = sum((r.get("completion_tokens") or 0) >= 1024 for r in rows)
        fin = sum(bool(r.get("finalizer_used")) for r in rows)
        mism = sum(r.get("pred") != r.get("pred_strict") for r in rows)
        acc = 100 * sum(r["correct"] for r in rows) / n
        print(f"  {label:18} {n:5d} {err:5d} {none:10d} {trunc:6d} {fin:10d} {mism:14d} {acc:6.2f}%")


def check_manifests(arms):
    """Anything that differs between arms besides the intended variable is a confound."""
    print("\n" + "=" * 78)
    print("5. MANIFEST DIFF  (fields that must be identical across arms)")
    print("=" * 78)
    must_match = ["dataset", "num", "seed", "tools", "skim_mode", "prompt_style",
                  "max_rounds", "temperature", "top_p", "top_k", "presence_penalty",
                  "frequency_penalty", "repetition_penalty", "max_tokens",
                  "finalizer_max_tokens", "crop_source", "unify_time_format"]
    mans = {}
    for label, run, *_ in arms:
        p = os.path.join(OUT, run, "run_manifest.json")
        if os.path.exists(p):
            mans[label] = json.load(open(p))
    if not mans:
        print("  no manifests")
        return
    ref_label = list(mans)[0]
    for k in must_match:
        vals = {l: m.get(k) for l, m in mans.items()}
        if len(set(json.dumps(v, sort_keys=True) for v in vals.values())) > 1:
            print(f"  DIFFERS  {k}: {vals}")
    print(f"  all other pinned fields identical across {len(mans)} arms "
          f"(reference {ref_label})")
    print("  intended differences:")
    for l, m in mans.items():
        print(f"    {l:18} frames={m.get('frames'):4} proxy={os.path.basename(m.get('proxy_dir') or '')} "
              f"port={(m.get('base_url') or '').split(':')[-1].split('/')[0]}")


def main():
    present = [a for a in ARMS if load(a[1]) is not None]
    print(f"arms found on disk: {len(present)}/{len(ARMS)}")
    check_timeline(present)
    check_pairing(present)
    check_budget(present)
    check_integrity(present)
    check_manifests(present)


if __name__ == "__main__":
    main()
