"""Minimal head-to-head: does our vLLM port select the SAME tokens as the reference?

    PYTHONPATH=/home/cfyang/hanklin python -m longvt_compression.vidcom2_vllm.compare_selectors \
        --videos 6 --frames 128 --q 0.75

WHY THIS IS A SEPARATE CHECK FROM test_retention.py
    test_retention.py compares the SCORING function only -- `vidcom2_scores` against the
    reference `compute_gaussian_scores` -- and finds them identical to 1e-7 with identical
    rankings. That was reported as "matches the reference", which was too strong: scoring is
    only the first half. The two implementations then turn the same scores into a kept set
    by different rules, and this script measures how far apart the kept sets actually are.

THE STRUCTURAL DIFFERENCE
    reference   ks = (scales * tpf).round().clamp(min=1)
                Each frame rounds independently. The total kept is whatever falls out, so
                realised retention floats around the nominal rate.

    ours        ks = _apportion(scales * tpf, target=compute_retained_tokens_count(...))
                Largest-remainder apportionment onto vLLM's exact expected count. vLLM sizes
                the prompt's video placeholders before the vision tower runs, so a mask whose
                True count differs by even one token fails in masked_scatter. The total is not
                negotiable for us; it is for the reference.

    Both consume the identical per-frame `scales` from `compute_scales`, so any disagreement
    comes from rounding and from the total-count constraint, never from the scores.

The embeddings are real: one forward of the actual Qwen3-VL vision tower over LVBench proxies
at the grid's own geometry, so the comparison is on the distribution the experiments ran on.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

MODEL = ("/local1/cfyang/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
         "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
REF = "/local1/cfyang/VidCom2"
PROXY = "/local1/cfyang/hanklin/outputs/lvbench_agent"
NATIVE_CAP = {32: 28835840, 64: 57671680, 128: 115343360}


def vision_embeds(frames, cap, device="cuda:0"):
    """Post-merger main-block video embeddings, exactly what the selector sees."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(device).eval()
    return proc.video_processor, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, default=6)
    ap.add_argument("--frames", type=int, default=128)
    ap.add_argument("--q", type=float, default=0.75, help="pruning rate; retention = 1-q")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    sys.path.insert(0, REF)
    from token_compressor.vidcom2.vidcom2 import (
        select_low_var_channels, compute_gaussian_scores, compute_scales,
        select_outlier_indices)
    from vllm.multimodal.evs import compute_retained_tokens_count
    from longvt_compression.vidcom2_vllm.retention import compute_retention_mask

    import decord
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(MODEL)
    vp = proc.video_processor
    print(f"loading vision tower onto {a.device} ...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(a.device).eval()
    visual = model.model.visual
    merge = visual.spatial_merge_size

    clips = sorted(glob.glob(f"{PROXY}/skim_proxies_hires_{a.frames}/*_n{a.frames}.mp4"))
    clips = clips[:a.videos]
    cap = NATIVE_CAP[a.frames]
    base = 1.0 - a.q

    print(f"\nframes={a.frames}  q={a.q}  nominal retention={base}  cap={cap:,}")
    print(f"{'video':22s} {'grid':>14s} {'tpf':>5s} {'total':>7s} "
          f"{'ref kept':>9s} {'ref R':>7s} {'ours kept':>10s} {'ours R':>7s} "
          f"{'overlap':>8s} {'frames differing':>17s}")

    agg = []
    for path in clips:
        vid = os.path.basename(path).rsplit("_n", 1)[0]
        vr = decord.VideoReader(path, num_threads=2)
        idx = np.linspace(0, len(vr) - 1, a.frames).astype(int)
        arr = vr.get_batch(idx).asnumpy()
        out = vp(videos=[list(arr)], do_sample_frames=False,
                 size={"longest_edge": cap, "shortest_edge": 4096}, return_tensors="pt")
        grid = out["video_grid_thw"]
        pix = out["pixel_values_videos"].to(a.device, dtype=torch.bfloat16)
        with torch.no_grad():
            emb = visual(pix, grid_thw=grid.to(a.device))
        if isinstance(emb, (tuple, list)):        # HF returns (main, deepstack)
            emb = emb[0]
        T, H, W = (int(v) for v in grid[0])
        tpf = (H // merge) * (W // merge)
        n = T * tpf
        emb = emb[:n]

        # ---- reference selection ------------------------------------------------
        x = emb.float()
        sel = select_low_var_channels(x)
        v_s, f_s = compute_gaussian_scores(sel, tpf)
        scales = compute_scales(-v_s.mean(dim=-1), base)
        ref_per_frame = select_outlier_indices(v_s + f_s, scales, tpf)
        ref_ks = [len(i) for i in ref_per_frame]
        ref_flat = set()
        for t, ii in enumerate(ref_per_frame):
            ref_flat.update((t * tpf + int(j)) for j in ii.tolist())

        # ---- our selection ------------------------------------------------------
        mask = compute_retention_mask(emb, (T, H, W), merge, a.q)
        ours_flat = set(torch.nonzero(mask).flatten().tolist())
        m2 = mask.view(T, tpf)
        our_ks = [int(m2[t].sum()) for t in range(T)]

        target = compute_retained_tokens_count(tokens_per_frame=tpf, num_frames=T, q=a.q)
        inter = len(ref_flat & ours_flat)
        union = len(ref_flat | ours_flat)
        diff_frames = sum(1 for t in range(T) if ref_ks[t] != our_ks[t])
        print(f"{vid[:22]:22s} {str([T,H,W]):>14s} {tpf:5d} {n:7,d} "
              f"{len(ref_flat):9,d} {len(ref_flat)/n:7.4f} {len(ours_flat):10,d} "
              f"{len(ours_flat)/n:7.4f} {100*inter/union:7.1f}% {diff_frames:6d}/{T:<10d}")
        agg.append(dict(vid=vid, n=n, T=T, tpf=tpf, target=target,
                        ref=len(ref_flat), ours=len(ours_flat),
                        jac=inter / union, inter=inter,
                        ref_ks=ref_ks, our_ks=our_ks, diff_frames=diff_frames))

    print("\n--- summary -------------------------------------------------------------")
    tot_ref = sum(r["ref"] for r in agg); tot_ours = sum(r["ours"] for r in agg)
    tot_n = sum(r["n"] for r in agg)
    print(f"  realised retention   reference {tot_ref/tot_n:.4f}   ours {tot_ours/tot_n:.4f}"
          f"   nominal {base:.4f}")
    print(f"  token-count gap      reference keeps {tot_ref - tot_ours:+,d} vs ours "
          f"({100*(tot_ref-tot_ours)/tot_ours:+.2f}%)")
    print(f"  ours hits vLLM's required count exactly: "
          f"{all(r['ours'] == r['target'] for r in agg)}")
    print(f"  reference hits it:   {all(r['ref'] == r['target'] for r in agg)}")
    print(f"  mean Jaccard overlap of the kept sets: {sum(r['jac'] for r in agg)/len(agg):.4f}")
    print(f"  frames with a different budget: "
          f"{sum(r['diff_frames'] for r in agg)}/{sum(r['T'] for r in agg)}")
    mx = max(agg, key=lambda r: abs(max(a1 - a2 for a1, a2 in zip(r['ref_ks'], r['our_ks']))))
    d = [a1 - a2 for a1, a2 in zip(mx['ref_ks'], mx['our_ks'])]
    print(f"  largest per-frame budget disagreement: {max(d)} tokens (video {mx['vid'][:18]}, "
          f"tpf={mx['tpf']})")
    print(f"  reference per-frame budgets, first 8:  {mx['ref_ks'][:8]}")
    print(f"  ours,                       first 8:  {mx['our_ks'][:8]}")


if __name__ == "__main__":
    main()
