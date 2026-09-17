"""Which per-frame signal actually points at the ground-truth evidence window?

    CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/home/cfyang/hanklin \
      python -m longvt_compression.fast_agent.signal_auc --videos 60

THE INVERSION
-------------
Measuring "does VidCom2 aim at the evidence" answered no (point-biserial r = -0.02, its
peak grid step lands inside the window 7.5% of the time against 9.9% for a random step).
That is a fact about one selector. The useful question is the inverse: given the features
the model already computes, IS there a signal that separates evidence frames from the
rest? If none does, no ViT-side selector can help and the direction is closed. If one
does, it names what a selector should be built on.

Scored per grid step, then ranked by AUC for separating in-window from out-of-window
steps within each question. AUC 0.5 is blind; the metric is per-question so long videos
with tiny windows do not swamp it.

Candidates, cheapest first:
  norm            ||e_t||                          -- is the evidence just "brighter"?
  centroid_dist   ||e_t - mean_t(e)||              -- VidCom2's notion of distinctive
  temporal_delta  ||e_t - e_{t-1}||                -- EVS's notion of change
  vidcom2         VidCom2's own frame_importance   -- the measured baseline
  query_sim       cos(e_t, question embedding)     -- the query-aware signal
  query_sim_max   max over tokens in the step      -- same, but peak rather than pooled
  clip_query      cos(CLIP_img(frame), CLIP_txt(q)) -- Q-Frame / AKS-style relevance

WHY `clip_query` IS NOT THE SAME MEASUREMENT AS `query_sim` (added 2026-08-28)
-----------------------------------------------------------------------------
`query_sim` pools Qwen3-VL's INPUT EMBEDDING LOOKUP for the question tokens
(`model.get_input_embeddings()`) and takes its cosine against pooled visual
embeddings. Those two tensors live in the same nominal space -- visual embeddings
are injected into the LM sequence -- but nothing ever trained a cosine between them
to mean relevance. `query_sim` being blind is therefore NOT evidence that
query-frame relevance is blind; it is evidence about an untrained similarity.

`clip_query` uses CLIP, which IS contrastively trained for exactly this comparison.
It reuses the repo's existing scorer (`videoarm/videoarm/video/aks_select.py`,
`_CLIPScorer`, the AKS CVPR-2025 port) so the image/text encoders, the L2
normalisation and the model choice are the ones already in use here.

Report the two separately. They are different hypotheses.

LABELLING -- two schemes, both reported
---------------------------------------
`point`  (ORIGINAL, unchanged): step t is positive iff `t0 <= (t+.5)*dur/T <= t1`,
         and the question is dropped unless `0 < sum(lab) < T`. Every previously
         banked AUC on this file used this. Kept verbatim so the new signal is
         directly comparable to the banked `query_sim 0.505`.
`ovl`    (DIAGNOSTIC, new): step t owns the half-open interval
         `[t*dur/T, (t+1)*dur/T)` -- a partition of the video -- and is positive iff
         that interval intersects the GT span. See the two problems this exposes in
         the report footer: `point` drops ~4 of every 5 questions and the ones it
         keeps are the wide-evidence ones.

Everything runs offline on the vision tower; no server, no generation.
"""
import argparse, json, math, os, statistics as st
from collections import defaultdict

MODEL = ("/local1/cfyang/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
         "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
PROXY = "/local1/cfyang/hanklin/outputs/lvbench_agent/skim_proxies_hires_128"
CAP1 = 25165824      # the stock video budget: 128 frames -> 64 grid steps x 180 tok
VIDEOARM = "/home/cfyang/hanklin/videoarm"   # holds the existing AKS _CLIPScorer


def auc(scores, labels):
    """Mann-Whitney AUC. labels 1 = inside the evidence window."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank = {}
    i = 0
    while i < len(order):                       # average ranks over ties
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            rank[order[k]] = r
        i = j + 1
    rsum = sum(rank[i] for i, l in enumerate(labels) if l)
    n1, n0 = len(pos), len(neg)
    return (rsum - n1 * (n1 + 1) / 2) / (n1 * n0)


CLIP_CTX = 77        # CLIP's text context length; longer queries are truncated


def clip_frame_feats(scorer, arr, batch=64):
    """L2-normalised CLIP image features for a (N,H,W,3) uint8 array.

    The same three lines as `_CLIPScorer._image_feats` (processor -> get_image_features
    -> L2 normalise), minus its path-keyed disk cache: frames are held in memory here
    and there is no jpeg path to key on. The encoder, preprocessing and normalisation
    are the scorer's own, so scores are identical to what `_CLIPScorer.score` returns.
    """
    import numpy as np
    import torch
    from PIL import Image
    out = []
    for i in range(0, len(arr), batch):
        imgs = [Image.fromarray(a) for a in arr[i:i + batch]]
        inp = scorer.processor(images=imgs, return_tensors="pt",
                               padding=True).to(scorer.device)
        with torch.no_grad():
            f = scorer.model.get_image_features(**inp)
        out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out)


def clip_text_feat(scorer, query):
    """(L2-normalised text feature, n_tokens). n_tokens >= CLIP_CTX means truncated."""
    import torch
    inp = scorer.processor(text=[query], return_tensors="pt", padding=True,
                           truncation=True).to(scorer.device)
    n_tok = int(inp["input_ids"].shape[1])
    with torch.no_grad():
        t = scorer.model.get_text_features(**inp)
    return (t / t.norm(dim=-1, keepdim=True)).cpu().numpy()[0], n_tok


def topk_hit(scores, labels, k):
    """Did any of the k highest-scoring steps land inside the evidence window?"""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return any(labels[i] for i in order)


def topk_random_rate(n, n_pos, k):
    """P(a uniformly random top-k contains >=1 positive) = 1 - C(n-p,k)/C(n,k).

    The honest control for topk_hit: with p positives out of n steps a blind ranker
    still hits sometimes, and how often depends on the question's evidence width.
    """
    if k >= n or n_pos <= 0:
        return 1.0
    if n - n_pos < k:
        return 1.0
    return 1.0 - math.comb(n - n_pos, k) / math.comb(n, k)


def inside_outside(scores, labels):
    """(mean_in, med_in, mean_out, med_out, standardised gap) for one question.

    The standardised gap is (mean_in - mean_out) / std(all scores) -- raw cosines sit
    at different absolute levels per question and per video, so the pooled raw means
    are only interpretable alongside a within-question normalisation.
    """
    ins = [s for s, l in zip(scores, labels) if l]
    out = [s for s, l in zip(scores, labels) if not l]
    if not ins or not out:
        return None
    sd = st.pstdev(scores)
    gap = (st.mean(ins) - st.mean(out)) / sd if sd > 0 else 0.0
    return (st.mean(ins), st.median(ins), st.mean(out), st.median(out), gap)


def boot_ci(vals, n=2000, seed=0):
    """Percentile bootstrap 95% CI on the mean of per-question values."""
    import random
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    m = sorted(st.mean([vals[rng.randrange(len(vals))] for _ in range(len(vals))])
               for _ in range(n))
    return (m[int(.025 * n)], m[int(.975 * n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, default=60)
    ap.add_argument("--frames", type=int, default=128)
    args = ap.parse_args()

    import numpy as np, torch, decord
    import torch.nn.functional as F
    from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration
    from . import data as D
    from ..vidcom2_vllm.retention import vidcom2_scores
    import longvt_compression.vidcom2_vllm.retention as R

    rows = D.load_lvbench(n=1549, seed=0)
    # `duration` is not carried on the dataset row; every completed arm recorded it, so
    # read it from one of them rather than re-probing 103 videos with ffprobe.
    DUR = {}
    with open("/local1/cfyang/hanklin/outputs/lvbench_agent/"
              "lvbench_d5_j_256f_uniform_seed0/results.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            DUR[r["videoID"]] = r["duration"]
    byvid = {}
    for r in rows:
        byvid.setdefault(r["videoID"], []).append(r)
    vids = [v for v in byvid if os.path.exists(f"{PROXY}/{v}_n{args.frames}.mp4")][: args.videos]
    print(f"{len(vids)} videos, {sum(len(byvid[v]) for v in vids)} questions", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    proc = AutoProcessor.from_pretrained(MODEL)
    vproc = proc.video_processor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16).to("cuda:0")
    model.eval()
    visual = model.model.visual
    embed = model.get_input_embeddings()
    merge = vproc.merge_size
    R._MAIN_WIDTH = model.config.text_config.hidden_size
    print(f"main embed width = {R._MAIN_WIDTH}", flush=True)

    # The repo's existing AKS CLIP scorer. Imported rather than reimplemented so the
    # encoders, preprocessing and L2 normalisation are the ones already in use here.
    import sys as _sys
    if VIDEOARM not in _sys.path:
        _sys.path.insert(0, VIDEOARM)
    from videoarm.video.aks_select import _CLIPScorer
    scorer = _CLIPScorer.get()
    clip_name = os.environ.get("VIDEOARM_AKS_CLIP_MODEL", "openai/clip-vit-base-patch32")
    print(f"CLIP scorer = {clip_name} on {scorer.device}", flush=True)

    SIGNALS = ["norm", "centroid_dist", "temporal_delta", "vidcom2",
               "query_sim", "query_sim_max", "clip_query"]
    QWEN_SIGNALS = {"query_sim", "query_sim_max"}
    KS = [1, 4, 8]                 # T = 64 grid steps -> 1/64, 1/16, 1/8 of the video

    # per-question records, one per labelling scheme
    rec = {"point": [], "ovl": []}
    drop = defaultdict(int)
    n_trunc = n_q_text = 0

    for vi, vid in enumerate(vids, 1):
        path = f"{PROXY}/{vid}_n{args.frames}.mp4"
        vr = decord.VideoReader(path, num_threads=2)
        idx = np.linspace(0, len(vr) - 1, num=min(args.frames, len(vr))).round().astype(int)
        arr = vr.get_batch(list(idx)).asnumpy()
        clip = torch.from_numpy(arr).permute(0, 3, 1, 2)
        out = vproc(videos=[clip], size={"longest_edge": CAP1, "shortest_edge": 4096},
                    do_sample_frames=False)
        pv = torch.tensor(np.array(out["pixel_values_videos"])).to("cuda:0", torch.bfloat16)
        thw = torch.tensor(out["video_grid_thw"]).to("cuda:0")
        with torch.no_grad():
            emb = visual(pv, grid_thw=thw)
        # HF returns (main_embeds, deepstack_embeds); vLLM concatenates them instead.
        # We want the main block either way -- that is what the reference VidCom2 scores.
        if isinstance(emb, (tuple, list)):
            emb = emb[0]
        emb = emb[..., : R._MAIN_WIDTH].float()
        T, H, W = (int(v) for v in out["video_grid_thw"][0])
        tpf = (H // merge) * (W // merge)
        e = emb.reshape(T, tpf, -1)
        pooled = e.mean(dim=1)                                # (T, hidden)

        cen = pooled.mean(dim=0, keepdim=True)
        sig = {
            "norm": pooled.norm(dim=-1),
            "centroid_dist": (pooled - cen).norm(dim=-1),
            "temporal_delta": torch.cat([torch.zeros(1, device=pooled.device),
                                         (pooled[1:] - pooled[:-1]).norm(dim=-1)]),
            "vidcom2": vidcom2_scores(emb, T, tpf)[0],
        }
        # CLIP image features, computed ONCE per video and pooled to grid steps the
        # same way Qwen3-VL derives a step's timestamp: mean over the merge_size
        # frames the temporal patch covers (processing_qwen3_vl.py::_calculate_timestamps).
        fpg = len(idx) // T
        cfeat = clip_frame_feats(scorer, arr)                  # (n_frames, 512)
        cstep = cfeat[: T * fpg].reshape(T, fpg, -1).mean(axis=1)
        cstep /= np.linalg.norm(cstep, axis=-1, keepdims=True)

        for q in byvid[vid]:
            ev = q.get("evidence"); dur = DUR.get(q["videoID"], 0.0)
            if not ev:
                drop["evidence_malformed"] += 1
                continue
            if dur <= 0:
                drop["no_duration"] += 1
                continue
            t0, t1 = float(ev[0]), float(ev[1])
            zero_w = (t1 <= t0)

            # ORIGINAL labelling, unchanged -- this is what every banked AUC used.
            lab_pt = [1 if t0 <= (t + .5) * dur / T <= t1 else 0 for t in range(T)]
            # DIAGNOSTIC labelling: step t owns [t*dur/T, (t+1)*dur/T), a partition of
            # the video, and is positive iff that interval meets the GT span.
            lab_ov = [1 if (t * dur / T) <= t1 and t0 < ((t + 1) * dur / T) else 0
                      for t in range(T)]
            ok_pt = 0 < sum(lab_pt) < T
            ok_ov = 0 < sum(lab_ov) < T
            if not ok_pt:
                drop["point_no_positive_step" if sum(lab_pt) == 0 else "point_all_steps"] += 1
                if zero_w:
                    drop["  ^ of which zero-width span"] += 1
            if not (ok_pt or ok_ov):
                continue

            ids = tok(D.format_question(q), return_tensors="pt").input_ids.to("cuda:0")
            with torch.no_grad():
                qv = embed(ids)[0].float().mean(dim=0, keepdim=True)
            sig["query_sim"] = F.cosine_similarity(pooled, qv, dim=-1)
            sig["query_sim_max"] = F.cosine_similarity(
                e.reshape(T * tpf, -1), qv, dim=-1).reshape(T, tpf).max(dim=1).values

            tfeat, n_tok = clip_text_feat(scorer, D.format_question(q))
            n_q_text += 1
            n_trunc += int(n_tok >= CLIP_CTX)
            sig["clip_query"] = torch.from_numpy(cstep @ tfeat)

            for scheme, lab, ok in (("point", lab_pt, ok_pt), ("ovl", lab_ov, ok_ov)):
                if not ok:
                    continue
                r = {"qid": q["question_id"], "vid": vid, "T": T,
                     "n_pos": sum(lab), "width": t1 - t0, "zero_w": zero_w,
                     "trivial": bool(q["localization_trivial"]), "sig": {}}
                for sg in SIGNALS:
                    sc = sig[sg].tolist()
                    a = auc(sc, lab)
                    if a is None:
                        continue
                    r["sig"][sg] = {
                        "auc": a,
                        "hit": {k: topk_hit(sc, lab, k) for k in KS},
                        "io": inside_outside(sc, lab),
                    }
                r["rand"] = {k: topk_random_rate(T, sum(lab), k) for k in KS}
                rec[scheme].append(r)
        del emb, e, pooled, pv, sig
        torch.cuda.empty_cache()
        if vi % 10 == 0:
            print(f"  [{vi}/{len(vids)}] point={len(rec['point'])} "
                  f"ovl={len(rec['ovl'])} questions scored", flush=True)

    def report(scheme, subset=None, title=""):
        R_ = [r for r in rec[scheme] if subset is None or subset(r)]
        if not R_:
            print(f"\n[{scheme}] {title}: no questions"); return
        widths = sorted(r["width"] for r in R_)
        print(f"\n{'='*78}\n[{scheme}] {title}   n={len(R_)} questions, "
              f"{len(set(r['vid'] for r in R_))} videos")
        print(f"  GT span width: median {st.median(widths):.0f}s  "
              f"mean {st.mean(widths):.0f}s   positives/question: median "
              f"{st.median([r['n_pos'] for r in R_]):.0f} of {R_[0]['T']} steps")

        print(f"\n  1) AUC vs GT evidence")
        print(f"     {'signal':15}{'meanAUC':>9}{'95% CI':>18}{'median':>9}{'>0.5':>7}   verdict")
        for sg in SIGNALS:
            v = [r["sig"][sg]["auc"] for r in R_ if sg in r["sig"]]
            if not v:
                continue
            m = st.mean(v); lo, hi = boot_ci(v)
            share = 100 * sum(1 for x in v if x > .5) / len(v)
            verdict = ("blind" if abs(m - .5) < .02 else
                       "weak" if abs(m - .5) < .06 else "USABLE SIGNAL")
            tag = " (Qwen, NOT aligned)" if sg in QWEN_SIGNALS else ""
            print(f"     {sg:15}{m:>9.4f}{f'[{lo:.3f},{hi:.3f}]':>18}"
                  f"{st.median(v):>9.4f}{share:>6.1f}%   {verdict}{tag}")

        print(f"\n  2) top-K evidence hit rate  (K of {R_[0]['T']} grid steps)")
        hdr = "".join(f"{f'K={k}':>16}" for k in KS)
        print(f"     {'signal':15}{hdr}")
        for sg in SIGNALS:
            cells = []
            for k in KS:
                v = [r["sig"][sg]["hit"][k] for r in R_ if sg in r["sig"]]
                rnd = st.mean([r["rand"][k] for r in R_])
                cells.append(f"{100*st.mean(v):>7.1f}% ({100*rnd:.1f})")
            print(f"     {sg:15}" + "".join(f"{c:>16}" for c in cells))
        print("     (value in parentheses = random-ranking baseline for the same "
              "questions)")

        print(f"\n  3) relevance inside vs outside GT evidence")
        print(f"     {'signal':15}{'mean_in':>10}{'med_in':>10}{'mean_out':>10}"
              f"{'med_out':>10}{'std.gap':>10}{'95% CI':>18}")
        for sg in SIGNALS:
            io = [r["sig"][sg]["io"] for r in R_ if sg in r["sig"] and r["sig"][sg]["io"]]
            if not io:
                continue
            gaps = [x[4] for x in io]
            lo, hi = boot_ci(gaps)
            print(f"     {sg:15}{st.mean([x[0] for x in io]):>10.4f}"
                  f"{st.mean([x[1] for x in io]):>10.4f}"
                  f"{st.mean([x[2] for x in io]):>10.4f}"
                  f"{st.mean([x[3] for x in io]):>10.4f}"
                  f"{st.mean(gaps):>10.4f}{f'[{lo:.3f},{hi:.3f}]':>18}")

    print(f"\n\n{'#'*78}")
    print("# CLIP query-frame relevance vs GT evidence -- LVBench, "
          f"{len(vids)} videos, {args.frames} frames")
    print(f"{'#'*78}")
    print(f"\nCLIP text truncation: {n_trunc}/{n_q_text} queries hit the "
          f"{CLIP_CTX}-token context ({100*n_trunc/max(n_q_text,1):.1f}%)")
    print("\nQuestions dropped, by reason:")
    for k, v in sorted(drop.items(), key=lambda x: -x[1]):
        print(f"  {k:34}{v:>6}")

    report("point", None, "ORIGINAL labelling - comparable to the banked query_sim 0.505")
    report("point", lambda r: not r["trivial"], "ORIGINAL, localization_trivial=False")
    report("ovl", None, "DIAGNOSTIC interval-overlap labelling")
    report("ovl", lambda r: not r["trivial"], "DIAGNOSTIC, localization_trivial=False")

    out = os.environ.get("SIGNAL_AUC_OUT")
    if out:
        with open(out, "w") as fh:
            json.dump({"videos": len(vids), "frames": args.frames,
                       "clip_model": clip_name, "drop": dict(drop),
                       "n_trunc": n_trunc, "n_q_text": n_q_text,
                       "records": rec}, fh)
        print(f"\nrecords -> {out}")


if __name__ == "__main__":
    main()
