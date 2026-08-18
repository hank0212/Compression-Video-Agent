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

Everything runs offline on the vision tower; no server, no generation.
"""
import argparse, json, math, os, statistics as st

MODEL = ("/local1/cfyang/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
         "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
PROXY = "/local1/cfyang/hanklin/outputs/lvbench_agent/skim_proxies_hires_128"
CAP1 = 25165824      # the stock video budget: 128 frames -> 64 grid steps x 180 tok


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

    SIGNALS = ["norm", "centroid_dist", "temporal_delta", "vidcom2",
               "query_sim", "query_sim_max"]
    got = {s: [] for s in SIGNALS}
    nq = 0

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
        for q in byvid[vid]:
            ev = q.get("evidence"); dur = DUR.get(q["videoID"], 0.0)
            if not ev or dur <= 0:
                continue
            t0, t1 = float(ev[0]), float(ev[1])
            lab = [1 if t0 <= (t + .5) * dur / T <= t1 else 0 for t in range(T)]
            if not (0 < sum(lab) < T):
                continue
            ids = tok(D.format_question(q), return_tensors="pt").input_ids.to("cuda:0")
            with torch.no_grad():
                qv = embed(ids)[0].float().mean(dim=0, keepdim=True)
            sig["query_sim"] = F.cosine_similarity(pooled, qv, dim=-1)
            sig["query_sim_max"] = F.cosine_similarity(
                e.reshape(T * tpf, -1), qv, dim=-1).reshape(T, tpf).max(dim=1).values
            for s in SIGNALS:
                a = auc(sig[s].tolist(), lab)
                if a is not None:
                    got[s].append(a)
            nq += 1
        del emb, e, pooled, pv, sig
        torch.cuda.empty_cache()
        if vi % 10 == 0:
            print(f"  [{vi}/{len(vids)}] {nq} questions scored", flush=True)

    print(f"\nAUC for separating in-evidence grid steps from the rest  (n={nq} questions)")
    print(f"  {'signal':16}{'mean AUC':>10}{'median':>9}{'>0.5':>8}{'   verdict'}")
    for s in SIGNALS:
        v = got[s]
        if not v:
            continue
        m = st.mean(v)
        share = 100 * sum(1 for x in v if x > 0.5) / len(v)
        verdict = ("blind" if abs(m - .5) < .02 else
                   "weak" if abs(m - .5) < .06 else "USABLE SIGNAL")
        print(f"  {s:16}{m:>10.4f}{st.median(v):>9.4f}{share:>7.1f}%   {verdict}")


if __name__ == "__main__":
    main()
