"""Measure the REAL visual-token budget of a skim config, before and after compression.

    PYTHONPATH=/home/cfyang/hanklin python -m longvt_compression.fast_agent.probe_budget \
        --frames 128 --vidcap 2 --rate 0.5 --videos 5

WHY THIS EXISTS
---------------
`prompt_tokens` from the OpenAI response is NOT the visual budget. Qwen3-VL's processor
interleaves a real `<t seconds>` marker before every grid step, so the text contribution
GROWS WITH FRAME COUNT: at 256 frames that is 128 markers, at 64 frames only 32. Comparing
two arms on `prompt_tokens` therefore compares "visual + a frame-count-dependent text tax",
and the tax is ~1,000 tokens between a 64f and a 256f arm -- the same order as the effects
being measured.

The quantity that matters is the number of embeddings entering the LLM after the
PatchMerger and after retention:

    visual_pre  = prod(grid_thw) / merge_size**2
    visual_post = compute_retained_tokens_count(tokens_per_frame, num_frames, q)

`compute_retained_tokens_count` is vLLM's own function, and the VidCom2 plugin is written
to reproduce it exactly (retention.py:149), so the same number is correct for both
selectors. Everything here is computed from the SAME processor and the SAME
mm_processor_kwargs the server is launched with, so it is a measurement of the served
config, not an estimate of it.
"""

import argparse
import json
import os

DEFAULT_CAP = 25165824          # video_preprocessor_config.json size.longest_edge
MODEL = ("/local1/cfyang/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
         "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
PROXY_ROOT = "/local1/cfyang/hanklin/outputs/lvbench_agent"


def mm_kwargs(vidcap: int, keep_max_pixels: bool = False) -> dict:
    """What serve_arm.sh passes, minus the parts the VIDEO processor ignores.

    `max_pixels`/`min_pixels` come back as "Unused or unrecognized kwargs" from
    Qwen3VLVideoProcessor -- they are IMAGE-processor knobs. For video the only pixel
    control is `size.longest_edge`, the whole-clip budget. Pass --keep-max-pixels to
    reproduce the warning and confirm it changes nothing.
    """
    kw = {}
    if keep_max_pixels:
        kw.update({"max_pixels": 786432, "min_pixels": 3136})
    if vidcap != 1:
        kw["size"] = {"longest_edge": DEFAULT_CAP * vidcap, "shortest_edge": 4096}
    # vLLM presamples frames itself via --media-io-kwargs num_frames and then calls the
    # processor with do_sample_frames=False (qwen3_vl.py:1021). Omitting this makes the
    # processor resample to its own default and reports a grid_t of ~11 instead of N/2.
    kw["do_sample_frames"] = False
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--vidcap", type=int, default=1, help="multiplier on size.longest_edge")
    ap.add_argument("--rate", type=float, default=0.0, help="--video-pruning-rate (0 = none)")
    ap.add_argument("--proxy-dir", default=None)
    ap.add_argument("--videos", type=int, default=5)
    ap.add_argument("--keep-max-pixels", action="store_true")
    args = ap.parse_args()

    pdir = args.proxy_dir or f"{PROXY_ROOT}/skim_proxies_hires_{args.frames}"
    if args.frames == 32:
        pdir = args.proxy_dir or f"{PROXY_ROOT}/skim_proxies_hires"

    import numpy as np
    import torch
    import decord
    from transformers import AutoProcessor
    from vllm.multimodal.evs import compute_retained_tokens_count

    def read_uniform(path: str, n: int) -> torch.Tensor:
        """n frames evenly spaced, as vLLM's num_frames media-io kwarg does.

        The proxies are built with exactly n frames, so this is normally the identity --
        but it is written as a resample so a mismatched proxy shows up as a decoded-frame
        count that differs from `--frames`, rather than silently changing the budget.
        """
        vr = decord.VideoReader(path, num_threads=2)
        total = len(vr)
        idx = np.linspace(0, total - 1, num=min(n, total)).round().astype(int)
        arr = vr.get_batch(list(idx)).asnumpy()          # (T, H, W, C)
        return torch.from_numpy(arr).permute(0, 3, 1, 2)  # (T, C, H, W)

    kw = mm_kwargs(args.vidcap, args.keep_max_pixels)
    proc = AutoProcessor.from_pretrained(MODEL)
    vproc = proc.video_processor
    merge = vproc.merge_size

    files = sorted(f for f in os.listdir(pdir) if f.endswith(".mp4"))[: args.videos]
    print(f"config: frames={args.frames} vidcap={args.vidcap}x rate={args.rate} "
          f"proxy_dir={os.path.basename(pdir)}")
    print(f"mm_processor_kwargs = {json.dumps(kw)}")
    print(f"video cap = {DEFAULT_CAP * args.vidcap:,} px  ->  ceiling "
          f"{DEFAULT_CAP * args.vidcap // 2 // 1024:,} visual tokens\n")

    hdr = (f"{'video':16}{'proxy wh':>12}{'decoded':>8}{'grid_thw':>16}{'proc wh':>12}"
           f"{'pre':>8}{'post':>8}{'ret':>7}{'tok/step':>9}")
    print(hdr)
    print("-" * len(hdr))

    tots = []
    for f in files:
        path = os.path.join(pdir, f)
        vid = read_uniform(path, args.frames)
        decoded = vid.shape[0]
        out = vproc(videos=[vid], **kw)
        thw = out["video_grid_thw"][0].tolist()
        t, h, w = thw
        pre = (t * h * w) // (merge * merge)
        tpf = (h * w) // (merge * merge)          # tokens per grid step
        if args.rate > 0:
            post = compute_retained_tokens_count(tokens_per_frame=tpf, num_frames=t,
                                                 q=args.rate)
        else:
            post = pre
        ph, pw = vid.shape[-2], vid.shape[-1]
        print(f"{f[:15]:16}{str(ph)+'x'+str(pw):>12}{decoded:>8}{str(thw):>16}"
              f"{str(h*16)+'x'+str(w*16):>12}{pre:>8,}{post:>8,}"
              f"{post/pre:>7.3f}{post/t:>9.1f}")
        tots.append((pre, post, post / t))

    if tots:
        n = len(tots)
        print("-" * len(hdr))
        print(f"{'MEAN':16}{'':12}{'':8}{'':16}{'':12}"
              f"{sum(x[0] for x in tots)/n:>8,.0f}{sum(x[1] for x in tots)/n:>8,.0f}"
              f"{sum(x[1] for x in tots)/sum(x[0] for x in tots):>7.3f}"
              f"{sum(x[2] for x in tots)/n:>9.1f}")
        print("\nPOST is the number to match across arms. Do NOT use prompt_tokens:\n"
              f"  timestamp markers alone add ~{tots[0][0] and (args.frames//2)} of them, "
              "one per grid step, and that count scales with --frames.")


if __name__ == "__main__":
    main()
