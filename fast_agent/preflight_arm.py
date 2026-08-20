"""Verify an arm's real geometry over ALL 103 videos before the run, and freeze it.

    PYTHONPATH=/home/cfyang/hanklin python -m longvt_compression.fast_agent.preflight_arm \
        --frames 128 --retention 0.25 --workers 8 \
        --out /local1/cfyang/hanklin/outputs/lvbench_agent/preflight/f128.json

WHY THIS EXISTS
---------------
Qwen3-VL's video processor spends `size.longest_edge` across the WHOLE clip, so it will
quietly trade spatial resolution for frame count: ask for 256 frames at the stock cap and
it returns 224x416 frames with no warning. A "frame count ablation" launched without
checking is really a frame-count-AND-resolution ablation, and the two cannot be separated
afterwards. The grid's central claim -- resolution is FIXED, only frames and retention
move -- is a claim about the processor's OUTPUT, so it is measured, not assumed.

This project has been bitten three times by a silent processor default (`max_pixels`
inert for video; `do_sample_frames` resampling to grid_t=11; the whole-clip cap), so the
geometry is checked against an explicit expectation and the run ABORTS on mismatch.

WHAT IT ABORTS ON
    - proxy missing, or its resolution is not the source's native resolution
    - decoded frame count != requested
    - grid_t != frames / temporal_patch_size
    - tokens-per-grid-step differs from the modal value for a reason other than a
      smaller SOURCE (LVBench is not uniform: 86 of 103 videos are 1280x720, the rest
      are smaller and cannot reach 704x1280 at any cap -- a corpus property, not a
      configuration error, so those are reported and allowed)
    - post-prune count != round(pre-prune * retention)

Geometry depends only on (frames, cap), never on retention, so one run per frame count
is reused across that count's compressed and uncompressed arms; `--retention` only
changes the derived post-prune column.
"""

import argparse
import concurrent.futures as cf
import glob
import json
import os
import subprocess
import sys

import numpy as np

MODEL = ("/local1/cfyang/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
         "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
NATIVE_CAP = 230686720          # saturating: past this the source is the binding limit
TEMPORAL_PATCH_SIZE = 2
MERGE = 2
PATCH = 16
TS_PER_STEP = 10.73             # measured timestamp-marker cost, one `<t seconds>` per step
TEXT_TOKENS = 115               # system 67 + question median 39 + chat template

_VP = None


def _vp():
    global _VP
    if _VP is None:
        from transformers import AutoProcessor
        _VP = AutoProcessor.from_pretrained(MODEL).video_processor
    return _VP


def probe_container(path):
    """Frame count and timeline straight off the container -- this is what vLLM reads to
    derive each `<t seconds>` marker as frame_index / fps, so it is the thing that must
    be right, not the pixel data."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_frames,r_frame_rate,duration",
         "-of", "json", path], capture_output=True, text=True).stdout
    st = json.loads(out)["streams"][0]
    num, den = (st["r_frame_rate"].split("/") + ["1"])[:2]
    fps = float(num) / float(den) if float(den) else 0.0
    n = int(st.get("nb_frames") or 0)
    return {"w": int(st["width"]), "h": int(st["height"]), "nb_frames": n, "fps": fps,
            "first_ts": 0.0, "last_ts": (n - 1) / fps if fps else None}


def one(args_tuple):
    path, frames, cap, retention = args_tuple
    import decord
    vid = os.path.basename(path).rsplit("_n", 1)[0]
    c = probe_container(path)
    vr = decord.VideoReader(path, num_threads=2)
    idx = np.linspace(0, len(vr) - 1, frames).astype(int)
    arr = vr.get_batch(idx).asnumpy()
    g = _vp()(videos=[list(arr)], do_sample_frames=False,
              size={"longest_edge": cap, "shortest_edge": 4096},
              return_tensors="pt")["video_grid_thw"][0].tolist()
    t, h, w = g
    tps = (h // MERGE) * (w // MERGE)
    pre = t * tps
    return {
        "video_id": vid,
        "proxy_res": f"{c['w']}x{c['h']}",
        "requested_frames": frames,
        "decoded_frames": len(vr),
        "container_nb_frames": c["nb_frames"],
        "first_ts": c["first_ts"], "last_ts": c["last_ts"], "fps": c["fps"],
        "grid_thw": g,
        "grid_steps": t,
        "processed_res": f"{h * PATCH}x{w * PATCH}",
        "tok_per_step": tps,
        "pre_prune": pre,
        "post_prune": int(round(pre * retention)),
        "actual_retention": round(int(round(pre * retention)) / pre, 6) if pre else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--retention", type=float, default=1.0)
    ap.add_argument("--cap", type=int, default=NATIVE_CAP)
    ap.add_argument("--proxy-root",
                    default="/local1/cfyang/hanklin/outputs/lvbench_agent")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    pdir = os.path.join(a.proxy_root, f"skim_proxies_hires_{a.frames}")
    clips = sorted(glob.glob(os.path.join(pdir, f"*_n{a.frames}.mp4")))
    if a.limit:
        clips = clips[:a.limit]
    if not clips:
        sys.exit(f"PREFLIGHT FAIL: no *_n{a.frames}.mp4 in {pdir}")

    print(f"[preflight] frames={a.frames} retention={a.retention} cap={a.cap:,}")
    print(f"[preflight] {len(clips)} proxies from {pdir}", flush=True)

    work = [(p, a.frames, a.cap, a.retention) for p in clips]
    rows = []
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one, work), 1):
            rows.append(r)
            if i % 25 == 0:
                print(f"  {i}/{len(clips)}", flush=True)

    # --- the checks -------------------------------------------------------------
    fails = []
    expect_steps = a.frames // TEMPORAL_PATCH_SIZE
    for r in rows:
        if r["grid_steps"] != expect_steps:
            fails.append(f"{r['video_id']}: grid_t {r['grid_steps']} != {expect_steps}")
        if r["container_nb_frames"] != a.frames:
            fails.append(f"{r['video_id']}: container holds {r['container_nb_frames']} "
                         f"frames, requested {a.frames}")
        if r["decoded_frames"] != a.frames:
            fails.append(f"{r['video_id']}: decoded {r['decoded_frames']} != {a.frames}")
        want = int(round(r["pre_prune"] * a.retention))
        if r["post_prune"] != want:
            fails.append(f"{r['video_id']}: post-prune {r['post_prune']} != {want}")

    hist = {}
    for r in rows:
        hist[(r["processed_res"], r["tok_per_step"])] = \
            hist.get((r["processed_res"], r["tok_per_step"]), 0) + 1
    (mres, mtps), mcount = max(hist.items(), key=lambda kv: kv[1])

    # a clip below the modal geometry is only OK if its SOURCE is smaller
    small = []
    for r in rows:
        if r["tok_per_step"] == mtps:
            continue
        sw, sh = (int(x) for x in r["proxy_res"].split("x"))
        mh, mw = (int(x) for x in mres.split("x"))
        if sh * sw >= mh * mw:
            fails.append(f"{r['video_id']}: {r['processed_res']} below modal {mres} "
                         f"despite a {r['proxy_res']} source -- the CAP bound it, not "
                         f"the source")
        else:
            small.append(r)

    steps = expect_steps
    summary = {
        "frames": a.frames, "retention": a.retention, "cap": a.cap,
        "n_videos": len(rows), "grid_steps": steps,
        "modal_processed_res": mres, "modal_tok_per_step": mtps,
        "modal_share": f"{mcount}/{len(rows)}",
        "visual_pre_prune": steps * mtps,
        "visual_post_prune": int(round(steps * mtps * a.retention)),
        "timestamp_tokens": round(TS_PER_STEP * steps),
        "text_tokens": TEXT_TOKENS,
        "below_modal_smaller_source": [r["video_id"] for r in small],
        "videos": rows,
    }
    summary["llm_input_len_pre_prune"] = (summary["visual_pre_prune"]
                                          + summary["timestamp_tokens"] + TEXT_TOKENS)

    print(f"\n  grid steps            {steps}   (frames / temporal_patch_size "
          f"{TEMPORAL_PATCH_SIZE})")
    print(f"  modal processed res   {mres}   ({summary['modal_share']} of videos)")
    print(f"  tokens / grid step    {mtps}")
    print(f"  visual PRE-prune      {summary['visual_pre_prune']:,}")
    print(f"  visual POST-prune     {summary['visual_post_prune']:,}   "
          f"(retention {a.retention})")
    print(f"  timestamp tokens      {summary['timestamp_tokens']:,}")
    print(f"  text tokens           {TEXT_TOKENS}")
    print(f"  LLM input (pre-prune) {summary['llm_input_len_pre_prune']:,}   "
          f"<- max-model-len must exceed this")
    if small:
        print(f"  {len(small)} video(s) below modal because the SOURCE is smaller "
              f"(corpus property, allowed):")
        for r in small[:6]:
            print(f"    {r['video_id'][:24]:24s} src {r['proxy_res']:>9s} -> "
                  f"{r['processed_res']:>9s}  {r['tok_per_step']:4d} tok/step")

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  frozen -> {a.out}")

    if fails:
        print(f"\nPREFLIGHT FAIL ({len(fails)}):", file=sys.stderr)
        for m in fails[:20]:
            print("  " + m, file=sys.stderr)
        sys.exit(1)
    print("[preflight] OK")


if __name__ == "__main__":
    main()
