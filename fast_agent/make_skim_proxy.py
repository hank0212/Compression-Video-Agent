"""Pre-build one small `--frames`-frame proxy mp4 per LVBench video, for arm B.

  /local1/cfyang/miniconda3/envs/vllm/bin/python \
      -m longvt_compression.fast_agent.make_skim_proxy --frames 256 --workers 12

WHY
---
Arm B's skim must be VIDEO modality (vLLM prunes video only, never images), so the
server decodes the file itself. LVBench is 103 videos / 62 GB, median 412 MB and
1 hour long, and the agent re-sends the same video on every tool round -- measured
~39 s per request, i.e. >60 h for one arm. A proxy is decoded once per *video*
instead of once per *request*.

THE TIMELINE MUST BE PRESERVED, NOT JUST THE PIXELS
---------------------------------------------------
Qwen3-VL interleaves a real `<t seconds>` marker before every video frame, and the
model aims its crop from those markers. vLLM derives them as `frame_index / fps`
read from the container. So a naive proxy written at, say, 8 fps would label a
3664-second video as 32 seconds long and every crop the model asks for would be
wrong by two orders of magnitude -- silently, since the crop tool would happily
return frames from 0-32 s.

The proxy is therefore written at `fps = n_frames / duration` (0.07 fps for a
1-hour video) with explicit pts on a 1/90000 time base, so frame i sits at exactly
`i * duration / n_frames` seconds -- the same instant it occupies in the source.
`--verify` re-opens each proxy and checks that round-trip.
"""

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction

TIME_BASE = Fraction(1, 90000)
DEFAULT_DIR = "/local1/cfyang/hanklin/outputs/lvbench_agent/skim_proxies"


def proxy_path(out_dir: str, video_id: str, n: int) -> str:
    return os.path.join(out_dir, f"{video_id}_n{n}.mp4")


def _probe(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames:format=duration", path],
        capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    st = d["stream"][0] if isinstance(d.get("stream"), list) else d["streams"][0]
    return {"w": int(st["width"]), "h": int(st["height"]),
            "duration": float(d["format"]["duration"]),
            "nb_frames": int(st.get("nb_frames") or 0)}


def build_one(video_path: str, out_path: str, n: int, max_side: int = 448) -> dict:
    """Resample to n frames and rewrite them on the SOURCE timeline.

    `fps=n/duration` does both jobs in one sequential pass: it picks n uniformly
    spaced frames AND stamps the output at that rate, so the proxy's own duration
    comes back out as `n / (n/duration) = duration`. Nothing else has to preserve
    the timeline -- which is why this is one filter and not a pts-rewriting loop.
    """
    src = _probe(video_path)
    dur = src["duration"]
    if dur <= 0:
        raise RuntimeError(f"ffprobe reports duration {dur} for {video_path}")
    rate = Fraction(n, max(int(round(dur)), 1))
    scale = min(1.0, max_side / max(src["w"], src["h"]))
    ow, oh = (int(src["w"] * scale) // 2) * 2, (int(src["h"] * scale) // 2) * 2

    tmp = out_path + f".tmp{os.getpid()}"
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", video_path,
           "-vf", f"fps={rate.numerator}/{rate.denominator},scale={ow}:{oh}",
           "-frames:v", str(n), "-an", "-sn",
           "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-video_track_timescale", str(TIME_BASE.denominator),
           "-f", "mp4",          # the temp name ends in .tmpNNN, so state the format
           tmp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(tmp):
        raise RuntimeError(f"ffmpeg rc={r.returncode}: {r.stderr.strip()[:300]}")
    os.replace(tmp, out_path)
    got = _probe(out_path)
    return {"frames": got["nb_frames"], "duration": dur, "proxy_duration": got["duration"],
            "size_mb": os.path.getsize(out_path) / 1e6, "w": ow, "h": oh}


def verify_one(out_path: str, src_duration: float, n: int, tol: float = 2.0) -> str | None:
    """None if the proxy's own timeline reproduces the source's, else the complaint.

    This is the check that matters: vLLM reads `fps` and `duration` off the container
    and hands Qwen3-VL `frame_index / fps` as each frame's `<t seconds>` marker. If the
    proxy's duration drifts from the source's, every timestamp the model reads is wrong
    by that ratio -- and the crop tool, which uses the REAL video, would silently obey.
    """
    p = _probe(out_path)
    if p["nb_frames"] == 0:
        return "0 frames"
    if abs(p["nb_frames"] - n) > 1:
        return f"{p['nb_frames']} frames != {n}"
    slack = max(tol, 0.02 * src_duration)
    if abs(p["duration"] - src_duration) > slack:
        return f"proxy duration {p['duration']:.0f}s != source {src_duration:.0f}s"
    return None


def _job(a):
    vid, path, out_path, n, max_side = a
    t = time.time()
    try:
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            return vid, "cached", None, 0.0
        info = build_one(path, out_path, n, max_side=max_side)
        bad = verify_one(out_path, info["duration"], n)
        return vid, ("ok" if bad is None else "BAD"), (bad or
            f"{info['frames']}f {info['w']}x{info['h']} {info['size_mb']:.1f}MB "
            f"dur {info['duration']:.0f}s"), time.time() - t
    except Exception as e:
        return vid, "ERR", f"{type(e).__name__}: {e}", time.time() - t


def main():
    from . import data
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--out-dir", default=DEFAULT_DIR)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    # 448 is LongVT's regime: it ships up to 768 frames as images, so it needs the
    # per-frame cost low. That constraint is INHERITED, not chosen -- and at 448 a
    # frame is only ~91 tokens, ~8x below the ~720/frame that VidCom2 and Qwen3-VL's
    # own LVBench eval both use. Raising it is how you test a compressor at the
    # operating point it was actually validated at. Source videos are 1280x720, so
    # --max-side 1280 keeps full source resolution and the server's max_pixels does
    # the final clamping. Write high-res proxies to their OWN --out-dir: the filename
    # is `{vid}_n{frames}.mp4` with no resolution in it, so mixing them in one
    # directory would silently serve the wrong pixels.
    ap.add_argument("--max-side", dest="max_side", type=int, default=448)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = data.load_dataset("lvbench", n=1549, seed=0)
    vids = {}
    for r in rows:
        vids.setdefault(r["videoID"], r["video_path"])
    items = [(v, p, proxy_path(args.out_dir, v, args.frames), args.frames, args.max_side)
             for v, p in list(vids.items())[: args.limit]]
    print(f"[proxy] {len(items)} videos -> {args.out_dir} @ {args.frames} frames "
          f"max_side={args.max_side}", flush=True)

    t0, bad = time.time(), []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_job, it) for it in items]
        for k, f in enumerate(as_completed(futs), 1):
            vid, status, note, el = f.result()
            if status in ("BAD", "ERR"):
                bad.append((vid, status, note))
            print(f"[{k}/{len(items)}] {status:6} {vid} {note or ''} ({el:.0f}s)", flush=True)
    print(f"\n[proxy] done in {time.time() - t0:.0f}s | failures: {len(bad)}")
    for v, s, n in bad:
        print(f"  {s} {v}: {n}")


if __name__ == "__main__":
    main()
