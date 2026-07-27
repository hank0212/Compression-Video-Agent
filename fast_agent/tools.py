"""Video span decoding for the two tools + the initial view.

cv2-based decode (qwen_vl_utils' torchcodec/torchvision backends are broken on
this box — torchcodec can't identify the stream, torchvision/av dies in
swscaler). Sampling matches the project lineage: fps=1 over the span, capped
per-tool, uniform frame centers. Frames are smart-resized (Qwen formula: dims
multiple of 32, area <= MAX_PIXELS) so the processor's own resize is a no-op.
"""

import re
import math
import os
import subprocess
import time

import cv2
import numpy as np
import torch
from PIL import Image

from . import config

PROXY_DIR = os.path.join(config.OUTPUT_DIR, "proxies")


def _smart_size(h: int, w: int, factor: int = 32, max_pixels: int = config.MAX_PIXELS):
    """Qwen smart_resize: round dims to multiples of `factor`, keep area <= max_pixels."""
    if h * w > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        h, w = h / beta, w / beta
    h = max(factor, round(h / factor) * factor)
    w = max(factor, round(w / factor) * factor)
    while h * w > max_pixels:
        h, w = h - factor if h >= w else h, w - factor if w > h else w
    return int(h), int(w)


_BACKEND_CACHE: dict[str, str] = {}   # original path -> decodable path (self or proxy)


def _plan(video_path: str, start, end, max_frames: int, even: bool):
    """Common sampling plan: clamped span, target frame times, target size."""
    cap = cv2.VideoCapture(video_path)
    try:
        fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_v = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    finally:
        cap.release()
    dur = n_v / fps_v if fps_v > 0 else 0.0
    start = 0.0 if start is None else max(0.0, min(start, dur))
    end = dur if end is None else max(start, min(end, dur))
    span = max(end - start, 1e-6)
    n = int(round(span * config.FPS))
    n = max(2, min(n, max_frames))
    if even:
        n = max(2, n - (n % 2))
    centers = start + (np.arange(n) + 0.5) * span / n
    th, tw = _smart_size(h0 or 224, w0 or 224)
    return centers, fps_v, n_v, th, tw


def _decode_cv2(video_path, centers, fps_v, n_v, th, tw):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    try:
        idxs = np.clip((centers * fps_v).astype(int), 0, max(n_v - 1, 0))
        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, fr = cap.read()
            if not ok:
                continue
            fr = cv2.resize(fr, (tw, th), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        return frames
    finally:
        cap.release()


def _decode_pyav(video_path, centers, th, tw):
    """Software decode via PyAV (handles AV1 through libdav1d; cv2's ffmpeg only
    tries hardware AV1 on this box)."""
    import av

    frames = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        tb = stream.time_base
        for t in centers:
            container.seek(int(t / tb), stream=stream)  # keyframe at/before t
            got = None
            for frame in container.decode(stream):
                if frame.time is None or frame.time >= t - 1e-3:
                    got = frame
                    break
            if got is None:
                continue
            arr = got.to_ndarray(format="rgb24")
            frames.append(cv2.resize(arr, (tw, th), interpolation=cv2.INTER_AREA))
    return frames


def make_proxy(video_path: str) -> str:
    """One-time low-res h264 proxy for videos cv2 can't decode (AV1 — this box's
    cv2/ffmpeg lacks software AV1; system ffmpeg has libdav1d). 256px short side
    is lossless for us: every consumer resizes to <=224x224 anyway. Seek-per-frame
    pyav on hour-long sparse-keyframe AV1 measured ~10 min per 48 frames — the
    proxy makes all later decodes cv2-fast."""
    os.makedirs(PROXY_DIR, exist_ok=True)
    proxy = os.path.join(PROXY_DIR, os.path.basename(video_path))
    if os.path.exists(proxy) and os.path.getsize(proxy) > 0:
        return proxy

    # Race-safe: video-sharding already keeps one video on one worker, but guard
    # anyway. O_EXCL lock claims the transcode; other workers wait for the result.
    lock = proxy + ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        for _ in range(900):                       # someone else is transcoding
            if os.path.exists(proxy) and os.path.getsize(proxy) > 0:
                return proxy
            time.sleep(1)
        # stale lock (dead worker) -> fall through and transcode ourselves

    tmp = proxy + f".{os.getpid()}.tmp.mp4"         # unique per process
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", video_path,
             "-vf", "scale=-2:256", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "28", "-an", tmp],
            check=True, timeout=1800,
        )
        os.replace(tmp, proxy)                       # atomic
    finally:
        for f in (lock, tmp):
            try:
                os.remove(f)
            except OSError:
                pass
    return proxy


def resolve_decodable(video_path: str) -> str:
    """Return a cv2-decodable path for this video (original, or cached proxy —
    transcoding once if needed)."""
    cached = _BACKEND_CACHE.get(video_path)
    if cached is not None:
        return cached
    cap = cv2.VideoCapture(video_path)
    try:
        n_v = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(n_v // 2 - 1, 0))
        ok, _ = cap.read()
    finally:
        cap.release()
    path = video_path if ok else make_proxy(video_path)
    _BACKEND_CACHE[video_path] = path
    return path


def _decode_span_with_timestamps(
    video_path: str,
    start: float | None,
    end: float | None,
    max_frames: int,
    even: bool = False,
) -> tuple[list[np.ndarray], list[float]]:
    """Decode a span and retain the sampling-plan timestamps for diagnostics."""
    src = resolve_decodable(video_path)
    centers, fps_v, n_v, th, tw = _plan(src, start, end, max_frames, even)
    frames = _decode_cv2(src, centers, fps_v, n_v, th, tw)
    if len(frames) < 2:  # last resort: software decode of the original
        frames = _decode_pyav(video_path, centers, th, tw)
    if len(frames) < 2:
        raise RuntimeError(f"decoded {len(frames)} frames from {video_path}")
    if even and len(frames) % 2:
        frames = frames[:-1]
    return frames, centers[: len(frames)].astype(float).tolist()


def _decode_span(video_path: str, start: float | None, end: float | None,
                 max_frames: int, even: bool = False) -> list[np.ndarray]:
    frames, _timestamps = _decode_span_with_timestamps(
        video_path, start, end, max_frames, even
    )
    return frames


def initial_frames(video_path: str) -> list:
    """48 uniformly-sampled PIL frames of the whole video (image modality)."""
    return [Image.fromarray(f) for f in
            _decode_span(video_path, None, None, config.INITIAL_FRAMES)]


def initial_frames_with_timestamps(video_path: str):
    """Diagnostic initial-view API with the exact planned sample timestamps."""
    frames, timestamps = _decode_span_with_timestamps(
        video_path, None, None, config.INITIAL_FRAMES
    )
    return [Image.fromarray(f) for f in frames], timestamps


def crop_frames(video_path: str, start: float, end: float) -> list:
    """<=128 PIL frames at fps=1 over [start, end] (full-detail image modality)."""
    return [Image.fromarray(f) for f in
            _decode_span(video_path, start, end, config.CROP_MAX_FRAMES)]


def crop_frames_with_timestamps(video_path: str, start: float, end: float):
    """Diagnostic crop API; production callers keep using ``crop_frames``."""
    frames, timestamps = _decode_span_with_timestamps(
        video_path, start, end, config.CROP_MAX_FRAMES
    )
    return [Image.fromarray(f) for f in frames], timestamps


def compress_tensor(video_path: str, start: float, end: float):
    """((T,C,H,W) uint8 tensor, frame_times) over [start, end], up to 768 frames,
    T even (video modality for the FlashVID path). frame_times are the sampled
    frames' timestamps in seconds — needed for Qwen3-VL's per-frame timestamp
    text (`<{t:.1f} seconds>` before each temporal group)."""
    frames, times = _decode_span_with_timestamps(
        video_path, start, end, config.COMPRESS_MAX_FRAMES, even=True
    )
    return (
        torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous(),
        times,
    )


_TIME_STR_RE = re.compile(r"^\s*(\d+):([0-5]?\d)(?::([0-5]?\d))?\s*$")


def parse_time_arg(v) -> float | None:
    """Accept whatever format the model emits for a time argument and return seconds.
    Handles: 1036, 1036.5, "1036", "1036s", "17:16" (m:ss), "1:06:37" (h:mm:ss).
    Zero-shot models mix clock and second notation (a model read "17:16" as 17s and
    called crop_video(17,17); 2026-07-25) -- parsing beats erroring."""
    if isinstance(v, (int, float)):
        return float(v)
    if not isinstance(v, str):
        return None
    s = v.strip().lower().removesuffix("seconds").removesuffix("sec").removesuffix("s").strip()
    m = _TIME_STR_RE.match(s)          # match the SUFFIX-STRIPPED form, so "17:16s" parses
    if m:
        a, b, c = m.group(1), m.group(2), m.group(3)
        return float(int(a) * 3600 + int(b) * 60 + int(c)) if c else float(int(a) * 60 + int(b))
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def clamp_span(start, end, duration: float) -> tuple[float, float, str | None]:
    """Sanitize a model-emitted span. Returns (start, end, error_or_None)."""
    start, end = parse_time_arg(start), parse_time_arg(end)
    if start is None or end is None:
        return 0.0, 0.0, "start_time/end_time must be numbers of SECONDS (e.g. 1036)."
    start = max(0.0, min(start, duration))
    end = max(0.0, min(end, duration))
    if end - start < 1.0:
        return start, end, (
            f"Invalid span [{start:.0f}, {end:.0f}]s: end must exceed start by >=1s "
            f"within [0, {duration:.0f}]s."
        )
    return start, end, None
