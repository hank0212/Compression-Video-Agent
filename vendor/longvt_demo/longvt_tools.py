"""
Plumbing for the LongVT interleaved-tool-calling demo.

Mirrors examples/eval/single_inference.py exactly (same fps=1, 224px,
crop_video schema) but ALSO returns the decoded PIL frames so the notebook
can render what the model actually saw at each step.
"""
import base64
import json
import os
from io import BytesIO
from typing import List, Tuple

import torch
from qwen_vl_utils import fetch_video
from torchvision.transforms.functional import to_pil_image

# ---- paper-faithful sampling settings (see single_inference.py docstring) ----
FPS = 1
MAX_FRAMES_GLOBAL = 512
MAX_FRAMES_CROP = 128
MAX_PIXELS = 224 * 224  # 50176
MIN_PIXELS = 28 * 28

# The exact tool schema the released checkpoint was trained to call.
CROP_VIDEO_TOOL = {
    "type": "function",
    "function": {
        "name": "crop_video",
        "description": "Crop a video to a specified duration. Use this tool to zoom in on specific time segments for detailed analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "video_path": {"type": "string", "description": "Path to the video file"},
                "start_time": {"type": "number", "description": "Start time in seconds"},
                "end_time": {"type": "number", "description": "End time in seconds"},
            },
            "required": ["video_path", "start_time", "end_time"],
        },
    },
}

# ---- compression / coverage mode (PIXEL-SPACE stand-in; see note in compress_video) ----
MAX_FRAMES_COMPRESS = 256
COMPRESS_MAX_PIXELS = 112 * 112  # 1/4 the tokens-per-frame of the 224px full mode

COMPRESS_VIDEO_TOOL = {
    "type": "function",
    "function": {
        "name": "compress_video",
        "description": (
            "Get a wide, low-detail temporal OVERVIEW of a long span: many frames sampled across "
            "[start_time, end_time] at reduced resolution under a token budget. Use this to LOCATE "
            "where relevant events happen across a broad time range when the video is too sparse to "
            "answer directly; then call crop_video to zoom into the located window for detail."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "video_path": {"type": "string", "description": "Path to the video file"},
                "start_time": {"type": "number", "description": "Start time in seconds"},
                "end_time": {"type": "number", "description": "End time in seconds"},
            },
            "required": ["video_path", "start_time", "end_time"],
        },
    },
}

TOOL_PROMPT = (
    "Think first, call **crop_video** if needed, then answer. "
    "Format strictly as: <think>...</think> <tool_call>...</tool_call> (if needed) <answer>...</answer>."
)

TOOL_PROMPT_BOTH = (
    "Think first. You have two visual tools: **compress_video** gives a wide, low-detail overview to "
    "locate relevant time ranges, and **crop_video** zooms into a range for detail. Call whichever you "
    "need (compress to localize, then crop to inspect), then answer. "
    "Format strictly as: <think>...</think> <tool_call>...</tool_call> (if needed) <answer>...</answer>."
)


def _pils_to_b64(images) -> List[dict]:
    out = []
    for img in images:
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        out.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return out


def _fetch_pils(video_path, start=None, end=None, max_frames=MAX_FRAMES_GLOBAL, max_pixels=MAX_PIXELS) -> List:
    ele = {
        "type": "video",
        "video": f"file://{video_path}",
        "fps": FPS,
        "min_frames": 1,
        "max_frames": max_frames,
        "min_pixels": MIN_PIXELS,
        "max_pixels": max_pixels,
    }
    if start is not None:
        ele["video_start"] = start
        ele["video_end"] = end
    frames = fetch_video(ele).to(torch.uint8)
    return [to_pil_image(f) for f in frames]


def encode_global(video_path) -> Tuple[List[dict], List]:
    """Whole-video skim frames (fps=1, <=512, 224px). Returns (b64_contents, pil_frames)."""
    pil = _fetch_pils(video_path, max_frames=MAX_FRAMES_GLOBAL)
    return _pils_to_b64(pil), pil


def crop_video(video_path, start_time, end_time) -> Tuple[List[dict], List]:
    """Zoom into [start,end] (fps=1, <=128, 224px). Returns (b64_contents, pil_frames)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
    cap.release()
    if end_time > dur:
        end_time = dur
    pil = _fetch_pils(video_path, start=start_time, end=end_time, max_frames=MAX_FRAMES_CROP)
    return _pils_to_b64(pil), pil


def compress_video(video_path, start_time, end_time, max_frames=MAX_FRAMES_COMPRESS) -> Tuple[List[dict], List]:
    """Wide low-res coverage skim of [start,end] (fps=1, <=256 frames, 112px). Returns (b64_contents, pil_frames).

    NOTE: this is a PIXEL-SPACE approximation of compression — it trades spatial resolution for
    temporal coverage (the 'downscale coverage' baseline / arrangement-2). It is NOT FlashVID's
    ViT-internal ADTS token selection, which runs on the model's video_features + cls_attention
    *inside* the forward pass and therefore cannot be produced by a tool that returns pixels over
    the HTTP API. Swapping in real FlashVID is track 2 (in-process HF or a patched vLLM model class).
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
    cap.release()
    if end_time > dur:
        end_time = dur
    pil = _fetch_pils(video_path, start=start_time, end=end_time,
                      max_frames=max_frames, max_pixels=COMPRESS_MAX_PIXELS)
    return _pils_to_b64(pil), pil
