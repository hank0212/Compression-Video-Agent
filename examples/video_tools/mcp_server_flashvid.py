# FlashVID-integration variant of mcp_server.py: crop_video (unchanged, images)
# + compress_video (returns a [VIDEO_CLIP] marker; the harness client attaches
# the clip as video_url so vLLM's video path — where EVS/FlashVID pruning
# fires — carries it; see FLASHVID_TOOL_PLAN.md Option A).
#
# Kept as a SEPARATE file: mcp_server.py is re-spawned per tool call by running
# evals, so editing it would change live runs mid-flight.

import base64
import logging
import os
from io import BytesIO
from typing import Annotated

import cv2
import torch
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent
from pydantic import Field
from qwen_vl_utils import fetch_video
from torchvision.transforms.functional import to_pil_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MCP_SERVER_FV] %(levelname)s: %(message)s",
    handlers=[logging.FileHandler("/tmp/mcp_server_flashvid_debug.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

app = FastMCP("Video Tools MCP Server (FlashVID)", "0.1.0")


def _validate(video_path, start_time, end_time):
    if not video_path or str(video_path).strip() == "":
        raise ValueError("video_path parameter is required")
    if start_time is None or start_time < 0:
        raise ValueError(f"start_time must be non-negative, got {start_time}")
    if end_time is None or end_time <= start_time:
        raise ValueError(f"end_time ({end_time}) must be greater than start_time ({start_time})")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
    cap.release()
    if start_time >= duration:
        raise ValueError(f"start_time ({start_time}s) exceeds video duration ({duration:.2f}s)")
    return min(end_time, duration)


@app.tool(name="crop_video", description="Crop a video to a specified duration. Use this to ZOOM IN on a specific time segment for detailed, full-resolution analysis.")
def crop_video(
    video_path: Annotated[str, Field(description="Path to the video file")] = None,
    start_time: Annotated[float, Field(description="Start time in seconds")] = None,
    end_time: Annotated[float, Field(description="End time in seconds, must be > start_time")] = None,
) -> list[ImageContent]:
    logger.info(f"crop_video: {video_path} [{start_time}, {end_time}]")
    end_time = _validate(video_path, start_time, end_time)
    try:
        video_ele = {
            "type": "video",
            "video": f"file://{video_path}",
            "fps": 1,
            "min_frames": 1,
            "max_frames": 128,
            "min_pixels": 28 * 28,
            "max_pixels": 224 * 224,
            "video_start": start_time,
            "video_end": end_time,
        }
        video_frames = fetch_video(video_ele).to(torch.uint8)
        image_contents = []
        for frame in video_frames:
            buf = BytesIO()
            to_pil_image(frame).save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            image_contents.append(ImageContent(type="image", data=b64, mimeType="image/png"))
        return image_contents
    except Exception as e:
        raise RuntimeError(f"Failed to process video {video_path}: {str(e)}") from e


@app.tool(
    name="compress_video",
    description=(
        "Get a wide, compressed temporal OVERVIEW of a long time span: many frames "
        "across [start_time, end_time] at a reduced token budget. Use this to LOCATE "
        "where relevant events happen over a broad range; then call crop_video to zoom "
        "into the located window for full detail."
    ),
)
def compress_video(
    video_path: Annotated[str, Field(description="Path to the video file")] = None,
    start_time: Annotated[float, Field(description="Start time in seconds")] = None,
    end_time: Annotated[float, Field(description="End time in seconds, must be > start_time")] = None,
) -> list[TextContent]:
    """Validates the span and returns a [VIDEO_CLIP] marker. The eval client
    (async_openai with video_tool_names=compress_video) cuts the clip and
    attaches it as a video_url content part, so the serving engine's
    video-only pruning path (EVS / FlashVID plugin) compresses it."""
    logger.info(f"compress_video: {video_path} [{start_time}, {end_time}]")
    end_time = _validate(video_path, start_time, end_time)
    marker = f"[VIDEO_CLIP] video_path={video_path} start={start_time} end={end_time}"
    return [TextContent(type="text", text=marker)]


if __name__ == "__main__":
    app.run()
