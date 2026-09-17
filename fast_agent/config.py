"""fast_agent configuration — paths, budgets, prompts.

Design decisions (see ~/.claude/plans/ok-fix-the-prompt-magical-globe.md):
- zero-shot Qwen3-VL-8B-Instruct, local HF, flash-attn-2, batch=1 per worker
- LongVT-faithful loop: native tool schema (# Tools block), terse prompt, single-turn
  baseline, one bounded finalizer, dual (strict/lenient) scoring
- initial view: 64 frames as IMAGES (deliberately sparse; compression gets room to shine)
- crop_video: <=128 frames, fps=1, 224px, images (full detail)
- compress_video: video modality -> faithful FlashVID (DySeg+ADTS+TSTM), with
  fixed retention and a short-span floor so FINAL token count stays near the
  crop budget (matched-budget A/B)
"""

import os

# ---- paths ----
MODEL_SNAPSHOT = (
    "/local1/cfyang/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
    "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
FLASHVID_REPO = "/home/cfyang/hanklin/FlashVID"
VIDEOMME_PARQUET = (
    "/local1/cfyang/.cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/"
    "ead1408f75b618502df9a1d8e0950166bf0a2a0b/videomme/test-00000-of-00001.parquet"
)
VIDEO_DIR = "/local1/cfyang/.cache/huggingface/videomme/videomme/data"
OUTPUT_DIR = "/local1/cfyang/hanklin/outputs/fast_agent"

# LVBench (zai-org snapshot; already complete on disk — see RESEARCH.md 2026-07-23).
# The rich per-question evidence span lives ONLY in the .meta.jsonl (`time_reference`);
# the flattened LVBench.tsv drops it, so the loader reads the jsonl.
_LVBENCH_SNAP = (
    "/local1/cfyang/.cache/huggingface/hub/datasets--zai-org--LVBench/snapshots/"
    "0caedb92002cc268bad486449e551c76f0485670"
)
LVBENCH_META = os.path.join(_LVBENCH_SNAP, "video_info.meta.jsonl")
LVBENCH_VIDEO_DIR = os.path.join(_LVBENCH_SNAP, "videos")

# ---- sampling (paper-faithful base settings: fps=1, 224px) ----
FPS = 1
MAX_PIXELS = 224 * 224          # per-frame pixel budget (both tools + initial view)
MIN_PIXELS = 28 * 28
INITIAL_FRAMES = 64             # deliberately sparse initial view (user decision)
CROP_MAX_FRAMES = 128           # crop tool: full-detail budget ceiling
TEMPORAL_PATCH_SIZE = 2         # Qwen3-VL merges this many raw frames per grid-t step

# ---- FlashVID ----
# retention is FIXED (not auto-derived). To keep compress_video matched-budget
# against crop_video (same final token cost, more temporal coverage) rather
# than just cheaper, COMPRESS_MAX_FRAMES must scale inversely with retention.
# Naive derivation: temporal merge already halves raw-frame cost for free (2 raw
# frames -> 1 grid-t slice of P tokens, same P crop_video pays per single
# frame), so matching crop's P*CROP_MAX_FRAMES budget needs
#   N = CROP_MAX_FRAMES * TEMPORAL_PATCH_SIZE / retention
# and this part IS video-independent (P cancels). BUT FlashVID's actual
# kept/base ratio drifts above the nominal retention_ratio (segment-floor
# effect: min_segment_num=8 segments each keep a fixed ADTS token allocation
# regardless of span, so a fixed per-segment cost eats a bigger share of a
# smaller nominal budget) -- and that drift is NOT a fixed percentage, it gets
# worse as retention drops. Measured on a real 3169s video, sweeping N up to
# 3072 with the resolution-ceiling fix in place:
#   retention=0.2: naive N=1280, real crossover ~1224 frames (~5% over)
#   retention=0.1: naive N=2560, real crossover ~2127 frames (~20% over)
# So the naive formula is only a starting point -- use it to size a
# calibration sweep, then read the real crossover off measured kept_tokens.
# Calibrated values (recalibrate if retention changes):
_CALIBRATED_MAX_FRAMES = {0.2: 1224, 0.1: 2128}
FIXED_RETENTION = float(os.environ.get("FA_FIXED_RETENTION", "0.1"))
COMPRESS_MAX_FRAMES = (
    _CALIBRATED_MAX_FRAMES.get(FIXED_RETENTION)
    or (int(round(CROP_MAX_FRAMES * TEMPORAL_PATCH_SIZE / FIXED_RETENTION))
        if FIXED_RETENTION > 0 else 768)
)
# Below the calibrated input-frame cap, fixed retention alone would undershoot
# the crop budget; use the nominal matched-budget ratio for those short spans.
FLOOR_ENGAGE_FRAMES = COMPRESS_MAX_FRAMES
FLASHVID_KW = dict(
    alpha=0.7,                   # faithful default: selection + TSTM merge
    do_segment=True,
    segment_threshold=0.9,
    min_segment_num=8,
    complementary_segment=True,
    token_selection_method="attn_div_v2",
    temporal_threshold=0.8,
    expansion=1.0,               # keep budget math exact (walkthrough default 1.25)
)

# ---- compressor selection ----
# "flashvid" (default, query-agnostic CLS-saliency merge) | "semvid" (query-aware
# selection — Keeping the Evidence Chain, arXiv:2603.05663; lifted in semvid.py).
# Both consume the same ViT-output tensors and emit keep-indices into the full
# pad run, so the Clip/assemble path is identical across compressors.
COMPRESSOR = os.environ.get("FA_COMPRESSOR", "flashvid")
SEMVID_KW = dict(
    dyseg_c=0,
    dyseg_tau=0.0,               # released presets: threshold cuts effectively off
    stage1_topk_segments=0,      # soft allocation over all segments
    stage1_smooth_win=1,
    frame_weight_alpha=0.7,
    obj_ratio=0.4,               # VideoQA regime (their videoqa preset direction,
    mmr_lambda=0.3,              #  not the Charades grounding preset)
    min_tokens_per_frame=1,
    motion_query_beta=0.5,
)
SEMVID_QUERY_TOKEN_MAX = 64      # per-token query path if <= this many tokens,
                                 # else mean-pooled vector (upstream default 50)

# ---- oracle arm (LVBench only; forced coarse->fine from GT time_reference) ----
# Compress step covers an evidence-WIDENED window (evidence span +/- margin), clamped
# to the video; crop step targets the GT evidence span, widened to a min floor so a
# tight (or zero-width) reference still yields enough full-detail frames.
ORACLE_COMPRESS_MARGIN = 300.0   # +/- seconds around the evidence span for the compress scope
ORACLE_CROP_MIN_WIDTH = 16.0     # minimum crop-window width (seconds) for tight refs

# ---- agent loop ----
MAX_ROUNDS = 5                   # tool rounds before tools are withheld
MAX_NEW_TOKENS = 2048            # per round (1024 truncated frame-by-frame analyses
                                 # mid-thought -> deterministic restart loops)
GEN_TEMPERATURE = 0.0            # greedy

# ---- outputs ----
RUN_ROOT = os.path.join(OUTPUT_DIR, "runs")  # per-run dirs: results.jsonl + traj/*.json + media/

# ---- prompts (LongVT-faithful: terse; tool signatures come from the native # Tools block) ----
ANSWER_INSTR = (
    "Think first inside <think></think> tags. If you need to inspect the video more "
    "closely, call one tool; otherwise give your final answer as <answer>X</answer> "
    "where X is one of the option letters A, B, C, or D."
)

# Tombstone: REASONED mode (FA_REASONED) removed 2026-07. Its per-turn "write 2-5
# sentences" instruction caused frame-by-frame narration -> greedy repetition ->
# multi-round refusal loops. Kept as "" so existing imports don't break.
TOOL_RESULT_INSTR = ""

# OpenAI-format tool schemas fed to the model's native chat template (`tools=`), so a
# zero-shot Qwen3-VL sees the `# Tools` block hermes was trained on. The loop owns the
# video, so tools take only start_time/end_time (no video_path).
TOOL_SCHEMAS = {
    "crop_video": {
        "type": "function",
        "function": {
            "name": "crop_video",
            "description": (
                "Zoom in on a specific time span and view it at FULL detail (1 frame per "
                f"second, up to {CROP_MAX_FRAMES} frames). Use a TIGHT span (<=120 seconds) "
                "to read fine details, on-screen text, or faces."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "number", "description": "Start time in seconds."},
                    "end_time": {"type": "number", "description": "End time in seconds, must be greater than start_time."},
                },
                "required": ["start_time", "end_time"],
            },
        },
    },
}

# NOTE 2026-08-17 -- `compress_video` was REMOVED from this schema set.
# It advertised a compressed wide-span view, but the vLLM agent loop has no way to
# serve one: compression is a SERVER-side setting (--video-pruning-rate), not a
# per-call option, and the OpenAI chat API accepts only images or a whole-video URL.
# The old dispatch in run_agent.py silently answered every accepted tool call with
# tools.crop_frames regardless of the name, so a compress_video call returned an
# ordinary crop and the trajectory recorded it under the wrong tool name. No reported
# run enabled it (LONGVT_TOOL_SCHEMAS and INFORMED_TOOL_SCHEMAS both offer crop_video
# only, and 0 of 28 manifests list it), so no result changed -- but the schema is gone
# and the dispatch now raises rather than mis-serving. Re-adding it requires a real
# server-side compressed view, not a schema entry.

# ---- LongVT-faithful prompt (LongVT/examples/eval/single_inference.py) ----
# Verbatim from the reference eval, including the `video_path` parameter that our
# loop does not need: the schema text is part of the prompt the model conditions on,
# so trimming it would no longer be "LongVT style". The runner ignores whatever path
# the model echoes back and uses the row's own path.
LONGVT_TOOL_SCHEMAS = {
    "crop_video": {
        "type": "function",
        "function": {
            "name": "crop_video",
            "description": ("Crop a video to a specified duration. Use this tool to zoom "
                            "in on specific time segments for detailed analysis."),
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
    },
}

# LongVT's own NO-TOOL mode (single_inference.py, `--no_tool`). A no-tool run must not be
# handed the tool prompt: measured 2026-08-13, run1 was sent "call **crop_video** if needed"
# and a video path with no tool available, which is not a vanilla baseline at all.
LONGVT_SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include "
    "two parts: first, the reasoning process enclosed in <think>...</think> tags, then the "
    "final answer enclosed in <answer>...</answer> tags. Please provide a clear, concise "
    "response within <answer></answer> tags that directly addresses the question."
)

LONGVT_TOOL_PROMPT = (
    "Think first, call **crop_video** if needed, then answer. "
    "Format strictly as: <think>...</think> <tool_call>...</tool_call> (if needed) "
    "<answer>...</answer>."
)


# ---- "informed" prompt: environment FACTS, never a search strategy -------------
# WHY THIS EXISTS (measured 2026-08-13 on run2/run4, n=1,548 each)
#   median crop width          10 s     against a 128 s tool capacity
#   video duration median   3,666 s  ->  one crop sees 0.27% of the timeline
#   ~2.2 calls/question      ->        ~0.6% of the video ever inspected
#   98% of multi-call turns are a monotonic scan PLAN emitted in one turn
#     e.g. (0,10) (10,20) (20,30) ... until the 1024-token cap cuts it mid-word
#   genuine-search hit rate    2.8% (run2) / 2.2% (run4)
# The tool-call SYNTAX is fine (95% / 91% of blocks parse), and the calls are not
# repeats (0 of 350 turns had identical arguments). The model simply does not know
# (a) that the protocol is turn-based, or (b) what the tool can return.
#
# LongVT's prompt can omit all of this because LongVT-RFT was TRAINED on the
# interaction pattern. Zero-shot Qwen3-VL was not, and at a 2% hit rate there is no
# localization signal left for compression to improve or degrade -- which makes the
# whole run2-vs-run4 comparison unreadable.
#
# THE LINE THIS MUST NOT CROSS: state facts about the environment, never a policy.
# "crop_video returns up to 128 frames" is a fact. "Start with a wide crop, then
# narrow down" is the search behaviour we are trying to MEASURE -- writing it into
# the prompt would manufacture the result and void the ablation.
INFORMED_TOOL_SCHEMAS = {
    "crop_video": {
        "type": "function",
        "function": {
            "name": "crop_video",
            # Accurate description of what tools.crop_frames actually does, including
            # the long-span behaviour the LongVT schema never mentions -- the model
            # cannot ask for a coarse wide view if it does not know one is available.
            "description": (
                f"Return frames from a time span of this video. Frames are sampled at 1 "
                f"frame per second between start_time and end_time, up to "
                f"{CROP_MAX_FRAMES} frames. If the span is longer than {CROP_MAX_FRAMES} "
                f"seconds, {CROP_MAX_FRAMES} frames are sampled uniformly across the "
                f"whole span instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "number", "description": "Start time in seconds."},
                    "end_time": {"type": "number", "description": "End time in seconds, must be greater than start_time."},
                },
                # video_path dropped on purpose: the loop owns the video and ignores
                # whatever path the model echoes, so requiring it only invites a
                # hallucinated string in every call.
                "required": ["start_time", "end_time"],
            },
        },
    },
}


def informed_user_text(question: str, duration: float, max_rounds: int) -> str:
    """LongVT's format contract + the environment facts, and nothing else.

    The `<think>/<tool_call>/<answer>` sentence is kept verbatim from LongVT because it
    is the SCORING contract (data.extract_answer reads the tag), not a strategy hint.
    """
    return (
        f"{question}\n\n"
        f"This video is {duration:.0f} seconds long ({hms(duration)}).\n"
        f"crop_video returns frames from a time span you choose, sampled at 1 frame per "
        f"second, up to {CROP_MAX_FRAMES} frames; a span longer than {CROP_MAX_FRAMES} "
        f"seconds is sampled down to {CROP_MAX_FRAMES} frames spread across it.\n"
        f"You may call crop_video at most {max_rounds} times. Make one call per turn — "
        f"the frames are returned to you before you choose what to do next.\n\n"
        f"Think first, then answer. Format strictly as: <think>...</think> "
        f"<tool_call>...</tool_call> (if needed) <answer>...</answer>."
    )


def longvt_user_text(question: str, video_path: str) -> str:
    """`{question} {TOOL_PROMPT} The Video path for this video is: {video_path}`"""
    return f"{question} {LONGVT_TOOL_PROMPT} The Video path for this video is: {video_path}"


def hms(seconds: float) -> str:
    """Seconds -> H:MM:SS / M:SS, for stating both units wherever a time appears."""
    s = int(round(max(seconds, 0)))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


# The initial skim is IMAGE modality, which (unlike video modality) carries NO native
# `<t seconds>` marker per frame -- so without this legend the model must infer time
# from frame INDEX with no anchor, and cannot aim a crop. Measured 2026-07-25: the
# autonomous arm's GT-evidence coverage was 0.00 on 6/6 questions.
SKIM_TIMESTAMPS = os.environ.get("FA_SKIM_TIMESTAMPS", "1") == "1"


def initial_view_text(duration: float, n_frames: int, frame_times: list | None = None) -> str:
    base = (
        f"This video is {duration:.0f} seconds long ({hms(duration)}). The {n_frames} "
        f"frames above are uniformly sampled from 0s to {duration:.0f}s."
    )
    if not SKIM_TIMESTAMPS:
        return base
    step = duration / max(n_frames, 1)
    line = [base,
            f"They are in chronological order; frame i (1-indexed) is at about "
            f"(i-1)x{step:.1f} seconds."]
    if frame_times:
        times = ", ".join(f"{t:.0f}" for t in frame_times)
        line.append(f"Exact frame times in seconds, in order: {times}.")
    line.append("ALL TIMES YOU PASS TO TOOLS MUST BE IN SECONDS (not m:ss).")
    return " ".join(line)

def tool_instructions(duration: float, tools: tuple) -> str:
    """Short strategic hint in the user turn. Tool signatures + call format come from
    the native # Tools block (TOOL_SCHEMAS); this only nudges WHEN to use each."""
    if not tools:
        return ""
    hints = ["\nYou may inspect the video before answering."]
    if "compress_video" in tools and "crop_video" in tools:
        hints.append(
            "Strategy: use compress_video to skim a wide span and locate the relevant "
            "moment, then crop_video to zoom in at full detail. One tool call per turn."
        )
    elif "crop_video" in tools:
        hints.append("Use crop_video to zoom in on the relevant span. One tool call per turn.")
    return " ".join(hints)
