"""CPU unit tests for the pieces every result passes through.

Run:  pytest fast_agent/tests -m "not gpu" -q
"""

import subprocess

import pytest

from fast_agent import config, data, tools
from fast_agent.model import clip_tokens, retention_for_budget


# --------------------------------------------------------------------------
# extract_answer -- the scoring function. Every accuracy number in the project
# is whatever this returns, so its edge cases are the project's edge cases.

@pytest.mark.parametrize("text,expected", [
    ("<answer>B</answer>", "B"),
    ("<answer> (C) </answer>", "C"),
    ("<answer>d</answer>", "D"),
    ("blah <think>x</think> <answer>A</answer>", "A"),

    # Refusal: an explicit tag that is not an option letter. Must NOT be guessed past.
    ("<answer>Unknown</answer>", None),
    ("<answer>None of the above</answer>", None),

    # Multi-round text: callers join every round, and the LAST answer is the model's
    # final one (the finalizer turn appends after the earlier rounds).
    ("<answer>B</answer>\nmore\n<answer>C</answer>", "C"),

    # Repetition-degenerate refusal. The enumeration must be stripped before the
    # 200-char tail window is taken, or the cut lands mid-list and the trailing "D"
    # is scored as a confident answer.
    ("I have reviewed the video. " + "x" * 400 +
     " The correct answer is not A, B, C, or D.", None),
    ("It is not A, B, C, or D.", None),

    # Bare-letter fallback when no tag was emitted at all.
    ("After reviewing the frames, the answer is C", "C"),
    ("", None),
    ("no letters here at all", None),
])
def test_extract_answer(text, expected):
    assert data.extract_answer(text) == expected


def test_extract_answer_prefers_tag_over_trailing_letter():
    # A tag anywhere beats a stray letter later in the prose.
    assert data.extract_answer("<answer>A</answer> ... option B is also plausible") == "A"


# --------------------------------------------------------------------------
# Time parsing. Zero-shot models emit clock notation while the tools take seconds;
# a silent misparse sends the crop to the wrong minute (observed: "17:16" -> 17s).

@pytest.mark.parametrize("value,expected", [
    (1036, 1036.0), (1036.5, 1036.5), ("1036", 1036.0), ("1036s", 1036.0),
    ("17:16", 17 * 60 + 16), ("1:06:37", 3600 + 6 * 60 + 37),
    ("  90 seconds ", 90.0),
    (None, None), ("abc", None), ([], None),
])
def test_parse_time_arg(value, expected):
    assert tools.parse_time_arg(value) == expected


def test_clamp_span():
    s, e, err = tools.clamp_span(10, 50, 100)
    assert (s, e, err) == (10.0, 50.0, None)

    _, _, err = tools.clamp_span(10, 10, 100)          # zero width
    assert err is not None
    _, _, err = tools.clamp_span("x", 50, 100)         # unparseable
    assert err is not None

    s, e, err = tools.clamp_span(-5, 500, 100)         # clamped into the video
    assert (s, e, err) == (0.0, 100.0, None)


def test_parse_time_reference():
    assert data.parse_time_reference("01:00-02:30") == (60.0, 150.0)
    assert data.parse_time_reference("08:17-None") is None
    assert data.parse_time_reference("garbage") is None


def test_split_lvbench_question():
    stem, opts = data._split_lvbench_question(
        "What happens?\n(A) one\n(B) two\n(C) three\n(D) four")
    assert stem == "What happens?"
    assert opts == ["A. one", "B. two", "C. three", "D. four"]


# --------------------------------------------------------------------------
# Frame sizing and the token arithmetic the retention ladder is built on.

@pytest.mark.parametrize("h,w", [(1080, 1920), (720, 1280), (480, 640), (100, 100),
                                 (2160, 3840), (50, 900)])
def test_smart_size(h, w):
    th, tw = tools._smart_size(h, w)
    assert th % 32 == 0 and tw % 32 == 0, "grid must divide into 16px patches x 2 merge"
    assert th * tw <= config.MAX_PIXELS
    assert th > 0 and tw > 0


def test_clip_tokens_and_retention():
    import torch
    # one 288x160 frame -> (18 x 10) patches -> (9 x 5) merged = 45 tokens
    grid = torch.tensor([[1, 10, 18]])
    assert clip_tokens(grid) == 45

    assert retention_for_budget(1000, 100) == pytest.approx(0.1)
    assert retention_for_budget(100, 1000) == 1.0        # never above 1
    assert retention_for_budget(10**9, 1) >= 0.02        # never below the floor


def test_video_token_budget_arithmetic():
    """N video frames at retention r cost about (45 * N/2) * r tokens, because Qwen
    merges TEMPORAL_PATCH_SIZE raw frames into one grid-t group. The matched-budget
    ladder is derived from this, so pin it."""
    per_group = 45
    base = per_group * 640 // config.TEMPORAL_PATCH_SIZE      # 640 frames as video
    assert base == 14400
    assert round(base * 0.1) == 1440

    # The modality trap: 64 frames as IMAGES cost twice what 64 frames as VIDEO cost,
    # because only the video path merges frame pairs. Comparing across modalities at
    # "the same frame count" is therefore not a matched-budget comparison.
    as_video = per_group * 64 // config.TEMPORAL_PATCH_SIZE
    as_images = per_group * 64
    assert (as_video, as_images) == (1440, 2880)


# --------------------------------------------------------------------------
# Frame sampling on a real (tiny, generated) video.

@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    """20s, 10fps, 320x240 test pattern. Skips the tests if ffmpeg is unavailable."""
    p = tmp_path_factory.mktemp("vid") / "tiny.mp4"
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc=size=320x240:rate=10:duration=20",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)],
        capture_output=True,
    )
    if r.returncode != 0 or not p.exists():
        pytest.skip("ffmpeg not available")
    return str(p)


def test_sampling_counts_and_timestamps(tiny_video):
    frames, times = tools._decode_span_with_timestamps(tiny_video, None, None, 8)
    assert len(frames) == len(times) == 8
    assert times == sorted(times), "timestamps must be increasing"
    assert 0 <= times[0] and times[-1] <= 20.0
    assert frames[0].shape[2] == 3


def test_sampling_respects_span(tiny_video):
    frames, times = tools._decode_span_with_timestamps(tiny_video, 5.0, 10.0, 16)
    assert all(5.0 <= t <= 10.0 for t in times)
    assert len(frames) == len(times)


def test_sampling_even_flag(tiny_video):
    # video modality needs an even frame count (2 raw frames -> 1 grid-t group)
    _, times = tools._decode_span_with_timestamps(tiny_video, None, None, 7, even=True)
    assert len(times) % 2 == 0


def test_compress_tensor_shape_and_times(tiny_video):
    vt, times = tools.compress_tensor(tiny_video, None, None, max_frames=8)
    assert vt.ndim == 4 and vt.shape[1] == 3          # (T, C, H, W)
    assert vt.shape[0] == len(times) == 8
    assert vt.shape[0] % 2 == 0
    assert vt.dtype.is_floating_point is False        # uint8


def test_max_frames_is_an_argument_not_a_global(tiny_video):
    """Regression: the experiment scripts used to set config.COMPRESS_MAX_FRAMES
    before each call, which leaks into whatever runs next if anything raises."""
    before = config.COMPRESS_MAX_FRAMES
    vt, _ = tools.compress_tensor(tiny_video, None, None, max_frames=4)
    assert vt.shape[0] == 4
    assert config.COMPRESS_MAX_FRAMES == before, "compress_tensor mutated global config"


def test_decoders_return_their_own_timestamps(tiny_video):
    """Regression: a frame that fails to decode used to be dropped while the caller
    returned the first N planned times, relabelling every later frame."""
    src = tools.resolve_decodable(tiny_video)
    centers, fps_v, n_v, th, tw = tools._plan(src, None, None, 6, False)
    frames, times = tools._decode_cv2(src, centers, fps_v, n_v, th, tw)
    assert len(frames) == len(times)
    for t in times:                                   # every time is a planned one
        assert min(abs(t - c) for c in centers) < 1e-6
