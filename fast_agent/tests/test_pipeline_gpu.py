"""GPU tests for the FlashVID-as-tool pipeline.

These are the invariants that, if broken, corrupt every number in the project
without raising anything. They need a CUDA device and the model weights.

Run:  CUDA_VISIBLE_DEVICES=0 pytest fast_agent/tests -m gpu -q
"""

import subprocess

import numpy as np
import pytest
import torch
from PIL import Image

from fast_agent import config, tools
from fast_agent.agent_loop import assemble, decode
from fast_agent.model import Engine

pytestmark = pytest.mark.gpu

PROMPT = ("Describe what you see in one sentence, then answer <answer>A</answer>.")


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """60s, 10fps test pattern -- long enough that FlashVID sees >= 8 segments."""
    p = tmp_path_factory.mktemp("vid") / "clip.mp4"
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc=size=640x360:rate=10:duration=60",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)],
        capture_output=True)
    if r.returncode != 0 or not p.exists():
        pytest.skip("ffmpeg not available")
    return str(p)


@pytest.fixture(scope="module")
def engine():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return Engine()


@pytest.fixture(scope="module")
def pils(video):
    return tools.initial_frames(video, n_frames=8)


# ---------------------------------------------------------------------------
# 1. Does the FlashVID vision patch change the vision tower's output?
#
# We replace Qwen3VLVisionAttention/Block/Model.forward GLOBALLY so the tower also
# returns CLS attention. If that changed the hidden states, then every arm in this
# project -- including every uncompressed baseline -- has been running a different
# vision encoder than stock, and no number compares to anything published.
# This had never been checked.

def test_vision_patch_does_not_change_vit_output(engine, pils):
    assert not engine._vision_patched, "engine must start unpatched for this test"

    proc = engine.processor.image_processor(images=pils, return_tensors="pt")
    pv = proc["pixel_values"].to(engine.device, torch.bfloat16)
    grid = proc["image_grid_thw"].to(engine.device)

    with torch.inference_mode():
        stock = engine.model.model.visual(pv, grid)
        h_stock, ds_stock = stock[0].clone(), [d.clone() for d in stock[1]]

        engine.apply_vision_patches()
        patched = engine.model.model.visual(pv, grid)

    h_patched, ds_patched = patched[0], patched[1]
    assert len(patched) == 3, "patched tower must also return cls_attention"
    assert h_patched.shape == h_stock.shape
    assert torch.equal(h_stock, h_patched), (
        "FlashVID vision patch CHANGED the ViT hidden states -- every arm is affected")
    for a, b in zip(ds_stock, ds_patched):
        assert torch.equal(a, b), "patch changed a deepstack feature map"


# ---------------------------------------------------------------------------
# 2. Does our hand-built input match what the model would have built itself?

def test_assembly_is_token_identical_to_native(engine, pils):
    msgs = [{"role": "user",
             "content": [*({"type": "image"} for _ in pils), {"type": "text", "text": PROMPT}]}]
    text = engine.processor.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True)
    native = engine.processor(text=[text], images=pils, return_tensors="pt").to(engine.device)

    clip = engine.encode_images(pils)
    a = assemble(engine, [{"role": "user", "parts": [clip, PROMPT]}])

    assert a.ids.shape == native.input_ids.shape
    assert torch.equal(a.ids.cpu(), native.input_ids.cpu())


def test_decode_matches_stock_generate(engine, pils):
    # NOTE: stock `generate` CANNOT run while the vision patch is applied -- the
    # patched tower returns 3 values and stock get_image_features unpacks 2, so it
    # raises ValueError. Any comparison against stock must un-patch first. This is
    # safe precisely because test 1 shows the patch is output-neutral.
    msgs = [{"role": "user",
             "content": [*({"type": "image"} for _ in pils), {"type": "text", "text": PROMPT}]}]
    text = engine.processor.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True)
    native = engine.processor(text=[text], images=pils, return_tensors="pt").to(engine.device)
    engine.restore_vision()
    with torch.inference_mode():
        out = engine.model.generate(**native, max_new_tokens=32, do_sample=False)
    stock = engine.tokenizer.decode(out[0, native.input_ids.shape[1]:],
                                    skip_special_tokens=True)

    clip = engine.encode_images(pils)
    a = assemble(engine, [{"role": "user", "parts": [clip, PROMPT]}])
    ours = decode(engine, a, max_new_tokens=32)

    assert ours.strip() == stock.strip(), f"ours={ours!r} stock={stock!r}"


# ---------------------------------------------------------------------------
# 3. Compressed clips: does what FlashVID kept actually line up with what the LM sees?

@pytest.mark.parametrize("retention", [0.1, 0.5])
def test_compressed_clip_invariants(engine, video, retention):
    vt, times = tools.compress_tensor(video, None, None, max_frames=32)
    clip = engine.encode_video_compressed(vt, target_tokens=1, frame_times=times,
                                          retention=retention)
    m = clip.meta
    t, h, w = m["grid"]
    tpf = (h // 2) * (w // 2)

    assert m["base_tokens"] == t * tpf
    assert m["kept_tokens"] == clip.embeds.shape[0] == clip.keep_indices.shape[0]
    assert not m["bypassed"]
    assert 0 <= int(clip.keep_indices.min()) and int(clip.keep_indices.max()) < m["base_tokens"]
    # FlashVID overshoots its nominal ratio (per-frame ceil + segment floor) but must
    # never undershoot it, or the arm is quietly starved below its stated budget.
    actual = m["kept_tokens"] / m["base_tokens"]
    assert retention <= actual <= retention * 1.6, f"retention {retention} -> {actual:.3f}"
    for d in clip.deepstack:
        assert d.shape[0] == m["kept_tokens"], "deepstack rows != kept tokens"

    # and the whole thing must splice into a sequence without desyncing
    a = assemble(engine, [{"role": "user", "parts": [clip, PROMPT]}])
    assert int(a.visual_mask.sum()) == m["kept_tokens"]
    assert a.deepstack[0].shape[0] == m["kept_tokens"]


def test_retention_one_is_an_exact_identity(engine, video):
    """retention >= 1.0 must bypass the compressor entirely. Calling FlashVID with
    ratio 1.0 preserves the token COUNT but drops ~17% of patches and replaces them
    with duplicates -- which silently degraded every 'uncompressed' control arm."""
    vt, times = tools.compress_tensor(video, None, None, max_frames=32)
    clip = engine.encode_video_compressed(vt, target_tokens=1, frame_times=times,
                                          retention=1.0)
    m = clip.meta
    assert m["bypassed"] is True
    assert m["kept_tokens"] == m["base_tokens"]
    assert torch.equal(
        clip.keep_indices,
        torch.arange(m["base_tokens"], device=clip.keep_indices.device,
                     dtype=clip.keep_indices.dtype))
    assert clip.keep_indices.unique().numel() == m["base_tokens"], "no patch may be dropped"


def test_retention_is_an_argument_not_a_global(engine, video):
    """Regression: retention used to be set by assigning config.FIXED_RETENTION right
    before the call, so a raised exception left the value behind for the next arm."""
    before = (config.FIXED_RETENTION, config.FLOOR_ENGAGE_FRAMES, config.COMPRESS_MAX_FRAMES)
    vt, times = tools.compress_tensor(video, None, None, max_frames=32)
    engine.encode_video_compressed(vt, target_tokens=1, frame_times=times, retention=0.25)
    after = (config.FIXED_RETENTION, config.FLOOR_ENGAGE_FRAMES, config.COMPRESS_MAX_FRAMES)
    assert before == after, "encode_video_compressed mutated global config"
