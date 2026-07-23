"""Step-3 isolated checks (before agent-loop integration):

3a. VisionService.encode_video_compressed standalone — FlashVID fires on real
    CLS attention, kept token count ~= target (crop budget).
3b. Embedding injection — the LM produces coherent text from compressed tokens
    spliced through the projection, with the ViT called EXACTLY once per clip.
3b-interleave. One context holding full-res images (initial view) + a compressed
    video block — the single-ViT interleaving question, incl. DeepStack routing.

  CUDA_VISIBLE_DEVICES=3 python -m fast_agent.smoke_compress
"""

import time

import torch

from . import config, data, tools
from .agent_loop import assemble, crop_budget_tokens, decode
from .model import Engine


def main():
    rows = data.load_long_split(n=3, seed=0)
    row = rows[0]
    print(f"[3a] video={row['videoID']}")
    engine = Engine()

    # count ViT forwards to prove single-encode
    calls = {"n": 0}
    orig = type(engine.model.model.visual).forward

    def counting_forward(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    # NB: patch AFTER apply_vision_patches so we wrap the flashvid version
    engine.apply_vision_patches()
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    orig = Qwen3VLVisionModel.forward
    Qwen3VLVisionModel.forward = counting_forward

    target = crop_budget_tokens(engine)
    dur = data.video_duration(row["video_path"])
    t0 = time.time()
    vt, times = tools.compress_tensor(row["video_path"], 0, dur)
    t_dec = time.time() - t0
    t0 = time.time()
    clip = engine.encode_video_compressed(vt, target, times)
    t_enc = time.time() - t0
    m = clip.meta
    print(f"[3a] frames={vt.shape[0]} grid={m['grid']} base={m['base_tokens']} "
          f"retention={m['retention']:.3f} kept={m['kept_tokens']} target={target}")
    print(f"[3a] decode {t_dec:.1f}s, encode total {t_enc:.1f}s "
          f"(proc {m['t_proc']}s, ViT {m['t_vit']}s, flashvid {m['t_flashvid']}s), "
          f"ViT calls={calls['n']}")
    ratio = m["kept_tokens"] / max(target, 1)
    ok_budget = 0.5 <= ratio <= 1.5
    print(f"[3a] kept/target={ratio:.2f} -> {'OK' if ok_budget else 'OUT OF RANGE'}")
    assert clip.embeds.shape[0] == clip.keep_indices.shape[0]
    assert all(d.shape[0] == clip.embeds.shape[0] for d in clip.deepstack)

    # --- 3b: compressed-only context -> coherent description ---
    n_before = calls["n"]
    msgs = [{"role": "user", "parts": [
        clip, "\nDescribe briefly what happens in this video."]}]
    a = assemble(engine, msgs)
    out = decode(engine, a, max_new_tokens=128)
    print(f"[3b] ctx={a.embeds.shape[1]} tok (full would be ~{m['base_tokens']}+text)")
    print(f"[3b] output: {out[:400]!r}")
    print(f"[3b] extra ViT calls during assemble/decode: {calls['n'] - n_before} (must be 0)")
    assert calls["n"] == n_before, "double-encode detected!"

    # --- 3b-interleave: images + compressed video in ONE context ---
    pils = tools.initial_frames(row["video_path"])
    clip0 = engine.encode_images(pils)
    n_before = calls["n"]
    msgs = [{"role": "user", "parts": [
        clip0,
        f"\nThe {len(pils)} frames above sample the whole video. Below is a "
        "compressed overview of the same video:\n",
        clip,
        "\nDescribe briefly what happens in this video.",
    ]}]
    a = assemble(engine, msgs)
    out = decode(engine, a, max_new_tokens=128)
    print(f"[3b-i] ctx={a.embeds.shape[1]} tok, visual={int(a.visual_mask.sum())} "
          f"(= {clip0.n_tokens} img + {clip.n_tokens} vid)")
    print(f"[3b-i] output: {out[:400]!r}")
    print(f"[3b-i] extra ViT calls: {calls['n'] - n_before} (1 for initial encode only)")

    Qwen3VLVisionModel.forward = orig
    print("[smoke_compress] DONE")


if __name__ == "__main__":
    main()
