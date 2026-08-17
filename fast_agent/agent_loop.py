"""Our own decoding scheme: embedding-space assembly + manual greedy decode.

Every round re-prefills the full sequence from CACHED clip embeddings (each clip's
ViT encode happens exactly once, at tool-execution time). Compressed video clips
contribute only their kept-anchor positions: we lay out the FULL placeholder run,
compute M-RoPE over the full grid via get_rope_index (native), then gather the kept
positions — exactly FlashVID's own M-RoPE treatment (modeling_qwen3_vl.py:340-344),
without touching the text model.
"""

import json
import os
import re
import time
from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache

from . import config, data, tools
from . import trajectory as tj
from .model import Clip, Engine, clip_tokens


@dataclass
class Assembled:
    embeds: torch.Tensor          # (1, L, D)
    position_ids: torch.Tensor    # (3, 1, L)
    visual_mask: torch.Tensor     # (1, L) bool
    deepstack: list | None        # per-level (n_visual, D)
    next_pos: int                 # first decode position (M-RoPE text plane)
    ids: torch.Tensor             # (1, L) kept token ids (debug/parity)

IMG_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
VID_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"

_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
STOP_STRINGS = ("</tool_call>", "</answer>")


# ---------------------------------------------------------------------------
# messages: list of {"role": str, "parts": [str | Clip, ...]}

def _render_text(engine: Engine, messages: list, add_generation_prompt=True,
                 tools: list | None = None) -> tuple[str, list]:
    """Chat-template the conversation with literal placeholder strings standing in
    for clips. When `tools` is given, the model's native `# Tools` system block is
    injected (the hermes format it was trained on). Returns (text, ordered clips)."""
    clips, chat = [], []
    for m in messages:
        buf = []
        for p in m["parts"]:
            if isinstance(p, Clip):
                clips.append(p)
                if p.kind == "images":
                    buf.append(IMG_PLACEHOLDER * p.grid_thw.shape[0])
                else:
                    buf.append(VID_PLACEHOLDER)
            else:
                buf.append(p)
        chat.append({"role": m["role"], "content": "".join(buf)})
    kw = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    if tools:
        kw["tools"] = tools
    text = engine.processor.apply_chat_template(chat, **kw)
    return text, clips


def assemble(engine: Engine, messages: list, tools: list | None = None) -> Assembled:
    """Build (inputs_embeds, position_ids, visual_pos_masks, deepstack, next_pos)
    for one prefill. Token layout is identical to the native processor's for image
    clips (asserted in smoke); video clips use the full-grid layout then keep only
    FlashVID anchors."""
    dev = engine.device
    text, clips = _render_text(engine, messages, tools=tools)
    ids = engine.tokenizer(text, return_tensors="pt").input_ids[0].to(dev)

    # Expand singleton pad ids to full runs (per clip, in order).
    img_clips = [c for c in clips if c.kind == "images"]
    vid_clips = [c for c in clips if c.kind == "video"]
    out, img_i, img_frame_i, vid_i = [], 0, 0, 0
    skip_next_ve = False
    for tok in ids.tolist():
        if skip_next_ve:  # drop the <|vision_end|> that closed the video sentinel
            assert tok == engine.vision_end_id
            skip_next_ve = False
            continue
        if tok == engine.image_token_id:
            c = img_clips[img_i]
            
            t, h, w = c.grid_thw[img_frame_i].tolist()
            out += [tok] * (t * (h // 2) * (w // 2))
            img_frame_i += 1
            if img_frame_i == c.grid_thw.shape[0]:
                img_i, img_frame_i = img_i + 1, 0
        elif tok == engine.video_token_id:
            # Replace the whole <vs><|video_pad|><ve> sentinel with the clip's
            # native per-frame segment (timestamps + per-frame <vs>pads<ve> —
            # required by get_rope_index's per-frame grid split).
            assert out and out[-1] == engine.vision_start_id
            out.pop()
            out += vid_clips[vid_i].segment_ids.tolist()
            vid_i += 1
            skip_next_ve = True
        else:
            out.append(tok)
    full_ids = torch.tensor(out, dtype=torch.long, device=dev).unsqueeze(0)
    L = full_ids.shape[1]

    # M-RoPE over the FULL layout (native helper), then prune non-kept video pads.
    img_grids = torch.cat([c.grid_thw for c in img_clips]) if img_clips else None
    vid_grids = torch.cat([c.grid_thw for c in vid_clips]) if vid_clips else None
    position_ids, _ = engine.model.model.get_rope_index(
        full_ids, img_grids, vid_grids, attention_mask=torch.ones_like(full_ids)
    )

    embeds = engine.model.get_input_embeddings()(full_ids)
    visual_mask = torch.zeros(L, dtype=torch.bool, device=dev)

    img_pos = (full_ids[0] == engine.image_token_id).nonzero(as_tuple=True)[0]
    off = 0
    for c in img_clips:
        pos = img_pos[off : off + c.n_tokens]
        off += c.n_tokens
        embeds[0, pos] = c.embeds.to(embeds.dtype)
        visual_mask[pos] = True

    # Video clips: FlashVID keep_indices may contain DUPLICATE anchors (an ADTS
    # pick and a merge-tree anchor can share a position). Mirror FlashVID's own
    # semantics (modeling_qwen3_vl.py:325-344): scatter (last write wins), then
    # keep an INDEX LIST with duplicates — never a boolean mask, which dedupes
    # and desyncs from the deepstack row count.
    vid_pos = (full_ids[0] == engine.video_token_id).nonzero(as_tuple=True)[0]
    is_vid_pad = torch.zeros(L, dtype=torch.bool, device=dev)
    is_vid_pad[vid_pos] = True
    kept_video = []
    off = 0
    for c in vid_clips:
        pos = vid_pos[off : off + c.meta["base_tokens"]]
        assert pos.shape[0] == c.meta["base_tokens"], (
            f"video clip claims {c.meta['base_tokens']} base tokens but only "
            f"{pos.shape[0]} <|video_pad|> positions remain")
        off += c.meta["base_tokens"]
        kept = pos[c.keep_indices]              # duplicates preserved
        # Official FlashVID asserts exactly this before its scatter
        # (modeling_qwen3_vl.py:338). Without it a keep_indices/embeds mismatch
        # silently desyncs the deepstack rows from the visual positions, which
        # corrupts results rather than raising.
        assert kept.shape[0] == c.n_tokens, (
            f"kept positions {kept.shape[0]} != compressed tokens {c.n_tokens}")
        embeds[0, kept] = c.embeds.to(embeds.dtype)
        visual_mask[kept] = True
        kept_video.append(kept)

    non_video = torch.arange(L, device=dev)[~is_vid_pad]
    final_index = torch.sort(
        torch.cat([non_video] + kept_video)
    ).values if kept_video else torch.arange(L, device=dev)

    embeds = embeds[:, final_index]
    position_ids = position_ids[..., final_index]
    visual_final = visual_mask[final_index].unsqueeze(0)
    deepstack = None
    if clips:
        n_levels = len(clips[0].deepstack)
        deepstack = [
            torch.cat([c.deepstack[lv] for c in clips]).to(embeds.dtype)
            for lv in range(n_levels)
        ]
        # The LM indexes deepstack rows by position among the True entries of
        # visual_final, so the row count must equal that many -- and the clip order
        # must match ascending position order (it does: clips are appended in message
        # order). A mismatch here is the failure mode the assert above guards per-clip.
        assert deepstack[0].shape[0] == int(visual_final.sum()), (
            f"deepstack rows {deepstack[0].shape[0]} != visual positions "
            f"{int(visual_final.sum())}")
    next_pos = int(position_ids.max()) + 1
    return Assembled(embeds, position_ids, visual_final, deepstack, next_pos,
                     full_ids[:, final_index])


@torch.inference_mode()
def decode(engine: Engine, a: Assembled,
           max_new_tokens: int = config.MAX_NEW_TOKENS) -> str:
    """Greedy decode with manual KV cache; stops on </tool_call>, </answer>, or EOS."""
    lm = engine.model.model.language_model
    lm_head = engine.model.lm_head
    embed_tokens = engine.model.get_input_embeddings()
    L = a.embeds.shape[1]
    next_pos = a.next_pos

    past = DynamicCache()
    out = lm(
        inputs_embeds=a.embeds,
        position_ids=a.position_ids,
        past_key_values=past,
        cache_position=torch.arange(L, device=engine.device),
        use_cache=True,
        visual_pos_masks=a.visual_mask,
        deepstack_visual_embeds=a.deepstack,
    )
    gen = []
    next_id = int(lm_head(out.last_hidden_state[:, -1:]).argmax(-1))
    for s in range(max_new_tokens):
        if next_id in engine.eos_ids:
            break
        gen.append(next_id)
        # Only the tail can end with a stop string, so decode the tail -- decoding the
        # whole `gen` list every step is O(n^2) over a 2048-token budget. 16 tokens is
        # far more than the longest stop string ("</tool_call>") needs.
        if any(engine.tokenizer.decode(gen[-16:]).endswith(t) for t in STOP_STRINGS):
            break
        e = embed_tokens(torch.tensor([[next_id]], device=engine.device))
        pos = torch.full((3, 1, 1), next_pos + s, device=engine.device, dtype=torch.long)
        out = lm(
            inputs_embeds=e,
            position_ids=pos,
            past_key_values=past,
            cache_position=torch.tensor([L + s], device=engine.device),
            use_cache=True,
        )
        next_id = int(lm_head(out.last_hidden_state[:, -1:]).argmax(-1))
    return engine.tokenizer.decode(gen)


# ---------------------------------------------------------------------------
def parse_tool_call(text: str) -> dict | None:
    m = _TOOL_RE.search(text)
    if not m:
        return None
    try:
        d = json.loads(m.group(1))
        args = d.get("arguments") or d.get("parameters") or {}
        return {"name": d.get("name", ""), "args": args}
    except json.JSONDecodeError:
        return {"name": "__malformed__", "args": {}}


def crop_budget_tokens(engine: Engine) -> int:
    """Crop-tool token ceiling, derived not hardcoded: 128 frames at the actual
    processor grid. Uses a 224x224 probe frame."""
    from PIL import Image
    import numpy as np

    probe = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    proc = engine.processor.image_processor(images=[probe], return_tensors="pt")
    per_frame = clip_tokens(proc["image_grid_thw"])
    return per_frame * config.CROP_MAX_FRAMES


def run_sample(engine: Engine, row: dict, tool_names: tuple = (),
               max_rounds: int = config.MAX_ROUNDS, verbose: bool = False,
               record_dir: str | None = None) -> dict:
    """Drive one sample through a LongVT-style loop: generate -> execute the tool if
    one was called, else stop (baseline is single-turn automatically). Then ONE
    bounded finalizer turn if no answer letter emerged. Dual-scored: pred_strict
    (pre-finalizer, LongVT-faithful) and pred_lenient (post-finalizer, headline).
    Writes a replayable trajectory + montages under record_dir when given."""
    qid = str(row["question_id"])
    media = os.path.join(record_dir, "media", qid) if record_dir else None

    t_sample = time.time()
    dur = data.video_duration(row["video_path"])
    pils, skim_times = tools.initial_frames_with_timestamps(row["video_path"])
    clip0 = engine.encode_images(pils, meta={"role": "initial"})
    initial_montage = (tj.save_montage(pils, os.path.join(media, "initial.png"))
                       if media else None)

    # Native tool schemas -> `# Tools` block, present in EVERY round of a tool arm.
    schemas = [config.TOOL_SCHEMAS[t] for t in tool_names] if tool_names else None

    prompt = (
        data.format_question(row)
        + "\n\n" + config.initial_view_text(dur, len(pils), skim_times)
        + (config.tool_instructions(dur, tool_names) if tool_names else "")
        + "\n\n" + config.ANSWER_INSTR
    )
    messages = [{"role": "user", "parts": [clip0, prompt]}]
    per_frame = clip_tokens(clip0.grid_thw[:1])          # matched-budget target (crop=128 -> real compression)
    target_tokens = per_frame * config.CROP_MAX_FRAMES

    calls, texts, rounds = [], [], []
    n_exec = 0                                            # tool executions so far

    def gen(label) -> tuple[str, dict]:
        """One assemble+decode turn; appends assistant text, returns (text, rec)."""
        t0 = time.time()
        a = assemble(engine, messages, tools=schemas)
        t1 = time.time()
        text = decode(engine, a)
        t2 = time.time()
        texts.append(text)
        rec = {"round": label, "context_tokens": int(a.embeds.shape[1]),
               "gen_seconds": round(t2 - t0, 2),
               "timing": {"assemble_seconds": round(t1 - t0, 3),
                          "decode_seconds": round(t2 - t1, 3),
                          "round_seconds": round(t2 - t0, 3)},
               "thinking": text, "action": None, "tool_result": None}
        if verbose:
            print(f"  [R{label}] ({a.embeds.shape[1]} tok ctx) {text[:300]}")
        messages.append({"role": "assistant", "parts": [text]})
        return text, rec

    for rnd in range(max_rounds + 1):
        text, rec = gen(rnd)
        tc = parse_tool_call(text)
        has_answer = "<answer>" in text                  # answer wins over a same-turn tool call
        will_exec = (tc is not None and not has_answer
                     and tc["name"] in tool_names and n_exec < max_rounds)
        if not will_exec:
            rec["action"] = {"kind": "answer" if has_answer else "stop"}
            rounds.append(rec)
            break

        n_blocks = len(_TOOL_RE.findall(text))            # native hermes may emit several
        s, e, err = tools.clamp_span(
            tc["args"].get("start_time"), tc["args"].get("end_time"), dur
        )
        rec["action"] = {"kind": "tool_call", "name": tc["name"],
                         "start": s, "end": e, "error": err}
        if n_blocks > 1:
            rec["action"]["extra_tool_calls_dropped"] = n_blocks - 1
        calls.append({"name": tc["name"], "start": s, "end": e, "error": err})
        if err:
            rounds.append(rec)
            messages.append({"role": "user", "parts": [f"<tool_response>\n{err}\n</tool_response>"]})
            continue
        t_tool = time.time()  # decode + ViT encode (montage cost is negligible)
        if tc["name"] == "crop_video":
            frames = tools.crop_frames(row["video_path"], s, e)
            clip = engine.encode_images(frames, meta={"role": "crop", "span": (s, e)})
            note = (f"crop_video: {len(frames)} full-detail frames covering "
                    f"{s:.0f}s-{e:.0f}s (1 fps).")
            mont = (tj.save_montage(frames, os.path.join(media, f"r{rnd}_crop.png"))
                    if media else None)
            rec["tool_result"] = {"tool": "crop_video", "span": [s, e],
                                  "n_frames": len(frames), "montage": mont}
        else:  # compress_video
            vt, times = tools.compress_tensor(row["video_path"], s, e)
            clip = engine.encode_video_compressed(
                vt, target_tokens, times, meta={"role": "compress", "span": (s, e)},
                query_text=data.format_question(row),  # semvid only; flashvid ignores
            )
            note = (f"compress_video: compressed overview of {s:.0f}s-{e:.0f}s "
                    f"({vt.shape[0]} frames -> {clip.meta['kept_tokens']} tokens).")
            mont = (tj.save_montage(vt, os.path.join(media, f"r{rnd}_compress.png"))
                    if media else None)
            rec["tool_result"] = {"tool": "compress_video", "span": [s, e],
                                  "n_frames": int(vt.shape[0]),
                                  "kept_tokens": clip.meta["kept_tokens"],
                                  "base_tokens": clip.meta["base_tokens"],
                                  "retention": round(clip.meta["retention"], 3),
                                  "montage": mont}
            calls[-1].update({k: clip.meta[k] for k in
                              ("kept_tokens", "base_tokens", "retention")})
        rec["tool_seconds"] = round(time.time() - t_tool, 2)
        n_exec += 1
        rounds.append(rec)
        messages.append({"role": "user",
                         "parts": ["<tool_response>\n", clip,
                                   f"\n{note}{config.TOOL_RESULT_INSTR}\n</tool_response>"]})

    # Strict, LongVT-faithful score: whatever the model produced on its own.
    pred_strict = data.extract_answer("\n".join(texts))

    # One bounded finalizer turn (identical across arms) only if no letter surfaced.
    finalizer_used = False
    if pred_strict is None:
        finalizer_used = True
        messages.append({"role": "user", "parts": [
            "Reply with <answer>X</answer> where X is one of A, B, C, or D."]})
        _text, rec = gen("finalizer")
        rec["action"] = {"kind": "finalizer"}
        rounds.append(rec)

    pred_lenient = data.extract_answer("\n".join(texts))
    gold = row["answer"]
    seconds = round(time.time() - t_sample, 1)
    traj_path = None
    if record_dir:
        traj = {"question_id": qid, "task_type": row["task_type"],
                "question": row["question"], "options": list(row["options"]),
                "gold": gold, "pred": pred_lenient,
                "pred_strict": pred_strict, "pred_lenient": pred_lenient,
                "finalizer_used": finalizer_used,
                "correct": pred_lenient == gold, "correct_strict": pred_strict == gold,
                "videoID": row["videoID"], "video_path": row["video_path"],
                "duration": dur, "tools": list(tool_names), "seconds": seconds,
                "schema_version": 3,
                "timing": {"sample_seconds": seconds,
                           "round_seconds": round(sum(r["timing"]["round_seconds"] for r in rounds), 3),
                           "tool_seconds": round(sum(r.get("tool_seconds", 0) for r in rounds), 3)},
                "initial": {"n_frames": len(pils), "montage": initial_montage},
                "rounds": rounds}
        traj_path = tj.save_trajectory(traj, os.path.join(record_dir, "traj", f"{qid}.json"))

    return {
        "question_id": qid,
        "task_type": row["task_type"],
        "pred": pred_lenient,
        "pred_strict": pred_strict,
        "pred_lenient": pred_lenient,
        "finalizer_used": finalizer_used,
        "gold": gold,
        "correct": pred_lenient == gold,
        "correct_strict": pred_strict == gold,
        "rounds": len(texts),
        "tool_calls": calls,
        "duration": dur,
        "seconds": seconds,
        "traj_path": traj_path,
    }
