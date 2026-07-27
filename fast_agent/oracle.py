"""Oracle-guided arm (LVBench only): forced GT-localized evidence, then answer.

Diagnostic, NOT a deployable policy. The harness hands the agent the correct evidence
REGION (from LVBench's per-question `time_reference`) and asks it to answer -- isolating
"given perfect localization, is perception+reasoning enough?" from "did the agent look in
the right place?".

Three modes, all forcing the SAME GT evidence region so crop vs compress is a clean
matched-region ablation of evidence FORM (not of localization):
  - "crop"     : crop(GT region) -> answer        # PRIMARY: full-detail at the right place;
                                                    #  the cleanest "is perception enough" test.
  - "compress" : compress(GT region) -> answer    # coverage form (denser temporal sampling,
                                                    #  same token budget) -- matters on wide refs.
  - "both"     : compress + crop(GT region) -> answer   # does adding coverage to a correct crop help/hurt?

Everything else is byte-identical to arms 1/2: same 64-frame skim, same prompt text, same
tool schemas in context (both tools shown in every mode, so only the forced evidence
differs), same greedy decode, same answer parser + bounded finalizer, same trajectory
schema. Model-facing text is a normal <tool_call>/<tool_response>; the "oracle_forced"
marks live only in the log. A single final generation answers (no reason-between-steps:
RESEARCH.md 07-18/19 shows that mode is repetition-prone under greedy decode).
"""

import json
import os
import time

from . import config, data, tools
from . import trajectory as tj
from .agent_loop import assemble, decode
from .model import clip_tokens

# Tools shown in the # Tools block, PER MODE. This must mirror run_eval.ARM_TOOLS for the
# autonomous arm each mode is compared against: showing crop-only modes an extra
# compress_video schema (and the "use compress_video to skim, then crop to zoom" strategy
# hint from config.tool_instructions) gives oracle_crop a different prompt than the
# autonomous `crop` arm, confounding the single comparison the experiment exists to make.
MODE_TOOLS = {
    "crop": ("crop_video",),
    "compress": ("compress_video",),
    "both": ("compress_video", "crop_video"),
    "ctrl": ("crop_video",),          # wrong-location control: same tools as oracle_crop
}
ORACLE_MODES = tuple(MODE_TOOLS)


def oracle_spans(row: dict, duration: float, mode: str = "crop") -> dict | None:
    """Derive the forced REGION. For the evidence modes this is the GT evidence span
    (widened to a min floor), used by BOTH crop and compress so the two forms are compared
    at a matched region. For mode="ctrl" it is a SAME-WIDTH region deliberately placed
    AWAY from the evidence -- the control that separates "the right pixels" from "more
    pixels" (without it, oracle_crop vs baseline is equally explained by extra visual
    tokens of any content). None if no usable evidence."""
    ev = row.get("evidence")
    if not ev:
        return None
    es, ee = float(ev[0]), float(ev[1])
    if ee < es:                      # defensive: normalize reversed refs
        es, ee = ee, es
    cw = max(ee - es, config.ORACLE_CROP_MIN_WIDTH)
    mid = 0.5 * (es + ee)
    xs = max(0.0, mid - cw / 2)
    xe = min(duration, mid + cw / 2)
    if mode != "ctrl":
        return {"region": (xs, xe)}

    # Control: mirror the region about the video midpoint; if the mirror still overlaps
    # the evidence (evidence near the centre), slide it to the farthest non-overlapping
    # placement. Deterministic -- no seed, so the control is reproducible per question.
    w = xe - xs
    m_xs = max(0.0, min(duration - w, duration - xe))
    m_xe = m_xs + w
    if not (m_xe <= xs or m_xs >= xe):          # mirror overlaps evidence -> go to an edge
        left_room, right_room = xs, duration - xe
        if left_room >= right_room and left_room >= w:
            m_xs, m_xe = 0.0, w
        elif right_room >= w:
            m_xs, m_xe = duration - w, duration
        else:
            return None                          # video too short to place a clean control
    return {"region": (m_xs, m_xe), "evidence_region": (xs, xe)}


def run_oracle_sample(engine, row: dict, record_dir: str | None = None,
                      mode: str = "crop", verbose: bool = False) -> dict:
    """Drive one LVBench sample through a forced oracle trajectory at the GT evidence
    region. mode in {"crop","compress","both"} (see module docstring). Emits the same
    result dict + trajectory schema as agent_loop.run_sample so run_eval.write_summary
    and viz.py work unchanged."""
    if mode not in ORACLE_MODES:
        raise ValueError(f"unknown oracle mode {mode!r} (expected one of {ORACLE_MODES})")
    qid = str(row["question_id"])
    media = os.path.join(record_dir, "media", qid) if record_dir else None

    t_sample = time.time()
    dur = data.video_duration(row["video_path"])
    spans = oracle_spans(row, dur, mode)
    if spans is None:
        raise ValueError(f"oracle arm needs evidence; qid {qid} has none "
                         f"(time_reference={row.get('time_reference')!r})")
    tool_names = MODE_TOOLS[mode]

    pils, skim_times = tools.initial_frames_with_timestamps(row["video_path"])
    clip0 = engine.encode_images(pils, meta={"role": "initial"})
    initial_montage = (tj.save_montage(pils, os.path.join(media, "initial.png"))
                       if media else None)

    schemas = [config.TOOL_SCHEMAS[t] for t in tool_names]
    prompt = (
        data.format_question(row)
        + "\n\n" + config.initial_view_text(dur, len(pils), skim_times)
        + config.tool_instructions(dur, tool_names)
        + "\n\n" + config.ANSWER_INSTR
    )
    messages = [{"role": "user", "parts": [clip0, prompt]}]
    per_frame = clip_tokens(clip0.grid_thw[:1])
    target_tokens = per_frame * config.CROP_MAX_FRAMES

    calls, texts, rounds = [], [], []

    def gen(label):
        """One real assemble+decode turn (identical to run_sample.gen)."""
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
            print(f"  [R{label}] ({a.embeds.shape[1]} tok) {text[:200]}")
        messages.append({"role": "assistant", "parts": [text]})
        return text, rec

    def forced_step(name, s, e):
        """Harness-forced tool call at a GT-derived span. Synthesizes the assistant
        <tool_call> (no model generation), executes the tool, injects the <tool_response>
        exactly as the autonomous arm does. Marked oracle_forced in the log only."""
        rnd = len(rounds)
        s2, e2, err = tools.clamp_span(s, e, dur)
        args = {"start_time": round(s2 if not err else s, 1),
                "end_time": round(e2 if not err else e, 1)}
        call_text = ("<tool_call>\n"
                     + json.dumps({"name": name, "arguments": args})
                     + "\n</tool_call>")
        messages.append({"role": "assistant", "parts": [call_text]})
        rec = {"round": rnd, "oracle_forced": True, "context_tokens": None,
               "gen_seconds": 0.0,
               "timing": {"assemble_seconds": 0.0, "decode_seconds": 0.0,
                          "round_seconds": 0.0},
               "thinking": "(oracle: harness-forced tool call, no model generation)",
               "action": {"kind": "tool_call", "name": name, "start": s2, "end": e2,
                          "error": err, "oracle_forced": True},
               "tool_result": None}
        call = {"name": name, "start": s2, "end": e2, "error": err, "oracle_forced": True}
        if err:
            calls.append(call)
            rounds.append(rec)
            messages.append({"role": "user",
                             "parts": [f"<tool_response>\n{err}\n</tool_response>"]})
            return
        t_tool = time.time()
        if name == "crop_video":
            frames = tools.crop_frames(row["video_path"], s2, e2)
            clip = engine.encode_images(frames, meta={"role": "oracle_crop", "span": (s2, e2)})
            note = (f"crop_video: {len(frames)} full-detail frames covering "
                    f"{s2:.0f}s-{e2:.0f}s (1 fps).")
            mont = (tj.save_montage(frames, os.path.join(media, f"r{rnd}_crop.png"))
                    if media else None)
            rec["tool_result"] = {"tool": "crop_video", "span": [s2, e2],
                                  "n_frames": len(frames), "montage": mont}
        else:  # compress_video
            vt, times = tools.compress_tensor(row["video_path"], s2, e2)
            clip = engine.encode_video_compressed(
                vt, target_tokens, times, meta={"role": "oracle_compress", "span": (s2, e2)},
                query_text=data.format_question(row),  # semvid only; flashvid ignores
            )
            note = (f"compress_video: compressed overview of {s2:.0f}s-{e2:.0f}s "
                    f"({vt.shape[0]} frames -> {clip.meta['kept_tokens']} tokens).")
            mont = (tj.save_montage(vt, os.path.join(media, f"r{rnd}_compress.png"))
                    if media else None)
            rec["tool_result"] = {"tool": "compress_video", "span": [s2, e2],
                                  "n_frames": int(vt.shape[0]),
                                  "kept_tokens": clip.meta["kept_tokens"],
                                  "base_tokens": clip.meta["base_tokens"],
                                  "retention": round(clip.meta["retention"], 3),
                                  "montage": mont}
            call.update({k: clip.meta[k] for k in ("kept_tokens", "base_tokens", "retention")})
        rec["tool_seconds"] = round(time.time() - t_tool, 2)
        calls.append(call)
        rounds.append(rec)
        messages.append({"role": "user",
                         "parts": ["<tool_response>\n", clip,
                                   f"\n{note}{config.TOOL_RESULT_INSTR}\n</tool_response>"]})

    # Forced evidence, per mode (both = coverage form then detail form; ctrl = a
    # same-width crop AWAY from the evidence, so it is matched to oracle_crop on tool,
    # frame count and token budget and differs ONLY in location).
    region = spans["region"]
    if mode in ("compress", "both"):
        forced_step("compress_video", *region)
    if mode in ("crop", "both", "ctrl"):
        forced_step("crop_video", *region)

    # Forced answer: one real generation over skim + the forced evidence.
    messages.append({"role": "user", "parts": [
        "Based on the video evidence above, give your final answer as "
        "<answer>X</answer> where X is one of A, B, C, or D."]})
    _text, rec = gen("answer")
    rec["action"] = {"kind": "answer"}
    rounds.append(rec)

    pred_strict = data.extract_answer("\n".join(texts))
    finalizer_used = False
    if pred_strict is None:
        finalizer_used = True
        messages.append({"role": "user", "parts": [
            "Reply with <answer>X</answer> where X is one of A, B, C, or D."]})
        _t, frec = gen("finalizer")
        frec["action"] = {"kind": "finalizer"}
        rounds.append(frec)

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
                "duration": dur, "arm": f"oracle_{mode}",
                "tools": [c["name"] for c in calls],
                "seconds": seconds, "schema_version": 3,
                "oracle": {"mode": mode,
                           "tools_shown": list(tool_names),
                           "evidence": row.get("evidence"),
                           "time_reference": row.get("time_reference"),
                           "region": list(region),
                           "evidence_region": list(spans.get("evidence_region", region)),
                           "region_width": round(region[1] - region[0], 1),
                           "localization_trivial": row.get("localization_trivial"),
                           "compressor": config.COMPRESSOR,
                           "skim_timestamps": config.SKIM_TIMESTAMPS,
                           "unify_time_format": data.UNIFY_TIME_FORMAT,
                           "crop_min_width": config.ORACLE_CROP_MIN_WIDTH},
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
        "oracle_mode": mode,
        "oracle_region": list(region),
        "traj_path": traj_path,
    }
