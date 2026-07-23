"""One-sample diagnostic harness for the four explicitly separated debug modes.

This is intentionally not an evaluation runner.  It never loads more than one
dataset row and writes a transparent, human-readable record under
``debug_runs/<qid>/<mode>``.
"""

import argparse
import json
import traceback
from pathlib import Path

import torch

from . import config, data, tools
from .agent_loop import _TOOL_RE, _render_text, assemble, decode, parse_tool_call
from .model import Engine, clip_tokens
from .trajectory import save_montage


MODES = (
    "baseline_single_turn",
    "forced_crop",
    "forced_compress",
    "autonomous_tools",
)


class Artifacts:
    def __init__(self, root: Path, row: dict, mode: str, start: float, end: float):
        self.path = root / str(row["question_id"]) / mode
        self.visuals = self.path / "visualizations"
        self.visuals.mkdir(parents=True, exist_ok=True)
        self.generations = []
        self.prompts = []
        self.trace = []
        self.timestamps = {}
        self.shapes = {}
        self.config = {
            "mode": mode,
            "question_id": str(row["question_id"]),
            "video_id": row["videoID"],
            "video_path": row["video_path"],
            "model_snapshot": config.MODEL_SNAPSHOT,
            "manual_time_range_seconds": [start, end],
            "initial_frames": config.INITIAL_FRAMES,
            "crop_max_frames": config.CROP_MAX_FRAMES,
            "compress_max_frames": config.COMPRESS_MAX_FRAMES,
            "fixed_retention": config.FIXED_RETENTION,
            "max_new_tokens": config.MAX_NEW_TOKENS,
        }

    def finish(self, result: dict):
        self._json("config.json", self.config)
        self._json("parsed_result.json", result)
        self._json("frame_timestamps.json", self.timestamps)
        self._json("tensor_shapes.json", self.shapes)
        self._text("prompt.txt", "\n\n".join(self.prompts))
        self._text("raw_generations.txt", "\n\n".join(
            f"=== generation {i + 1} ===\n{text}" for i, text in enumerate(self.generations)
        ))
        self._text("trace.md", "\n\n".join(self.trace) + "\n")

    def fail(self, exc: Exception, component: str):
        self.trace.append(
            f"## Failure\n\nThe run stopped in the **{component}** component: "
            f"`{type(exc).__name__}: {exc}`."
        )
        self.finish({
            "status": "failed",
            "failure_component": component,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "parsed_answer": data.extract_answer("\n".join(self.generations)),
        })

    def _json(self, name: str, value):
        with open(self.path / name, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True)

    def _text(self, name: str, value: str):
        with open(self.path / name, "w") as f:
            f.write(value)


def _shape(x):
    return list(x.shape)


def _frame_info(frames) -> dict:
    first = frames[0]
    if torch.is_tensor(first):
        return {"count": len(frames), "resolution_hw": [int(first.shape[-2]), int(first.shape[-1])]}
    return {"count": len(frames), "resolution_hw": [first.height, first.width]}


def _answer_status(raw: str, parsed: str | None) -> dict:
    if parsed is not None:
        return {"status": "worked"}
    # The unchanged parser deliberately accepts only A-D.  Outputs such as
    # <answer>Unknown</answer> are therefore model behavior, not parser bugs.
    return {"status": "failed", "failure_component": "model behavior",
            "error": "no answer could be parsed from the model output"}


def _question_prompt(row, duration: float, n_initial: int, tool_names=()) -> str:
    return (
        data.format_question(row)
        + "\n\n" + config.initial_view_text(duration, n_initial)
        + (config.tool_instructions(duration, tool_names) if tool_names else "")
        + "\n\n" + config.ANSWER_INSTR
    )


def _record_generation(engine, messages, art: Artifacts, tools_schema=None, label="1"):
    rendered, _clips = _render_text(engine, messages, tools=tools_schema)
    art.prompts.append(f"=== model input for generation {label} ===\n{rendered}")
    assembled = assemble(engine, messages, tools=tools_schema)
    text = decode(engine, assembled)
    art.generations.append(text)
    art.shapes[f"generation_{label}"] = {
        "input_embeds": _shape(assembled.embeds),
        "position_ids_mrope": _shape(assembled.position_ids),
        "visual_mask": _shape(assembled.visual_mask),
        "kept_input_ids": _shape(assembled.ids),
        "deepstack": [_shape(x) for x in (assembled.deepstack or [])],
        "finite": {
            "input_embeds": bool(torch.isfinite(assembled.embeds).all().item()),
            "position_ids": bool(torch.isfinite(assembled.position_ids).all().item()),
        },
    }
    messages.append({"role": "assistant", "parts": [text]})
    return text


def _prepare_initial(engine, row, art: Artifacts):
    frames, timestamps = tools.initial_frames_with_timestamps(row["video_path"])
    art.timestamps["initial"] = timestamps
    art.shapes["initial_frames"] = _frame_info(frames)
    save_montage(frames, str(art.visuals / "initial.png"))
    return frames, engine.encode_images(frames, meta={"role": "initial"})


def run_baseline(engine, row, art: Artifacts, start: float, end: float):
    duration = data.video_duration(row["video_path"])
    frames, initial = _prepare_initial(engine, row, art)
    prompt = _question_prompt(row, duration, len(frames))
    messages = [{"role": "user", "parts": [initial, prompt]}]
    raw = _record_generation(engine, messages, art)
    parsed = data.extract_answer(raw)
    generations, tool_calls = len(art.generations), 0
    terminated_after_first = generations == 1
    assert generations == 1, f"baseline made {generations} generations"
    assert tool_calls == 0, f"baseline made {tool_calls} tool calls"
    assert terminated_after_first, "baseline did not terminate after generation one"
    art.trace.extend([
        "# Baseline single turn",
        "1. The harness decoded the initial whole-video view and saved `visualizations/initial.png`.",
        "2. It combined the unchanged question, initial visual input, and answer instruction.",
        "3. It performed exactly one model generation and parsed that output directly.",
        "4. It stopped. No tool loop and no finalizer are reachable in this mode.",
    ])
    return {
        **_answer_status(raw, parsed), "parsed_answer": parsed, "gold": row["answer"],
        "raw_output": raw, "generation_count": generations,
        "tool_call_count": tool_calls, "terminated_after_first_generation": True,
        "assertions": {"exactly_one_generation": True, "zero_tool_calls": True,
                       "terminated_after_first_generation": True},
    }


def run_forced_crop(engine, row, art: Artifacts, start: float, end: float):
    duration = data.video_duration(row["video_path"])
    initial_frames, initial = _prepare_initial(engine, row, art)
    crop, timestamps = tools.crop_frames_with_timestamps(row["video_path"], start, end)
    art.timestamps["forced_crop"] = timestamps
    art.shapes["crop_frames"] = _frame_info(crop)
    montage = art.visuals / "forced_crop.png"
    save_montage(crop, str(montage))
    crop_clip = engine.encode_images(crop, meta={"role": "forced_crop", "span": [start, end]})
    prompt = _question_prompt(row, duration, len(initial_frames))
    note = (f"\n\nForced diagnostic crop for {start:.1f}s-{end:.1f}s: "
            f"{len(crop)} full-detail frames. Use this visual evidence to answer.")
    messages = [{"role": "user", "parts": [initial, prompt, note, crop_clip]}]
    raw = _record_generation(engine, messages, art)
    parsed = data.extract_answer(raw)
    assert len(art.generations) == 1
    art.trace.extend([
        "# Forced crop",
        f"1. The harness manually decoded {len(crop)} frames from {start:.1f}s to {end:.1f}s. The model did not choose or call a tool.",
        "2. Exact planned timestamps are in `frame_timestamps.json`; frame count and resolution are in `tensor_shapes.json`.",
        "3. The crop montage is `visualizations/forced_crop.png`.",
        "4. `prompt.txt` is the exact rendered model input after inserting the forced crop.",
        "5. The harness generated once, saved the raw output, parsed it, and stopped.",
    ])
    return {**_answer_status(raw, parsed), "parsed_answer": parsed, "gold": row["answer"],
            "raw_output": raw, "generation_count": 1, "tool_call_count": 0,
            "forced_operation": {"name": "crop_video", "span": [start, end]},
            "frame_count": len(crop), "resolution_hw": art.shapes["crop_frames"]["resolution_hw"]}


def _visual_text_context(rendered: str, placeholder: str, window: int = 500):
    pos = rendered.rfind(placeholder)
    if pos < 0:
        return {"placeholder": placeholder, "found": False}
    return {"placeholder": placeholder, "found": True,
            "text_before": rendered[max(0, pos - window):pos],
            "text_after": rendered[pos + len(placeholder):pos + len(placeholder) + window]}


def run_forced_compress(engine, row, art: Artifacts, start: float, end: float):
    duration = data.video_duration(row["video_path"])
    initial_frames, initial = _prepare_initial(engine, row, art)
    video_tensor, timestamps = tools.compress_tensor(row["video_path"], start, end)
    art.timestamps["forced_compress"] = timestamps
    art.shapes["compression_input_frames"] = {
        "tensor": _shape(video_tensor), "count": int(video_tensor.shape[0]),
        "resolution_hw": [int(video_tensor.shape[-2]), int(video_tensor.shape[-1])],
    }
    save_montage(video_tensor, str(art.visuals / "sampled_source_frames.png"))
    per_frame = clip_tokens(initial.grid_thw[:1])
    target = per_frame * config.CROP_MAX_FRAMES
    clip = engine.encode_video_compressed(
        video_tensor, target, timestamps,
        meta={"role": "forced_compress", "span": [start, end]},
    )
    art.shapes["flashvid_boundary"] = clip.meta["diagnostics"]
    art.shapes["flashvid_tokens"] = {
        "before": clip.meta["base_tokens"], "after": clip.meta["kept_tokens"],
        "expected_budget": target, "retention": clip.meta["retention"],
        "keep_indices": _shape(clip.keep_indices),
    }
    prompt = _question_prompt(row, duration, len(initial_frames))
    note = (f"\n\nForced diagnostic compression for {start:.1f}s-{end:.1f}s: "
            f"{video_tensor.shape[0]} source frames became {clip.meta['kept_tokens']} visual tokens. "
            "Use this visual evidence to answer.")
    messages = [{"role": "user", "parts": [initial, prompt, note, clip]}]
    rendered, _ = _render_text(engine, messages)
    art.shapes["inserted_visual_text_context"] = _visual_text_context(
        rendered, "<|vision_start|><|video_pad|><|vision_end|>"
    )

    assertions = {
        "finite_tensors": all(clip.meta["diagnostics"]["finite"].values()),
        "valid_dimensions": video_tensor.ndim == 4 and clip.embeds.ndim == 2
                            and clip.keep_indices.ndim == 1,
        "nonzero_token_count": clip.meta["base_tokens"] > 0 and clip.meta["kept_tokens"] > 0,
        "expected_token_budget": clip.meta["kept_tokens"] <= target,
    }
    raw = _record_generation(engine, messages, art)
    parsed = data.extract_answer(raw)
    art.trace.extend([
        "# Forced compression",
        f"1. The harness manually decoded {video_tensor.shape[0]} source frames from {start:.1f}s to {end:.1f}s. The model did not choose or call a tool.",
        "2. It passed the recorded input tensor through the existing vision tower and unchanged FlashVID compressor.",
        f"3. FlashVID reduced {clip.meta['base_tokens']} visual tokens to {clip.meta['kept_tokens']} against a budget of {target}.",
        "4. Boundary tensor shapes, finite checks, kept indices, and M-RoPE shapes are in `tensor_shapes.json`.",
        "5. Sampled source frames are in `visualizations/sampled_source_frames.png`.",
        "6. `prompt.txt` and `inserted_visual_text_context` show the exact text around the inserted visual representation.",
        "7. The harness generated once, saved the raw output, parsed it, and stopped.",
    ])
    assertion_error = None
    try:
        assert assertions["finite_tensors"], "non-finite tensor at FlashVID boundary"
        assert assertions["valid_dimensions"], "invalid compression tensor dimensions"
        assert assertions["nonzero_token_count"], "zero visual token count"
        assert assertions["expected_token_budget"], (
            f"kept {clip.meta['kept_tokens']} tokens, above expected token budget {target}"
        )
    except AssertionError as exc:
        assertion_error = str(exc)
    status = (_answer_status(raw, parsed) if assertion_error is None else {
        "status": "failed", "failure_component": "compression path",
        "error": f"AssertionError: {assertion_error}",
    })
    return {**status, "parsed_answer": parsed, "gold": row["answer"],
            "raw_output": raw, "generation_count": 1, "tool_call_count": 0,
            "forced_operation": {"name": "compress_video", "span": [start, end]},
            "tokens_before": clip.meta["base_tokens"], "tokens_after": clip.meta["kept_tokens"],
            "expected_token_budget": target, "assertions": assertions}


def run_autonomous(engine, row, art: Artifacts, start: float, end: float):
    duration = data.video_duration(row["video_path"])
    initial_frames, initial = _prepare_initial(engine, row, art)
    tool_names = ("crop_video", "compress_video")
    schemas = [config.TOOL_SCHEMAS[name] for name in tool_names]
    prompt = _question_prompt(row, duration, len(initial_frames), tool_names)
    messages = [{"role": "user", "parts": [initial, prompt]}]
    calls, events, n_exec, stop = [], [], 0, None
    target = clip_tokens(initial.grid_thw[:1]) * config.CROP_MAX_FRAMES

    for rnd in range(config.MAX_ROUNDS + 1):
        raw = _record_generation(engine, messages, art, schemas, str(rnd + 1))
        tc = parse_tool_call(raw)
        has_answer = "<answer>" in raw
        event = {"generation": rnd + 1, "raw_output": raw, "parsed_answer": data.extract_answer(raw),
                 "parsed_tool_call": tc}
        will_exec = (tc is not None and not has_answer and tc["name"] in tool_names
                     and n_exec < config.MAX_ROUNDS)
        if not will_exec:
            stop = "answer" if has_answer else "no_executable_tool_call"
            event["stopping_reason"] = stop
            events.append(event)
            break
        s, e, err = tools.clamp_span(tc["args"].get("start_time"), tc["args"].get("end_time"), duration)
        call = {"name": tc["name"], "start": s, "end": e, "error": err,
                "raw_tool_blocks": len(_TOOL_RE.findall(raw))}
        event["tool_call"] = call
        calls.append(call)
        if err:
            result = {"error": err}
            messages.append({"role": "user", "parts": [f"<tool_response>\n{err}\n</tool_response>"]})
        elif tc["name"] == "crop_video":
            frames, times = tools.crop_frames_with_timestamps(row["video_path"], s, e)
            art.timestamps[f"tool_{n_exec + 1}_crop"] = times
            clip = engine.encode_images(frames, meta={"role": "crop", "span": [s, e]})
            montage = art.visuals / f"tool_{n_exec + 1}_crop.png"
            save_montage(frames, str(montage))
            result = {"frame_count": len(frames), "resolution": _frame_info(frames)["resolution_hw"],
                      "montage": str(montage)}
            messages.append({"role": "user", "parts": ["<tool_response>\n", clip,
                f"\ncrop_video: {len(frames)} full-detail frames covering {s:.0f}s-{e:.0f}s (1 fps).\n</tool_response>"]})
        else:
            vt, times = tools.compress_tensor(row["video_path"], s, e)
            art.timestamps[f"tool_{n_exec + 1}_compress"] = times
            clip = engine.encode_video_compressed(vt, target, times, meta={"role": "compress", "span": [s, e]})
            montage = art.visuals / f"tool_{n_exec + 1}_compress.png"
            save_montage(vt, str(montage))
            result = {"frame_count": int(vt.shape[0]), "resolution": [int(vt.shape[-2]), int(vt.shape[-1])],
                      "tokens_before": clip.meta["base_tokens"], "tokens_after": clip.meta["kept_tokens"],
                      "montage": str(montage)}
            messages.append({"role": "user", "parts": ["<tool_response>\n", clip,
                f"\ncompress_video: compressed overview of {s:.0f}s-{e:.0f}s "
                f"({vt.shape[0]} frames -> {clip.meta['kept_tokens']} tokens).\n</tool_response>"]})
        event["tool_result"] = result
        events.append(event)
        n_exec += 1

    parsed = data.extract_answer("\n".join(art.generations))
    if parsed is None:
        messages.append({"role": "user", "parts": [
            "Reply with <answer>X</answer> where X is one of A, B, C, or D."]})
        raw = _record_generation(engine, messages, art, schemas, "finalizer")
        parsed = data.extract_answer("\n".join(art.generations))
        events.append({"generation": "finalizer", "raw_output": raw,
                       "parsed_answer": data.extract_answer(raw), "stopping_reason": "bounded_finalizer"})
        stop = "bounded_finalizer"
    art.shapes["events"] = events
    art.trace.append("# Autonomous tools")
    for event in events:
        art.trace.append(
            f"## Generation {event['generation']}\n\n"
            f"Parsed tool call: `{event.get('parsed_tool_call')}`. "
            f"Tool result: `{event.get('tool_result')}`. "
            f"Stopping reason: `{event.get('stopping_reason')}`. "
            f"Parsed answer for this generation: `{event.get('parsed_answer')}`."
        )
    status = _answer_status("\n".join(art.generations), parsed)
    return {**status, "parsed_answer": parsed, "gold": row["answer"],
            "generation_count": len(art.generations), "tool_call_count": len(calls),
            "tool_calls": calls, "stopping_reason": stop, "events": events}


RUNNERS = {
    "baseline_single_turn": run_baseline,
    "forced_crop": run_forced_crop,
    "forced_compress": run_forced_compress,
    "autonomous_tools": run_autonomous,
}


def _classify_failure(exc: Exception, mode: str) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "cuda" in text or "nvidia" in text or "model" in text:
        return "harness"
    if mode == "forced_compress":
        return "compression path"
    if mode == "forced_crop":
        return "crop path"
    if "decode" in text or "frame" in text or "video" in text:
        return "crop path" if mode == "forced_crop" else "compression path"
    if "flashvid" in text or "token budget" in text or "tensor" in text:
        return "compression path"
    if "tool" in text and mode == "autonomous_tools":
        return "tool protocol"
    return "harness"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qid", default=None, help="exact question_id; default is seed-0 sample 1")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=300.0)
    parser.add_argument("--output-root", default="debug_runs")
    parser.add_argument("--mode", action="append", choices=MODES,
                        help="run only this mode (repeatable); default runs all four")
    args = parser.parse_args()
    rows = data.load_long_split()
    row = next((r for r in rows if str(r["question_id"]) == str(args.qid)), None) if args.qid else data.load_long_split(n=1, seed=0)[0]
    if row is None:
        raise SystemExit(f"question_id not found: {args.qid}")
    duration = data.video_duration(row["video_path"])
    start, end, err = tools.clamp_span(args.start, args.end, duration)
    if err:
        raise SystemExit(err)
    root = Path(args.output_root).resolve()
    selected_modes = tuple(args.mode) if args.mode else MODES

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; the configured local model requires a CUDA device")
        engine = Engine()
    except Exception as exc:
        for mode in selected_modes:
            art = Artifacts(root, row, mode, start, end)
            art.fail(exc, "harness")
        raise

    summary = {}
    for mode in selected_modes:
        art = Artifacts(root, row, mode, start, end)
        try:
            result = RUNNERS[mode](engine, row, art, start, end)
            art.finish(result)
            summary[mode] = result["status"]
        except Exception as exc:
            component = _classify_failure(exc, mode)
            art.fail(exc, component)
            summary[mode] = f"failed: {component}: {type(exc).__name__}: {exc}"
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(json.dumps({"question_id": str(row["question_id"]), "output": str(root), "modes": summary}, indent=2))


if __name__ == "__main__":
    main()
