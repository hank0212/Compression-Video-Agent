"""Agentic eval against a vLLM OpenAI endpoint. Same backbone as run_plain, plus tools.

  # arm A -- Qwen + cropping, no compression
  python -m longvt_compression.fast_agent.run_agent --tools crop_video \
      --base-url http://localhost:8010/v1 --tag arm_a

  # arm B -- VidCom2-compressed skim + cropping (server must run with the plugin)
  python -m longvt_compression.fast_agent.run_agent --tools crop_video \
      --skim-mode video --base-url http://localhost:8021/v1 --tag arm_b

WHY NOT agent_loop.run_sample
-----------------------------
`agent_loop` drives the model through HF `inputs_embeds` + a hand-written greedy
decode. That path is what produced the flaky benchmark numbers, and it cannot use
the vLLM-side compression at all (the pruning hook lives inside the vLLM model
forward). This module keeps agent_loop's *semantics* -- LongVT prompt, native
`# Tools` block, one tool per turn, bounded finalizer, strict/lenient dual score --
and swaps the transport for HTTP.

Everything that decides the score is imported from the same modules run_plain uses,
so the no-tool baseline and these arms differ ONLY in `--tools` / `--skim-mode`:

  data.load_dataset / format_question / extract_answer
  tools.initial_frames_with_timestamps / crop_frames / clamp_span
  config.initial_view_text / tool_instructions / ANSWER_INSTR / TOOL_SCHEMAS

LOCALIZATION BOOKKEEPING
------------------------
Every tool call is stored next to the question's GT evidence span, with the
coverage of that span already computed (`covered_frac`, `iou`, `centre_offset`).
The metrics the task asks for -- call coverage %, per-question hit rate, and
accuracy split by landed/not-landed -- are then a groupby over results.jsonl and
need no re-reading of trajectories.
"""

import argparse
import base64
import io
import json
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from . import config, data, make_skim_proxy, run_plain, tools

DEFAULT_BASE_URL = os.environ.get("FA_BASE_URL", "http://localhost:8010/v1")
DEFAULT_MODEL = os.environ.get("FA_MODEL", "qwen3vl")

_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# GT evidence spans are frequently narrower than one crop can resolve (LVBench
# median 14s, p25 4s, 263 of them zero-width). IoU against a 4s span is ~0.03 even
# for a perfectly aimed call, which measures the tool's granularity rather than the
# model's aim. Floor the span before IoU -- same constant oracle.py uses.
IOU_MIN_WIDTH = 16.0


def _b64_image(pil, quality: int = 90) -> str:
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# localization bookkeeping

def _span_metrics(calls: list, evidence, duration: float) -> dict:
    """Per-question localization summary against the GT evidence span.

    covered_frac: |union(calls) ∩ evidence| / |evidence|  -- 1.0 means the model
        looked at ALL of the evidence, which is what "landed" means here.
    best_iou:     max IoU over calls, evidence floored to IOU_MIN_WIDTH.
    centre_offset: |centre(nearest call) - centre(evidence)| / duration.
    """
    ok = [c for c in calls if c.get("error") is None and c.get("end", 0) > c.get("start", 0)]
    out = {"n_calls": len(calls), "n_valid_calls": len(ok),
           "looked_frac": round(sum(c["end"] - c["start"] for c in ok) / max(duration, 1e-6), 4),
           "covered_frac": None, "landed": None, "best_iou": None,
           "centre_offset": None, "evidence": evidence}
    if not evidence or len(evidence) != 2 or evidence[0] is None or evidence[1] is None:
        return out
    e0, e1 = float(evidence[0]), float(evidence[1])
    if e1 < e0:
        e0, e1 = e1, e0
    # widen a tight/zero-width reference symmetrically for IoU only; coverage is
    # measured against the reference as given, floored to 1s so it is not degenerate
    ew = max(e1 - e0, IOU_MIN_WIDTH)
    c = (e0 + e1) / 2
    f0, f1 = c - ew / 2, c + ew / 2

    cov_lo, cov_hi = e0, max(e1, e0 + 1.0)
    covered = sum(max(0.0, min(x["end"], cov_hi) - max(x["start"], cov_lo)) for x in ok)
    out["covered_frac"] = round(min(covered / (cov_hi - cov_lo), 1.0), 4)
    out["landed"] = out["covered_frac"] >= 1.0
    if ok:
        out["best_iou"] = round(max(
            max(0.0, min(x["end"], f1) - max(x["start"], f0))
            / max(1e-6, max(x["end"], f1) - min(x["start"], f0)) for x in ok), 4)
        out["centre_offset"] = round(min(
            abs((x["start"] + x["end"]) / 2 - c) for x in ok) / max(duration, 1e-6), 4)
    return out


# ---------------------------------------------------------------------------
# one question

def _skim_content(row: dict, args) -> tuple[list, list, float, int]:
    """(content blocks, frame times, duration, n_frames) for the initial view.

    images: 64 JPEGs, no server-side pruning possible (vLLM prunes video only).
    video:  the source file by path, resampled server-side to --video-frames and
            pruned by whatever --video-pruning-method the server was started with.
            Passing the path (not base64) keeps the mp4 timeline intact, so the
            per-frame `<t seconds>` markers Qwen3-VL emits are REAL video times --
            re-encoding a resampled clip would silently rescale them.
    """
    dur = data.video_duration(row["video_path"])
    if args.skim_mode == "images":
        pils, times = tools.initial_frames_with_timestamps(
            row["video_path"], n_frames=args.frames)
        return ([{"type": "image_url", "image_url": {"url": _b64_image(p)}} for p in pils],
                times, dur, len(pils))
    n = args.frames
    src = row["video_path"]
    if args.proxy_dir:
        # Prefer the pre-built n-frame proxy. The server otherwise re-decodes a
        # median-412 MB, 1-hour mp4 on EVERY tool round (measured ~39 s per request,
        # >60 h for one arm). The proxy keeps the source timeline -- verified through
        # vLLM's own loader: 256 frames of a 7,547 s video come back at fps 0.0339
        # with timestamps 0.0 .. 7517.5 s -- so the `<t seconds>` markers the model
        # aims its crops from are unchanged. `make_skim_proxy.py` builds them.
        p = make_skim_proxy.proxy_path(args.proxy_dir, row["videoID"], n)
        if os.path.exists(p):
            src = p
        elif args.require_proxy:
            raise FileNotFoundError(f"no skim proxy for {row['videoID']}: {p}")
    url = "file://" + os.path.abspath(src)
    times = [(i + 0.5) * dur / n for i in range(n)]
    return ([{"type": "video_url", "video_url": {"url": url}}], times, dur, n)


_SCHEMA_SRC = {
    "longvt": config.LONGVT_TOOL_SCHEMAS,      # verbatim, incl. the unused video_path arg
    "informed": config.INFORMED_TOOL_SCHEMAS,  # accurate capacity + long-span behaviour
}


def schemas_for(args):
    src = _SCHEMA_SRC.get(args.prompt_style, config.TOOL_SCHEMAS)
    return [src[t] for t in args.tools] if args.tools else None


def build_first_turn(row: dict, args) -> tuple[list, list, float, int, str]:
    """Opening user turn. Returns (content, times, dur, n, prompt_text).

    style="longvt" reproduces LongVT/examples/eval/single_inference.py verbatim:
        "{question} Think first, call **crop_video** if needed, then answer. Format
         strictly as: <think>...</think> <tool_call>...</tool_call> (if needed)
         <answer>...</answer>. The Video path for this video is: {path}"
        No duration, no timestamp legend, no strategy hint -- LongVT-RFT was TRAINED
        to aim a crop from bare frames, so the reference prompt gives it nothing else.

    style="fastagent" adds the duration + exact per-frame times + a strategy line.
        Zero-shot Qwen3-VL is not that trained model; with image modality it has no
        time anchor at all, and measured GT coverage without the legend was 0.00 on 6/6.
        Kept as the switchable alternative so the legend's effect is measurable rather
        than assumed.
    """
    content, times, dur, n = _skim_content(row, args)
    if args.prompt_style == "informed":
        # Facts only: duration, tool capacity, long-span behaviour, round budget, and
        # that the protocol is turn-based. No search strategy -- see the note above
        # config.INFORMED_TOOL_SCHEMAS for why that boundary is load-bearing.
        # With no tools this style is meaningless (there is no environment to describe),
        # so it falls through to the same no-tool prompt the longvt style uses, keeping
        # run1/run3 byte-identical across styles and off the re-run list.
        if not args.tools:
            prompt = data.format_question(row)
            return (content + [{"type": "text", "text": prompt}], times, dur, n, prompt)
        prompt = config.informed_user_text(
            data.format_question(row), dur, args.max_rounds)
        return content + [{"type": "text", "text": prompt}], times, dur, n, prompt

    if args.prompt_style == "longvt":
        if not args.tools:
            # LongVT's own no-tool mode. A vanilla arm must NOT be handed the tool prompt:
            # until 2026-08-13 run1 was told "call **crop_video** if needed" and given a
            # video path while holding no tool, which is not a vanilla baseline.
            # the <think>/<answer> contract lives in a SYSTEM turn here, exactly as in
            # single_inference.py's no-tool branch; answer_one prepends it.
            prompt = data.format_question(row)
            return (content + [{"type": "text", "text": prompt}], times, dur, n, prompt)
        prompt = config.longvt_user_text(data.format_question(row), row["video_path"])
        # Opt-in additive, OFF by default in this style -- and IMAGES ONLY.
        # Video modality already carries per-frame `<t seconds>` markers, but vLLM
        # computes them as `frame_index / fps` and then AVERAGES them in pairs
        # (temporal_patch_size=2), giving 128 values like 14.7, 73.7, 132.7 ... A legend
        # computed here would say 15, 44, 74 ... -- two conflicting time labellings of
        # the same frames in one prompt, which is worse than either alone.
        if args.skim_timestamps and args.skim_mode == "images":
            prompt += "\n\n" + config.initial_view_text(dur, n, times)
        return content + [{"type": "text", "text": prompt}], times, dur, n, prompt

    # The exact-times legend exists because IMAGE modality carries no time marker.
    # Video modality already interleaves a real `<t seconds>` token before every
    # frame, so repeating 256 numbers would cost ~700 tokens to restate what the
    # model can already read -- and would compete with the native markers.
    legend = times if (args.skim_timestamps and args.skim_mode == "images") else None
    view = config.initial_view_text(dur, n, legend)
    if args.skim_mode == "video":
        view += (" Each frame above is labelled with its own timestamp in seconds. "
                 "ALL TIMES YOU PASS TO TOOLS MUST BE IN SECONDS (not m:ss).")
    prompt = (
        data.format_question(row)
        + "\n\n" + view
        + config.tool_instructions(dur, tuple(args.tools))
        + "\n\n" + config.ANSWER_INSTR
    )
    return content + [{"type": "text", "text": prompt}], times, dur, n, prompt


def _call(client, model, messages, schemas, args, max_tokens):
    """One chat completion, retried on transport failures.

    Measured on the first full run2/ctrl attempt: 2.8% and 13.5% of questions died on
    `APITimeoutError` / a 500, and the whole question was then written as pred=None --
    i.e. scored as WRONG. That silently biases the arm downward, and the resume logic
    would skip the row on a re-run because it is already present.

    The server was not overloaded when this happened (0 preemptions, max 3 waiting), so
    these are individual stalled requests, not queue pressure. Retrying the single call
    is therefore the fix; lowering concurrency would only have hidden it. The client
    timeout is deliberately short so a stall fails fast enough to retry inside the same
    question rather than burning 30 minutes.
    """
    last = None
    # Backoff reaches ~5 min in total: long enough to ride out a watchdog server restart
    # (SIGTERM, wait for the GPU to release, reload weights, capture CUDA graphs) instead
    # of failing every in-flight question the moment a server is bounced.
    for delay in (5, 15, 30, 60, 90, 120):
        try:
            return _call_once(client, model, messages, schemas, args, max_tokens)
        except Exception as e:                       # transport only; parse errors raise later
            last = e
            if type(e).__name__ not in ("APITimeoutError", "APIConnectionError",
                                        "InternalServerError", "APIStatusError"):
                raise
            time.sleep(delay)
    raise last


def _call_once(client, model, messages, schemas, args, max_tokens):
    extra = {"repetition_penalty": args.repetition_penalty,
             "include_stop_str_in_output": True}
    if args.top_k > 0:
        extra["top_k"] = args.top_k
    kw = {}
    if schemas:
        # LongVT's reference eval uses tool_choice="auto", which needs the server
        # started with --enable-auto-tool-choice --tool-call-parser hermes. Without
        # them vLLM rejects the request outright (400: '"auto" tool choice requires
        # --enable-auto-tool-choice'), so this is a serving prerequisite, not a nicety.
        kw["tools"] = schemas
        kw["tool_choice"] = "auto"
    return client.chat.completions.create(
        model=model, messages=messages,
        temperature=args.temperature, top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        max_tokens=max_tokens, seed=args.seed,
        stop=["</answer>"], extra_body=extra, **kw)


def _assistant_text(choice) -> tuple[str, dict | None, dict]:
    """(transcript text, parsed tool call or None, assistant message to append back).

    vLLM's hermes parser lifts `<tool_call>` out of `content` into `.tool_calls`, so
    `content` alone is NOT the model's output. The returned message mirrors LongVT's
    own loop (assistant turn carries `tool_calls`), while the transcript text re-inlines
    the call so `data.extract_answer` and the trajectory see one continuous response.
    """
    msg = choice.message
    text = msg.content or ""
    tc = None
    raw = list(msg.tool_calls or [])
    for t in raw:
        try:
            argd = json.loads(t.function.arguments or "{}")
        except json.JSONDecodeError:
            argd = {}
        text += f"\n<tool_call>\n{json.dumps({'name': t.function.name, 'arguments': argd})}\n</tool_call>"
        if tc is None:
            tc = {"name": t.function.name, "args": argd, "id": t.id}
    if tc is None:                                  # parser off / model wrote raw tags
        m = _TOOL_RE.search(text)
        if m:
            try:
                d = json.loads(m.group(1))
                tc = {"name": d.get("name", ""),
                      "args": d.get("arguments") or d.get("parameters") or {},
                      "id": None}
            except json.JSONDecodeError:
                tc = {"name": "__malformed__", "args": {}, "id": None}
    out = {"role": "assistant", "content": msg.content or ""}
    if raw:
        out["tool_calls"] = [{"id": t.id, "type": "function",
                              "function": {"name": t.function.name,
                                           "arguments": t.function.arguments}} for t in raw]
    else:
        out["content"] = text
    return text, tc, out


def oracle_span(row: dict, duration: float, mode: str) -> tuple[float, float] | None:
    """Forced crop region for run4. None when the question has no usable evidence.

    Same geometry as `oracle.py::oracle_spans` (which is the HF-side twin and cannot be
    imported here -- it pulls in the transformers model at module scope). Duplicated
    deliberately; keep the two in step.

    mode="oracle": the GT evidence span, widened to ORACLE_CROP_MIN_WIDTH so a tight or
        zero-width reference (263 of LVBench's are zero-width) still yields enough frames,
        and capped at the crop budget.
    mode="ctrl":   the SAME width placed as far from the evidence as the video allows.
        Without this control, "oracle beats run2" is equally explained by the extra visual
        tokens of *any* content -- it separates the right pixels from more pixels.
    """
    ev = row.get("evidence")
    if not ev or len(ev) != 2 or ev[0] is None or ev[1] is None:
        return None
    es, ee = float(ev[0]), float(ev[1])
    if ee < es:
        es, ee = ee, es
    w = min(max(ee - es, config.ORACLE_CROP_MIN_WIDTH), float(config.CROP_MAX_FRAMES))
    mid = 0.5 * (es + ee)
    xs, xe = max(0.0, mid - w / 2), min(duration, mid + w / 2)
    if mode != "ctrl":
        return (xs, xe)

    m_xs = max(0.0, min(duration - (xe - xs), duration - xe))
    m_xe = m_xs + (xe - xs)
    if not (m_xe <= xs or m_xs >= xe):          # mirror still overlaps -> go to an edge
        if xs >= duration - xe and xs >= (xe - xs):
            m_xs, m_xe = 0.0, xe - xs
        elif duration - xe >= (xe - xs):
            m_xs, m_xe = duration - (xe - xs), duration
        else:
            return None                          # video too short for a clean control
    return (m_xs, m_xe)


def answer_one(client, model: str, row: dict, args) -> dict:
    t_sample = time.time()
    tool_names = tuple(args.tools)
    schemas = schemas_for(args)

    t0 = time.time()
    content, skim_times, dur, n_frames, _ = build_first_turn(row, args)
    t_decode = time.time() - t0

    messages = [{"role": "user", "content": content}]
    if not tool_names and args.prompt_style == "longvt":
        messages.insert(0, {"role": "system", "content": config.LONGVT_SYSTEM_PROMPT})
    rounds, calls, texts = [], [], []
    n_exec = 0
    usage_tot = {"prompt_tokens": 0, "completion_tokens": 0}

    # run4: hand the evidence over BEFORE the model speaks, so localization is removed
    # from the loop entirely rather than merely graded. Tools stay available so the
    # affordances are identical to run2 and the only difference is the free crop.
    forced = None
    if args.crop_source != "model":
        forced = oracle_span(row, dur, args.crop_source)
    if forced:
        fs, fe = forced
        frames = tools.crop_frames(row["video_path"], fs, fe)
        calls.append({"round": -1, "name": "crop_video", "start": fs, "end": fe,
                      "error": None, "n_frames": len(frames), "forced": args.crop_source})
        messages.append({"role": "user", "content":
                         [{"type": "image_url", "image_url": {"url": _b64_image(p)}}
                          for p in frames]
                         + [{"type": "text", "text":
                             f"Cropped {fs:.0f}s-{fe:.0f}s, got {len(frames)} frames."}]})
        rounds.append({"round": "forced_crop", "text": "", "action":
                       {"kind": "forced_crop", "mode": args.crop_source,
                        "start": fs, "end": fe, "n_frames": len(frames)},
                       "tool_result": None, "prompt_tokens": None,
                       "completion_tokens": None, "finish_reason": None,
                       "gen_seconds": 0.0})

    def gen(label, use_tools=True, max_tokens=None):
        t = time.time()
        resp = _call(client, model, messages,
                     schemas if use_tools else None, args,
                     max_tokens or args.max_tokens)
        text, tc, amsg = _assistant_text(resp.choices[0])
        texts.append(text)
        messages.append(amsg)
        u = resp.usage
        usage_tot["prompt_tokens"] = max(usage_tot["prompt_tokens"],
                                         getattr(u, "prompt_tokens", 0) or 0)
        usage_tot["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
        return text, tc, {"round": label, "text": text,
                          "prompt_tokens": getattr(u, "prompt_tokens", None),
                          "completion_tokens": getattr(u, "completion_tokens", None),
                          "finish_reason": resp.choices[0].finish_reason,
                          "gen_seconds": round(time.time() - t, 2),
                          "action": None, "tool_result": None}

    overflowed = False
    for rnd in range(args.max_rounds + 1):
        try:
            text, tc, rec = gen(rnd)
        except Exception as e:
            # Context overflow is a legitimate end of the trajectory, not a failed
            # question. It only bites the wide-skim arms (run5's 12,150-token skim plus
            # five 6,016-token crops exceeds 40,960), and the old behaviour was fatal:
            # BadRequestError is not retryable, so the question was written as an error,
            # dropped by the queue, re-run, and failed identically forever.
            # Keep the answer the model can still give from the opening turn.
            if "maximum context length" not in str(e) and "Input length" not in str(e):
                raise
            overflowed = True
            messages[:] = messages[:1]
            rounds.append({"round": rnd, "text": "", "action": {"kind": "context_overflow"},
                           "tool_result": None, "prompt_tokens": None,
                           "completion_tokens": None, "finish_reason": "context_overflow",
                           "gen_seconds": 0.0})
            break
        has_answer = "<answer>" in text          # an answer wins over a same-turn call
        will_exec = (tc is not None and not has_answer
                     and tc["name"] in tool_names and n_exec < args.max_rounds)
        if not will_exec:
            rec["action"] = {"kind": "answer" if has_answer else "stop"}
            rounds.append(rec)
            break

        s, e, err = tools.clamp_span(tc["args"].get("start_time"),
                                     tc["args"].get("end_time"), dur)
        rec["action"] = {"kind": "tool_call", "name": tc["name"],
                         "start": s, "end": e, "error": err}
        calls.append({"round": rnd, "name": tc["name"], "start": s, "end": e, "error": err})

        def reply(blocks):
            """Tool result turn. LongVT answers with role="tool" + tool_call_id; that
            needs an id, which only exists when the hermes parser produced the call.
            A regex-recovered call has none, so fall back to a user turn wrapped in the
            same <tool_response> tags the chat template would have emitted anyway."""
            if tc.get("id"):
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": blocks})
            else:
                messages.append({"role": "user", "content":
                                 [{"type": "text", "text": "<tool_response>\n"}] + blocks
                                 + [{"type": "text", "text": "\n</tool_response>"}]})

        if err:
            rounds.append(rec)
            reply([{"type": "text", "text": err}])
            continue

        t_tool = time.time()
        frames = tools.crop_frames(row["video_path"], s, e)
        rec["tool_result"] = {"tool": tc["name"], "span": [s, e], "n_frames": len(frames)}
        rec["tool_seconds"] = round(time.time() - t_tool, 2)
        calls[-1]["n_frames"] = len(frames)
        n_exec += 1
        rounds.append(rec)
        reply([{"type": "image_url", "image_url": {"url": _b64_image(p)}} for p in frames]
              + [{"type": "text", "text":
                  f"Cropped {s:.0f}s-{e:.0f}s, got {len(frames)} frames."}])

    pred_strict = data.extract_answer("\n".join(texts))
    finalizer_used = False
    if pred_strict is None:
        finalizer_used = True
        # The finalizer has to FORCE a letter, not merely request one. Measured on the
        # first probes: 25% of questions end in a genuine refusal ("None of the above",
        # finish_reason=stop, 5 tokens) because a crop that landed in the wrong place
        # hands the model confident negative evidence -- it concludes the answer is not
        # in the video and declines. The no-tool baseline never sees that evidence, so
        # it guesses and scores ~25% on those. Leaving the refusals in would charge the
        # tool arm for a behaviour the control was never exposed to.
        # `finalizer_used` and `pred_strict` keep the refusal itself measurable.
        messages.append({"role": "user", "content": [{"type": "text", "text":
            "You must commit to one of the options even if the frames you inspected "
            "were inconclusive; guess if necessary. " + run_plain.OFFICIAL_INSTR}]})
        fin_text, _tc, rec = gen("finalizer", use_tools=False,
                                 max_tokens=args.finalizer_max_tokens)
        rec["action"] = {"kind": "finalizer"}
        rounds.append(rec)

    # Score the finalizer's OWN text first. Scoring the concatenation lets an earlier
    # `<answer>None of the above</answer>` trip extract_answer's refusal branch and
    # return None even when the finalizer replied with a clean letter -- that alone
    # accounted for every remaining no-answer on the probe (finalizers were 'C','B',
    # 'A','A'). The finalizer turn is by construction the model's last word.
    pred = None
    if finalizer_used:
        pred = data.extract_answer(fin_text)
    if pred is None:
        pred = data.extract_answer("\n".join(texts))
    loc = _span_metrics(calls, row.get("evidence"), dur)
    return {
        "question_id": str(row["question_id"]), "task_type": row["task_type"],
        "videoID": row["videoID"], "duration": dur,
        "pred": pred, "pred_strict": pred_strict, "gold": row["answer"],
        "correct": pred == row["answer"], "correct_strict": pred_strict == row["answer"],
        "finalizer_used": finalizer_used,
        "n_frames": n_frames, "rounds": len(texts),
        "tool_calls": calls, **{k: v for k, v in loc.items() if k != "evidence"},
        "evidence": row.get("evidence"),
        "time_reference": row.get("time_reference"),
        "localization_trivial": row.get("localization_trivial"),
        "prompt_tokens": usage_tot["prompt_tokens"],
        "completion_tokens": usage_tot["completion_tokens"],
        "decode_seconds": round(t_decode, 2),
        "seconds": round(time.time() - t_sample, 1),
        "_rounds": rounds,
    }


# ---------------------------------------------------------------------------

def write_summary(run_dir: str) -> dict:
    by_task, total = defaultdict(lambda: [0, 0]), [0, 0]
    none_pred = n_called = n_landed = 0
    land = [0, 0]      # correct, n   among landed
    miss = [0, 0]      # correct, n   among not-landed
    cov_sum = cov_n = 0.0
    with open(os.path.join(run_dir, "results.jsonl")) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ok = bool(r.get("correct"))
            by_task[r.get("task_type", "?")][0] += ok
            by_task[r.get("task_type", "?")][1] += 1
            total[0] += ok
            total[1] += 1
            none_pred += r.get("pred") is None
            n_called += bool(r.get("n_valid_calls"))
            if r.get("covered_frac") is not None:
                cov_sum += r["covered_frac"]
                cov_n += 1
                (land if r.get("landed") else miss)[0] += ok
                (land if r.get("landed") else miss)[1] += 1
                n_landed += bool(r.get("landed"))
    n = max(total[1], 1)
    s = {"n": total[1], "accuracy": round(100 * total[0] / n, 2), "no_answer": none_pred,
         "questions_with_a_call": n_called,
         "localization": {
             "scored": int(cov_n),
             "mean_coverage": round(cov_sum / max(cov_n, 1), 4),
             "hit_rate": round(100 * n_landed / max(cov_n, 1), 2),
             "acc_landed": round(100 * land[0] / max(land[1], 1), 2), "n_landed": land[1],
             "acc_not_landed": round(100 * miss[0] / max(miss[1], 1), 2), "n_not_landed": miss[1]},
         "by_task": {t: {"correct": c, "n": m, "acc": round(100 * c / max(m, 1), 1)}
                     for t, (c, m) in sorted(by_task.items())}}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lvbench", choices=["lvbench", "videomme"])
    ap.add_argument("--num", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=time.strftime("%m%d"))
    ap.add_argument("--tools", nargs="*", default=["crop_video"],
                    choices=list(config.TOOL_SCHEMAS))
    ap.add_argument("--skim-mode", dest="skim_mode", default="images",
                    choices=["images", "video"],
                    help="images = 64 JPEGs (no server-side pruning is possible); "
                         "video = source file by path, resampled+pruned server-side")
    ap.add_argument("--frames", type=int, default=config.INITIAL_FRAMES)
    ap.add_argument("--proxy-dir", dest="proxy_dir", default=make_skim_proxy.DEFAULT_DIR,
                    help="pre-built n-frame skim proxies (--skim-mode video only)")
    ap.add_argument("--require-proxy", dest="require_proxy", action="store_true",
                    help="fail instead of silently falling back to the 62 GB source")
    ap.add_argument("--prompt-style", dest="prompt_style", default="longvt",
                    choices=["longvt", "informed", "fastagent"],
                    help="longvt = LongVT/examples/eval/single_inference.py verbatim; "
                         "informed = longvt's format contract + environment FACTS "
                         "(duration, tool capacity, round budget, turn-based protocol) "
                         "and no search strategy; "
                         "fastagent = adds duration + per-frame timestamp legend")
    ap.add_argument("--skim-timestamps", dest="skim_timestamps", type=int, default=0,
                    help="append the exact per-frame time legend (LongVT does NOT); "
                         "ignored for --skim-mode video, which carries native markers")
    ap.add_argument("--max-rounds", dest="max_rounds", type=int, default=config.MAX_ROUNDS)
    ap.add_argument("--crop-source", dest="crop_source", default="model",
                    choices=["model", "oracle", "ctrl"],
                    help="model = the model aims its own crops (run1-run3); "
                         "oracle = a free crop of the GT evidence span is injected before "
                         "the first turn (run4, the unbiased ceiling); "
                         "ctrl = same-width crop placed AWAY from the evidence")
    # identical to the vanilla baseline except max_tokens: 32 cannot hold a tool call
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.8)
    ap.add_argument("--top-k", dest="top_k", type=int, default=20)
    ap.add_argument("--presence-penalty", dest="presence_penalty", type=float, default=0.0)
    ap.add_argument("--frequency-penalty", dest="frequency_penalty", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", dest="repetition_penalty", type=float, default=1.0)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=1024)
    ap.add_argument("--finalizer-max-tokens", dest="finalizer_max_tokens", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out-root", default="/local1/cfyang/hanklin/outputs/lvbench_agent")
    ap.add_argument("--print-prompt", action="store_true",
                    help="render turn 1 for the first question and exit -- no model call")
    args = ap.parse_args()

    rows = data.load_dataset(args.dataset, n=args.num, seed=args.seed)

    if args.print_prompt:
        _c, times, dur, n, prompt = build_first_turn(rows[0], args)
        kinds = defaultdict(int)
        for b in _c:
            kinds[b["type"]] += 1
        print("=" * 78)
        print(f"question_id={rows[0]['question_id']}  video={rows[0]['videoID']}  "
              f"duration={dur:.0f}s  evidence={rows[0].get('evidence')}")
        print(f"turn-1 content blocks: {dict(kinds)}   skim_mode={args.skim_mode} "
              f"frames={n}   prompt_style={args.prompt_style}")
        print("=" * 78)
        print(prompt)
        print("=" * 78)
        print("# Tools block (native schema):")
        print(json.dumps(schemas_for(args), indent=1))
        return

    from openai import OpenAI
    # 300 s is ~30x the median round; a request slower than that is stalled, not busy.
    # Failing fast lets _call retry it inside the question instead of losing the question.
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=300, max_retries=0)

    run_dir = os.path.join(args.out_root,
                           f"{args.dataset}_{args.tag}_seed{args.seed}")
    os.makedirs(os.path.join(run_dir, "traj"), exist_ok=True)
    results_path = os.path.join(run_dir, "results.jsonl")

    manifest = {k: v for k, v in vars(args).items()}
    manifest.update({"arm": "agent", "max_pixels": config.MAX_PIXELS,
                     "crop_max_frames": config.CROP_MAX_FRAMES,
                     "unify_time_format": data.UNIFY_TIME_FORMAT,
                     "launched": time.strftime("%Y-%m-%d %H:%M:%S")})
    with open(os.path.join(run_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    done = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                try:
                    done.add(str(json.loads(line)["question_id"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    todo = [r for r in rows if str(r["question_id"]) not in done]
    print(f"[agent] {args.dataset} n={len(rows)} todo={len(todo)} (skip {len(done)}) "
          f"tools={args.tools} skim={args.skim_mode}/{args.frames} "
          f"T={args.temperature} top_p={args.top_p} top_k={args.top_k} "
          f"pp={args.presence_penalty} max_tok={args.max_tokens} seed={args.seed}\n"
          f"        -> {run_dir}", flush=True)
    if done:
        print("        NOTE: resuming. On a fresh run a nonzero skip means the "
              "directory is contaminated -- stop and purge it.", flush=True)

    lock = threading.Lock()
    t0, n_done = time.time(), [0]

    def work(row):
        try:
            r = answer_one(client, args.model, row, args)
        except Exception as e:
            r = {"question_id": str(row["question_id"]), "task_type": row.get("task_type"),
                 "pred": None, "gold": row["answer"], "correct": False,
                 "error": f"{type(e).__name__}: {e}", "_rounds": []}
        rounds = r.pop("_rounds", [])
        qid = r["question_id"]
        tp = os.path.join(run_dir, "traj", f"{qid}.json")
        with open(tp, "w") as tf:
            json.dump({"question_id": qid, "question": row["question"],
                       "options": row["options"], "gold": row["answer"],
                       "pred": r.get("pred"), "pred_strict": r.get("pred_strict"),
                       "videoID": row["videoID"], "duration": r.get("duration"),
                       "evidence": row.get("evidence"),
                       "time_reference": row.get("time_reference"),
                       "tool_calls": r.get("tool_calls"),
                       "covered_frac": r.get("covered_frac"),
                       "landed": r.get("landed"), "best_iou": r.get("best_iou"),
                       "rounds": rounds}, tf, indent=1)
        r["traj_path"] = tp
        with lock:
            with open(results_path, "a") as rf:
                rf.write(json.dumps(r) + "\n")
            n_done[0] += 1
            s = write_summary(run_dir)
            el = time.time() - t0
            cf = r.get("covered_frac")
            print(f"[{n_done[0]}/{len(todo)}] {qid} pred={r.get('pred')} gold={r['gold']} "
                  f"{'OK ' if r.get('correct') else 'X  '}| calls={r.get('n_valid_calls')} "
                  f"cov={cf if cf is None else f'{cf:.2f}'} "
                  f"| acc {s['accuracy']}% hit {s['localization']['hit_rate']}% n={s['n']} "
                  f"| {el / max(n_done[0], 1):.1f}s/q"
                  + (f" | ERR {r['error'][:60]}" if r.get("error") else ""), flush=True)
        return r

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))

    s = write_summary(run_dir)
    L = s["localization"]
    print(f"\n[SUMMARY] accuracy {s['accuracy']}% over n={s['n']} "
          f"| no-answer {s['no_answer']} | {time.time() - t0:.0f}s total")
    print(f"  questions with >=1 valid call : {s['questions_with_a_call']}/{s['n']}")
    print(f"  mean GT-evidence coverage     : {L['mean_coverage']:.3f}")
    print(f"  landed (coverage == 1.0)      : {L['hit_rate']}%  ({L['n_landed']}/{L['scored']})")
    print(f"  accuracy | landed             : {L['acc_landed']}%  (n={L['n_landed']})")
    print(f"  accuracy | NOT landed         : {L['acc_not_landed']}%  (n={L['n_not_landed']})")
    for t, d in s["by_task"].items():
        print(f"  {t:<28} {d['correct']:>3}/{d['n']:<3} = {d['acc']}%")


if __name__ == "__main__":
    main()
