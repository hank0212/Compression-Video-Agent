"""Run the ACTUAL trained LongVT agent (LongVT-RFT, Qwen2.5-VL-7B) on VideoMME-long
samples via its native crop_video tool loop, recording full trajectories + montages.

This is the *trained tool-calling* reference (SFT->RL->RFT), NOT the zero-shot
Qwen3-VL fast_agent. It talks to a vLLM server over the OpenAI API; tool outputs
cross the boundary as base64 frames (pixels), exactly as single_inference.py /
compress_smoke.py do. Because pixels are all the tool boundary carries, "compression"
here is the PIXEL-SPACE coverage stand-in (downscale-for-coverage), NOT real
ViT-internal FlashVID -- that needs in-process HF (track 2), see RESEARCH.md 2026-07-08.

Reuses fast_agent for: proxy-safe AV1 decode (tools), montage/trajectory writing
(trajectory), and sample selection (data.load_long_split, seed 0) -- so the 10
samples are a SUBSET of the fast_agent run2 set and directly cross-comparable to the
Qwen3-VL numbers.

Two arms on the SAME samples:
  crop     = the real LongVT agent, exactly as trained (crop_video only)
  compress = crop + pixel-space compress_video coverage stand-in (does the trained
             model adopt a brand-new tool zero-shot, and does it help?)

Usage (from longvt_compression/):
  python -m longvt_agent.run_longvt --n 10 --arms crop,compress --tag demo \
         --base-url http://localhost:8000/v1
"""

import argparse
import base64
import glob
import json
import os
import time
from collections import defaultdict
from io import BytesIO

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image

from fast_agent import data as fadata
from fast_agent import tools as fatools
from fast_agent import trajectory as tj

# ---- budgets (LongVT-faithful: fps=1, 224px crop; global capped to fit 32k / 600-image server) ----
GLOBAL_FRAMES = 192          # whole-video skim (capped: 512 would fill the 32k context alone)
CROP_MAX_FRAMES = 128        # crop_video full-detail budget (LongVT default)
CROP_MAX_PIXELS = 224 * 224
COMPRESS_MAX_FRAMES = 256    # pixel-space coverage stand-in: more frames...
COMPRESS_MAX_PIXELS = 112 * 112  # ...at 1/4 the per-frame tokens
MAX_TOOL_ROUNDS = 4
MAX_TOKENS = 2048
OUTPUT_ROOT = "/local1/cfyang/hanklin/outputs/longvt_agent/runs"

# ---- exact schemas / prompts the released checkpoint was trained to call ----
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

ARM_TOOLS = {
    "crop": [CROP_VIDEO_TOOL],
    "compress": [COMPRESS_VIDEO_TOOL, CROP_VIDEO_TOOL],
}
ARM_PROMPT = {"crop": TOOL_PROMPT, "compress": TOOL_PROMPT_BOTH}


# ---------------------------------------------------------------------------
def decode_frames(video_path, start, end, max_frames, max_pixels):
    """fps=1 span decode -> list[PIL], proxy/pyav-safe (reuses fast_agent.tools)."""
    src = fatools.resolve_decodable(video_path)
    cap = cv2.VideoCapture(src)
    try:
        fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_v = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    finally:
        cap.release()
    dur = n_v / fps_v if fps_v > 0 else 0.0
    s = 0.0 if start is None else max(0.0, min(start, dur))
    e = dur if end is None else max(s, min(end, dur))
    span = max(e - s, 1e-6)
    n = max(2, min(int(round(span)), max_frames))       # fps=1
    centers = s + (np.arange(n) + 0.5) * span / n
    th, tw = fatools._smart_size(h0 or 224, w0 or 224, max_pixels=max_pixels)
    frames = fatools._decode_cv2(src, centers, fps_v, n_v, th, tw)
    if len(frames) < 2:
        frames = fatools._decode_pyav(video_path, centers, th, tw)
    return [Image.fromarray(f) for f in frames]


def pils_to_b64(images):
    out = []
    for img in images:
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        out.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return out


def exec_tool(name, video_path, s, e):
    if name == "compress_video":
        pil = decode_frames(video_path, s, e, COMPRESS_MAX_FRAMES, COMPRESS_MAX_PIXELS)
    else:  # crop_video
        pil = decode_frames(video_path, s, e, CROP_MAX_FRAMES, CROP_MAX_PIXELS)
    return pils_to_b64(pil), pil


# ---------------------------------------------------------------------------
def run_sample(client, model, row, arm, record_dir, verbose=False):
    qid = str(row["question_id"])
    video = row["video_path"]
    media = os.path.join(record_dir, "media", qid)
    tools = ARM_TOOLS[arm]
    dispatch = {t["function"]["name"] for t in tools}

    t_sample = time.time()
    dur = fadata.video_duration(video)
    gpils = decode_frames(video, None, None, GLOBAL_FRAMES, CROP_MAX_PIXELS)
    gb64 = pils_to_b64(gpils)
    initial_montage = tj.save_montage(gpils, os.path.join(media, "initial.png"))

    q = fadata.format_question(row)
    prompt = f"{q}\n{ARM_PROMPT[arm]} The Video path for this video is: {video}"
    msgs = [{"role": "user", "content": gb64 + [{"type": "text", "text": prompt}]}]

    rounds, texts, calls = [], [], []
    for rnd in range(MAX_TOOL_ROUNDS + 1):
        withhold = rnd == MAX_TOOL_ROUNDS      # last turn: force an answer (no tools)
        kw = dict(model=model, messages=msgs, max_tokens=MAX_TOKENS, temperature=0)
        if not withhold:
            kw.update(tools=tools, tool_choice="auto")
        t0 = time.time()
        r = client.chat.completions.create(**kw)
        m = r.choices[0].message
        fr = r.choices[0].finish_reason
        content = m.content or ""
        texts.append(content)
        rec = {"round": rnd, "gen_seconds": round(time.time() - t0, 2),
               "finish_reason": fr, "thinking": content,
               "action": None, "tool_results": []}
        if verbose:
            print(f"  [{arm} R{rnd}] fr={fr} :: {content[:200]}")

        if fr == "tool_calls" and m.tool_calls and not withhold:
            msgs.append({"role": "assistant", "content": content,
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name,
                                                      "arguments": tc.function.arguments}}
                                        for tc in m.tool_calls]})
            for i, tc in enumerate(m.tool_calls):
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                s, e, err = fatools.clamp_span(args.get("start_time"), args.get("end_time"), dur)
                if name not in dispatch:
                    msgs.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": [{"type": "text", "text": f"Unknown tool {name}."}]})
                    rec["tool_results"].append({"tool": name, "error": "unknown_tool"})
                    continue
                if err:
                    msgs.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": [{"type": "text", "text": err}]})
                    rec["tool_results"].append({"tool": name, "span": [s, e], "error": err})
                    calls.append({"name": name, "start": s, "end": e, "error": err, "round": rnd})
                    continue
                b64, pil = exec_tool(name, video, s, e)
                mont = tj.save_montage(pil, os.path.join(media, f"r{rnd}_{i}_{name}.png"))
                note = f"{name} {s:.0f}s-{e:.0f}s, got {len(b64)} frames."
                msgs.append({"role": "tool", "tool_call_id": tc.id,
                             "content": b64 + [{"type": "text", "text": note}]})
                calls.append({"name": name, "start": s, "end": e, "n_frames": len(pil), "round": rnd})
                rec["tool_results"].append({"tool": name, "span": [s, e],
                                            "n_frames": len(pil), "montage": mont})
            first = rec["tool_results"][0]
            rec["action"] = {"kind": "tool_call", "name": first["tool"],
                             "span": first.get("span"),
                             "n_calls": len(m.tool_calls)}
            rounds.append(rec)
        else:
            rec["action"] = {"kind": "answer" if "<answer>" in content else "stop"}
            rounds.append(rec)
            break

    pred = fadata.extract_answer("\n".join(texts))
    gold = row["answer"]
    seconds = round(time.time() - t_sample, 1)
    traj = {"question_id": qid, "task_type": row["task_type"],
            "question": row["question"], "options": list(row["options"]),
            "gold": gold, "pred": pred, "correct": pred == gold,
            "videoID": row["videoID"], "video_path": video, "duration": dur,
            "arm": arm, "tools": sorted(dispatch), "model": model,
            "seconds": seconds, "n_rounds": len(texts),
            "tool_calls": calls, "schema_version": 1,
            "initial": {"n_frames": len(gpils), "montage": initial_montage},
            "rounds": rounds}
    traj_path = tj.save_trajectory(traj, os.path.join(record_dir, "traj", f"{qid}.json"))
    return {"question_id": qid, "task_type": row["task_type"], "arm": arm,
            "pred": pred, "gold": gold, "correct": pred == gold,
            "n_rounds": len(texts), "tool_calls": calls,
            "seconds": seconds, "traj_path": traj_path}


def write_summary(run_dir, arm):
    by_task, total = defaultdict(lambda: [0, 0]), [0, 0]
    tool_hist = defaultdict(int)
    used_any = 0
    for p in glob.glob(os.path.join(run_dir, "results.jsonl")):
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                ok = bool(r.get("correct"))
                by_task[r["task_type"]][0] += ok
                by_task[r["task_type"]][1] += 1
                total[0] += ok
                total[1] += 1
                tcs = r.get("tool_calls", [])
                used_any += bool(tcs)
                for c in tcs:
                    tool_hist[c["name"]] += 1
    n = max(total[1], 1)
    summary = {"arm": arm, "n": total[1],
               "accuracy": round(100 * total[0] / n, 2),
               "tool_adoption_rate": round(100 * used_any / n, 1),
               "by_task": {t: {"correct": c, "n": nt, "acc": round(100 * c / max(nt, 1), 1)}
                           for t, (c, nt) in sorted(by_task.items())},
               "tool_calls": dict(tool_hist)}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="crop,compress")
    ap.add_argument("--tag", default="demo")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

    client = OpenAI(api_key="EMPTY", base_url=args.base_url)
    model = client.models.list().data[0].id
    print(f"[longvt] model={model} base={args.base_url}")

    rows = fadata.load_long_split(n=args.n, seed=args.seed)
    print(f"[longvt] {len(rows)} samples (seed={args.seed}); qids: "
          f"{[r['question_id'] for r in rows]}")

    for arm in args.arms.split(","):
        arm = arm.strip()
        run_dir = os.path.join(OUTPUT_ROOT, f"{arm}_{args.tag}")
        os.makedirs(run_dir, exist_ok=True)
        results_path = os.path.join(run_dir, "results.jsonl")
        done = set()
        if os.path.exists(results_path):
            with open(results_path) as f:
                done = {json.loads(l)["question_id"] for l in f if l.strip()}
        todo = [r for r in rows if str(r["question_id"]) not in done]
        with open(os.path.join(run_dir, "run_manifest.json"), "w") as mf:
            json.dump({"arm": arm, "tag": args.tag, "n": args.n, "seed": args.seed,
                       "model": model, "global_frames": GLOBAL_FRAMES,
                       "crop_max_frames": CROP_MAX_FRAMES,
                       "compress_max_frames": COMPRESS_MAX_FRAMES,
                       "compress_max_pixels": COMPRESS_MAX_PIXELS,
                       "max_tool_rounds": MAX_TOOL_ROUNDS,
                       "max_tokens": MAX_TOKENS}, mf, indent=1)
        print(f"\n[longvt] === arm={arm} -> {run_dir}  todo={len(todo)} ===")
        with open(results_path, "a") as f:
            for i, row in enumerate(todo):
                t = time.time()
                try:
                    r = run_sample(client, model, row, arm, run_dir, verbose=args.verbose)
                except Exception as ex:
                    import traceback
                    traceback.print_exc()
                    r = {"question_id": str(row["question_id"]), "task_type": row["task_type"],
                         "arm": arm, "pred": None, "gold": row["answer"], "correct": False,
                         "n_rounds": 0, "tool_calls": [], "seconds": 0,
                         "error": f"{type(ex).__name__}: {ex}"}
                f.write(json.dumps(r) + "\n")
                f.flush()
                s = write_summary(run_dir, arm)
                print(f"  [{i+1}/{len(todo)}] {row['question_id']} pred={r.get('pred')} "
                      f"gold={r['gold']} ok={r.get('correct')} rounds={r.get('n_rounds')} "
                      f"tools={[c['name'] for c in r.get('tool_calls', [])]} "
                      f"{time.time()-t:.0f}s | acc {s['accuracy']}% n={s['n']}")
        s = write_summary(run_dir, arm)
        print(f"[SUMMARY {arm}/{args.tag}] acc {s['accuracy']}% over n={s['n']} | "
              f"tool_adoption {s['tool_adoption_rate']}% | calls {s['tool_calls']}")


if __name__ == "__main__":
    main()
