"""Build the LongVT interleaved-tool-calling explainer notebook."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md(r"""# LongVT — Interleaved Multimodal Chain-of-Tool-Thought (iMCoTT), by example

**Paper:** *LongVT: Incentivizing Thinking with Long Videos via Native Tool Calling* (CVPR 2026, EvolvingLMMs-Lab).
**Model here:** `longvideotool/LongVT-RFT` — a **Qwen2.5-VL-7B** fine-tuned (SFT → RL → RFT) to *natively call a `crop_video` tool* while reasoning.

## The one idea
A long video has thousands of frames. If you sample uniformly, you either blow up the context or miss the
2-second moment that answers the question. LongVT instead teaches the model to **reason global-to-local**:

1. **Skim** the whole video at low fps / low resolution (a coarse overview).
2. **Think** about where the answer probably lives.
3. **Call `crop_video(start, end)`** to *zoom into* that time window and get fresh, denser frames.
4. **Think again** on the zoomed-in frames, and either crop again or **answer**.

The trace is *interleaved*: `think → tool_call → (new frames) → think → … → answer`. That interleaving of
**text reasoning** and **freshly-retrieved visual evidence** is the "Multimodal Chain-of-Tool-Thought".

This notebook runs the released checkpoint on one real Video-MME question and **shows every step**: the frames
the model saw, what it decided to crop, and the reasoning it produced.
""")

md(r"""## 0. Prerequisite: the model server must be running

We serve the checkpoint with vLLM (tool-calling enabled). From `demo/`:

```bash
bash serve_longvt.sh 5 8000      # GPU 5, port 8000
```

The cell below just checks it's up.""")

code(r"""import os, json, textwrap
from openai import OpenAI

API_BASE = "http://localhost:8000/v1"
os.environ["no_proxy"] = "localhost,127.0.0.1"
client = OpenAI(api_key="EMPTY", base_url=API_BASE)
MODEL = client.models.list().data[0].id
print("Server up. Serving model:", MODEL)""")

md(r"""## 1. The question

A real Video-MME (medium split) **OCR** question. OCR questions are a great demo because the answer is a piece of
on-screen text that is unreadable at 224px overview resolution — the model is *forced* to crop in and zoom to read it.""")

code(r"""VIDEO = "/local1/cfyang/.cache/huggingface/videomme/videomme/data/kSBB5PsRV-k.mp4"
QUESTION = "When the video talks about sea level rise, what is the time span?"
OPTIONS = ["A. 1300 AD - 2016 AD.", "B. 1200 BC - 2019 AD.", "C. 1300 BC - 2016 AD.", "D. 1200 AD - 2019 AD."]
GT = "C"
OPTIONS_TEXT = "\n".join(OPTIONS)
print(QUESTION); print(OPTIONS_TEXT); print("Ground truth:", GT)""")

md(r"""## 2. The tool and the sampling settings

These are copied verbatim from the paper's `single_inference.py` (via `longvt_tools.py`):

- **Global skim:** `fps=1`, up to **512** frames, **224×224** px.
- **`crop_video(start,end)`:** re-samples *only that window* at `fps=1`, up to **128** frames, 224px — i.e. a
  window that was 1 frame in the overview now gets many frames. That's the "zoom".
- The tool schema below is the exact function signature the checkpoint was trained to emit.""")

code(r"""from longvt_tools import encode_global, crop_video, CROP_VIDEO_TOOL, TOOL_PROMPT
print("Tool schema the model calls:")
print(json.dumps(CROP_VIDEO_TOOL, indent=2))
print("\nInstruction appended to the question:\n", TOOL_PROMPT)""")

md(r"""## 3. A tiny renderer

Helper to montage a list of frames inline so we can see exactly what the model saw at each step.""")

code(r"""import numpy as np
import matplotlib.pyplot as plt

def show_montage(frames, n=16, title=""):
    if not frames:
        print("(no frames)"); return
    if len(frames) > n:
        idx = np.linspace(0, len(frames)-1, n).astype(int)
        sel = [frames[i] for i in idx]
        cap = f"{title}  (showing {n} of {len(frames)} frames, evenly sampled)"
    else:
        sel = frames; cap = f"{title}  ({len(frames)} frames)"
    cols = min(8, len(sel)); rows = (len(sel)+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.7, rows*1.7))
    axes = np.array(axes).reshape(-1)
    for ax in axes: ax.axis("off")
    for ax, im in zip(axes, sel):
        ax.imshow(im)
    fig.suptitle(cap, fontsize=11)
    plt.tight_layout(); plt.show()""")

md(r"""## 4. Step 0 — the global skim

Decode the whole video at 1 fps / 224px. This is the *only* visual context the model starts with; everything
finer comes from tool calls. Notice how low-res it is — you cannot read small on-screen text here.""")

code(r"""global_b64, global_pil = encode_global(VIDEO)
print(f"Global skim: {len(global_pil)} frames @1fps, 224px  (video is ~{len(global_pil)}s long)")
show_montage(global_pil, n=16, title="Global skim (whole video @1fps/224px)")""")

md(r"""## 5. The interleaved loop

This is the heart of LongVT. Each iteration:
1. send the running conversation to the model (with the `crop_video` tool available),
2. if it emits a `<tool_call>`, **execute the crop** and feed the new frames back as a `tool` message,
3. otherwise it produced an `<answer>` and we stop.

We instrument the loop to record every event (assistant reasoning, each crop, and the frames each crop returned).""")

code(r"""def run_interleaved(video_path, question, options_text, max_rounds=5, verbose=True):
    user_text = (f"{question}\n{options_text}\n{TOOL_PROMPT} "
                 f"The Video path for this video is: {video_path}")
    messages = [{"role": "user", "content": global_b64 + [{"type": "text", "text": user_text}]}]
    events = [{"kind": "global", "frames": global_pil}]

    for rnd in range(max_rounds + 1):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages,
            tools=[CROP_VIDEO_TOOL], tool_choice="auto",
            max_tokens=1024, temperature=0,
        )
        msg = resp.choices[0].message
        finish = resp.choices[0].finish_reason
        events.append({"kind": "assistant", "round": rnd, "text": msg.content or "", "finish": finish})
        if verbose:
            print(f"\n===== ROUND {rnd}  (finish_reason={finish}) =====")
            print(textwrap.fill((msg.content or "").strip(), 100))

        if finish == "tool_calls" and msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": tc.id, "type": "function",
                                             "function": {"name": tc.function.name,
                                                          "arguments": tc.function.arguments}}
                                            for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                if verbose:
                    print(f"  >>> TOOL CALL: crop_video(start={args.get('start_time')}, end={args.get('end_time')})")
                b64, pil = crop_video(args["video_path"], args["start_time"], args["end_time"])
                events.append({"kind": "tool", "round": rnd, "args": args, "frames": pil})
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": b64 + [{"type": "text",
                                     "text": f"Cropped {args['start_time']}s-{args['end_time']}s, got {len(b64)} frames."}]})
        else:
            break
    return events

events = run_interleaved(VIDEO, QUESTION, OPTIONS_TEXT)""")

md(r"""## 6. The interleaved trace, visualized

Now replay the recorded events *in order* — the reasoning text interleaved with the actual frames the crop tool
fetched. Watch the resolution: the crop frames cover a short window, so on-screen text becomes legible.""")

code(r"""def render(events):
    for ev in events:
        if ev["kind"] == "global":
            print("STEP 0 — GLOBAL SKIM")
            show_montage(ev["frames"], n=12, title="what the model starts with")
        elif ev["kind"] == "assistant":
            tag = "FINAL ANSWER TURN" if ev["finish"] != "tool_calls" else "THINK"
            print(f"\n--- ROUND {ev['round']}: {tag} ---")
            print(textwrap.fill(ev["text"].strip(), 100))
        elif ev["kind"] == "tool":
            a = ev["args"]
            print(f"\n     zoomed in -> crop_video({a['start_time']}s .. {a['end_time']}s) "
                  f"returned {len(ev['frames'])} frames:")
            show_montage(ev["frames"], n=12, title=f"crop [{a['start_time']}s, {a['end_time']}s]")

render(events)

# final answer vs ground truth
final = events[-1]["text"]
import re
m = re.search(r"<answer>(.*?)</answer>", final, re.S)
pred = (m.group(1).strip() if m else final.strip())
print("\n" + "="*60)
print("PREDICTED:", pred[:200])
print("GROUND TRUTH:", GT)
n_crops = sum(1 for e in events if e["kind"] == "tool")
print(f"# crop_video calls: {n_crops}")""")

md(r"""## 7. Contrast: the same model, tool disabled ("skim-only")

Ask the same question but forbid cropping — the model must answer from the blurry overview alone.

**Read the result honestly:** on an *easy* question the model may still get it right from the skim (or from
world-knowledge priors — note how its Round-0 `<think>` above started with *"if I remember correctly…"* and only
*then* cropped to **verify** by reading the on-screen scale). So here the tool acts as **visual grounding /
verification**. The tool becomes *decisive* on questions the model cannot answer from a blurry overview or priors —
fine-grained counting, small text, exact temporal ordering — where the skim-only answer degrades to a guess.""")

code(r"""user_text = f"{QUESTION}\n{OPTIONS_TEXT}\nAnswer with the letter."
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": global_b64 + [{"type": "text", "text": user_text}]}],
    max_tokens=512, temperature=0,   # no tools passed -> cannot crop
)
print("SKIM-ONLY (no crop) answer:\n", resp.choices[0].message.content.strip()[:600])
print("\nGround truth:", GT)""")

md(r"""## 8. What to take away

- **The tool call is a *native action*, not a wrapper.** The checkpoint was trained (SFT+RL+RFT) so that emitting
  `<tool_call>crop_video(...)</tool_call>` mid-reasoning is part of its policy — it decides *when* and *where* to zoom.
- **Coarse-to-fine beats uniform sampling** at fixed compute: the overview localizes the moment; the crop spends the
  frame budget only where it matters, at higher effective temporal density. (On easy questions the skim alone may
  already suffice — the payoff grows with video length and answer fineness.)
- **The training data for this behavior is synthesized** by `data/launch/imcott_generate.py`: a teacher model is
  given the answer *and* its ground-truth time window, then asked to write a plausible global-skim + crop trace that
  lands there. See that file for how these traces are manufactured (the public version stubs the teacher API call —
  the prompts are the real artifact).

**Next, for research:** the crop tool returns *uniform* frames of the chosen window. The obvious lever is *which*
frames it returns (query-guided selection) — but that's a separate investigation, not part of understanding the paper.
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3 (vllm)", "language": "python", "name": "python3"}
out = "/home/cfyang/hanklin/LongVT/demo/longvt_interleaved_demo.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")
