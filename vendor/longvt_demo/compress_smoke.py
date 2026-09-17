"""One-sample probe: does LongVT-RFT (trained only on crop_video) adopt a
brand-new compress_video tool zero-shot? Mirrors smoke.py but registers BOTH tools.
NOTE: compress_video here is the PIXEL-SPACE coverage stand-in, not real FlashVID (track 2).
"""
import os, json, textwrap
os.environ["no_proxy"] = "localhost,127.0.0.1"
from openai import OpenAI
from longvt_tools import (
    _fetch_pils, _pils_to_b64, crop_video, compress_video,
    CROP_VIDEO_TOOL, COMPRESS_VIDEO_TOOL, TOOL_PROMPT_BOTH,
)

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
MODEL = client.models.list().data[0].id
VIDEO = "/local1/cfyang/.cache/huggingface/videomme/videomme/data/kSBB5PsRV-k.mp4"
Q = "When the video talks about sea level rise, what is the time span?"
OPTS = "A. 1300 AD - 2016 AD.\nB. 1200 BC - 2019 AD.\nC. 1300 BC - 2016 AD.\nD. 1200 AD - 2019 AD."

# Modest global skim (<=192) to leave image-budget headroom for tool frames (server cap 600).
gpil = _fetch_pils(VIDEO, max_frames=192)
gb64 = _pils_to_b64(gpil)
print(f"[probe] global frames: {len(gpil)}")

msgs = [{"role": "user", "content": gb64 + [
    {"type": "text", "text": f"{Q}\n{OPTS}\n{TOOL_PROMPT_BOTH} The Video path for this video is: {VIDEO}"}]}]

TOOLS = [COMPRESS_VIDEO_TOOL, CROP_VIDEO_TOOL]
DISPATCH = {"compress_video": compress_video, "crop_video": crop_video}
called = []

for rnd in range(4):
    r = client.chat.completions.create(model=MODEL, messages=msgs, tools=TOOLS,
                                       tool_choice="auto", max_tokens=1024, temperature=0)
    m = r.choices[0].message
    fr = r.choices[0].finish_reason
    print(f"\n[R{rnd}] finish={fr}")
    print(textwrap.fill((m.content or "").strip(), 100))
    if fr == "tool_calls" and m.tool_calls:
        msgs.append({"role": "assistant", "content": m.content or "",
                     "tool_calls": [{"id": tc.id, "type": "function",
                                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                    for tc in m.tool_calls]})
        for tc in m.tool_calls:
            name = tc.function.name
            a = json.loads(tc.function.arguments)
            called.append(name)
            print(f"  TOOL={name}  ARGS={a}")
            fn = DISPATCH.get(name)
            if fn is None:
                msgs.append({"role": "tool", "tool_call_id": tc.id,
                             "content": [{"type": "text", "text": f"Unknown tool {name}."}]})
                continue
            b64, pil = fn(a["video_path"], a["start_time"], a["end_time"])
            print(f"  -> {len(pil)} frames")
            msgs.append({"role": "tool", "tool_call_id": tc.id,
                         "content": b64 + [{"type": "text",
                                            "text": f"{name} {a['start_time']}s-{a['end_time']}s, got {len(b64)} frames."}]})
    else:
        break

print(f"\n[probe] tools called in order: {called}")
print("[probe] DONE")
