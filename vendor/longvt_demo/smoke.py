import os, json, textwrap
os.environ["no_proxy"]="localhost,127.0.0.1"
from openai import OpenAI
from longvt_tools import encode_global, crop_video, CROP_VIDEO_TOOL, TOOL_PROMPT
client=OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
MODEL=client.models.list().data[0].id
VIDEO="/local1/cfyang/.cache/huggingface/videomme/videomme/data/kSBB5PsRV-k.mp4"
Q="When the video talks about sea level rise, what is the time span?"
OPTS="A. 1300 AD - 2016 AD.\nB. 1200 BC - 2019 AD.\nC. 1300 BC - 2016 AD.\nD. 1200 AD - 2019 AD."
print("[smoke] encoding global frames...")
gb64, gpil = encode_global(VIDEO)
print(f"[smoke] global frames: {len(gpil)}")
msgs=[{"role":"user","content": gb64 + [{"type":"text","text":f"{Q}\n{OPTS}\n{TOOL_PROMPT} The Video path for this video is: {VIDEO}"}]}]
for rnd in range(4):
    r=client.chat.completions.create(model=MODEL, messages=msgs, tools=[CROP_VIDEO_TOOL], tool_choice="auto", max_tokens=1024, temperature=0)
    m=r.choices[0].message; fr=r.choices[0].finish_reason
    print(f"\n[R{rnd}] finish={fr}")
    print(textwrap.fill((m.content or "").strip(),100))
    if fr=="tool_calls" and m.tool_calls:
        msgs.append({"role":"assistant","content":m.content or "","tool_calls":[{"id":tc.id,"type":"function","function":{"name":tc.function.name,"arguments":tc.function.arguments}} for tc in m.tool_calls]})
        for tc in m.tool_calls:
            a=json.loads(tc.function.arguments); print("  TOOL:", a)
            b64,pil=crop_video(a["video_path"],a["start_time"],a["end_time"])
            print(f"  -> {len(pil)} frames")
            msgs.append({"role":"tool","tool_call_id":tc.id,"content": b64 + [{"type":"text","text":f"Cropped {a['start_time']}s-{a['end_time']}s, got {len(b64)} frames."}]})
    else:
        break
print("\n[smoke] DONE")
