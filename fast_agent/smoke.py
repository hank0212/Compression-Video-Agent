"""3-sample smoke per arm + engine self-checks.

Usage:
  CUDA_VISIBLE_DEVICES=3 python -m fast_agent.smoke --arm baseline
  ... --arm crop | --arm compress   (later steps)

Self-checks (baseline arm only, sample 0):
  A. assembly parity — our template+expand+splice produces token-identical
     input_ids to the native processor path.
  B. decode parity — our embeds-assembly + manual greedy loop reproduces stock
     model.generate() text (stock runs FIRST, before vision patches).
"""

import argparse

import torch

from . import config, data, tools
from .agent_loop import assemble, decode, run_sample
from .model import Engine

ARM_TOOLS = {
    "baseline": (),
    "crop": ("crop_video",),
    "compress": ("crop_video", "compress_video"),
}


def parity(engine: Engine, row: dict):
    dur = data.video_duration(row["video_path"])
    pils = tools.initial_frames(row["video_path"])
    prompt = (
        data.format_question(row)
        + "\n\n" + config.initial_view_text(dur, len(pils))
        + "\n\n" + config.ANSWER_INSTR
    )

    # --- stock path FIRST (vision still unpatched) ---
    msgs = [{
        "role": "user",
        "content": [*({"type": "image"} for _ in pils), {"type": "text", "text": prompt}],
    }]
    text = engine.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = engine.processor(text=[text], images=pils, return_tensors="pt").to(engine.device)
    with torch.inference_mode():
        out = engine.model.generate(**inputs, max_new_tokens=64, do_sample=False)
    stock = engine.tokenizer.decode(
        out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )

    # --- our path (patches vision on first encode) ---
    clip0 = engine.encode_images(pils)
    a = assemble(engine, [{"role": "user", "parts": [clip0, prompt]}])

    ok_ids = a.ids.shape == inputs.input_ids.shape and torch.equal(
        a.ids.cpu(), inputs.input_ids.cpu()
    )
    print(f"[parity A] token-identical assembly: {ok_ids} "
          f"(ours {tuple(a.ids.shape)} vs stock {tuple(inputs.input_ids.shape)})")
    if not ok_ids and a.ids.shape == inputs.input_ids.shape:
        diff = (a.ids.cpu() != inputs.input_ids.cpu()).nonzero()
        print(f"  first diffs at {diff[:5].tolist()}")

    ours = decode(engine, a, max_new_tokens=64)
    ok_txt = ours.strip() == stock.strip()
    print(f"[parity B] decode matches stock generate: {ok_txt}")
    print(f"  stock: {stock[:160]!r}")
    print(f"  ours : {ours[:160]!r}")
    return ok_ids, ok_txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="baseline", choices=list(ARM_TOOLS))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args()

    rows = data.load_long_split(n=args.n, seed=0)
    print(f"[smoke] {len(rows)} samples, arm={args.arm}")
    engine = Engine()
    print("[smoke] model loaded")

    if args.arm == "baseline" and not args.skip_parity:
        parity(engine, rows[0])

    n_correct = 0
    for i, row in enumerate(rows):
        print(f"\n=== [{i+1}/{len(rows)}] {row['question_id']} ({row['task_type']}) "
              f"video={row['videoID']}")
        r = run_sample(engine, row, tool_names=ARM_TOOLS[args.arm], verbose=True)
        print(f"  -> pred={r['pred']} gold={r['gold']} correct={r['correct']} "
              f"rounds={r['rounds']} "
              f"calls={[(c['name'], round(c['start']), round(c['end'])) for c in r['tool_calls']]}")
        n_correct += bool(r["correct"])

    print(f"\n[SMOKE {args.arm}] {n_correct}/{len(rows)} correct, all answered="
          f"{all(True for _ in rows)}")


if __name__ == "__main__":
    main()
