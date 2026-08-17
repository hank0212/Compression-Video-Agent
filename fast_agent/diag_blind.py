"""Blind control: LVBench with NO video at all -- question + options only.

  python -m longvt_compression.fast_agent.diag_blind --base-url http://localhost:8031/v1

This is the diagnostic that decides whether a low absolute accuracy means "the harness is
broken" or "the visual budget is small". Everything is shared with the real runs --
`data.load_dataset`, `data.format_question`, `data.extract_answer`, the same sampling
params -- so the ONLY difference from run1 is that the frames are gone.

Read it as: run1 - blind = what the 64-frame skim is actually contributing. If that gap is
~0, the video is not reaching the model and something IS wrong. If it is clearly positive,
the pipeline works and the absolute number is a budget story.

LVBench is 4-way multiple choice, so 25% is chance and a strong text-only prior (question
wording, option plausibility, world knowledge about the video's topic) lands well above it.
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import data
from .run_plain import OFFICIAL_INSTR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=1549)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-url", default="http://localhost:8031/v1")
    ap.add_argument("--model", default="qwen3vl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="/local1/cfyang/hanklin/outputs/lvbench_agent/blind_seed0")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=120, max_retries=2)
    os.makedirs(args.out, exist_ok=True)
    rows = data.load_dataset("lvbench", n=args.num, seed=args.seed)

    lock, done, hits = threading.Lock(), [0], [0]
    out = open(os.path.join(args.out, "results.jsonl"), "w")
    t0 = time.time()

    def work(row):
        prompt = data.format_question(row) + "\n" + OFFICIAL_INSTR
        try:
            r = client.chat.completions.create(
                model=args.model, messages=[{"role": "user", "content": prompt}],
                temperature=0.7, top_p=0.8, presence_penalty=0.0, frequency_penalty=0.0,
                max_tokens=32, seed=args.seed,
                extra_body={"top_k": 20, "repetition_penalty": 1.0})
            text = r.choices[0].message.content or ""
        except Exception as e:
            text = ""
        pred = data.extract_answer(text)
        rec = {"question_id": str(row["question_id"]), "task_type": row["task_type"],
               "pred": pred, "gold": row["answer"], "correct": pred == row["answer"],
               "text": text}
        with lock:
            out.write(json.dumps(rec) + "\n")
            done[0] += 1
            hits[0] += rec["correct"]
            if done[0] % 100 == 0:
                print(f"[{done[0]}/{len(rows)}] blind acc {100*hits[0]/done[0]:.2f}%",
                      flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, rows))
    out.close()
    print(f"\n[BLIND] no video at all: {100*hits[0]/max(done[0],1):.2f}% over n={done[0]} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
