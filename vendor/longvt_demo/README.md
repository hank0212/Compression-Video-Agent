# LongVT demo — understanding the interleaved tool-calling

Files here reproduce the released **LongVT-RFT** (Qwen2.5-VL-7B) paper on one Video-MME question
and visualize the "Multimodal Chain-of-Tool-Thought" (skim → think → `crop_video` → see → answer).

## What's here
- `serve_longvt.sh` — serve the checkpoint with vLLM (existing `vllm` conda env). `bash serve_longvt.sh <GPU> <PORT>`
- `longvt_tools.py` — paper-faithful frame plumbing (global skim + `crop_video`), also returns PIL frames for display
- `longvt_interleaved_demo.ipynb` — **the explainer notebook** (already executed, outputs + frame montages baked in)
- `smoke.py` — minimal end-to-end sanity check
- `pick_video.py` — pick a Video-MME QA from the parquet

## Run it
```bash
cd /home/cfyang/hanklin/LongVT/demo
bash serve_longvt.sh 5 8000            # start server on GPU5:8000 (needs ~22GB free)
# wait for "READY"; curl -s localhost:8000/v1/models
conda run -n vllm python smoke.py       # quick check
# then open longvt_interleaved_demo.ipynb (kernel = vllm env) and Run All
```

## Stop the server (shared box — NEVER kill -9 a GPU process)
```bash
pkill -f "vllm serve /local1/cfyang/LongVT-RFT"   # SIGTERM; wait, re-check nvidia-smi
```

## Verified result
Video `kSBB5PsRV-k` (OCR, "sea level rise time span"): model skims 284 frames, thinks, calls
`crop_video(77s,92s)`, reads the on-screen scale in the zoomed frames, answers **C** (correct, GT=C).
Note: on this easy question skim-only *also* answers C — the crop here is grounding/verification; the
tool is decisive on harder fine-detail questions.

## How the training data is made (studied, not run)
`../data/launch/imcott_generate.py` = **ground-truth-guided teacher distillation**: each QA carries a GT
answer + GT time window; a teacher (Gemini-2.5-Pro) writes a coarse-to-fine trace whose crop windows are
*forced* to the GT window. ⚠️ The public `run_imcott_generation` is a **stub** (teacher API call not wired
in; placeholder strings) — the prompts are the real artifact.
