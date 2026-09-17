# Project Configuration

## CRITICAL GPU SAFETY RULES — READ FIRST
**NEVER use `kill -9` (SIGKILL) or `pkill -9` on any process touching a GPU** (vLLM, torch, CUDA, python training jobs, etc.). SIGKILL bypasses cleanup handlers, and the NVIDIA driver often fails to release GPU memory — leaving zombie processes that hold tens of GB until a sysadmin runs `nvidia-smi --gpu-reset` or reboots the server. This is a shared 8x A6000 box; orphaned GPU memory blocks other users' work.

**Instead, always escalate gracefully:**
1. `kill <pid>` (SIGTERM) — let the process clean up CUDA context
2. Wait 10–30 seconds, re-check with `nvidia-smi`
3. If still stuck, `kill -INT` or `kill -HUP`, then wait again
4. Only as a last resort, ask the user before any forceful action
t
Same rule applies to `killall -9`, `pkill -9`, `kill -SIGKILL`, Ctrl-\ (SIGQUIT on a CUDA process can also leak memory), `docker kill` on GPU containers, and `systemctl kill -s SIGKILL`. If a GPU process is unresponsive, STOP and tell the user — do not force it.

## Research context lives in a separate file — READ IT AT THE START OF EVERY CONVERSATION
**At the start of every conversation, read /home/cfyang/hanklin/RESEARCH.md before doing anything else.** Its top section is `# ACTIVE TASK` — that is the task currently being worked on, including the metrics, the fixed decisions (model, compressor, serving stack, temperature, repeat count), and what still has to be built. Do not propose or start work that contradicts it without saying so explicitly.

The rest of the file holds current experiments, findings, what we've built, open questions, and the north star. CLAUDE.md only holds always-relevant infrastructure and rules. Keep RESEARCH.md the single source of truth for research state and update it as results land; the PreCompact and research-keyword hooks also remind you to load and update it.

## BEFORE RUNNING ANY EXPERIMENT — use the canonical settings
**Read `# THE FIVE RUNS — the experiment` in RESEARCH.md and use those settings every time.** They are not defaults to re-derive per run; they are what makes runs comparable to each other and to the ones already banked. Summary, so a deviation is obvious:

- **run1** vanilla (64-frame video, no tools) · **run2** + `crop_video` · **run3** 256-frame skim at retention **0.25**, no tools · **run4** compressed skim + `crop_video` · **run5** oracle (crop forced to the GT evidence span). Each step changes exactly one thing.
- A **no-tool arm must get the no-tool prompt.** LongVT ships both; sending the tool prompt to a vanilla arm invalidates it. Print the prompt with `--print-prompt` before every launch.
- **vLLM only, never HF inference.** Qwen3-VL-8B-Instruct.
- **Skim is VIDEO modality, never images** — that is where Qwen3-VL's per-frame `<t seconds>` markers come from, and it is the only modality vLLM prunes. An image skim silently removes the model's ability to aim a crop (measured: hit rate 10% vs 24%).
- Crop tool: ≤128 frames, fps 1, returned as **images** (so never pruned).
- **T=0.7**, top_p 0.8, top_k 20, penalties 0/0/1.0, max_tokens 1024 (finalizer 32), ≤5 tool rounds, LongVT-verbatim prompt.
- **3 seeds per run**, report mean ± spread. All **1,549** LVBench questions. **Always store trajectories.**
- Video skims need pre-built proxies (`fast_agent/make_skim_proxy.py`) and `--require-proxy`; without them one arm takes 60+ hours instead of 2.

If a run has to deviate, say so explicitly and record the deviation in RESEARCH.md — do not quietly change a setting.

## Compute
- 8x RTX A6000 (48 GB each); all typically occupied — check `nvidia-smi` first
- conda env: `vllm`
- Large outputs: `/local1/cfyang/hanklin/outputs/`
- Experiment work dir: `/local1/cfyang/hanklin/experiments/`
- /home has ~796 GB free but keep large outputs on /local1 for consistency
- cfyang's vLLM server uses 4 GPUs; other users (e.g. oceanusm) often use the rest

## Running Infrastructure

### vLLM servers (verify with `curl localhost:<port>/v1/models` — this list goes stale)
- **Port 8222: LongVT-RFT** (Qwen2.5-VL-7B fine-tune, 32k ctx, GPU 5) — as of 2026-07-11. NOT Qwen3-VL anymore.
- **Port 8010: Qwen/Qwen3-VL-8B-Instruct** (DP3, GPUs 3/4/7, 131k ctx, hermes tool-calling) — baseline-eval server; launcher `longvt_compression/examples/eval/serve_qwen3vl_baseline.sh`
- Weights: /local1/cfyang/models--Qwen--Qwen3-VL-8B-Instruct/; Qwen2.5-VL-7B-Instruct at /local1/cfyang/models--Qwen--Qwen2.5-VL-7B-Instruct/
- Env: `OPENAI_API_KEY=EMPTY`, `OPENAI_BASE_URL=http://localhost:<port>/v1`

## Key Paths
- VideoARM code: /home/cfyang/hanklin/videoarm/
- VideoARM eval script: /home/cfyang/hanklin/videoarm/run_videomme.py
- LongVT code (current chapter's base): /home/cfyang/hanklin/LongVT/
- LongVT-RFT server launcher: /home/cfyang/hanklin/LongVT/demo/serve_longvt.sh
- VideoMME videos (MP4): /local1/cfyang/.cache/huggingface/videomme/videomme/data/
- VideoMME parquet: /local1/cfyang/.cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/videomme/test-00000-of-00001.parquet
- VideoMME pre-extracted frames (JPEGs, not MP4s): /local1/cfyang/LMUData/images/Video-MME/
- Model cache: /local1/cfyang/
- Output dir: /local1/cfyang/hanklin/outputs/

## How to interact with the user
- User is early in independent research. When asked to do something, evaluate whether it's worth the time or headed in a wrong direction. If so, say so honestly with a concrete reason — don't just flag for the sake of flagging. Offer an alternative when possible. If the request is sound, just do it.
