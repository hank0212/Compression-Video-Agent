# HANDOFF — picking this chapter up on a new machine

Written for a coding agent with no prior context, and for the author as a reference. If you
are an agent: read this file top to bottom before running anything, then read `RESEARCH.md`
(checked in at the repo root) for the research state. `CLAUDE.md` at the repo root carries
the safety rules and canonical experiment settings, and is meant to be auto-loaded.

Original machine: `kraken`, 8× RTX A6000 48 GB, work under `/home/cfyang/hanklin/` and
`/local1/cfyang/`. Every absolute path in this repo points there and **will not exist on the
new box.** §3 tells you which ones to rewrite.

---

## 1. What this project is, in one page

**Question.** A video agent gets a cheap global "skim" of a long video, then may call
`crop_video` to spend its remaining token budget on a narrow time window. Does compressing
the ViT output — so the skim covers more of the video at the same token cost — make the agent
better at *finding* the evidence?

**Answer so far: no, and the reason is more interesting than the answer.** On LVBench with
Qwen3-VL-8B-Instruct:

| comparison | n | result |
|---|---|---|
| 4× temporal coverage (64 → 256 frames, uncompressed) | 253 | **+4.74 pp** |
| compressing that coverage to 25% (VidCom2 r=0.25) | 253 | **−7.90 pp** |
| 6.4× per-frame resolution (arm e vs run1) | 1,549 | **−0.07 pp** |
| the model aiming its own crop (run2 − run1) | 1,549 | **−0.20 pp** |
| **an oracle aiming the crop at the GT span (run5 − run1)** | 1,549 | **+15.62 pp** |

Read it as: **temporal coverage is the binding constraint, spatial fidelity is not, and
almost all the value of the crop tool is locked behind a localization ability the model does
not have.** The compressor is not the bottleneck; knowing where to look is.

**The one crack in that wall, and the reason the next experiment exists.** Every internal
signal tested for "which frames hold the evidence" scored at chance (AUC 0.485–0.509),
including a cosine against Qwen3-VL's own question embeddings. An *aligned* scorer — CLIP,
which is contrastively trained for exactly this comparison — scores **AUC 0.643**, top-1
evidence hit 15.7% against a 5.9% random baseline. That is the first non-blind signal this
chapter has found. It licenses building a temporal allocator. It licenses **no** accuracy
claim — nothing was generated. Full writeup: `RESEARCH.md § MEASURED 2026-08-28`.

---

## 2. Repo map

### In this repo

| path | what |
|---|---|
| `fast_agent/run_agent.py` | **the experiment harness.** Agentic loop against a vLLM OpenAI endpoint: skim → optional `crop_video` rounds → bounded finalizer. Every banked number came out of this. |
| `fast_agent/run_plain.py` | same backbone, no tools. Single skim, one answer. |
| `fast_agent/config.py` | all paths, sampling params, and the verbatim LongVT prompts (tool **and** no-tool). **Start every path rewrite here.** |
| `fast_agent/data.py` | LVBench / VideoMME loaders, GT evidence-span parsing, question formatting. |
| `fast_agent/tools.py` | `crop_video`: decodes the source mp4, resizes to 224² client-side, returns **images**. |
| `fast_agent/make_skim_proxy.py` | builds the small per-video proxy mp4s. Not optional — see §5. |
| `fast_agent/serve_arm.sh` | one vLLM launcher per arm. The per-arm sizing block is load-bearing. |
| `fast_agent/run_grid.sh` | serve → wait for health → run → integrity-check → SIGTERM, chained over arms. |
| `fast_agent/signal_auc.py` | offline signal diagnostics incl. `clip_query`. No server, no generation, ~9 min on one GPU. |
| `fast_agent/preflight.py`, `probe_budget.py`, `preflight_arm.py` | measure real token counts by driving vLLM's own processor. Use these instead of trusting a flag. |
| `fast_agent/diagnose_runs.py`, `analyze_grid.py`, `analyze_pairs.py` | read `results.jsonl` + `traj/` and produce the tables. |
| `vidcom2_vllm/` | VidCom2 as a vLLM plugin. `patch.py` rebinds `compute_retention_mask`; `plugin.py` is the entry point that makes it survive into spawned workers. |
| `results/all_runs.csv`, `results/manifests/` | checked-in extract of every run's settings and score, so numbers are auditable without re-running ~30 GPU-hours. |
| `RESEARCH.md` | **single source of truth for research state.** 155 KB. Read `# ACTIVE TASK` and `# THE FIVE RUNS` first. |
| `vendor/` | work that lives in other people's clones and cannot be pushed upstream. See `vendor/README.md`. |

### Not in this repo, needed to run

| what | where it was | how to get it |
|---|---|---|
| Qwen3-VL-8B-Instruct | `/local1/cfyang/models--Qwen--Qwen3-VL-8B-Instruct/` | HF: `Qwen/Qwen3-VL-8B-Instruct`, 17 GB |
| LVBench videos | `/local1/.../datasets--zai-org--LVBench/snapshots/<sha>/videos/` | HF: `zai-org/LVBench`, **58 GB** — see §4 |
| LVBench metadata | same snapshot, `video_info.meta.jsonl` | same repo, small. **Required** — holds `time_reference`, the GT evidence span the flattened `.tsv` drops |
| skim proxies | `/local1/cfyang/hanklin/outputs/lvbench_agent/skim_proxies*` | rebuild with `make_skim_proxy.py`, CPU only |
| banked runs + trajectories | `/local1/cfyang/hanklin/outputs/lvbench_agent/` | **256 MB for all 68 runs** — copy these, do not re-run |
| `videoarm` clone | `/home/cfyang/hanklin/videoarm` | only `signal_auc.py` needs it, for `_CLIPScorer` |

---

## 3. Paths you must rewrite

These are hardcoded to the old box. In rough order of how fast they will bite:

1. `fast_agent/config.py` — `MODEL_SNAPSHOT`, `_LVBENCH_SNAP`, `OUTPUT_DIR`, `VIDEO_DIR`,
   `VIDEOMME_PARQUET`, `FLASHVID_REPO`.
2. `fast_agent/serve_arm.sh` — `PY`, `MODEL`, `LOG`, `--allowed-local-media-path`, and the
   `GPU=`/`PORT=` assignments in the per-arm `case` block.
3. `fast_agent/signal_auc.py` — `MODEL`, `PROXY`, `VIDEOARM` (module-level constants).
4. `fast_agent/run_grid.sh` — `cd /home/cfyang/hanklin`, `PYTHONPATH`, `$LOG`.

The harness is imported as `longvt_compression.fast_agent.*`, so the **parent** of this repo
must be on `PYTHONPATH` and the repo directory must be named `longvt_compression`.

---

## 4. Machine setup

### Environment — the version pin is not optional

```bash
conda create -n vllm python=3.12 -y && conda activate vllm
pip install vllm==0.19.0          # wheel, NOT a source build
pip install opencv-python av pandas numpy pillow openai
pip install -e vidcom2_vllm/      # registers the vllm.general_plugins entry point
```

Verified on the original box: `vllm 0.19.0`, `torch 2.10.0+cu129`, `transformers 4.57.6`,
Python 3.12.

**Why 0.19.0 exactly.** `vidcom2_vllm/patch.py` works by rebinding `compute_retention_mask`
inside three *named* modules (`qwen3_vl`, `qwen3_vl_moe`, `qwen2_5_vl`), because those modules
do `from vllm.multimodal.evs import compute_retention_mask` at import time — so patching
`vllm.multimodal.evs` alone has no effect on the model. Any vLLM version that moves that
import, or that ships VidCom2 natively (upstream PR #47750, `main`-only as of this writing),
will make the patch either fail loudly or, worse, silently no-op while the launcher reports
success. **If you must upgrade vLLM, re-run the structural check in
`VIDCOM2_VALIDATION_PLAN.md` before trusting any compressed arm.**

### GPU

Weights 16.5 GB, KV cache **144 KiB/token**. Tool arms run at `--max-model-len 40960`,
so KV alone is ~5.9 GB.

| card | verdict |
|---|---|
| 48 GB (A6000, original) | reference configuration |
| 40 GB (A100) | fine |
| 24 GB (4090 / A5000) | **tool arms are not practical.** One sequence fits; concurrency does not, and 1,549 questions × 5 rounds becomes untenable. Lowering `--max-model-len` truncates exactly the long trajectories that carry the localization signal. Try FP8 before changing anything else. |

A single GPU is enough. The original `DP3` setup was throughput, not a requirement.

### Data

```bash
# 58 GB. Without allow_patterns this pulls ~173 GB — the repo also ships the same
# videos again as all_videos_split.zip.*
hf download zai-org/LVBench --repo-type dataset \
    --include "videos/*" "video_info.meta.jsonl" --local-dir <LVBENCH_DIR>

hf download Qwen/Qwen3-VL-8B-Instruct --local-dir <MODEL_DIR>   # 17 GB
```

**Minimum viable transfer, if disk or bandwidth is tight:** the 256 MB of banked runs alone
lets you reproduce every table and figure without a GPU. Start there and confirm the analysis
scripts run before downloading anything large.

**Proxy-only mode (3 GB instead of 58 GB)** runs the no-tool arms (run1, run3) without the
source videos, but needs a small patch: `data.py::load_lvbench` filters questions by
`os.listdir(LVBENCH_VIDEO_DIR)`, and `run_agent.py` calls `video_duration()` on the raw mp4
even when a proxy is used. Point both at the proxy — this is sound because `make_skim_proxy`
guarantees the proxy reproduces the source timeline, and `--verify` checks it. **The crop arms
(run2, run4, run5) cannot be made proxy-only**: `tools.py` decodes the source mp4 directly.

---

## 5. How to run the experiment

### The five runs

Each row changes **exactly one thing** from the row above. They share the loader, prompt
family, crop tool, scoring and sampling params. Deviating makes a run incomparable to the
banked ones — if you must deviate, say so explicitly and record it in `RESEARCH.md`.

| run | skim | tools | output dir |
|---|---|---|---|
| **run1** vanilla | 64-frame video | none | `lvbench_r1_seed0` |
| **run2** + crop | 64-frame video | `crop_video` | `lvbench_r2_seed0` |
| **run3** + compression | 256-frame video, VidCom2 r=0.25 | none | `lvbench_c3_notool_seed0` |
| **run4** + both | 256-frame video, VidCom2 r=0.25 | `crop_video` | `lvbench_r4_seed0` |
| **run5** oracle | 64-frame video | crop **forced to the GT span** | `lvbench_r5_seed0` |

Fixed across all five: Qwen3-VL-8B-Instruct, vLLM only (**never HF inference**), skim in
**video** modality, crop ≤128 frames at fps 1 returned as **images**, T=0.7 / top_p 0.8 /
top_k 20 / penalties 0-0-1.0, `max_tokens` 1024 (finalizer 32), ≤5 tool rounds, LongVT-verbatim
prompt, 3 seeds, all 1,549 questions, trajectories always stored.

### Launch

```bash
# 1. serve. Arm letter picks frames + pruning + port; see the case block in serve_arm.sh.
GPU_FORCE=0 bash fast_agent/serve_arm.sh a          # uncompressed 64f  -> :8030
GPU_FORCE=1 bash fast_agent/serve_arm.sh b          # VidCom2 256f     -> :8032

# 2. confirm it is actually up and actually pruning
curl -s localhost:8030/v1/models | head -c 200
grep -a "vidcom2_vllm\] ACTIVE\|video_pruning_rate" <LOG>/server_arm_b.log | tail -2

# 3. PRINT THE PROMPT. Every time. See the gotcha below.
python -m longvt_compression.fast_agent.run_agent --dataset lvbench --print-prompt ...

# 4. run (run2 shown: 64-frame video skim + crop tool)
PYTHONPATH=<parent-of-this-repo> python -m longvt_compression.fast_agent.run_agent \
  --dataset lvbench --skim-mode video --frames 64 --tools crop_video \
  --proxy-dir <OUT>/skim_proxies --require-proxy \
  --num 1549 --seed 0 --tag run2 \
  --base-url http://localhost:8030/v1 --workers 8
```

Drop `--tools` for run1/run3; use `--frames 256` and the compressed server for run3/run4;
run5 adds the oracle crop source. `run_grid.sh` chains serve → health-wait → run → integrity
check → **SIGTERM** across arms and is the safer way to launch a batch.

Proxies first, or one arm takes 60+ hours instead of 2:

```bash
python -m longvt_compression.fast_agent.make_skim_proxy --frames 256 --workers 12 --verify
```

### Reading results

Each run directory holds `results.jsonl`, `run_manifest.json`, `summary.json`, and
`traj/<qid>.json` (one per question, with the GT evidence span and per-call coverage).

Two headline metrics: **accuracy**, and **coverage** — `covered_frac` = |union(crop spans) ∩
GT evidence| / |GT evidence|, where `landed` means `covered_frac == 1.0`.

---

## 6. Gotchas that have already cost real time

Each of these produced a wrong number that was believed for a while.

1. **Always check for transport errors before quoting an accuracy.**
   ```bash
   python -c "import json;rs=[json.loads(l) for l in open('results.jsonl')];print(sum('error' in r for r in rs), len(rs))"
   ```
   A run launched against a dead port wrote 223 `APIConnectionError` rows as
   `pred=None, correct=False` and scored 17.8% — below 4-way chance — which would have made
   the compression effect read as −21 pp instead of −1 pp. **This failure mode has bitten
   this chapter twice.**

2. **A no-tool arm must get the no-tool prompt.** LongVT ships both. `build_first_turn` used
   to apply the tool prompt whenever `--prompt-style longvt` was set, regardless of `--tools`,
   so the first run1 was told *"call crop_video if needed"* while holding no tool. Any no-tool
   arm from before 2026-08-13 is invalid. `--print-prompt` catches this in one second.

3. **The skim must be VIDEO modality, never images.** Qwen3-VL's per-frame `<t seconds>`
   markers come from the video path, and video is the only modality vLLM prunes. An image
   skim silently removes the model's ability to aim a crop — measured hit rate 10% vs 24%.

4. **`--mm-processor-kwargs '{"max_pixels":...}'` is inert for video in vLLM 0.19.** Sweeping
   it produces a byte-identical `video_grid_thw`. Only `{"size": {"longest_edge": N}}` moves
   the grid. Several arms ran at the checkpoint default believing the flag had applied.

5. **Video `max_pixels` is a whole-video budget, not per-frame** — so asking for more frames
   silently lowers per-frame resolution. 64 frames → 112 tok/frame; 256 frames → 91 tok/frame,
   same video. This is why run1 and run3 are **not** budget-matched (3,584 vs 2,912 tokens,
   19% apart against run3), despite an earlier note claiming they were.

6. **`--video-pruning-rate 0.75` means retention 0.25.** The flag is the *pruned* fraction.

7. **Directory names are authoritative; the `tag` field in `run_manifest.json` is not.**
   `lvbench_r5_seed0` carries `"tag": "run4"` and `lvbench_r4_seed0` carries `"tag": "armB"`.

8. **Never `kill -9` a GPU process.** Shared box, and SIGKILL leaks GPU memory that only a
   `nvidia-smi --gpu-reset` or reboot recovers. SIGTERM, wait 10–30 s, re-check `nvidia-smi`.
   `run_grid.sh` already does this correctly.

9. **Never use HF inference for a scored arm.** vLLM only. HF is fine for offline diagnostics
   (`signal_auc.py`) and for the official-implementation smoke test.

---

## 7. Starting RL — read this before promising anything

**Status: not built, never run here, and not runnable on one node as LongVT ships it.**
Setting expectations honestly matters more than a command that fails at hour three.

**What exists.** The `LongVT` clone vendors `verl` (a full RL framework) plus a GRPO recipe at
`examples/video_tools/longvt_7b_rl_train.sh` with configs in `examples/video_tools/config/`
(`timer1_multiturn_grpo.yaml`). The training target is the same thing this chapter measured:
a policy that skims globally, then calls `crop_video` into detail.

**What blocks it, concretely, from that script:**

| blocker | detail |
|---|---|
| **scale** | `NNODES=8` × `trainer.n_gpus_per_node=8` = **64 GPUs**. Plus `ulysses_sequence_parallel_size=4` and `rollout.n=16` at `max_prompt_length=36000`. |
| **data** | `DATA_PATH='/path/to/your/data'` is still the unfilled placeholder — **it was never configured.** The target set is VideoSIAH 1.7k; `/local1/cfyang/hanklin/videosiah_eval/` is an **empty** scaffold. |
| **serving stack** | rollout is `sglang`, not vLLM. Nothing in this chapter has ever run sglang. |
| **side services** | an MCP tool server (`mcp_server.py`, `mcp_tool_config.yaml`) and an `LLM_AS_A_JUDGE_BASE` endpoint must both be up for rewards to compute. |

**So the honest path, in order:**

1. **Do not start from `longvt_7b_rl_train.sh`.** Start from a verl recipe sized for one node —
   `recipe/one_step_off_policy/grpo_3b_gsm8k_fsdp2_2_6.sh` is the closest working template —
   and get *any* GRPO step to run end to end on text before adding video.
2. **Get the data.** Nothing can be trained until VideoSIAH (or a substitute) is on disk and
   in verl's expected format. This is the real long pole, not the compute.
3. **Decide what the reward is.** This chapter's own result argues the reward should target
   **localization**, not just answer correctness: the oracle gap is +15.62 pp and the model's
   own aiming is worth −0.20 pp. `covered_frac` is already computed per trajectory and is a
   ready-made dense reward signal. That is a genuinely well-posed RL problem and it is the
   strongest argument for doing RL here at all.
4. **Expect to need more than 8 GPUs** for a 7–8B policy with 36k-token prompts and 16 rollouts.
   Shrinking the backbone to make RL fit is the obvious move and the wrong one — see below.

**Do not shrink the model to make RL fit.** All 68 banked arms (n=1,549 each) are
Qwen3-VL-8B. Changing the backbone invalidates every comparison in `RESEARCH.md` and the
bottleneck this chapter identified is localization, not capacity — a smaller model localizes
*worse*. If VRAM is the constraint, quantize first.

---

## 8. Suggested first hour on the new machine

1. Clone this repo, copy the 256 MB of banked runs, run `diagnose_runs.py` and `analyze_grid.py`
   against them. **No GPU, no videos, no downloads.** If the tables match §1, the code moved
   cleanly.
2. Rewrite the paths in §3.
3. Build the env, start one server, `--print-prompt`, and check the prompt matches the arm.
4. Re-run 100 questions of run1 seed 0 and compare accuracy against the banked `results.jsonl`.
   **This is the only check that proves the move did not break anything.**
5. Only then download the 58 GB and rebuild proxies.

## 9. Open threads

- **Build the temporal allocator.** The CLIP signal (AUC 0.643) licenses the U / Q / U+Q
  allocation experiment. This is the live next step.
- **Run arm f.** Configured, not banked. `arm f − arm e` tests VidCom2 at its own operating
  point (~180 surviving tok/frame) rather than the 22.8 the banked 256-frame arm ran at, and
  separates "compression hurts" from "we ran it 8× outside its regime."
- **`SOFTMAX_TEMP=0.01` in `vidcom2_vllm/retention.py`** makes VidCom2's adaptive frame budget
  degenerate: 124 of 128 frames get an identical 22-token budget. Check it against the
  reference implementation.
- **Ablate the CLIP query on stem-only.** The query currently includes the four options
  because `query_sim` was given the same string; for CLIP they are probably noise. One line.
- **LongCLIP.** 5.6% of queries exceed CLIP-B/32's 77-token context.
