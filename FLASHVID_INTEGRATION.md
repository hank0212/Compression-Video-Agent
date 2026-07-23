# FlashVID → Baseline-Agent Integration — Handoff Document

*Written 2026-07-11 (evening). Self-contained: everything a fresh agent session needs
to understand the context, what is already built/verified, and exactly what remains.
Companion evidence doc: [FLASHVID_TOOL_PLAN.md](FLASHVID_TOOL_PLAN.md) (probe results
backing every design claim here). Research state: `/home/cfyang/hanklin/RESEARCH.md`.*

---

## 1. Research context (why this exists)

- **North star** (RESEARCH.md): visual compression as an **action the agent takes**,
  not fixed preprocessing — the agent allocates bandwidth: wide+coarse (locate) vs
  narrow+detailed (inspect) vs answer.
- **Current chapter**: build a **zero-shot Qwen3-VL-8B baseline agent** inside
  LongVT's tool-calling pipeline (NOT LongVT's trained Qwen2.5-VL checkpoint), then
  add compression as a second tool, inspect failure modes, and later generate
  SFT/RL data with the same pipeline.
- **Key archive finding** (VideoARM study, closed 2026-07-05): hand-wired compression
  never beat a well-navigated agent; FlashVID-map and AKS-map **tied** (57.8% on
  matched-633) but won *different* task types (coverage vs detail). The learned
  WHEN-to-compress decision is the thesis; this integration provides the mechanism.
- **Why "external tool"**: real FlashVID consumes ViT-internal state
  (`video_features` + `cls_attention` inside the forward —
  `FlashVID/flashvid/modeling_qwen3_vl.py:299-311`). The LongVT tool boundary
  carries pixels. So the integration must either route around the boundary
  (Option A) or restrict to pixel-representable selection (Option B). Both below.

## 2. System as it stands (what already runs)

### The baseline agent (built 2026-07-11, currently mid-eval)

Zero-shot `Qwen/Qwen3-VL-8B-Instruct` driving LongVT's `crop_video` loop through
lmms-eval. Full file pipeline, execution order:

| stage | file |
|---|---|
| serve model (all GPU inference) | `examples/eval/serve_qwen3vl_baseline.sh` → vLLM 0.19, port **8010**, DP3 GPUs 3/4/7, 131k ctx, hermes tool parser, `--limit-mm-per-prompt.image 2048` |
| launch eval | `examples/eval/run_qwen3vl_baseline.sh` (all env vars + model_args live here) |
| task def | `lmms_eval_tasks/videomme/videomme_long_tool.yaml` → task `videomme_long_reward_tool` (900 long-split Qs, `max_new_tokens: 8192`) |
| doc → messages | `lmms_eval_tasks/videomme/utils.py` (`videomme_tool_doc_to_messages`: video + question + TOOL_PROMPT) |
| frame encoding | `FlashVID/lmms-eval/lmms_eval/protocol.py` (`to_qwen3_vl_openai_messages`: client-side decord decode, 1fps ≤768f 224px, base64 images + `<t seconds>` timestamp texts) |
| **agent loop** | `FlashVID/lmms-eval/lmms_eval/models/chat/async_openai.py` (`maybe_forward_with_tool`) |
| tool subprocess | `examples/video_tools/mcp_server.py` (`crop_video`: 1fps ≤128f 224px → base64 PNGs; **fresh python spawn per call**) |
| scoring | `custom_rewards/lmms_lab_recipe.py` (`compute_score`: `<answer>` extract → MCQ match → LLM-judge fallback) |
| results | `eval_logs/Qwen__Qwen3-VL-8B-Instruct/*_samples_*.jsonl` + `*_results.json` |

The **judge** is the same Qwen3-VL-8B on 8010 (`OPENAI_BASE_URL` env — NOTE:
`OPENAI_API_BASE` = model-under-test, `OPENAI_BASE_URL` = judge; two different vars).
Port **8222 serves LongVT-RFT** (Qwen2.5-VL), not usable as judge.

### Harness patches already made (all in the user's fork `FlashVID/lmms-eval`, all opt-in via model_args, stock defaults unchanged)

In `lmms_eval/models/chat/async_openai.py`:

| model_arg | default | what it does / why it exists |
|---|---|---|
| `retain_tool_history=True` | False | stock loop DROPS previous rounds' tool frames from context (`messages + tool_messages` transient) → zero-shot agent re-crops forever (30+ rounds). This keeps all evidence in context. |
| `max_tool_rounds=8` | None | after N rounds, tools are withheld from the next request → model must answer. |
| `force_final_answer=True` | False | if the finished trace has no `<answer>` tag, one extra tool-less turn demands it (separates capability from format compliance for pre-SFT models). |
| `max_total_images=1900` | None | rounds cap doesn't bound images (parallel tool calls per round × 128 frames) → server 400 "At most 2048 image(s)". Once budget hit, tool results become a text notice + tools withheld. |
| retry logic in `_process` | — | 4 attempts w/ backoff on "Already borrowed" (vLLM HF-tokenizer race) / 500s; non-transient → `[EVAL_ERROR] ...` string, sample scored 0, run survives. |
| `video_tool_names=""` | "" | **PARTIALLY APPLIED** — the `__init__` parameter exists but is NOT yet stored on self nor used in the loop. This is where Option A implementation resumes (§4 step 2). |

Also patched: `lmms_eval/evaluator.py` (0-dim → shape-(1,) gather tensor; gloo/CPU
clients crashed otherwise) and run script uses `CUDA_VISIBLE_DEVICES=""` +
hard-prepended flashvid-env PATH (see §6 footguns).

Incremental persistence: `LMMS_EVAL_USE_CACHE=True` + `LMMS_EVAL_HOME=/local1/cfyang/hanklin/outputs/lmms_eval_home`
(set in run script). Every finished sample appends to
`$LMMS_EVAL_HOME/eval_cache/<hash>/[task]_rank{r}_world_size{w}.jsonl`; restarts skip
cached doc_ids (**resume requires same process count** — files keyed by rank/world_size).
Mid-run scorer: `scratchpad/partial_score.py` (running accuracy + tool-usage stats from
the live cache, real `compute_score`).

### In flight right now (do not disturb)

- **Full 900-Q VideoMME-long baseline run**, launched 08:26, ~65% at 20:26, ETA
  ~2-4am. Detached (`setsid`), log `/local1/cfyang/hanklin/outputs/full900_qwen3vl_videomme_long.log`.
  Predates the cache flag → its results exist ONLY in process memory until the end.
  **Killing it loses everything.** 1 known failed sample (image-limit, will be
  replayed post-run with the patched loop and spliced).
- A 24-sample cache-enabled validation run (3 procs) alongside it.
- **`examples/video_tools/mcp_server.py` must not be edited while any eval runs**
  (re-spawned from disk per tool call → edits leak mid-run). That's why the new tool
  lives in a separate file.

## 3. The integration design (probe-verified)

### The keystone fact
vLLM 0.19's Qwen3-VL EVS hook (`compute_retention_mask`, gated on
`--video-pruning-rate`) fires **only on video-modality inputs**; images bypass it
(`vllm/model_executor/models/qwen3_vl.py:73-76, 1104, 1925` in the `vllm` conda env).
The archived plugin `_archive/flashvid_plugin` swaps that one symbol for FlashVID
DySeg+ADTS selection — verified to patch vLLM **0.19** cleanly (signature match;
`register()` swap confirmed in-process).

### Option A — modality-routed engine compression (RECOMMENDED, partially built)
One server, two bandwidths, selected per tool call:
- `crop_video` returns **images** → full-resolution tokens (bit-identical to today).
- `compress_video` returns its clip **as video** → engine-side FlashVID prunes it to
  `(1-r)` tokens before the LLM sees it.
- The agent's **choice of tool = the compression decision**. No model changes, no
  new prompt format.

Verified enablers: the live 8010 server accepts base64 `video_url` in **tool-role
messages** and the model reads it correctly (~1.1k tokens for a 6s/320p clip).
lmms-eval's MCP client only converts image/text → hence the marker mechanism below.

### Option B — sidecar FlashVID frame selector (fallback; RL-safe)
`compress_video` backed by a **persistent** sidecar holding just the Qwen3-VL vision
tower (576M params, 1.5 GB, 7.6s load — measured) + FlashVID's real last-block
attention salience; DySeg+ADTS with `alpha=1.0` (pure selection) → return top-K
frames **as images**. 2.65s per 10-min-span call (measured,
`scratchpad/probe_sidecar_flashvid.py`). Works unchanged in lmms-eval AND verl RL
rollout. Loses within-frame pruning/merge — archive says merge only tied selection,
so the loss is small. Build this if (or where) Option A's video path can't reach.

### Option C — in-process HF patched model
FlashVID's native route. Full fidelity, but no vLLM serving → no tool-loop parity,
no RL rollout. Reference/ablation arm only.

## 4. Implementation state & remaining steps (Option A)

**DONE:**
1. ✅ `examples/video_tools/mcp_server_flashvid.py` — new MCP server file:
   `crop_video` (identical behavior, images) + `compress_video` (validates span,
   returns `[VIDEO_CLIP] video_path=<p> start=<s> end=<e>` TextContent marker).
2. ✅ `video_tool_names` parameter added to `AsyncOpenAIChat.__init__` signature
   (currently unused — see next).

**REMAINING (in order):**

3. **Client marker handling** in `async_openai.py`:
   - store `self.video_tool_names = set(filter(None, video_tool_names.split(",")))`
     (+ comment) next to the other feature flags;
   - in the tool-call loop of `maybe_forward_with_tool`: when
     `call.function.name in self.video_tool_names`, parse the `[VIDEO_CLIP]` marker
     from the tool result, cut the clip client-side
     (`/usr/bin/ffmpeg -ss <start> -t <dur> -i <path> -vf "scale=min(448\,iw):-2" -an
     -c:v libx264 -preset veryfast -crf 28 <workdir>/clip_<id>.mp4`, run via
     `asyncio.to_thread(subprocess.run, ...)`), base64 it, and append tool content
     `[{"type":"text","text":"Compressed overview of <s>s-<e>s:"},
       {"type":"video_url","video_url":{"url":"data:video/mp4;base64,..."}}]`
     instead of the image-conversion path. On ffmpeg failure → text error content
     (never crash the sample). Videos do NOT count toward `max_total_images`; if a
     separate budget is wanted later, add `max_total_videos`.
4. **Plugin reinstall + sentinel fix**:
   - `/local1/cfyang/miniconda3/envs/vllm/bin/pip install -e /home/cfyang/hanklin/_archive/flashvid_plugin`
     (current editable install points at the pre-archive path → ModuleNotFoundError);
   - edit `_archive/flashvid_plugin/flashvid_vllm/__init__.py` `_SENTINEL` default to
     `/home/cfyang/hanklin/_archive/flashvid_plugin/FLASHVID_ON`. The sentinel FILE
     (not env) is the reliable gate because vLLM DP workers scrub custom env vars.
5. **Serve script FlashVID mode** (`serve_qwen3vl_baseline.sh`), gated e.g.
   `FLASHVID_COMPRESS=1`, adding: touch the sentinel; `--video-pruning-rate ${PRUNE:-0.7}`;
   `--media-io-kwargs '{"video": {"num_frames": 256}}'` (the compressed overview
   should sample MANY frames — pruning then cuts tokens);
   `--limit-mm-per-prompt.video 4` (default video limit is too low for multi-call
   conversations; keep `.image 2048`). Without the gate the script must stay
   bit-identical to today's baseline server.
6. **Run script variant**: point `MCP_PATH` at `mcp_server_flashvid.py`, add
   `video_tool_names=compress_video` to model_args. Keep the plain baseline script
   untouched for control runs.
7. **Validate (Probe 3)** — needs GPUs free (after the 900-run ends):
   - restart server in FlashVID mode; look for plugin log line
     `[FLASHVID] mask FIRED: T=... kept K/N` on first video request;
   - send the same 6s test clip (`scratchpad/tiny_clip.mp4` pattern) as video_url →
     `usage.prompt_tokens` should drop ≈ `r` vs the un-pruned measurement (~1.1k);
     send as images → token count unchanged (modality split confirmed);
   - 24-sample smoke with both tools (`--limit 24`, cache on, `partial_score.py`
     mid-run): check the model actually CALLS compress_video zero-shot (archive
     precedent: LongVT-RFT adopted a same-schema compress tool zero-shot on move 1),
     traces interleave compress→crop, no 400s.
8. **A/B when ready**: baseline (crop only; the 900-run now finishing is the control
   arm) vs +compress_video, matched token budget, same judge/protocol. Long-split
   VideoMME first, then VideoSIAH-Eval (videos not yet downloaded —
   `/local1/cfyang/hanklin/videosiah_eval/` has only the 96KB QA parquet).

**OPEN QUESTIONS (flag before the training phase):**
- verl/sglang RL rollout: does its tool-response path accept video content? One-day
  spike; if not, RL uses Option B (sidecar) while eval A/Bs both.
- Per-request pruning rate: vLLM has no per-request override; v1 fixes `r`
  server-side, budget varies via span/frame count in the call.
- Plugin salience: the vLLM plugin's v1 uses an L2-norm proxy instead of real CLS
  attention (its documented limitation; the archived 57.8% was achieved WITH this
  proxy). Real-attention exposure inside vLLM is a later upgrade; the sidecar probe
  already demonstrates real-attention selection if needed.

## 5. Failure-mode work (the other half of the ask)

After the 900-run completes:
1. `scratchpad/audit_full900.py` — faithfulness audit (900/900, dupes,
   `[EVAL_ERROR]`s, no-`<answer>` rate, judge-grant inspection, per-task table).
2. Replay failed doc_ids through the patched loop, splice into samples jsonl.
3. **Crop-usage failure census** over the traces (pattern:
   `videoarm/analysis/9b_flashvid_ab/census.md`): categorize wrong answers into
   (a) never cropped the answer-bearing range (coverage failure — the compress tool's
   target case), (b) cropped right, misread (perception), (c) cropped right, read
   right, answered wrong (reasoning), (d) format/loop artifacts. The archive's
   version of this census (59% never-gathered) is what motivated the whole chapter.
   Traces contain full tool history: `<tool_call>name args</tool_call>` +
   `<image_url>` markers + all think text.

## 6. Footguns (each cost real time today — do not rediscover)

1. **PATH**: login profile puts `dvd_tool` env first; it beats `conda run -n X`.
   Always hard-prepend `/local1/cfyang/miniconda3/envs/flashvid/bin` (also fixes the
   bare `python` MCP spawn) and `unset VIRTUAL_ENV`.
2. **Never edit `mcp_server.py` (or any MCP server file) while a run is live** —
   re-spawned from disk per call.
3. Editing `async_openai.py` does NOT affect running ranks (module in memory) — safe
   to edit, but running runs keep old behavior; failed samples need post-hoc replay.
4. vLLM **DP workers scrub custom env vars** → the plugin's sentinel-file gate, not
   `FLASHVID_ENABLE`, is what actually works under `--data-parallel-size > 1`.
5. lmms-eval writes results **only at run end** unless `LMMS_EVAL_USE_CACHE=True`;
   cache resume requires the same `--num_processes`.
6. CPU-only eval clients (CUDA_VISIBLE_DEVICES="") fall back to gloo → needed the
   evaluator.py gather fix (already applied). Keep clients off GPUs — they otherwise
   open NCCL contexts on other users' cards.
7. "Already borrowed" 400s = vLLM tokenizer race under concurrency; retries handle it.
8. VideoMME data symlinks: task expects `$HF_HOME/videomme/data/*.mp4`; symlinks
   created at `/local1/cfyang/.cache/huggingface/videomme/{data,subtitle}`. All 900
   long-split mp4s are h264 (AV1 census done — no black-frame risk on this copy).
9. `find` on `/local1/cfyang/.cache/huggingface/videomme/data` needs `-L` (symlink).
10. GPU safety (CLAUDE.md): NEVER `kill -9` GPU processes. SIGTERM → wait →
    `nvidia-smi`. Today's server restarts released memory cleanly this way.

## 7. Quick reference

| thing | value |
|---|---|
| model-under-test + judge server | `http://localhost:8010/v1`, `Qwen/Qwen3-VL-8B-Instruct` |
| LongVT-RFT demo server (leave alone) | port 8222, GPU 5 |
| eval env (lmms-eval editable + mcp + math-verify) | conda `flashvid` (py3.10) |
| serving env (vLLM 0.19) | conda `vllm` (py3.12) |
| FlashVID repo (HF patches, real algorithm) | `/home/cfyang/hanklin/FlashVID/flashvid/` |
| vLLM plugin (EVS symbol swap) | `/home/cfyang/hanklin/_archive/flashvid_plugin/` |
| probe scripts | session scratchpad: `probe_sidecar_flashvid.py`, `partial_score.py`, `audit_full900.py`, `tiny_clip.mp4` |
| outputs / logs | `/local1/cfyang/hanklin/outputs/` |
| measured numbers | sidecar: 576M/1.5GB/2.65s per call; 6s clip via chat API ≈ 1.1k tokens; smoke-8 baseline: 7/8 acc with fixes (3/8 stock); full-900 in flight |
