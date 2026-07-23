# FlashVID as an external tool in the LongVT pipeline — integration plan

*2026-07-11. Method plan only — nothing here is implemented yet. Every claim tagged
[VERIFIED] was tested today with a small probe (scripts in scratchpad, commands
reproduced below); [OPEN] means untested.*

## The constraint that shapes everything

Real FlashVID consumes **ViT-internal state** — per-token `video_features` +
last-block `cls_attention` — inside the model forward
([FlashVID/flashvid/modeling_qwen3_vl.py:299-311](../FlashVID/flashvid/modeling_qwen3_vl.py)).
The LongVT tool boundary (MCP → OpenAI chat API) carries **pixels**. So "FlashVID
as an external tool" cannot mean "the tool returns compressed tokens." It can mean
one of two things, and both are viable on this stack today:

- **(A)** the tool's *result modality* routes the clip through an engine-side
  FlashVID path (compression happens inside vLLM, invisibly to the tool protocol), or
- **(B)** the tool runs FlashVID's *selection* out-of-process and returns the
  surviving frames as ordinary images (compression output stays pixel-representable).

## Verified facts (today's probes)

| # | Fact | Evidence |
|---|------|----------|
| 1 | vLLM 0.19's Qwen3-VL still has the EVS hook (`compute_retention_mask`, gated on `--video-pruning-rate`) and it fires **only on video-modality inputs** — image inputs bypass it entirely | `vllm/model_executor/models/qwen3_vl.py:73-76,1104,1925` in the `vllm` env |
| 2 | The archived plugin (`_archive/flashvid_plugin`) still patches vLLM 0.19 cleanly: signature match, symbol swap confirmed in-process. Only its editable install is stale (points at the pre-archive path) | `FLASHVID_ENABLE=1 PYTHONPATH=_archive/flashvid_plugin python -c "import flashvid_vllm; flashvid_vllm.register()"` → `patched: True` |
| 3 | The **live 8010 server accepts `video_url` (base64 data URL) in both user and tool-role messages**; the model reads the content correctly; a 6s/320p clip cost ~1.1k prompt tokens | probe against running server, answered clip content correctly from inside a `role:"tool"` message |
| 4 | A **sidecar vision tower is cheap**: Qwen3-VL-8B's ViT alone = 576M params, 1.5 GB VRAM, 7.6s one-time load. A tool-shaped call (10-min span, 64 frames, real FlashVID DySeg+ADTS on real last-block attention) = **2.65s total**, decode-dominated (decode 1.5s, ViT 0.33s, selection 0.06s) | `scratchpad/probe_sidecar_flashvid.py` on GPU5, flashvid env (transformers 4.57.3 + flash-attn 2.7.4) |
| 5 | lmms-eval's MCP client converts tool results to `image_url`/text only — a video-returning tool needs a small client-side change (MCP has no video content type) | `FlashVID/lmms-eval/lmms_eval/mcp/client.py`, `models/chat/async_openai.py` |

## Option A — modality-routed engine compression (recommended for eval)

**One server, two visual bandwidths, selected per tool call.** Serve 8010 with
`FLASHVID_ENABLE=1` + `--video-pruning-rate r` (plugin swaps EVS selection for
FlashVID DySeg+ADTS). Then, by fact 1:

- `crop_video` keeps returning **images** → full-resolution tokens (today's behavior, bit-identical);
- a new `compress_video(video_path, start, end)` returns its clip as **video** →
  engine-side FlashVID prunes it to `(1-r)` of tokens before the LLM sees it.

The agent's *choice of tool* is literally the compression decision — exactly the
"interleaved compressed-global / full-crop modes as tool actions" the chapter
thesis calls for, with zero changes to the model or the tool-call format.

Changes required (small, all in code we already own/patched — listed, not implemented):
1. `async_openai.py`: when the called tool is `compress_video`, attach the tool
   result as a `video_url` content part (client fetches the clip and base64s it;
   the MCP tool itself just validates and echoes the span — fact 5). ~20 lines.
2. Re-point the plugin's editable install at `_archive/flashvid_plugin` (or move
   the dir back); update the `FLASHVID_ON` sentinel path in it and in
   `server_scripts/start_flashvid.sh`.
3. `serve_qwen3vl_baseline.sh`: add the env gate + `--video-pruning-rate`.
4. Optional stronger arm: send the *global* view as video too → compressed global
   skim (many more frames at the same token budget). The archive's FlashVID-map
   result (57.8% on matched-633, tied with AKS-map) is direct evidence this arm
   is competitive.

Properties: per-call **budget** is controlled by span length / frame count in the
tool call; the **rate** `r` is server-global for v1 ([OPEN] vLLM has no
per-request pruning override — acceptable: budget varies, rate fixed). EVS's
reserved-seat accounting keeps prompt sizes predictable. The plugin degrades to
stock EVS on any error (its design), so the failure mode is "different
compression," not a crash.

Known gaps to validate ([OPEN], queued as Probe 3 when GPUs free up):
- Restart 8010 with plugin + pruning; rerun today's video probes; confirm the
  `[FLASHVID] mask FIRED` stderr line, prompt_tokens drop ≈ r on video inputs,
  and image inputs unchanged.
- Salience source: the plugin's v1 uses an L2-norm proxy instead of real CLS
  attention inside vLLM (its documented limitation). The archived 57.8% result
  was obtained WITH this proxy — good enough to start; real-attention exposure is
  a later upgrade.
- [OPEN] payload size: a 256-frame 112px clip re-encoded to mp4 is a few MB of
  base64 — fine for localhost; measure at Probe 3.
- [OPEN] **RL-side parity**: verl's sglang rollout consumes tool results via its
  own multimodal path (`process_image`); whether it accepts video content needs a
  spike *before* the training phase. If it doesn't, training uses Option B and
  eval A/B-tests both.

## Option B — sidecar FlashVID selector (pixel boundary, RL-ready today)

`compress_video` is a normal MCP tool backed by a **persistent** sidecar process
(NOT the current spawn-per-call pattern) holding the 576M vision tower. Per call:
decode span → tower forward → DySeg+ADTS with `alpha=1.0` (pure selection, no
merge) → aggregate kept-tokens per frame → return the **top-K frames as images**
(with timestamp texts), K set by the token budget.

- Works unchanged in *both* the lmms-eval harness and verl RL rollout (both
  already consume image lists) — no serving-stack changes at all.
- Real FlashVID salience (actual last-block attention — fact 4), 2.65s/call.
- What's lost vs A: within-frame token pruning (frames return whole) and
  merge-based compression. The archive says this loses little: merge only *tied*
  selection (57.8% = 57.8%, p=1.00), and selection won the tasks this chapter
  cares about.
- Cost: ~1.5 GB VRAM co-tenant + double ViT compute (sidecar scores, main model
  re-encodes the survivors).

## Option C — in-process HF patched model (reference only)

FlashVID's native route (patch `Qwen3VLForConditionalGeneration.generate`).
Full-fidelity token-level compression, but abandons vLLM serving: no tool-call
loop parity, ~10× slower eval, unusable for RL rollout. Keep solely as a
ground-truth ablation arm if A/B results are confusing.

## Recommendation and sequencing

1. **Build Option A on the eval stack first** (it's ~4 small changes, all in
   already-patched files). A/B on VideoMME-long (today's zero-shot baseline = the
   control arm, same harness/judge/seeds): baseline vs +`compress_video`, matched
   token budget.
2. **Decide the RL path with a one-day spike**: test video content through verl's
   sglang rollout. Falls back to Option B for training if unsupported — B is also
   the cleaner scientific comparison arm (selection-to-pixels vs engine
   merge+prune at the same budget re-runs the archive's central question *inside*
   the agent loop).
3. Defer C unless needed as ground truth.

## Measured numbers to design against

| quantity | value |
|---|---|
| sidecar tower load (one-time) | 7.6s, 1.5 GB VRAM |
| sidecar tool call (64f / 10-min span) | 2.65s (decode 1.50 + prep 0.75 + ViT 0.33 + select 0.06) |
| 6s/320p clip as video through chat API | ~1.1k prompt tokens, accepted in tool role |
| `crop_video` today (128f, images) | ~10-60s per call incl. per-call python spawn (existing baseline behavior) |
| zero-shot agent per sample (capped 8 rounds) | ~1-12 min, generation-dominated |

## Probe scripts (reproducible)

- Video-in-tool-message acceptance: inline probe, this session — 6s clip at
  `scratchpad/tiny_clip.mp4`, both message roles → 200 OK + correct description.
- Sidecar latency/selection: `scratchpad/probe_sidecar_flashvid.py`
  (`CUDA_VISIBLE_DEVICES=5 conda-run flashvid python probe_sidecar_flashvid.py <video.mp4>`).
- Plugin-on-0.19: PYTHONPATH import + `register()` symbol-swap check (fact 2).
