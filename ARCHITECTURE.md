# Architecture — orientation for a reviewer

This document is written for someone (or some agent) who has to read this code cold and
say whether the experiments it produced are valid. It covers what the code does, what
claims it has to support, the invariants that must not break, and the places where the
author is least confident and would most like a second opinion.

---

## 1. What is being measured

One question: **at a fixed visual-token budget, does ViT-side token compression help a
video-language model answer questions about hour-long video?**

The comparison that answers it is always the same shape. Take a video, sample N frames,
spend a fixed number of visual tokens on them, and vary only *how those tokens are chosen*:

- **uniform** — shrink every frame until the total fits the budget. No selection at all.
- **VidCom2** — keep the tokens furthest from the whole-video and per-frame centroids.
- **EVS** — keep the tokens that changed most from the previous frame.

If a selector is doing useful work it beats uniform at the same budget. Measured on
LVBench with Qwen3-VL-8B, neither does, at either of two operating points. An oracle that
crops to the ground-truth evidence span instead scores +13.23 pp using half the tokens.

Everything in this repository exists to make that comparison trustworthy, which mostly
means making it hard for an arm to differ from another arm in some way nobody noticed.

---

## 2. The one thing to understand before reading any code

**Qwen3-VL budgets the whole video, not each frame.**

`video_preprocessor_config.json` sets `size.longest_edge = 25,165,824` pixels. Divide by
`temporal_patch_size` (2) and by 1,024 px per output token (patch 16 × merge 2 = 32×32 px
per token) and you get a hard ceiling of **12,288 visual tokens for the entire clip,
regardless of how many frames you send**.

Consequences that shape the whole design:

- Asking for more frames does not buy more tokens. It divides the same budget more thinly.
  32 frames → 768 tokens per grid step; 256 frames → 96.
- Frames pair up: `temporal_patch_size = 2`, so N frames become N/2 *grid steps*, and
  token counts are per grid step, not per frame. Reviewers should watch for off-by-2 in
  any arithmetic here.
- To give an arm more budget you must raise `size.longest_edge` via
  `--mm-processor-kwargs`. `serve_arm.sh` calls this the "cap", `1×`/`2×`/`4×`.
- `max_pixels` **does nothing for video.** It is an image-processor knob;
  `Qwen3VLVideoProcessor` reports it as unrecognized and the measured budget is identical
  with and without it. It is still passed so launch strings match earlier arms verbatim.

---

## 3. Data flow

```
LVBench parquet                     103 videos, 1,549 questions, median 3,666 s
      |
      |  data.py            load, split question from options, parse the GT
      |                     evidence span out of `time_reference`
      v
make_skim_proxy.py          ONE small N-frame mp4 per video, written at
      |                     fps = N / duration so the source timeline survives
      v
  proxy .mp4  ---- file:// ---->  vLLM server (serve_arm.sh)
                                    |  decodes, resizes to the cap,
                                    |  inserts one <t seconds> marker per grid step,
                                    |  optionally prunes (VidCom2 / EVS)
                                    v
run_agent.py  <---- OpenAI chat completions ----
      |  turn loop: skim -> optional crop_video rounds -> <answer>
      |  tools.py serves crops as IMAGES (never pruned)
      v
results.jsonl + traj/*.json
      |
      +--> diagnose_runs.py     is this run comparable to the others?
      +--> analyze_pairs.py     paired deltas, McNemar, bootstrap CI
      +--> probe_budget.py      what budget did this config ACTUALLY use?
```

---

## 4. Module map

**The live path — every number in the report came through here.**

| File | Responsibility | Notes for a reviewer |
| --- | --- | --- |
| `data.py` | LVBench loading, `question + options` normalisation, `<answer>` extraction, `time_reference` → `(t0, t1)` seconds | `_split_lvbench_question` is where option formatting could silently diverge from upstream |
| `config.py` | Prompts and tool schemas. LongVT-verbatim, plus an "informed" variant that states environment facts only | The no-tool arm must get the no-tool prompt; `--print-prompt` proves it |
| `make_skim_proxy.py` | Pre-builds the proxy mp4s | The timeline preservation lives here. See §5 |
| `serve_arm.sh` | Every vLLM server configuration, arms `a`..`p`, with the measured budget recorded per arm | The single source of truth for what each arm actually served |
| `tools.py` | `crop_video`: decode a span at fps 1, ≤128 frames, return as images | Images are returned deliberately — vLLM prunes video only, so a crop must not be pruned |
| `run_agent.py` | The turn loop, concurrency, forced-crop (oracle / decoy) modes, results and trajectory writing | ~700 lines and the biggest single thing to review |
| `run_plain.py` | Single-shot eval; also carries an lmms-eval-compatible prompt and scorer for reproducing a published number | |
| `longvt_scoring.py` | lmms-eval's `extract_mcq_answer`, vendored verbatim | Vendored rather than reimplemented on purpose |

**Verification tools — added after four runs had already been silently corrupted.**

| File | What it protects against |
| --- | --- |
| `probe_budget.py` | Believing `prompt_tokens` is the visual budget. See §6 |
| `diagnose_runs.py` | Arms that differ in something nobody noticed: timeline drift, mismatched question sets, duplicate rows, transport errors, unequal hyperparameters |
| `preflight.py` | Launching against a server whose flags do not match the intended arm, or whose pruning plugin never activated |
| `analyze_pairs.py` | Unpaired or wrongly-paired comparisons. See §7 |

**The compressor plugin.**

`vidcom2_vllm/` implements VidCom2 as an out-of-tree vLLM plugin. `patch.py` rebinds
`vllm.multimodal.evs.compute_retention_mask` inside the model module — it patches
`qwen3_vl`'s already-bound reference rather than the `evs` module, because `qwen3_vl.py`
does `from vllm.multimodal.evs import compute_retention_mask` at import time and rebinding
the source module afterwards would have no effect.

The patch is a **no-op unless `FA_PRUNE_METHOD=vidcom2`**, which is how the EVS arms are
served: same launch command, one environment variable different, zero code difference.
That is deliberate — it removes "the two compressors were served differently" as an
explanation for any gap between them.

---

## 5. The invariants

Each of these was learned by having a run silently produce a plausible wrong number.

**The proxy must reproduce the source timeline.** Qwen3-VL derives each frame's
`<t seconds>` marker from `frame_index / fps` read off the container. A proxy written at a
convenient frame rate would relabel a 3,664-second video as 32 seconds; every crop the
model then requested would be wrong by two orders of magnitude, and the crop tool — which
reads the *real* video — would happily serve frames from the wrong place. Proxies are
written at `fps = n_frames / duration` and re-probed to confirm. `diagnose_runs.py` checks
drift across all 103 videos.

**Proxy resolution is fixed at build time.** The filename is `{video_id}_n{frames}.mp4` and
carries no resolution. Mixing resolutions in one directory serves the wrong pixels with no
error. Each `--max-side` needs its own `--out-dir`.

**A no-tool arm must get the no-tool prompt.** Sending the tool prompt to an arm with no
tools invalidates it as a vanilla baseline. Verify with `--print-prompt`, which prints the
content blocks and the tool schema (`null` when there are no tools).

**Pair on `question_id`, never on line order.** `results.jsonl` is written in completion
order under N workers. Pairing positionally once produced 351/289 flips where the truth was
157/95 — accuracies were unaffected, every significance test was wrong.

**An interrupted run resumes by appending, not de-duplicating.** Two failed launches plus
one clean pass produced 777 rows for 500 questions, 277 duplicated, 3 with disagreeing
predictions under T=0.7. `diagnose_runs.py` counts duplicates; any run with some should be
re-run clean.

**Below-chance accuracy means transport failure until proven otherwise.** One run was
launched against a port with no server; 1,362 connection errors were recorded as
`pred=None, correct=False` and it scored 4.71% against a 24.6% chance floor.

---

## 6. Why `prompt_tokens` is not the budget

The processor interleaves a `<t seconds>` marker before **every grid step**, so the text
contribution scales with frame count: 128 markers at 256 frames, 32 at 64 frames.

| Arm | `prompt_tokens` | real visual tokens | timestamp + text |
| --- | ---: | ---: | ---: |
| 64 frames, 1× cap | 12,106 | 11,648 | 458 |
| 256 frames, 1× cap | 13,136 | 11,648 | 1,488 |

Measured on `prompt_tokens` these arms look 8.5% apart in budget. Their visual budgets are
identical. The difference is the same order as the effects being measured.

Within a fixed frame count the tax is constant and `prompt_tokens` differences are safe.
Across frame counts they are not. `probe_budget.py` computes the real figure by running the
same processor with the same kwargs the server is launched with, and derives the
post-compression count with vLLM's own `compute_retained_tokens_count` — the same function
`retention.py` is written to reproduce exactly, so one number is correct for both selectors.

**One subtlety worth checking:** vLLM presamples frames itself via
`--media-io-kwargs num_frames` and then calls the processor with `do_sample_frames=False`.
Omitting that flag offline makes the processor resample to its own default and report
`grid_t = 11` instead of `N/2`. The probe sets it. This is exactly the class of divergence
between the offline replication and the served path that a reviewer should hunt for more of.

---

## 7. Statistics

Comparisons are paired on `question_id` and report:

- accuracy of each arm on the shared question set,
- flip counts (A-only-correct / B-only-correct),
- continuity-corrected McNemar |z|,
- a percentile bootstrap 95% CI on the paired delta, resampling question IDs.

Sampling is T=0.7, so per-question answers are noisy: across three seeds of one config,
aggregate accuracy has a standard deviation of 0.05 pp at n=1,549 while **26.1% of
individual answers churn**. Aggregate accuracy is therefore safe to compare across arms;
per-question flip analysis only means something when paired.

Reported effects are small — most in the 1–4 pp range with confidence intervals that
include zero. The claims are phrased accordingly ("neither selector shows any advantage"),
and the weight is carried by consistency across four measurements, two operating points and
two unrelated mechanisms, not by any single significant result.

---

## 8. Where the author is least confident

These are the review targets, in order.

1. **Does `probe_budget.py` faithfully replicate the served path?** It reconstructs the
   processor call offline. One divergence (`do_sample_frames`) was found and fixed; there
   could be others. Everything about budget matching between arms rests on this being
   right. The strongest check available would be reading the true post-PatchMerger
   embedding count out of the server process rather than recomputing it — that was not
   done.

2. **Does `vidcom2_vllm/retention.py` actually reproduce VidCom2?** It is a
   reimplementation, not a port. It reconciles a per-frame budget to an exact total via
   largest-remainder apportionment because the reference rounds independently per frame and
   drifts off any target — a deliberate divergence, documented in the file. The plugin
   reproduces VidCom2's published VideoMME-long numbers within 0.35 pp, which is evidence
   but not proof. `test_retention.py` covers the mask arithmetic.

3. **Are the arms genuinely matched?** `diagnose_runs.py` checks 17 pinned fields, but
   `gpu-memory-utilization` was *not* held constant (0.55/0.70/0.80, forced by vision-tower
   OOM). The argument that it only sizes the KV cache and never the arithmetic is
   plausible; it has not been tested by re-running an arm at two values.

4. **`run_agent.py` is the largest unreviewed surface.** Turn loop, tool dispatch, the
   forced-crop modes, context-overflow handling, concurrency. Bugs here would affect the
   tool arms more than the no-tool arms that carry the headline result, but they would
   affect the oracle number, which is the largest effect reported.

5. **17 of 103 videos are below 720p.** For those, source resolution binds before the token
   cap, so "matched budget" holds for 1,345 of 1,549 questions and not the rest. Stratified
   analysis shows the frame-count effect is the same size in both strata, so this does not
   appear to drive anything — but it is a real heterogeneity in what "matched" means.

6. **The oracle comparison changes two things.** It concentrates the budget *and* delivers
   the evidence as full-resolution images through the crop tool rather than as skim frames.
   A wrong-location control (`ctrl`, crop forced to a decoy span, 37.06%) exists to
   separate "extra pixels" from "right pixels", and it lands below the model-chosen crop
   arm — but the two factors are not fully disentangled.

---

## 9. What is deliberately not here

Removed rather than shipped: exploratory and presentation notebooks, per-case debug traces
and visualisations, logs from an earlier VideoMME chapter, design notes for a compressor
that never entered the results, judge servers, one-off launch and babysitting scripts, and
a legacy local-HuggingFace inference path (`model.py` / `agent_loop.py` / `oracle.py` /
`semvid.py`) that no reported run used — every arm went through the vLLM server.

The kept set was computed as the import closure of the entry points the README documents,
then checked by hand, not chosen by eye. Two pure token-math helpers were moved out of the
deleted `model.py` into `probe_budget.py`, where the rest of the token arithmetic lives.
