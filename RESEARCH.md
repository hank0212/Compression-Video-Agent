# Research Notes

> **Auto-loaded** when you ask Claude about anything research/experiment-related (the UserPromptSubmit hook detects keywords like *eval, experiment, benchmark, baseline, VideoARM, VideoMME, compression, ViT, ablation, paper*). Otherwise it sits quietly. Keep this file the single source of truth for research state. PreCompact hook reminds Claude to update this file before context is compacted.

## North star
Compression/perception as an **action a video agent takes**, not a fixed preprocessing step — the agent learns to allocate its visual bandwidth (wide+coarse vs narrow+detailed) in service of answering better.

---

# THE FIVE RUNS — the experiment (canonical, set 2026-08-13)

Everything else in this file is background or diagnostics. **These five are the experiment.**
Each row changes exactly one thing from the row above it, and all five share the same
loader, prompt family, crop tool, scoring, and sampling params.

| run | skim | tools | output dir |
|---|---|---|---|
| **run1** vanilla | 64-frame video | none | `lvbench_r1_seed0` |
| **run2** + crop | 64-frame video | `crop_video` | `lvbench_r2_seed0` |
| **run3** + compression | 256-frame video, **VidCom2 r=0.25** | none | `lvbench_c3_notool_seed0` |
| **run4** + compression + crop | 256-frame video, **VidCom2 r=0.25** | `crop_video` | `lvbench_r4_seed0` |
| **run5** oracle | 64-frame video | `crop_video` **forced to the GT span** | `lvbench_r5_seed0` |

run1→run2 = what the tool is worth. run1→run3 = what compression does to the skim alone.
run3→run4 = the tool on top of a compressed skim. run2→run5 = **the localization gap**:
what perfect aiming would be worth.

> ⚠️ **run3's directory was corrected 2026-08-13.** It previously pointed at
> `lvbench_r3_seed0`, which is a **dead run**: it was launched against port 8032 with no
> server on it, so **223 of its 410 rows are `APIConnectionError`** written as
> `pred=None, correct=False`. That scores 17.8% — *below 4-way chance* — and would have
> made run1→run3 read as −21 pp instead of the true −1.03 pp. The valid arm is
> `lvbench_c3_notool_seed0`: n=1,549, **0 errors**, pruning verified from token counts
> (skim-only median 4,445 tok vs 13,180 unpruned). Delete `lvbench_r3_seed0`.
>
> **Always check `sum('error' in r)` over results.jsonl before quoting an accuracy.**
> This is the second time this exact failure mode has bitten this chapter.

### Fixed settings — identical across all five
| | |
|---|---|
| model | Qwen3-VL-8B-Instruct, **vLLM only, never HF** |
| skim modality | **video** (never images — that is where the per-frame `<t seconds>` markers come from, and video is the only modality vLLM prunes) |
| skim source | pre-built proxies, `--require-proxy` (`make_skim_proxy.py`) |
| budget | ⚠️ **NOT matched — corrected 2026-08-14, see "MEASURED PIXEL BUDGET" below.** Measured on `qAIRFyR6NyQ`: run1/2/5 = **3,584** tok · run3/4 = **2,912** tok (19% apart, against run3). The old "2,688 exactly matched, verified" row was never measured per video. |
| crop tool | **LongVT's own settings**: ≤128 frames, fps 1, `max_pixels` 224², returned as **images** so they are never pruned |
| prompt (tool arms) | LongVT verbatim (`config.longvt_user_text` + `LONGVT_TOOL_SCHEMAS`) |
| prompt (vanilla arms) | LongVT's **no-tool** prompt (`config.LONGVT_SYSTEM_PROMPT` + question only) |
| sampling | T=0.7, top_p 0.8, top_k 20, penalties 0/0/1.0, max_tokens 1024, finalizer 32 |
| rounds | ≤5 tool rounds + one bounded finalizer |
| trajectories | **always stored** at `<run>/traj/<qid>.json`, with the GT evidence span and per-call coverage |
| dataset | LVBench, all **1,549** questions |

### The two headline metrics
- **overlap / hit** — `covered_frac` = |union(crop spans) ∩ GT evidence| / |GT evidence|;
  **landed** = `covered_frac == 1.0`. Reported as mean coverage and hit rate.
- **accuracy**, overall and split by landed vs not-landed.

### GOTCHA — a vanilla arm must not get the tool prompt (found 2026-08-13, run1 re-run)
The first run1 was sent LongVT's **tool** prompt — *"Think first, call **crop_video** if
needed … The Video path for this video is: /local1/…"* — while holding no tool at all.
`build_first_turn` applied the tool prompt whenever `--prompt-style longvt` was set,
regardless of `--tools`. LongVT ships a separate no-tool prompt for exactly this
(`single_inference.py`, `--no_tool`: a system turn carrying the `<think>/<answer>` contract
and nothing else); `run_agent.py` now branches on `args.tools`. **Any no-tool arm run before
2026-08-13 is invalid as a vanilla baseline.**

Generalisation worth keeping: *print the prompt for every arm before launching it.*
`--print-prompt` exists for this and would have caught it in one second.

### MEASURED PIXEL BUDGET — three corrections (found 2026-08-14)

All three were measured by driving vLLM's **real** `Qwen3VLMultiModalProcessor` on the
actual proxies. Reproduce any of them in `longvt_compression/OWNERSHIP.ipynb` §5, or with
`python -m longvt_compression.fast_agent.preflight --measure-video <proxy> --frames N`.

**1. `--mm-processor-kwargs '{"max_pixels":...}'` is INERT for video in vLLM 0.19.**
Sweeping it over `{}`, 50176, 200704, 786432 gives a byte-identical `video_grid_thw`.
vLLM *does* translate the kwarg into `size["longest_edge"]`, but only inside
`_get_vision_info`, which is used to *estimate* token counts for scheduling; the real
processor call passes `mm_kwargs` straight to `Qwen3VLVideoProcessor.preprocess`, whose
signature takes `size` and silently drops `max_pixels`. Only `{"size": {"longest_edge": …}}`
moves the grid.
→ **Arms a–f all ran at the checkpoint default `size.longest_edge = 25,165,824`**, not at
LongVT's 224² and not at arm e/f's 786432. Arms e/f got their ~720 tok/frame from the
hi-res proxy plus the default budget — *not* from the flag. `serve_arm.sh`'s `MMKW` lines
are decoration for the video path. (The crop tool is unaffected: `tools.py::_smart_size`
resizes crops to 224² client-side before they are ever sent.)

**2. Video `max_pixels` is a WHOLE-VIDEO budget, not per-frame.**
`smart_resize` compares `t_bar * h_bar * w_bar` against `max_pixels`
(`transformers/models/qwen3_vl/video_processing_qwen3_vl.py:55`). So **asking for more
frames automatically lowers per-frame resolution.** Measured on the same 448×250 proxy:
64 frames → 256×448 px → **112 tok/frame**; 256 frames → 224×416 px → **91 tok/frame**.
That is the mechanism behind correction 3.

**3. run1 and run3 are not budget-matched.** On `qAIRFyR6NyQ`:

| arm | grid T | tok/frame | pre | post | surviving tok/frame |
|---|---|---|---|---|---|
| run1/2/5 64f | 32 | 112 | 3,584 | 3,584 | **112.0** |
| run3/4 256f r=0.25 | 128 | 91 | 11,648 | 2,912 | **22.8** |
| arm e 32f hi-res | 16 | 720 | 11,520 | 11,520 | **720.0** |
| arm f 32f hi-res r=0.25 | 16 | 720 | 11,520 | 2,880 | **180.0** |

run3 spends 19% **fewer** visual tokens than run1, not the same. And run3 vs arm f — both
`r=0.25`, both ~2,900 tokens — differ **7.9×** in surviving tokens per frame. `r` is a
ratio; what a frame can still describe depends on the absolute count. VidCom2's validated
operating point is ~180 tok/frame after pruning; **run3 sits at 22.8, ~8× below it.**
Any statement of the form "VidCom2 at r=0.25 hurt accuracy" must carry that qualifier.

Also worth recording: **tokens/frame is video-dependent** (aspect ratio × frame count), so
no single number can be quoted for the dataset. `serve_arm.sh`'s comments quote 47 and 49
for arms whose measured values here are 112 and 91.

**Follow-ups this opens:**
- measure the run1-vs-run3 budget gap across all 103 videos before quoting any "matched budget"
- **run arm f.** Arm e (32f hi-res, no pruning) is banked at 39.70% / n=1,549. Arm f is
  configured but not banked. `arm f − arm e` is the direct test of VidCom2 *at its own
  operating point*, and separates "compression hurts" from "we ran it 8× outside its regime".
- check whether `SOFTMAX_TEMP=0.01` in `vidcom2_vllm/retention.py` matches the reference
  implementation: at that temperature the per-frame budget is degenerate (measured: **124
  of 128 frames get an identical 22-token budget**; effective frames sharing the allocation
  bonus = 1.25 of 128). VidCom2's "adaptive frame budget" is doing essentially nothing here.

### Matched-subset results (2026-08-14) — always compare on the same question ids

| comparison | n | result |
|---|---|---|
| run1 64f · **diag 256f uncompressed** · run3 256f r=0.25 | 253 | 39.92% · **44.66%** · 36.76% |
| run1 64f · arm e 32f hi-res uncompressed · run3 | 1,549 | 39.77% · 39.70% · 37.96% |
| run1 · run2 (model aims) · **run5 (oracle aims)** | 1,549 | 39.77% · 39.57% · **55.39%** |

Reading: 4× temporal coverage is worth **+4.74 pp**; compressing it to 25% gives back
**−7.90 pp**. 6.4× per-frame resolution is worth **−0.07 pp** (nothing). The model's own
cropping is worth **−0.20 pp**; perfect cropping is worth **+15.62 pp**. On LVBench,
**temporal coverage is the binding constraint and spatial fidelity is not** — and almost
all of the tool's value is locked behind localization the model cannot currently do.

### VIDCOM2 IMPLEMENTATION SMOKE TEST — PASSED (2026-08-22)

**Question asked:** is our vLLM VidCom2 plugin obviously behaving differently from the
authors' official implementation? (Not: are they mathematically equivalent.)

**Design.** 2x2 on a fixed 100-question LVBench subset (seed 0, 66 videos), everything
identical except the implementation: same checkpoint, same `skim_proxies_hires_64` proxy
file, same 64 frames, same lmms-eval `lvbench` MCQ prompt, same system prompt
("You are a helpful assistant."), greedy, `max_new_tokens=16`, same parser
(`extract_characters_regex`, vendored). NO LongVT prompt, NO `<think>`, NO T=0.7, NO agent.
Official arm = HF `Qwen3VLForConditionalGeneration` + `token_compressor/vidcom2` from the
repo's `qwen` branch, patched onto `model.model.forward` exactly as their lmms-eval
wrapper does. Code: scratchpad `smoke/{common,run_official,run_ours,analyze}.py`;
results `/local1/cfyang/hanklin/outputs/vidcom2_smoke/`.

| | vanilla | VidCom2 R=0.25 | delta |
|---|---|---|---|
| **official HF** | 44.0% | 47.0% | **+3.0 pp** (CI [-3.0, +10.0]) |
| **ours vLLM** | 44.0% | 43.0% | **-1.0 pp** (CI [-7.0, +5.0]) |

`D = delta_official - delta_ours = +4.0 pp`, 95% CI **[-5.0, +13.0]** — contains 0.

**Structural check (the part that actually settles it).**

| | pre | post | ratio |
|---|---|---|---|
| official | 24,573 | 6,142 | **0.2500** |
| ours | 24,573 | 6,143 | **0.2500** |

Per-question post-prune count differs by at most **19 of ~24,573 tokens (0.08%)**, mean +1.1
— exactly the size expected from `round()` vs our largest-remainder apportionment, and
nothing else. `prompt_tokens` were **identical on 100/100 questions**, so prompt text, chat
template and `video_grid_thw` are byte-identical between the stacks.

**Verdict: no VidCom2 integration bug.** Compression fires, the ratio is exact, the vanilla
arms are indistinguishable (44.0% vs 44.0%, 97.0% prediction agreement), and the
compression deltas differ by less than this test can resolve.

**Detection limit, stated up front:** at n=100 the CI half-width on `D` is ~9 pp. This test
rules out a gross bug (the "-3 vs -13" case). It CANNOT certify a 4 pp difference. See the
power table in `longvt_compression/VIDCOM2_VALIDATION_PLAN.md` §6.3: a +/-2 pp equivalence
band needs n ~ 1,900-3,800, i.e. more than all of LVBench.

**The one residual signal.** Prediction agreement drops **97.0% (vanilla) -> 79.0%
(compressed)**. Within-stack prediction churn from compression is 15% (official) vs 22%
(ours). The two compressors are demonstrably selecting different token subsets. That is
expected by design, but 18 points is more than a <=19-token allocation difference would
naively explain. If this ever needs settling, Stage 0A/0B of the validation plan (offline
scorer parity on one shared `video_embeds`) is the ~30-minute experiment that answers it.

**RESEARCH FINDING worth more than the validation.** At **64 frames, native resolution
(880 tok/grid-step, 220 surviving)**, VidCom2 at R=0.25 costs **nothing** in either
implementation (+3.0 and -1.0, both CIs spanning 0). The banked 256-frame arm at
**91 tok/grid-step (22.8 surviving)** lost **-7.90 pp**. Consistent with: VidCom2's failure
depends on the per-temporal-step token regime, not on the retention ratio. This is the
`arm f - arm e` question and it now has a first data point. NOTE the earlier claim that
"~180 tok/frame is VidCom2's validated regime" was never sourced -- it was our own arm f
measurement, not the authors'. Retracted.

### TWO CONFOUNDS FOUND WHILE SETTING THIS UP (2026-08-22) — both affect banked arms

**1. `serve_arm.sh cap_for()` makes the unpruned and pruned grid arms DIFFERENT resolutions.**
`cap_for()` returns `CAP_GENEROUS = 230,686,720` when `RATE` is empty and
`(N/2)*880*2048` when it is set. For 64 frames that is 230,686,720 (u64) vs 57,671,680 (v64).
A 1280x720 source -- **53 of the 66 videos in this subset** -- saturates at 900 tok/grid-step
under the first and is clipped to **880** under the second. So **`u64` and `v64` are not
resolution-matched**, by ~2.3%, against the compressed arm. Same argument applies to
u32/v32 and u128/v128. The smoke test pinned both stacks at 57,671,680 via `MAXPIX_FORCE`.
**Any `u64 - v64` number already banked carries this offset.**

**2. Concurrency + `max_num_batched_tokens` crashes the UNPRUNED path.**
Running the 64f native arm with 4 concurrent requests (~28.6k visual tokens each) against
`--max-num-batched-tokens 32768` made vLLM chunk a prefill **through the middle of a video's
placeholder run**; `_compute_deepstack_embeds` then died with
`ValueError: Error during masked scatter operation` and took EngineCore with it
(`server_arm_u64.log`, 2026-08-22 05:41). GPU memory was released cleanly. Fix used:
`--workers 1`. Anything driving an unpruned native-resolution arm concurrently is exposed
to this.

### THE OFFICIAL lmms-eval QWEN3-VL PATH LOSES 10x THE FRAMES (measured 2026-08-22)

Their wrapper hands the HF processor a **pre-sampled frame tensor with no `video_metadata`**.
Under transformers 4.57.6, measured on one of our 64-frame proxies:

| processor call | grid_thw | video tokens | timestamps |
|---|---|---|---|
| with metadata, `do_sample_frames=False` | `[32,16,28]` | 3,584 | 41.7 … 5,208.0 s (correct) |
| **no metadata (their default path)** | `[3,16,28]` | **336** | 0.3 … 2.6 s |
| no metadata, `do_sample_frames=False` | `[32,16,28]` | 3,584 | 0.0 … 2.6 s (wrong) |

`do_sample_frames` defaults to `True`, and with `metadata.fps` defaulting to 24 the processor
recomputes `num_frames = int(total/24*2)` and **resamples 64 frames down to 6**. Timestamps
become `index/24`, so a 5,333 s video is labelled 0-2.6 s.

This is a **video-loader / temporal-metadata defect in the official path, NOT a VidCom2
property**, and it is version-dependent -- we cannot claim it affected their published
numbers. But it means: (a) the smoke test had to pass explicit metadata and
`do_sample_frames=False` on both sides (recorded deviation), and (b) **the official HF path
must not be used as a temporal-reasoning reference on LVBench** without this fix.

### ONBOARDING / DEBUGGING ENTRY POINTS (added 2026-08-14)
- `longvt_compression/OWNERSHIP.ipynb` — executable trace of one LVBench question
  (qid 8, `qAIRFyR6NyQ`) through every stage, with measured tensor shapes and token counts,
  the VidCom2 and EVS implementations read line by line, and a claim-tagging convention
  (RUNTIME / SOURCE / PAPER / OURS). Run it before touching the pipeline.
- `longvt_compression/fast_agent/preflight.py` — run before every expensive arm. Verifies
  env, server launch flags, compressor proof-of-life from **EngineCore**, proxy timeline,
  measured grid/retention, prompt-vs-tools agreement, and output-dir cleanliness.

---

## MEASURED 2026-08-28 — CLIP query-frame relevance DOES localize evidence (AUC 0.64–0.66)

**Cheap offline diagnostic, no server, no generation, ~9 min on one GPU.** Asked one
question: does an *aligned* image-text scorer separate in-evidence grid steps from the
rest, where every previously tested signal was blind? **Yes, decisively.**

### The command and where the output is
```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/home/cfyang/hanklin \
  SIGNAL_AUC_OUT=/local1/cfyang/hanklin/outputs/clip_signal_auc/records_v60_f128.json \
  python -m longvt_compression.fast_agent.signal_auc --videos 60
```
- log: `/local1/cfyang/hanklin/outputs/clip_signal_auc/run_v60_f128.log`
- per-question records: `.../clip_signal_auc/records_v60_f128.json`
- scorer: the repo's existing AKS port, `videoarm/videoarm/video/aks_select.py::_CLIPScorer`,
  `openai/clip-vit-base-patch32`. **Not** LongCLIP — not installed, deliberately not yet.
- query text = `data.format_question(q)` (stem + the four options), **identical** to what
  `query_sim` is given, which is what makes the two directly comparable.

### Headline — 60 videos, 128 frames, T=64 grid steps

| signal | AUC (orig. labelling, n=322) | AUC (interval labelling, n=866) | AUC (interval, non-trivial, n=743) |
|---|---|---|---|
| norm | 0.509 | 0.500 | 0.502 |
| centroid_dist | 0.485 | 0.489 | 0.488 |
| temporal_delta | 0.509 | 0.494 | 0.490 |
| vidcom2 | 0.495 | 0.507 | 0.501 |
| query_sim (Qwen, **unaligned**) | 0.505 | 0.513 | 0.517 |
| query_sim_max (Qwen, **unaligned**) | 0.483 | 0.487 | 0.488 |
| **clip_query (aligned)** | **0.643** [0.614, 0.670] | **0.656** [0.640, 0.674] | **0.654** [0.636, 0.673] |

The n=322 column **exactly reproduces the banked 2026-08-17 numbers** (norm 0.509,
centroid_dist 0.485, temporal_delta 0.509, vidcom2 0.495, query_sim 0.505,
query_sim_max 0.483) — the original labelling is replicated verbatim, so `clip_query`
is measured on the same rows the "all blind" claim was made on.

### Top-K evidence hit rate (K of 64 steps; parenthesis = random baseline, same questions)

| signal | K=1 | K=4 | K=8 |
|---|---|---|---|
| best non-CLIP (interval, non-trivial) | 7.4% (5.9) | 16.4% (14.9) | 25.6% (24.0) |
| **clip_query** | **15.7%** (5.9) | **33.0%** (14.9) | **45.0%** (24.0) |

CLIP is ~2.7× random at K=1 and ~1.9× at K=8. Every other signal is at random.

### Relevance inside vs outside the GT span (interval labelling, non-trivial, n=743)
`clip_query` mean cosine **0.2564 inside vs 0.2426 outside**; within-question standardised
gap **+0.63 sd** [0.551, 0.707]. Every other signal's gap is within ±0.14 sd of zero.

### Sanity checks run BEFORE the 60-video pass (all passed)
1. **Same video, different questions → different rankings.** 4/4 distinct top-5 sets on
   both probe videos. Semantically right, too: on `aJI8XTa_DII` the two questions about
   Putin's first speech both rank step 1 (t=74 s) first; GT spans are 53–86 s and 55–68 s.
2. **Top frames inspected** for 10 questions. Successes are visual-content questions
   ("group of maids wearing yellow hats" → rank 1). Failures are temporal-ordinal ones
   ("When does the little mermaid appear for the **first** time?" → rank 64/64) — CLIP
   scores appearance, not ordinality. This is the expected failure mode, not a bug.
3. **No frame/timestamp indexing bug.** Over 148 questions / 12 videos, (CLIP peak − GT
   centre) is **median +3.05 grid steps, sd 27.1** — spread around zero, not piled at a
   constant. A constant off-by-k would pile at ±k. Median rank of the best in-evidence
   step is **12/64** against a random 32.

### TWO LABELLING PROBLEMS FOUND IN `signal_auc.py` (both pre-existing, neither fixed in place)

**(a) The grid-step timestamp is off by dur/(4T).** The file labels step `t` at
`(t+.5)*dur/T`. Qwen3-VL's own convention averages the frame pair the temporal patch
covers (`transformers/models/qwen3_vl/processing_qwen3_vl.py::_calculate_timestamps`,
`merge_size=2`), giving `(t+.25)*dur/T`. Measured drift: **+29.5 s on a 7,547 s video**,
a quarter of a grid step. Small, but LVBench evidence spans have a **median width of 11 s**.

**(b) Point-containment labelling silently drops 4 of every 5 questions, and keeps the
wide ones.** `lab[t] = t0 <= centre(t) <= t1` needs a grid centre to fall *inside* the
span. At T=64 over a median 3,666 s video the spacing is ~57 s, so a 11 s span almost
never contains one. Of 925 questions: **551 dropped with no positive step** (150 of them
zero-width spans), 52 with all steps positive, **322 survive = 34.8%**. Those survivors
have **median GT width 70 s vs 11 s for the full set** — the banked n=322 is a
wide-evidence subsample, i.e. the questions where localization matters least.

The `ovl` scheme added alongside it (step `t` owns `[t*dur/T, (t+1)*dur/T)`, a partition;
positive iff that interval meets the GT span) keeps **866/925** and handles zero-width
spans naturally. **Both schemes are reported; the original is untouched** so every banked
number stays comparable. CLIP's advantage is present under both, so it is not an artifact
of either labelling.

### Zero-width / malformed spans — explicit handling
LVBench ships 263/1,549 zero-width `time_reference`s (`t0 == t1`) and 2 malformed.
`data.parse_time_reference` returns `None` for malformed → dropped (counted as
`evidence_malformed`). Zero-width spans are **always dropped by the original labelling**
(a point cannot contain a grid centre) — 150 of the 551 drops here. The `ovl` scheme
keeps them: the point lands in exactly one step.

### Caveats to carry forward
- **CLIP-B/32, 77-token context: 49/871 queries (5.6%) truncated.** Low, but non-zero, and
  it is the argument for LongCLIP rather than a reason to distrust this result.
- The query includes the four options because `query_sim` was given the same string. For
  CLIP the options are probably noise; an ablation on stem-only is one line and not yet run.
- 60 of 103 videos, 925 of 1,549 questions. Not the full set.
- **AUC 0.65 is a signal, not an oracle.** The GT-crop oracle is worth +15.62 pp; a
  ranker that puts the evidence in its top 8 of 64 only 45% of the time captures a
  fraction of that. This licenses building the allocator, it does not predict the gain.

### What this does and does not license
- **Does:** the U/Q/U+Q temporal-allocation experiment is worth building. There is a real
  query-frame signal here, and it is the first non-blind signal this chapter has found.
- **Does not:** any claim about accuracy. Nothing was generated; no arm was run.
- Nothing was changed in `vidcom2_vllm/`, `SOFTMAX_TEMP` is untouched, no vLLM change,
  no LongCLIP installed, no QA run launched.

---

# ACTIVE TASK — Does compression improve temporal localization? (set 2026-08-10)

**Read this section first in every conversation.** Everything below it is background for it.

## The question

**Does giving Qwen3-VL-8B a compressed global view make it better at finding the right moment in a long video?**

LVBench ships a ground-truth evidence span per question, so "did it look in the right place" is directly measurable — that is why this is an LVBench task and not a VideoMME one.

## Metrics (identical definitions in both arms)

Per question the agent emits crop windows; score them against the GT evidence span:

| metric | definition |
|---|---|
| **tool-call coverage** | % of **tool calls** whose window overlaps the GT evidence span |
| **localization hit rate** | % of **questions** where the union of all crop windows covers the GT evidence (this is the "did it actually locate the key frame" number) |
| **accuracy \| landed** | accuracy on the questions where it landed on the evidence |
| **accuracy \| not landed** | accuracy on the questions where it did not |
| supporting | crop-window width, calls per question, % of video ever inspected, centre-offset, best-IoU |

Report coverage **per call** and hit rate **per question** separately — they are different denominators and were conflated before.

## THE THREE RUNS — canonical ladder (set 2026-08-12). Use these settings for every experiment.

Each step adds exactly one thing. Anything that differs between runs other than the named
change is a bug in the setup, not a result.

| run | skim (global view) | tools | visual tokens in the skim |
|---|---|---|---|
| **run1** Qwen on LVBench | 64-frame **video** | none | 64 frames, uncompressed |
| **run2** + cropping tool | 64-frame **video** | `crop_video` (≤128 frames, fps 1, full detail) | identical to run1 |
| **run3** + compression | 256-frame **video**, VidCom2 **retention 0.25** | `crop_video`, identical | 4× the frames, **same token count as run1/run2** |

run3's whole design: *same budget, 4× the temporal coverage.* If compression is worth
anything for localization, that is where it shows up.

**run4 — ground-truth / oracle crop. PLANNED, NOT YET RUN.**

| run | skim | tools | crop target |
|---|---|---|---|
| **run4** oracle | same as run2 (64-frame video) | `crop_video` | **forced to the GT `time_reference` span**, not chosen by the model |

run4 is a *control*, not a method: it removes localization from the loop entirely and
answers "if the model always looked in the right place, how well would it do?"

**Why it is needed — `acc | landed` is selection-biased and the bias is already measurable.**
`acc | landed` only covers questions the model could find on its own, which are the easier
ones. Adding the timestamp legend raised hit rate 10.15% → 23.59% while `acc | landed`
*fell* 58.60 → 49.59: the 208 extra landings were the questions it previously could not
even locate. So `acc | landed` cannot be read as a ceiling — it drifts down as localization
improves. run4 forces the same crop on **every** question and gives the unbiased ceiling.

That turns the headline into a decomposition:

```
run1  (no tool)          ── baseline
run2  (model aims)       ── baseline + what the model's own localization is worth
run4  (oracle aims)      ── baseline + what PERFECT localization would be worth
                            run4 − run2 = the size of the localization gap
run3  (compressed skim)  ── how much of that gap compression closes
```

Implementation: `fast_agent/oracle.py` already has the forced coarse→fine logic
(`ORACLE_CROP_MIN_WIDTH=16 s` floor for tight/zero-width refs) but still runs on the HF
backend; it needs porting to `run_agent.py` as a `--crop-source oracle` mode. Run it only
after run1–run3 are banked, and never report it as a method result.

### Fixed settings — do not vary these between runs
| | value |
|---|---|
| model | Qwen3-VL-8B-Instruct |
| serving | **vLLM only. Never HF inference.** |
| skim modality | **video** (`{"type": "video_url"}`), never images — see below |
| skim frames | 64 (run1, run2) / 256 (run3) |
| retention | 1.0 (run1, run2) / **0.25** (run3, `--video-pruning-rate 0.75`) |
| compressor | **VidCom2**, not FlashVID |
| crop tool | ≤128 frames, fps 1, `max_pixels` 224² — returned as **images**, so never pruned |
| temperature | **0.7**, top_p 0.8, top_k 20, presence/frequency penalty 0, repetition_penalty 1.0 |
| max_tokens | 1024 per tool turn, 32 for the finalizer |
| max tool rounds | 5 |
| prompt | LongVT-verbatim (`config.longvt_user_text`) |
| repeats | **3 seeds per run**, report mean ± spread — at T=0.7 one run is not a measurement |
| trajectories | **always stored** (`<run>/traj/<qid>.json`), with the GT evidence span and per-call coverage |
| dataset | LVBench, all **1,549** questions |

### WHY THE SKIM IS VIDEO MODALITY (CORRECTED 2026-08-13 — the old reason was wrong)

> ⚠️ **The 2026-08-12 version of this section said "LongVT passes `{"type": "video", "url":
> path}`, not images — feeding images was a fidelity bug against LongVT." That is
> backwards. LongVT feeds IMAGES.** The claim was made by reading the lmms-eval *task* file
> and stopping there. The task file's dict is lmms-eval's internal message format, not what
> goes over the wire.

**The traced path** (`EvolvingLMMs-Lab/lmms-eval`, main @ 2026-08-13):

```
lvbench_tool.yaml
  → utils.lvbench_tool_doc_to_messages     emits {"type":"video","url":path}   ← internal only
  → async_openai.maybe_forward_with_tool   is_qwen3_vl=False (run_eval.sh's LongVT branch)
  → protocol.ChatMessages.to_openai_messages(video_kwargs)
  → protocol.py:126-141   fetch_video(fps=1, max_frames=768, max_pixels=50176)
                          then ONE base64 image_url PER FRAME
```

`pass_video_url` defaults to `False` and `async_openai` never sets it. LongVT's own
reference implementation does the same thing explicitly
(`examples/eval/single_inference.py::encode_video_frames` — `fetch_video` then
`{"type":"image_url"}` per frame). This also resolves the contradiction with the 2026-08-06
note further down, which already said LongVT trained on image modality only.

The Qwen3-VL branch (`protocol.py:168-191`, `to_qwen3_vl_openai_messages`) is *also* images
— it just interleaves a hand-written `<{t:.1f} seconds>` text block before each frame. So
**the timestamp legend this harness added is what lmms-eval itself does**, not a deviation.

**The real reason we use video modality: vLLM prunes video only.** Images pass through
untouched, so VidCom2/EVS cannot be applied to an image skim at all. That is a constraint
imposed by the compressor, not a fidelity argument — and it must be stated that way,
because it means **every compression arm is off-distribution relative to LongVT** and the
uncompressed video arm (run1/run2) is the only legitimate baseline for them.

What survives from the old section: the timestamp mechanism itself is real. Qwen3-VL's
processor writes a literal marker before every frame's vision block
(`transformers/models/qwen3_vl/processing_qwen3_vl.py:221`):

```python
video_placeholder += f"<{curr_time:.1f} seconds>"
video_placeholder += vision_start + "<|placeholder|>" * frame_seqlen + vision_end
```

and the measured **hit rate 10.15% vs 23.59%** (below) stands as a fact about *this*
harness: an image skim with no legend gives the model no time anchor. Only the "LongVT does
it this way" justification attached to that number was wrong.

`crop_video` returning **images** is still what keeps the zoomed-in evidence at full detail
in every arm, and is identical to LongVT's MCP server (`examples/video_tools/mcp_server.py`:
fps 1, `max_frames` 128, `max_pixels` 224²).

### Serving prerequisites (`fast_agent/serve_arm.sh`)
- `--enable-auto-tool-choice --tool-call-parser hermes` — LongVT drives the loop with
  `tool_choice="auto"`; without these vLLM returns 400 on every request.
- `--media-io-kwargs '{"video":{"num_frames":N}}'` — **not optional.** Without it vLLM
  resamples to its own default and a 256-frame request silently costs the same as 64.
- `--limit-mm-per-prompt '{"image":768,...}'` and `--max-model-len 40960` — one crop is
  128 images / 6,016 tokens; five rounds is 33k. The old 80-image / 16k server would have
  truncated exactly the long trajectories that carry the localization signal.
- `--allowed-local-media-path /local1/cfyang` — for the `file://` skim.
- Pruning rate is **server-level, not per-request** → run2 and run3 need separate servers.

### Skim proxies (`fast_agent/make_skim_proxy.py`) — mandatory for video skims
LVBench is 103 videos / 62 GB, median 412 MB and 3,664 s, and the agent re-sends the video
every tool round → **~39 s per request, >60 h per arm**. Pre-build one N-frame proxy per
video (3–5 MB, ~25 s each) and the same arm runs in ~2 h.

The proxy is written with `ffmpeg -vf fps=N/duration`, which picks N uniform frames *and*
stamps them at that rate, so the proxy's duration equals the source's. **This is
load-bearing**: vLLM derives every `<t seconds>` marker as `frame_index / fps` read off the
container, so a proxy at a normal frame rate would label a 3,664 s video as ~32 s and every
crop would be wrong by two orders of magnitude — silently, since `crop_video` reads the real
file. Verified through vLLM's own loader: `fps=0.03392, duration=7547.0, timestamps
0.0 … 7517.5`. Always pass `--require-proxy`; the fallback to the 62 GB source is silent.

## Fixed decisions

| | decision | why |
|---|---|---|
| model | **Qwen3-VL-8B**, unchanged | continuity with the existing LVBench run |
| compressor | **VidCom2**, **not FlashVID** | community-standard baseline, upstreamed to vLLM; FlashVID is retired for this task |
| serving | **vLLM only. No HF inference.** | the HF in-process path made the last benchmark buggy (refusal-prone image skim, custom `inputs_embeds` decode). Do not reintroduce it. |
| temperature | **0.7** | |
| repeats | **3 runs per experiment**, report mean and spread across runs | at T=0.7 a single run is not a measurement |
| trajectories | **must be stored** for every run | failure-mode inspection is a deliverable, not an afterthought |

## VidCom2-in-vLLM — verification result (2026-08-10)

**"Officially adopted by vLLM" is true, but it is not in any released wheel.**

| | finding |
|---|---|
| upstream | merged as [PR #47750](https://github.com/vllm-project/vllm/pull/47750), 2026-07-28 — **`main` only**, one day after v0.26.0 was cut |
| installed here | `/local1/cfyang/miniconda3/envs/vllm` = **vLLM 0.19.0**. Has `video_pruning_rate` and `multimodal/evs.py` (so **EVS works today**), but **no `VideoPruningMethod`, zero `vidcom2` hits** |
| ~~consequence~~ | ~~using VidCom2 in vLLM requires building vLLM from `main`~~ — **SUPERSEDED 2026-08-11, see "VidCom2 IS RUNNING IN vLLM 0.19" below. No build is needed:** 0.19 already calls the hook and upstream's entry point has the same signature as the EVS one already present, so an out-of-tree plugin is enough. |
| EVS | still available for free (`--video-pruning-rate`, zero code) and now serves as the **control compressor** for A/B against VidCom2 — not as a substitute for it |

**ENV GOTCHA — `conda run -n vllm` silently resolves to the `dvd_tool` env** (python 3.10, vLLM 0.16.0), not the real `vllm` env (python 3.12, vLLM 0.19.0). Call `/local1/cfyang/miniconda3/envs/vllm/bin/python` by absolute path when the vLLM version matters. Several older notes that say "0.16" were written under this confusion.

## What has to be built

1. **vLLM from `main` with VidCom2**, served with tool calling (`--tool-call-parser hermes`) and the video-pruning flags. Verify the retention actually applies (VidCom2's reference impl silently disables compression at `batch_size != 1` — the same class of bug as the old FlashVID `r=1.0` bypass; always confirm token counts, never trust the flag).
2. **Point the LVBench agent harness at the vLLM endpoint** instead of the in-process HF model. The harness (`longvt_compression/fast_agent/`) already has the agent loop, LVBench loader with `time_reference`, and the localization scorer — the model backend is the piece that changes.
3. **Trajectory storage mechanism.** `fast_agent` already writes one JSON per question under `<run>/traj/<qid>.json` (rounds, the model's own reasoning text, every tool call with its window, tool results) and the analysis notebooks read it. Keep that format; the work is making it survive the HF→vLLM switch, plus:
   - **run identity in the path** — `<run>_<arm>_seed<k>/traj/` — because 3 repeats × 2 arms now share a qid namespace;
   - store the **sampling params and seed** in `run_manifest.json`;
   - **`traj_path` must be written into `results.jsonl`** and followed when loading. Trajectories get reused across runs and then live in *another* run's directory; globbing `<run>/traj/*.json` silently dropped 60/200 questions in the last analysis.
4. **Aggregation across the 3 repeats** — per-run metric, then mean ± spread. Do not pool the runs into one bag; that hides run-to-run variance, which at T=0.7 is the thing being controlled for.

## MEASURED 2026-08-11 — vanilla no-tool baseline on full LVBench, vLLM

**Qwen3-VL-8B-Instruct, 64 uniform frames, no tools, official-style prompt: 41.70% ± 0.05pp (3 seeds: 41.64 / 41.70 / 41.77), n=1,549 each.**
Runs `outputs/lvbench_plain/lvbench_plain_full_seed{0,1,2}/`, runner `fast_agent/run_plain.py`.
~29 min wall each, 1.1 s/question, **0 refusals, 0 errors**, essentially all outputs exactly 2 tokens.

Config: vLLM 0.19 stock (GPU 5) · T=0.7, top_p=0.8, top_k=20, presence/frequency penalty 0, repetition_penalty 1.0, max_tokens=32 · prompt = frames + question + options + *"directly provide the letter … without giving any explanation"* (no `<think>`, no duration/timestamp legend, no tool text, no clock→seconds rewrite) · 64 frames @ 224², **47 tok/frame → prompt median 3,078 tok**.

| reference | value | why it differs |
|---|---|---|
| **this run** | **41.6%** | 64 frames, ~2,880 visual tokens |
| official Qwen3-VL-8B-Instruct | 58.0 | 2 fps, ≤2,048 frames, 640 tok/frame, ≤224K visual tokens — **~78× the visual budget** |
| chance / always-B / always-D | 25.0 / 24.1 / 26.6 | |
| old 64-frame baseline (greedy, HF, CoT prompt) | 31.7% | +10pp comes from the clean prompt + eliminating 23% refusals/loops |

**Do not read the 16.4pp gap to 58.0 as a harness defect — it is the frame budget, by construction.**

### Two seeds: the aggregate is stable, individual answers are not

| seed | 0 | 1 | 2 | |
|---|---|---|---|---|
| accuracy (n=1,549) | 41.64% | 41.70% | 41.77% | **mean 41.70%, sd 0.05pp, range 0.13pp** |

| | |
|---|---|
| same answer in **all 3** seeds | **73.9%** — i.e. **26.1% of questions churn** |
| seed0→seed1 | 64 fixed / 63 broke (net +1) |

**At T=0.7 the headline is reproducible to ~0.1pp while a quarter of the individual answers churn across three seeds.** Accuracy is safe to compare across arms at n=1,549; *per-question* flip analysis is not, unless it is done across seeds — a 64/63 split is exactly what pure noise looks like.

### Behaviour: an option-position prior, but NOT the "B prior" a single seed suggests

pred/gold ratio, and recall, per seed:

pred/gold ratio per seed:

| | A | B | C | D |
|---|---|---|---|---|
| seed 0 | 0.75× | **1.72×** | 1.05× | **0.52×** |
| seed 1 | 0.70× | 1.36× | **1.52×** | **0.47×** |
| seed 2 | 0.61× | 0.98× | **1.94×** | **0.52×** |

Recall: seed0 A/B/C/D = 33.9 / 63.1 / 44.5 / 26.7%; seed1 = 31.6 / 51.6 / 59.2 / 26.0%.

**CORRECTION to the first-seed read, now confirmed at 3 seeds.** Seed 0 alone looks like a clean "defaults to B" story (B 1.72×). By seed 2 the B preference has evaporated entirely (0.98×) and the mass has moved to C (1.94×). **The B-vs-C split is sampling noise, not a property of the model.** What IS stable across all three seeds:

- **D is heavily avoided** — 0.52× / 0.47× / 0.52×, recall ~26%, despite D being the *most common* gold (26.6%).
- **A is under-picked** — 0.75× / 0.70× / 0.61×.
- The over-picked mass sits on the middle options; which one absorbs it is seed-dependent.

So the real bias is **positional: middle options over-selected, the last option (D) systematically avoided** — not a letter preference. A single-seed reading would have written up the wrong mechanism, which is the argument for the multi-seed protocol.

**Consequence unchanged:** part of the 41.7% is option-position prior rather than visual grounding, and every arm inherits it. **A shuffled-option control is still needed before accuracy is used as the headline metric** — and it should be run at ≥2 seeds.

**Questions that state their own timestamp are HARDER without tools: 34.8% (n=204) vs 42.7% (n=1,345).** Knowing "what happens at 17:16" is useless with only 64 uniform frames — those questions are *only* localization-trivial when a crop tool exists. Good sanity check that the tool arm should move exactly this stratum.

Per task type: key information retrieval 46.4% · reasoning 44.9% · event understanding 41.4% · entity recognition 40.5% · temporal grounding 36.9% · summarization 34.5% (n=55).

### vLLM gotcha found here (applies to every future run)

**vLLM re-processes uploaded images with its own pixel defaults and silently overrides `config.MAX_PIXELS`** — 64 frames came out at 68 tok/frame instead of 45. Fix: serve with `--mm-processor-kwargs '{"max_pixels":50176,"min_pixels":3136}'`. Without it the two arms' visual budgets are set by the server, not by us, and "compression" comparisons would be measuring the wrong thing. Also: a stale `flashvid_vllm` plugin is still registered in the env and logs a harmless `ModuleNotFoundError` at every server start.

**No trajectories to analyse in this arm** — `max_tokens=32` with "no explanation" yields a bare letter by design. Failure-mode analysis needs the agentic arm.

### PROBLEMS ENCOUNTERED building this baseline (all fixed)

1. **vLLM silently overrode our per-frame pixel budget.** It re-processes uploaded JPEGs with its own `max_pixels` defaults, giving **68 tok/frame instead of the intended 45** — `config.MAX_PIXELS` had no effect. Fix: `--mm-processor-kwargs '{"max_pixels":50176,"min_pixels":3136}'`. Left unfixed, the *server* would be setting each arm's visual budget and any compression comparison would measure the wrong thing. **Always measure tokens/frame empirically (1 image vs 2), never assume the config was honoured.**
2. **A cancelled command still ran and contaminated a fresh run.** An interrupted launch had already detached via `nohup`; it produced 17 rows under the OLD config (LongVT prompt, CoT, `max_tokens=2048`) which the next run then **silently resumed** because the tag matched. Caught only by `skip 17` in the startup line and confirmed by completion-token counts (81–358 vs the expected 2). Fix: purge the run dir; treat any nonzero `skip` on a supposedly fresh run as a stop-the-line event.
3. **Repetition-loop degeneracy, ~7–10% of questions.** With a CoT prompt the model fell into token-level loops ("frame 458, frame 459, …"), burning all 2048 tokens with **no `<answer>` at all** — distinct from a genuine refusal. Raising `max_tokens` does not help (legit answers need p99 ≈ 1,697 of 2,048; loops just run longer). Qwen's own recommended `presence_penalty` for 8B/4B/2B is 2.0 and exists for exactly this. Moot in the final config (`max_tokens=32`, no CoT) but will return the moment the agentic arm re-enables reasoning.
4. **Refusal ≠ truncation.** 27 cases answered `<answer>None of the above</answer>` (`finish_reason='stop'`, complete reasoning) while 11 were loop truncations (`finish_reason='length'`). Both surface as `pred=None`. The parser correctly scores refusals as None rather than guessing a letter — but the two must be separated in any diagnosis.
5. `--disable-log-requests` does not exist in vLLM 0.19 (silent `argparse` exit, no server).
6. `conda run -n vllm` resolves to the **`dvd_tool`** env (python 3.10, vLLM 0.16), not the real vllm env (python 3.12, vLLM 0.19). Use `/local1/cfyang/miniconda3/envs/vllm/bin/python` by absolute path.

## VidCom2 IS RUNNING IN vLLM 0.19 — verified end-to-end 2026-08-11 (no source build)

**Result: VidCom2 pruning works inside stock vLLM 0.19 as an out-of-tree plugin. No recompilation, no `main` build.** Package `longvt_compression/vidcom2_vllm/`, tests `test_retention.py` (6/6 pass).

The upstream PR was never needed. vLLM 0.19 **already calls** the exact hook VidCom2 needs — `qwen3_vl.py::_postprocess_video_embeds_evs` line 1925 — and upstream's VidCom2 entry point has the **same signature** as the EVS one already there (`compute_retention_mask(embeds, grid_thw, spatial_merge_size, q)`). So the whole port is one function plus a way to select it. The addition is pure Python; the 8 shipped `.so` files are untouched.

**Serving** (identical command to stock; the env var selects the algorithm):
```
FA_PRUNE_METHOD=vidcom2 vllm serve <model> --video-pruning-rate 0.5 \
  --mm-processor-kwargs '{"max_pixels":50176,"min_pixels":3136}' ...
```

### Verification evidence

| check | result |
|---|---|
| count exactness vs `evs.compute_retained_tokens_count` | **126/126** cases (T∈{1..64} × grids × q∈{0…0.95}) |
| scoring vs authors' released code | **max abs diff 0.0**, top-k identical 36/36 |
| runs in the process that owns the model | `(EngineCore pid=1475994) [vidcom2_vllm] ACTIVE … thw=(16,14,14), q=0.5` |
| token accounting | tpf 49 × T 16 = 784 visual; target `int(784·0.5)`=392; **941 − 392 = 549 = observed** |
| EVS vs VidCom2 A/B, same clip, T=0, q=0.5 | same 549 tokens, **different outputs** → genuinely different selection |

### PROBLEMS ENCOUNTERED (all fixed; each was silent)

1. **The monkeypatch was lost in the engine-core process — and reported success anyway.** vLLM 0.19 **spawns** `VLLM::EngineCore` (tell: a `multiprocessing.resource_tracker` process alongside it), so the child re-imports vLLM and any patch applied in the launcher is gone. The launcher printed "VidCom2 installed", the server answered normally, and **stock EVS was running the whole time**. Proven retroactively: the EVS control's output is byte-identical to that first "VidCom2" attempt. **Fix: register under the `vllm.general_plugins` entry-point group**, the only group vLLM loads *in all processes (process0, engine core, workers)*. **Never trust a patch that only prints from the parent** — make the patched function itself announce its pid on first call.
2. **Patch target must be the importing module, not the source.** `qwen3_vl.py` does `from vllm.multimodal.evs import compute_retention_mask` at module scope, binding the object. Rebinding `vllm.multimodal.evs.compute_retention_mask` alone changes nothing.
3. **Exact-count contract.** vLLM sizes the `<|video_pad|>` run in the *processor*, before the mask exists. The reference impl rounds a per-frame budget independently so its total drifts — which would corrupt generation. Fix: keep EVS's `compute_retained_tokens_count` as the single source of truth and apportion that exact total (largest-remainder water-fill).
4. **Reference's "≥1 token per frame" is unsatisfiable when target < T.** Relaxed the floor to 0 only in that regime (for tpf=45 this needs q>0.978, i.e. never in practice).
5. **My own apportionment bug**: a `for…else` exited the top-up loop after one pass, silently under-filling (25 vs 27). Caught only because the count test swept 126 configurations — a single spot check would have passed.
6. Port 8011 collides with another user's gemma server; use 8021+.
7. Stale `flashvid_vllm` plugin is still registered in this env but its module was archived → harmless `ModuleNotFoundError` on **every** server start. Remove with `pip uninstall flashvid-vllm` when convenient.

### Still to do before arm B

- ~~Wire the crop tool + agent loop to a vLLM backend (arm A)~~ — done 2026-08-12, see below.
- **Cross-check against Pipeline A** (the HF reference eval) at matched settings — same video, same q, compare surviving token counts. Given this repo's history of silent compression bypasses (FlashVID r=1.0; VidCom2's own `batch_size!=1`), the two implementations must be reconciled before any number is trusted.
- Note the pruning rate is **server-level, not per-request** (`multimodal_config`, read once at model init) → arm A and arm B need separate servers.

## BUILT 2026-08-12 — vLLM-backed agentic loop (`fast_agent/run_agent.py`)

The missing backend. Replaces `agent_loop.run_sample`'s HF `inputs_embeds` + hand-written
greedy decode with HTTP chat completions, keeping the semantics identical: LongVT prompt,
native `# Tools` block, one tool per turn, `MAX_ROUNDS=5`, one bounded finalizer,
strict/lenient dual score. Scoring, frame sampling and question rendering are the *same
imports* `run_plain.py` uses, so the no-tool baseline and the tool arms differ only in
`--tools` / `--skim-mode`.

- `--skim-mode images` (arm A): 64 JPEGs. vLLM prunes **video only**, so this arm is
  structurally uncompressible — which is the point.
- `--skim-mode video` (arm B): the source file passed as a `file://` URL, resampled
  server-side to 256 frames and pruned. Passing the **path, not base64**, keeps the mp4
  timeline intact so Qwen3-VL's native per-frame `<t seconds>` markers are real video
  times; re-encoding a resampled clip would silently rescale every timestamp.
- Localization bookkeeping is computed **at write time**: every call is stored next to the
  GT evidence span with `covered_frac` / `landed` / `best_iou` / `centre_offset` already
  resolved, so the three task metrics are a groupby over `results.jsonl`.
- Hermes parser is deliberately **not** enabled on the servers. Without it `<tool_call>`
  stays in `content`, the regex path picks it up, and the assistant turn is re-appended
  byte-identical to the training format. `_assistant_text` handles both paths anyway.

### Serving (`fast_agent/serve_arm.sh`) — two problems the old servers would have hit
1. `--limit-mm-per-prompt {"image":80}` — one `crop_video` returns up to **128** images;
   64 skim + 128 crop = 192. Raised to 768.
2. `--max-model-len 16384` — skim 3,008 tok + one crop 6,016 tok = 9k, so the **second**
   crop overflows. At `MAX_ROUNDS=5` the worst case is 3,008 + 5×6,016 = 33k → 40960.
   Truncating here would kill exactly the long trajectories that carry the localization signal.
3. `--allowed-local-media-path /local1/cfyang` — required for arm B's `file://` skim.
4. Placed on **GPU 4 (:8030, arm A)** and **GPU 7 (:8031, arm B)**; GPU 5 (the old :8010)
   is shared with another user's job at 95% util and would have made the arm slow.

### Budget — MEASURED on real LVBench video, correcting the earlier estimate
The earlier "arm B gets half the visual tokens" note was computed on a **square** test
video and is wrong for this dataset. LVBench is 16:9, so `smart_resize` under
`max_pixels=50176` gives 224×416 → grid 14×26 → 91 tokens per grid step, not 49:

| | arm A (64 images) | arm B (256-frame video, q=0.75) |
|---|---|---|
| grid | 64 × (14×26)/4 | thw=(128,14,26) — 256 frames paired to 128 steps |
| visual tokens | 64 × 47 = **3,008** | 128 × 91 = 11,648 base → **2,912 kept** |
| + timestamps | — | ~1,390 |
| prompt total | ~3,020 | ~4,300 |

So the arms are **matched on visual tokens (3,008 vs 2,912, 3% apart)** and arm B pays
~1,300 extra text tokens for its native timestamp markers. That is a better-posed A/B
than the spec promised. Confirmed live: `[vidcom2_vllm] ACTIVE in pid=1906321
(first mask: thw=(128, 14, 26), q=0.75)`.

### The skim proxy — arm B was 60+ hours before this
Arm B's skim must be video modality (vLLM prunes video only), so the **server** decodes
the file. LVBench is 103 videos / 62 GB, median 412 MB and 3,664 s, and the agent re-sends
the video every tool round → measured **~39 s per request**, >60 h for one arm.

`fast_agent/make_skim_proxy.py` pre-builds one 256-frame proxy per video (103 files,
3–5 MB each, ~25 s each). One `ffmpeg -vf fps=N/duration` pass does both jobs: it picks N
uniformly spaced frames **and** stamps them at that rate, so the proxy's own duration comes
back out as `N / (N/duration) = duration`.

**Why the timeline is the load-bearing part.** vLLM hands Qwen3-VL `frame_index / fps` as
each frame's `<t seconds>` marker (`qwen3_vl.py::_calculate_timestamps`), reading fps off
the container. A proxy written at a normal frame rate would label a 3,664 s video as ~32 s,
and every crop the model asked for would be wrong by two orders of magnitude — silently,
because `crop_video` reads the REAL file and would happily return frames from 0–32 s.

Verified through vLLM's own loader on the 7,547 s proxy:
`fps=0.03392, total_num_frames=256, duration=7547.0, timestamps 0.0 … 7517.5`. Correct.
(A first attempt to verify by *asking the model* for the last timestamp returned 748 s and
looked like a 10× bug — it was not; reading one entry out of a 128-number list is a needle
task the model simply fails. Trust the loader, not the model, for this class of check.)

`--require-proxy` exists because the fallback is silent and expensive: without it a missing
proxy quietly reverts to the 62 GB source.

### Probe findings, arm A (n=20, three iterations to get the harness right)
1. **400 on every request**: `tool_choice="auto"` needs `--enable-auto-tool-choice
   --tool-call-parser hermes`. (`tool_choice="none"` does still render the full `# Tools`
   block — 146 prompt tokens vs 9 — so the block is not what auto-tool-choice buys.)
2. **25% refusals, and they are real** (`finish_reason=stop`, 5 tokens, "None of the
   above") — not truncation. A crop that lands in the wrong place hands the model
   *confident negative evidence*; it concludes the answer is absent and declines. The
   no-tool baseline never sees that evidence, guesses, and scores ~25% on those. The
   finalizer now forces a commitment; `pred_strict` keeps the refusal measurable.
3. **The finalizer's letter was being discarded.** Scoring the concatenated transcript let
   an earlier `<answer>None of the above</answer>` trip `extract_answer`'s refusal branch
   and return None, even though the finalizer had replied 'C'/'B'/'A'/'A'. Now the
   finalizer turn is scored on its own first. → 0 no-answers.
4. **The model scans linearly instead of aiming.** On a 3,167 s video with evidence at
   2,618 s it started at 25 s and walked forward in 50 s steps — 5 rounds covers 250 s of
   3,167 s. This is the substantive finding the metrics are meant to capture, not a bug.

## MEASURED 2026-08-12 — arm A full LVBench, LongVT-verbatim prompt (n=1,549)

`outputs/lvbench_agent/lvbench_armA_seed0/`. Qwen3-VL-8B + `crop_video`, 64-image skim,
LongVT prompt exactly as published, T=0.7 seed 0.

| metric | value |
|---|---|
| accuracy | **40.28%** (vanilla no-tool control: 41.70%) |
| questions that called the tool | 1,540 / 1,549 |
| mean GT-evidence coverage | 0.118 |
| **landed** (coverage = 1.0) | **10.15%** (157/1,547) |
| **accuracy \| landed** | **58.60%** (n=157) |
| accuracy \| not landed | 38.27% (n=1,390) |
| no-answer | 1 |

**THE RESULT.** Giving the model a crop tool does not help on aggregate — 40.3% vs 41.7%,
slightly *worse* than not having it. But conditional on the crop actually covering the
evidence, accuracy is **58.6%**, a **+20.3pp** gap over the not-landed cases and level with
the official full-budget Qwen3-VL-8B number (58.0). The tool works; **the model cannot aim
it**. Only 10% of questions end up looking at the evidence at all.

That reframes the chapter's question. The bottleneck is not per-frame detail, and not the
compressor's fidelity — it is temporal localization. Which is exactly what the ACTIVE TASK
set out to measure, and it now has a number: **10.15% hit rate, worth 20pp when it hits.**

## WHY SOME COMPRESSORS PORT TO vLLM AND OTHERS DO NOT (2026-08-24)

### The reduction chain: three stages, not one
Qwen3-VL already compresses twice before any external method runs. For 128 frames at the
native 704x1280 geometry:

    raw 16px patches                                450,560
    /2  temporal merge  (3D conv, temporal_patch_size=2)  225,280   grid_t = 64
    /4  spatial merge   (PatchMerger, spatial_merge_size=2) 56,320   <- VidCom2 starts here
    /4  VidCom2 r=0.25                                     14,080
                                                    total 32x

Not double compression: the built-in stages change the granularity of the representation
(patches -> tokens); VidCom2 removes a subset of the resulting tokens. Different operations,
applied in sequence. Verified from config (patch_size 16, temporal_patch_size 2,
spatial_merge_size 2, out_hidden_size 4096) and from the reference's own shapes
(`frame_tokens = (h*w) // spatial_merge_size**2`, and `t` taken straight from grid_thw).

### SEMANTIC DRIFT: VidCom2's "frame" is a grid step, not a frame
The algorithm's units are named `frame_tokens`, `frame_center`, per-frame budgets. On
Qwen3-VL those operate on grid steps -- pairs of real frames already fused by the 3D conv.
The original design targets LLaVA-OV, where MODEL_SPECS hardcodes tpf=196 and one "frame"
is one actual image; the Qwen port takes `t` from grid_thw (= N/2) with no comment.

On hour-long video the two frames fused into one grid step are:
    32 frames -> 229 s apart      64 -> 115 s      128 -> 57 s
So at U32, VidCom2's "frame center" is the mean of two unrelated scenes, and its criterion
("keep tokens unlike that centre") loses its intended meaning. This is NOT a porting bug --
the official Qwen3-VL implementation does the same -- it is drift from moving a short-clip
method to a long-video model. VidCom2's own benchmarks make it harmless: MVBench is ~16 s,
so 32 frames means two fused frames half a second apart.

CANNOT currently explain our penalties: if this were the driver the cost should fall
monotonically with frame count, and measured it is -4.13 / -2.39 / -4.78 at 32/64/128.
Partial explanation at most. Clean test would be temporal_patch_size=1, which needs a config
change and a full rerun.

### Why attention-based compressors cannot run on vLLM at all
vLLM's Qwen3-VL dispatches to a fused attention backend (`vit_attn_backend`) and NEVER
materialises an attention matrix: `output_attentions`, `attn_weights`, `need_weights` all
appear ZERO times in `vllm/model_executor/models/qwen3_vl.py`. The reference compressors
that need it: HoliTom 12 references to `attn_weights`, VisionZip 9, FastV 5.

This blocks BOTH the ViT side (VisionZip, HoliTom patch
`_forward_vision_block_with_attention`) and the LLM side (FastV patches
`_fastv_text_model_forward`). It is not specific to CLS attention -- and note Qwen3-VL's ViT
has NO CLS token at all; its config carries no such field.

    Any compressor that needs attention weights is HF-only under vLLM's fused kernels.
    VidCom2 ports because it uses embedding geometry (Gaussian distance to a video/frame
    centre) and never touches attention.

### The placeholder constraint costs far less than expected
vLLM sizes the prompt's `<|video_pad|>` placeholders before the vision tower runs, from
`compute_retained_tokens_count(tpf, T, q) = max(tpf, int(tpf*T*(1-q)))`, and masked_scatter
fails if the mask's True count differs by one. vLLM's native EVS fits this because it is a
GLOBAL top-k (score by consecutive-frame dissimilarity, force-keep all of frame 0, slice the
sorted list) -- "how many" is trivially separable from "which". VidCom2 is a PER-FRAME
allocator whose total is the sum of independent roundings, i.e. content-dependent, hence
`_apportion` (largest remainder).

MEASURED cost of that constraint -- much smaller than the intuition that it is "very lossy":

    total kept tokens     +0.09%   (reference realises 0.2502 vs our exact 0.2500)
    per-frame budget      88 of 384 frames differ, by at most 1 token
    allocation shape      preserved: per-frame CoV 0.0655 (ours) vs 0.0617 (fp32 reference)

It is nearly free because `scales = base*(1 + probs - probs.mean())` has mean exactly
`base` by construction, so VidCom2's own total already sits on vLLM's formula.

**The genuinely lossy part is elsewhere: bf16 scoring** (see the transplant-fidelity section)
-- ~5% of kept tokens move, because near the cut the gap between neighbouring ranks is
0.000426 while bf16's rounding error is 0.011099, 26x larger. fp32 takes Jaccard against the
reference from 0.906 to 0.9986. Ranking: placeholder constraint ~0.1%, bf16 ~5%.

## TRANSPLANT FIDELITY — ours vs VidCom2 official (measured 2026-08-24)

Earlier claim "our port matches the reference bit-for-bit" was TOO STRONG. The test
(`test_retention.py`) compares the SCORING function only; it never compared the selected
token set. Measured head-to-head on real vision-tower output (6 LVBench proxies, 128f,
native res, q=0.75) with `vidcom2_vllm/compare_selectors.py`:

| difference | size | cause | avoidable |
|---|---|---|---|
| total kept tokens | +0.09% (ref 0.2502 vs our 0.2500) | ref floats, we are pinned | structural |
| per-frame budget | 88/384 frames, max 1 token | independent rounding vs Hamilton | structural |
| WHICH tokens | ~5% of kept (Jaccard 0.906) | **bf16 vs fp32 scoring arithmetic** | YES |

### Why the total is pinned (the architectural mismatch)
vLLM's native EVS scores frames by consecutive-frame dissimilarity, force-keeps ALL of
frame 0 (`255 * ones` for the first frame), then takes a GLOBAL top-k. For that design
"how many" is trivially separable from "which" -- you slice a sorted list. So vLLM sizes
the prompt's `<|video_pad|>` placeholders at prompt-build time from
`compute_retained_tokens_count(tpf, T, q) = max(tpf, int(tpf*T*(1-q)))`, before the vision
tower runs, and `masked_scatter` fails if the mask's True count differs by even one.

VidCom2 is a PER-FRAME allocator: `ks = round(scales*tpf)` per frame, total = sum of
independent roundings, i.e. CONTENT-DEPENDENT. That is the incompatibility -- not the
scoring signal. Any global-top-k pruner (FastV/FlashVID-style CLS-attention scoring) drops
into vLLM's interface unchanged; VidCom2 does not. Hence `_apportion` (largest remainder)
to convert a per-frame algorithm onto a globally fixed count.

NOTE: vLLM's EVS force-keeps frame 0 entirely; VidCom2 has no such rule and neither does
our port. The EVS arms in the grid therefore carry that structural prior; the VidCom2 arms
do not. This is a real difference between the two compressors, not a porting error.

### The float-precision finding (the only avoidable one)
Reference does `x.float()` before scoring; ours kept the input dtype (bf16) unless
VIDCOM2_FP32=1. All seven grid arms ran in bf16. Measured on frame 5 of one clip,
880 tokens, keeping the 220 lowest-scoring:

    bf16 vs fp32 score error       median 0.011099   p99 0.038458
    gap between neighbouring ranks   median 0.000426  (near the cut)
    -> the numerical error is 26x the gap it must resolve

So the ordering AT THE BOUNDARY is effectively randomised by rounding. Example: token 219
is fp32-rank 217 (kept) but bf16-rank 221 (dropped).

Switching our scoring to fp32: Jaccard vs reference 0.906 -> 0.9986, max per-frame budget
disagreement 24 -> 1 token.

### Does it matter? Probably not, but NOT YET MEASURED
The swapped tokens sit at median rank 218-219 of 880 when the cut is at 220 -- all at the
boundary -- and the score gap between swapped-in and swapped-out is only 7-9% of the IQR of
all scores. Per-frame budgets and the temporal allocation SHAPE are unchanged (CoV 0.0655
bf16 vs 0.0617 fp32), so the "VidCom2 allocation is blind to the evidence window"
conclusion is NOT a numerical artifact -- I checked this specifically after wrongly
inferring it was.

**Open, one arm of compute:** rerun V128 with VIDCOM2_FP32=1 (~2.7 h) and compare against
the banked 40.87. In fp32 our selection overlaps the reference 99.9%, so that number is
effectively the official implementation's LVBench score.

### Context: our penalties match VidCom2's own published Qwen3-VL numbers
The "99.6% of original performance" headline is LLaVA-OV. The same README reports Qwen3-VL
at max_num_frames=32, R_RATIO=0.25: MVBench -1.6, LongVideoBench -2.3, MLVU -2.9,
VideoMME overall -2.1 (short -3.9, medium -1.3, long -0.9); mean -2.1 pp, 96.7% ratio.
Ours: -4.13 (32f), -2.39 (64f), -4.78 (128f). Same order; we sit at the hard end because
LVBench's base accuracy is 39.77 (chance 25) against their ~64, so the same pp drop removes
far more above-chance signal: 72% retained at 32f against their 94.7% mean.

## FINAL 2026-08-20 — THE FRAMES x RETENTION GRID (native resolution, n=1,549 x 7)

This supersedes the 2026-08-18 iso-token grid for every claim about frame count. That
grid held the TOKEN BUDGET fixed and let the processor shrink each frame as frames were
added (32f 640x1152 -> 256f 224x416), so its "temporal density saturates at 128 frames"
was two effects cancelling. **That claim is withdrawn.**

### Design — two variables, nothing else
Spatial resolution is FIXED at the source's native 704x1280, 880 tokens per grid step,
for the 87 of 103 videos whose source allows it. Verified per arm over all 103 videos by
`fast_agent/preflight_arm.py`, which aborts on any geometry mismatch and freezes
grid_thw / decoded frames / pre- and post-prune counts to JSON.

| arm | frames | retention | visual tokens | prompt_tokens (predicted) | acc |
|---|---|---|---|---|---|
| U32 | 32 | 1.00 | 14,080 | 14,366 (14,367) | 39.77 |
| V32 | 32 | 0.25 | 3,520 | 3,807 (3,807) | 35.64 |
| U64 | 64 | 1.00 | 28,160 | 28,618 (28,618) | 42.03 |
| V64 | 64 | 0.25 | 7,040 | 7,498 (7,498) | 39.64 |
| V64-50 | 64 | 0.50 | 14,080 | 14,538 (14,538) | 41.32 |
| U128 | 128 | 1.00 | 56,320 | 57,121 (57,122) | **45.64** |
| V128 | 128 | 0.25 | 14,080 | 14,881 (14,882) | 40.87 |

All seven: n=1,549, 0 transport errors, 0 duplicates, no tools, single turn, T=0.7.

### Experiment A — frames, at fixed resolution. DOES NOT SATURATE.
| comparison | delta | 95% CI | broke/fixed | p |
|---|---|---|---|---|
| U32 -> U64 | +2.26 | [+0.06, +4.52] | 147/182 | 0.061 |
| U64 -> U128 | **+3.62** | [+1.42, +5.87] | 126/182 | **0.0017** |
| U32 -> U128 | **+5.87** | [+3.55, +8.20] | 128/219 | **<0.00001** |

The 64->128 step is LARGER than 32->64. At 128 frames the model still misses the evidence
window outright on 42% of questions, so there is headroom left.

### Experiment B — compression at fixed frames. ALWAYS COSTS.
| frames | full | 25% | delta | 95% CI | p |
|---|---|---|---|---|---|
| 32 | 39.77 | 35.64 | -4.13 | [-6.07, -2.13] | 0.00007 |
| 64 | 42.03 | 39.64 | -2.39 | [-4.33, -0.52] | 0.017 |
| 128 | 45.64 | 40.87 | -4.78 | [-6.78, -2.84] | <0.00001 |

By evidence-IoU stratum at 128 frames: needle -3.52 (n=568), middle -5.44 (n=331),
broad -5.56 (n=648). The cost lands on REDUNDANCY, not on needles.

### The iso-budget diagonal — flat
All three deliver 14,080 downstream visual tokens: U32 39.77, V64-50 41.32, V128 40.87.
No pair significant (p = 0.21 / 0.38 / 0.74). At a fixed budget it does not measurably
matter how you get there.

### THE DECOMPOSITION — the most decision-relevant result
Sampling is NESTED: `make_skim_proxy` writes at `fps = N/round(duration)`, so
`t_i(32) = i*dur/32 = t_4i(128)` exactly. Confirmed from the proxies' real packet
timestamps over all 103 videos, **max offset 0.00 ms**. So `U32 hit -> V128 miss` is
impossible and the 2x2 collapses to three cells. Using U128 as the middle term:

| coverage cell | n | frames U32->U128 | compression U128->V128 | net U32->V128 |
|---|---|---|---|---|
| both miss | 654 | **+7.19** (p=.0003) | -3.98 (p=.012) | +3.21 |
| miss -> hit | 335 | **+6.87** (p=.009) | -5.67 (p=.015) | +1.19 |
| both hit | 558 | **+3.76** (p=.062) | -5.20 (p=.003) | -1.43 |

**The frame benefit is FLAT across coverage states and LARGEST where the annotated window
was never reached.** If containment were the mechanism the middle row would dominate. It
does not. Whatever more frames buy, it is not "sampling inside the GT evidence span".

The aggregate +1.10pp null is not "nothing happens" — it is +7 and -5 cancelling.

Candidate mechanisms worth testing instead of containment:
- `temporal_patch_size=2` fuses two frames into one grid step. At 32 frames those two are
  **157 s apart**; at 128 frames, 39 s. U32's grid steps are hard fusions of temporally
  unrelated content. Ablate `temporal_patch_size`.
- More `<t seconds>` markers = better temporal anchoring (16 vs 64 markers).
- The binary hit label is coarse: in "both miss" V128 still closes from 28.2 s to 8.7 s
  from the evidence centre. Score by distance, not containment.

### Task type: nothing survives its control
Control = V64-50 -> V128 (also 14,080 tokens, overall -0.45pp p=0.74). Of twelve
task-type x coverage-cell tests, **only summarization inside "both hit"** exceeds its
control (-17.39 vs -8.70, n=46, 1 fix / 9 breaks, p=0.027). One uncorrected cell of
twelve on 46 questions -- a lead, not a finding. Everything else is matched or exceeded
by the control, including retrieval's eye-catching +11.63 against a control of +18.60.

Flip-set profiling is under-powered here: only 20.4% of "breaks" reproduce across frame
counts (32 observed vs 13.4 expected, z=+5.61 -- real but weak), leaving 61 consistent
breaks to split six ways. Use FULL-SAMPLE stratified penalties instead of profiling the
flip set; same question, 25x the power.

### Cost
| arm | visual | sec/q | wall-clock |
|---|---|---|---|
| V32 | 3,520 | 9.8 | 0.6 h |
| U32 | 14,080 | 21.6 | 1.6 h |
| V128 | 14,080 | 37.0 | 2.7 h |
| U64 | 28,160 | 57.2 | 3.5 h |
| U128 | 56,320 | 170.2 | 9.3 h |

Compressed arms still push the whole clip through the vision tower; they save LM context,
not vision compute. U128 buys +4.78 over V128 for 4.6x the time and 4x the context.

### Serving invariants learned building this (all cost hours)
- **The cap must differ for pruned and unpruned arms.** vLLM sizes its startup PROFILING
  dummy video from the processor ceiling (`longest_edge/2/1024`) and IGNORES
  `--media-io-kwargs num_frames`. A PRUNED arm needs a TIGHT cap or the dummy overruns
  `max-num-batched-tokens` and `_postprocess_video_embeds_evs` dies in masked_scatter
  before serving anything. An UNPRUNED arm needs a GENEROUS cap or
  `_compute_deepstack_embeds` dies in masked_scatter at INFERENCE time on every
  full-resolution video (the 16 smaller-source videos still succeed). Tested 1.0x and
  1.1x headroom: both fail; 16x works. Not our plugin -- vLLM's own EVS fails identically.
- **`--limit-mm-per-prompt` sizes the profiling run.** Leaving a no-tool arm at the tool
  arms' `{"image":768,"video":2}` profiles 768 dummy images it will never receive and
  OOMs a native-resolution arm at startup. Grid arms use `{"image":0,"video":1}`.
- **Raising `gpu-memory-utilization` can CAUSE the OOM.** vLLM subtracts the measured
  activation peak from that budget; at 0.92 the u128 profiling had 5.17 GiB left and
  needed 5.22. Lowering it too far starves KV instead. Size from actual free memory.
- **TP does not rescue u128**: the vision-tower forward on 56,320 tokens needs a ~5 GiB
  contiguous allocation per rank, so a card with a neighbour on it is still the binding
  constraint.
- **`set -e` + an informational `grep` kills a chain.** An uncompressed arm has no pruning
  line to find; grep returns 1 and the chain dies right after the server comes up.
- **A pipeline's exit status is its LAST command's**, so `preflight | grep` reported
  grep's status and swallowed a failed preflight. `set -o pipefail`.
- **`r_frame_rate` is the container's nominal rate and can be garbage** (`32/1`, `169/12`
  on some proxies). Read real packet PTS to verify sampling.

## MEASURED 2026-08-18 — THE ISO-BUDGET TRIAD, post-bugfix (n=1,549, seed 0)

This supersedes every VidCom2 accuracy number dated before 2026-08-17. Three
implementation bugs were found by external review, independently verified, and fixed
(see "Three VidCom2 bugs" below); the 256-frame arm was rerun from scratch as
`lvbench_pI2_256f_vidcom2_025_FIXED_seed0`. The fix moved that arm **+1.48 pp**
(42.54 → 44.03), which is larger than the effect it was being used to measure.

### The budget is measured, not assumed
`fast_agent/probe_budget.py`, same processor and same `mm_processor_kwargs` the server is
launched with. `prompt_tokens` is NOT used — it carries a frame-count-dependent timestamp
tax of ~1,000 tokens between a 64f and a 256f arm.

| arm | frames | cap | retention | grid steps | px/frame | tok/step | **visual tokens** |
|---|---|---|---|---|---|---|---|
| g  | 64  | 1x | 1.00 | 32  | 448x832 | 364 | **11,648** |
| j  | 256 | 1x | 1.00 | 128 | 224x416 | 91  | **11,648** |
| i2 | 256 | 4x | 0.25 | 128 | 448x832 | 91  | **11,648** (46,592 pre-prune) |
| k  | 256 | 4x | 0.25 (EVS) | 128 | 448x832 | 91 | **11,648** |

All four spend the identical budget. They differ only in **how**: few sharp frames,
many blurry frames, or many sharp frames with 3/4 of the patches dropped.

### Result — nothing is significant at n=1,549

| arm | acc |
|---|---|
| g  64f hi-res              | 42.16 |
| j  256f lo-res             | **44.16** |
| i2 256f hi-res + VidCom2   | **44.03** |
| k  256f hi-res + EVS       | 43.32 |
| (i  256f VidCom2, PRE-FIX) | 42.54 — void |

Paired on `question_id`, continuity-corrected McNemar, 4,000-resample bootstrap CI:

| comparison | Δ | 95% CI | \|z\| | p |
|---|---|---|---|---|
| 64f → 256f lo-res      | +2.00 | [−0.19, +4.26] | 1.74 | 0.082 |
| 64f → 256f + VidCom2   | +1.87 | [−0.26, +4.00] | 1.61 | 0.107 |
| 256f lo-res → +VidCom2 | −0.13 | [−2.13, +1.87] | 0.06 | 0.949 |
| 256f lo-res → +EVS     | −0.84 | [−2.91, +1.23] | 0.72 | 0.469 |

**Every CI is about ±2 pp wide.** LVBench at n=1,549 cannot resolve effects smaller than
~2 pp, and every effect in this chapter is smaller than that. State this as a power
limitation before quoting any of these deltas.

### The one thing that survives — narrow evidence spans
Stratified by the best-cell IoU between the 256-frame sampling grid and the GT evidence
span (n=1,547 with usable spans):

| stratum | n | 64f | 256f lo-res | +VidCom2 | +EVS |
|---|---|---|---|---|---|
| IoU < 0.10 (needle) | 586 | 44.88 | 47.44 | **49.15** | 48.12 |
| 0.10–0.25           | 285 | 36.49 | 38.25 | 36.84 | 36.84 |
| IoU ≥ 0.25 (easy)   | 676 | 42.31 | 43.93 | 42.75 | 42.01 |

64f → 256f+VidCom2 on the needle stratum is **+4.27 pp, |z| = 2.32 (p = 0.020)** — the
largest and only nominally significant effect in the grid. **It does not survive
Bonferroni across the 3 strata (threshold |z| = 2.39).** Treat it as the one direction
worth a targeted follow-up, not as a result.

Mechanism it is consistent with: compression buys temporal density without paying in
spatial detail, and that only matters when the evidence is short enough to be missed.
Where the evidence span is wide, every arm ties and compression is mildly negative.

### What is now settled about query-agnostic compression
At matched budget, VidCom2's learned patch selection is **indistinguishable from naive
uniform downscaling** (−0.13 pp, |z| = 0.06). It is not harmful — the earlier "−1.6 pp
penalty" was a bug artifact plus noise — but it buys nothing.

This is corroborated mechanistically, not just by accuracy:
- **Allocation is blind to the evidence.** Point-biserial r between VidCom2's per-grid-step
  budget and "is this step inside the GT span" = **−0.022**. Peak step lands inside the
  evidence **7.5%** of the time vs **9.9%** for random. Budget share inside the evidence
  9.153% vs 9.226% uniform (**−0.07 pp**). CoV 0.198, so it does vary temporally — just
  not usefully. (`vidcom2_alloc.jsonl`, 119 questions.)
- **So is every other cheap per-frame signal.** Mann-Whitney AUC for separating
  in-evidence grid steps from the rest, n=322 questions: norm 0.509, centroid_dist 0.485,
  temporal_delta 0.509, vidcom2 0.495, query_sim 0.505, query_sim_max 0.483. All blind.
  Estimator validated by positive control (perfect oracle 1.000, noisy 0.755, noise 0.500).
  (`fast_agent/signal_auc.py`.) **Retracts** the earlier n=11 smoke-test reading.
  > ⚠️ **SCOPE NARROWED 2026-08-28.** "All blind" covers only these six signals. An
  > *aligned* image-text scorer (CLIP) on the same n=322 rows scores **AUC 0.643**.
  > `query_sim` here is Qwen3-VL's INPUT EMBEDDING LOOKUP for the question, mean-pooled,
  > cosined against pooled visual embeddings — nothing ever trained that cosine to mean
  > relevance, so its 0.505 is not evidence about query-frame relevance in general.
  > See "MEASURED 2026-08-28 — CLIP query-frame relevance" near the top of this file.

Narrow claim only: **query-agnostic compression cannot allocate budget toward evidence in
the temporal dimension.** It does NOT follow that query-agnostic compression cannot help —
the spatial dimension is untested, and the needle stratum above is a hint that it might.

### Three VidCom2 bugs, found 2026-08-17, all fixed and regression-tested
1. **Scored all 16,384 channels.** The vision tower returns
   `cat([merger_out] + 3 DeepStack blocks)` = 4 x hidden_size. Scoring the concatenation
   mixes four differently-scaled feature spaces. Fixed: score only the final 4,096-D main
   block, apply the mask to all four. `main_embed_width()` resolves hidden_size from the
   served model's `config.json` (scanned out of `sys.argv` — `get_current_vllm_config()`
   is only valid during engine init, not at forward time). Server now logs
   `scoring the final 4096 of 16384 channels`; its presence is how a run is proven fixed.
2. **`_apportion` was not Hamilton.** Rewritten to genuine largest-remainder with
   cap/floor handling. Test `test_apportion_is_hamilton` asserts the <1-token guarantee,
   which holds only when `sum(ideal) == target` and no frame is capped.
3. **Unconditional FP32 upcast.** Now keeps input dtype; FP32 only under `VIDCOM2_FP32=1`.

Plus a dispatch bug in `run_agent.py`: `compress_video` was offered by the schema but
silently fell through to `crop_video`. The tool is removed from `TOOL_SCHEMAS` and any
non-`crop_video` name now raises.

**Canary:** the bugs were non-biasing at 128 frames (pre- vs post-fix within noise), but
at 256 frames the fix is worth +1.48 pp. Pre-2026-08-17 VidCom2 numbers are void.

### Invariants this chapter established (do not relearn these)
- **Pair on `question_id`, never on JSONL line order.** Positional pairing once reported
  351/289 flips where the truth was 157/95.
- **`prompt_tokens` is not the visual budget.** Use `probe_budget.py`.
- **`max_pixels` is inert for video** — `Qwen3VLVideoProcessor` reports it as unrecognized;
  measured budget identical with and without. The only video knob is `size.longest_edge`.
- **`do_sample_frames=False`** must be passed offline, or the processor resamples and
  reports grid_t ~11 instead of N/2.
- Qwen3-VL caps video at **12,288 visual tokens regardless of frame count**
  (`size.longest_edge = 25,165,824` / temporal_patch_size 2 / 1024 px-per-token). Adding
  frames thins the budget rather than adding to it. Override via the `size` key.
- `encoder_cache_size` is hardwired to `max_num_batched_tokens`
  (`config/scheduler.py:235`); auto-grow under-estimates when the cap is overridden.
- **Never run two processes against the same `--tag`.** Both read `done` at startup and
  neither sees the other's writes — this, not the resume logic, produced the 777-row /
  277-duplicate `lvbench_pA_CONTAMINATED_resume_dupes`.
- **Never `pkill -f` on a port string** — the pattern matches the killing script's own
  command line. Capture the PID from `serve_arm.sh` output and `kill` that.

### Open at the time of writing
- Three 128-frame arms (`pA_128f_uni_1x`, `pD_128f_evs_050`,
  `pC2_128f_vidcom2_050_FIXED`) are at n=500 and being extended to n=1,549 by appending
  the missing 1,049 questions under identical server settings. Do not quote the
  128-frame row until they land.
- Untested and the only signal with a reason to carry query information: FastV-style
  LM-layer-K text→visual attention. Would go into `signal_auc.py` as `lm_attention`.

## CANONICAL LADDER — results as they land (seed 0, n=1,549)

All video-modality, 64/256-frame proxies, LongVT-verbatim prompt, T=0.7.

| run | what | acc | Δ run1 | hit% | acc \| landed | acc \| miss | status |
|---|---|---|---|---|---|---|---|
| **run1** 64-frame video, no tools | baseline | **38.99** | — | – | – | – | done, 0 err |
| *ctrl* crop at the **wrong** place | control | **37.06** | **−1.93** | 4.14 | 54.69 (n=64) | 36.35 | done, 0 err |
| **run2** + `crop_video`, model aims | the policy | **39.57** | **+0.58** | 11.70 | 53.04 (n=181) | 37.85 | done, 0 err |
| **run3** 256f VidCom2 r=.25 + crop | compressed skim | **33.59** | **−5.40** | 9.96 | 45.45 (n=154) | 32.33 | done, 0 err |
| **run4** + crop at the **GT** place | ceiling | **55.39** | **+16.40** | 85.59 | 57.70 (n=1324) | 42.15 | done, 0 err |
| **run5** 256f, **no** pruning + crop | frame-count control | **43.29** | — | 11.75 | — | — | done n=**400**, 0 err |
| **c3_notool** 256f VidCom2, **no** crop | compression-alone control | **37.96** | **−1.03** | – | – | – | done, 0 err |

Two arms added 2026-08-13 that change how the table reads — see "THE 2x2 IS CLOSED" below:

- **`c3_notool`** is the cell that was missing. Compression *without* the tool costs only
  **−1.03 pp**; the −5.40 pp on run3 therefore is not a perception loss, it is an
  interaction with the agent loop.
- **`run5`** is on a 400-question subsample, so it is **not** comparable to the n=1,549
  column above. On its own matched 400: run2 40.25 → run5 42.75 → run3 31.75, i.e. 4x the
  frames unpruned is **+2.5 pp** and the pruning is **−11.0 pp**.

`ctrl`'s 37.06 and L0's 37.06 are unrelated arms that coincidentally tie — do not conflate.

### THE HEADLINE — localization is the entire bottleneck, and it substitutes for budget

**Perfect aim is worth +15.8 pp** over the model's own aim (55.39 vs 39.57) and +16.4 pp
over no tool at all. For scale: the **official Qwen3-VL-8B LVBench number is 58.0**, obtained
at 2 fps / ≤2,048 frames / 640 tok per frame — about **78× the visual budget this ladder
uses**. run4 reaches **55.39 with 1,280 skim tokens plus one correctly-placed 128-frame
crop.** Knowing *where* to look recovers almost everything that 78× more pixels buys.

Meanwhile the model's own aim converts almost none of it: **+0.58 pp**, because it reaches
the evidence on 11.7% of questions.

**The gain is location, not pixels.** `ctrl` gives the same number of extra full-detail
frames placed as far from the evidence as the video allows and scores **1.93 pp BELOW the
no-tool baseline** — misplaced visual tokens are actively harmful. So run4's +16.4 pp cannot
be read as "more pixels help."

**The selection bias ran the opposite way to the one run4 was built to check.** run2's
`acc | landed` is 53.04% (n=181); run4's unbiased 57.70% (n=1,324) is *higher*. The model's
own successful landings score **worse** than handed ones — it arrives after several wrong
crops, and the losing context it accumulates costs it ~4.7 pp even when it finally gets
there. Compare also `acc | miss`: 42.15% in run4 (the ~14% of evidence spans too wide for
one 128 s crop) versus 32–38% elsewhere.

**Compression is the one intervention that actively hurt** (−5.40 pp vs baseline, −5.98 vs
run2) *and* it localized worse (9.96% vs 11.70%), which is the opposite of the premise the
chapter was built on. run5 is running to separate "4× frames is worthless" from
"compression destroys what 4× frames buys" — at n=126 it was tracking **~46%**, i.e. well
above both run2 and run3, which points at the second reading.

**The control lands: the oracle's gain is location, not pixels.** `ctrl` hands the model the
same number of extra full-detail frames as run4, placed as far from the evidence as the
video allows. It scores **37.06%, i.e. 1.93 pp BELOW the no-tool baseline** — extra visual
tokens at the wrong place are actively harmful, not neutral. So the oracle's ≈ +13 pp
cannot be attributed to "more pixels"; it is the *right* pixels.

Second-order result from ctrl: the model still aimed its own crops on **99.0%** of
questions, yet only **4.14%** landed, against 11.70% in run2. Being handed a confidently
wrong region cuts its own successful search by ~2.8×: it anchors on what it was shown.

**run4's hit rate is 85.4%, not 100%, and that is a finding about the tool, not a bug.**
`crop_video` is capped at 128 frames at 1 fps = 128 s. **~14% of LVBench questions have an
evidence span wider than the tool can cover in one call**, so even a perfectly aimed crop
cannot see all of it.

**run5 (diagnostic, added 2026-08-13, running):** 256-frame video skim, **no pruning**, +
crop. run2 and run3 are matched on visual tokens (2,688 each), so run3 buys its 4× frames
by spending ¼ the detail per frame — and loses 6 pp. Two readings survive that: *4× frames
is worthless*, or *4× frames helps but compression's damage more than cancels it*. run5
sees the same 256 frames at full detail (**12,150 prompt tokens vs run3's 4,086**; visual
10,752 vs 2,688, exactly 4×) and separates them. Server arm `d`, port 8031.

**CORRECTION to the per-frame token figure.** An earlier note computed 40 tokens per grid
step by applying `--mm-processor-kwargs max_pixels=50176` to the video path. That is wrong:
the **video** processor uses its own budget (`size.longest_edge`), not the image
`max_pixels`. Measured on the server:

| | prompt tokens, video block + minimal text |
|---|---|
| 64-frame proxy @ `num_frames=64` | 3,044 |
| 256-frame proxy @ `num_frames=256` | 12,150 |
| **256-frame SOURCE mp4** @ `num_frames=256` | **12,918** |

→ **~84 visual tokens per grid step**, plus ~11 per step for the `<t seconds>` marker and
vision delimiters. So run2 = 32 × 84 = **2,688** and run3 = 128 × 84 × 0.25 = **2,688** —
still exactly matched; the ratio argument never depended on the per-step figure. And the
448 px proxy costs only **6%** against decoding the full source, which settles the worry
that the proxy was throttling fidelity.

| run5 vs | reading |
|---|---|
| ≈ run3 (33.6) | more frames do not help; compression is exonerated |
| ≫ run3, ≈ run2 (39.6) | compression's damage exactly cancels the frames it buys |
| ≫ run2 | frames genuinely help — and the compressor is what destroys the gain |

### run4 FINAL (n=1,549, 0 errors) — the headline of this chapter

**55.39%** (strict 51.84%, finalizer needed on only 9.3% vs run2's 29.3%).

| | acc | n |
|---|---|---|
| landed (oracle crop covered the evidence) | **57.70** | 1,324 |
| not landed | 42.15 | 223 |

The 223 non-landings are **not** aiming failures — they are questions whose evidence span is
wider than `crop_video` can cover in one 128 s call (**median width 2,074 s, p90 5,867 s**).
Their 42.15% is itself above run2's overall 39.57%, i.e. even a partial oracle crop helps.

**THE RESULT.** Localization, not visual budget, is what separates this setup from the
published number:

| | visual tokens | acc |
|---|---|---|
| blind (no video) | 0 | 24.66 |
| run1 (64-frame skim, no tools) | 2,688 | 38.99 |
| run2 (model aims its own crop) | 2,688 + ~6,000 crop | 39.57 |
| **run4 (oracle aims the crop)** | **2,688 + ~6,000 crop** | **55.39** |
| published Qwen3-VL-8B-Instruct | ≤224,000 | 58.0 |

run4 uses **~4% of the published visual budget and lands within 2.6 pp of it.** Told where
to look, a 64-frame skim plus one 128-frame crop is nearly as good as 2 fps over the whole
video. **Knowing where to look is worth almost as much as 25× more tokens** — and the
model's own aim converts essentially none of that (+0.58 pp).

That is the decomposition the chapter was built to produce:

```
24.66  blind
38.99  + a coarse global view                (+14.3  perception)
39.57  + a crop the model aims itself        (+0.58  the model's localization)
55.39  + a crop aimed correctly              (+15.8  THE GAP -- what localization is worth)
58.0   published, at 25x the tokens          (+2.6   what budget buys on top)
```

### LongVT-RFT as an extra arm (added 2026-08-13, running)

The chapter's headline is "aiming is worth +15.8 pp and zero-shot Qwen3-VL converts +0.58 of
it". LongVT-RFT is the *trained* answer to exactly that, so measuring it on the same GT
evidence spans turns the result from "the model cannot aim" into "training recovers X% of
the gap". Checkpoint `/local1/cfyang/LongVT-RFT` (16 GB), served by
`fast_agent/serve_longvt_rft.sh` on :8033.

**How LongVT actually evaluates** (`examples/eval/run_eval.sh`, `lmms_eval_tasks/lvbench/`):
**IMAGE modality** — `fetch_video(fps=1, max_frames=768, max_pixels=50176)` client-side, then
one base64 `image_url` per frame (`lmms-eval/lmms_eval/protocol.py:126-141`; `pass_video_url`
defaults False and is never set). Then vLLM + hermes + auto tool choice, DP8, `crop_video`
via an MCP server, `max_new_tokens: 4096`, **`USE_LLM_JUDGE=True`**, and — from
`async_openai`'s own defaults, since the task yaml sets none — **temperature 0** and an
**unbounded** tool loop (`while finish_reason == "tool_calls"`).
See the corrected modality section above; the earlier "video modality" reading of this same
file was wrong.

**Run it through OUR harness, not theirs** — two hard reasons:
1. lmms-eval emits only `acc_score` / `format_score`. It computes **no** coverage, hit rate,
   or landed/not-landed split — the metrics this whole task exists to produce.
2. Their scoring is an **LLM judge**; ours is exact-letter. Mixing them would make the
   number incomparable with run1–run4.
Our `run_agent.py` already uses their verbatim prompt, their tool schema, hermes with
`tool_choice="auto"`, and a 5-round cap, so swapping the weights swaps only the weights.

**The fact that makes this run interesting.** Qwen2.5-VL emits **no per-frame `<t seconds>`
text markers** (verified: zero hits in `processing_qwen2_5_vl.py`; Qwen3-VL writes one before
every frame). LongVT-RFT therefore has *no textual time anchor at all* — its localization is
a purely learned frame-index → time mapping. That is precisely the skill RFT trains, and it
explains why LongVT's published prompt needs no timestamp legend while the same prompt on
zero-shot Qwen3-VL gave a 10% hit rate.

**Confound to state with any result:** LongVT-RFT is **Qwen2.5-VL-7B**; our runs are
Qwen3-VL-8B. A difference mixes base model with training. The control is stock
Qwen2.5-VL-7B at identical settings — **that control is now running as L0, see below**.

> **Path gotcha:** `/local1/cfyang/models--Qwen--Qwen2.5-VL-7B-Instruct/` (the path in
> CLAUDE.md and above) is a **stub** — its snapshot dir contains `config.json` and nothing
> else, so vLLM loads and then dies in the tokenizer with
> `TypeError: expected str, bytes or os.PathLike object, not NoneType`. The real 16 GB
> checkout is at
> `/local1/cfyang/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5`.

**Smoke (n=12) said hit 25.0%; the full run says 8.82% at n=829.** The smoke was noise.

**64 frames is OFF-DISTRIBUTION for LongVT-RFT — do not read the 64-frame number as
"training does not help".** RFT was trained at `fps=1, max_frames=768`. Its localization is
a *learned frame-index -> time mapping*, and that mapping was learned at 768-frame sampling
density. At 64 frames on a 3,664 s video the frames are ~57 s apart, so the learned mapping
does not apply. This was an oversight in how the arm was specified.

Two runs are therefore needed, answering different questions:

| arm | question | status |
|---|---|---|
| LongVT-RFT @ **64 frames** | at *our* budget, does the trained model aim better? | running -- so far **no** (hit 8.82% vs run2's 11.70%) |
| LongVT-RFT @ **768 frames** | at *their* operating point, what did RFT actually buy? | 768-frame proxies building |

768 frames is feasible: Qwen2.5-VL is patch 14 / merge 2, so `max_pixels=50176` is ~64
tokens per frame -> 384 grid steps x 64 = **24,576** visual tokens, plus a 128-image crop
(~8,192) is ~33 k, inside our 40,960 context.

*Operational note:* the first smoke had 6/12 `404 model does not exist` — a startup race,
not a config error. vLLM answers a `/v1/chat/completions` probe before the model registry is
fully populated. Re-running after the server settled gave 0 errors.

### LongVT's OWN published LVBench numbers (paper Table 2, read 2026-08-13)

From arXiv:2511.20785. **Sparse = 64 uniformly sampled frames; dense = 512 or 768 (the
better of the two is reported).** Both are image modality (see the corrected section above).

| model | sparse (64f) | dense (512/768) |
|---|---|---|
| Qwen2.5-VL-7B (LongVT's own base) | **30.7** | **40.9** |
| Video-R1-7B | 37.2 | 40.1 |
| VideoRFT-7B | 34.7 | 18.7 |
| Video-Thinker-7B | **52.2** | **54.3** |
| LongVT-7B-SFT | 36.0 | 41.1 |
| LongVT-7B-RL | 37.8 | 41.4 |
| LongVT-7B-RFT | — (not reported) | 41.3 |
| GPT-4o | — | 62.0 |

**Three things this changes about the target:**

1. **LongVT's LVBench gain is +7.1 pp at 64 frames (30.7 → 37.8) and +0.5 pp at dense**
   (40.9 → 41.4). The sparse row is where its method shows; the dense row is nearly a wash.
2. **LongVT is not SOTA on LVBench** — Video-Thinker-7B leads at 54.3 dense / 52.2 sparse.
3. **"Beat LongVT on LVBench" is not a defensible claim for us.** Our vanilla Qwen3-VL-8B at
   64 image frames is 41.70, already above every LongVT cell — but that is a *different and
   newer base model* (Qwen3-VL-8B vs Qwen2.5-VL-7B) and, for the compression arms, a
   *different modality*. Two confounds stacked. The claim that survives is the internal
   delta: what compression does to localization at a fixed token budget.

**Scorer caveat, and it is not small.** LongVT's vanilla row is scored by upstream
lmms-eval's `lvbench` task (`extract_mcq_answer`, exact match, **no judge**) while its own
tool rows use `custom_rewards/lmms_lab_recipe.py::compute_score`, which falls back to an
**LLM judge** twice. Its published gap therefore mixes model and scorer. Any table we
produce should score every arm the same way and say which way.

**Deliberate deviations from LongVT in this harness** (stated, not accidental):

| | LongVT | here | why |
|---|---|---|---|
| skim modality | images | **video** | vLLM prunes video only — VidCom2 needs it |
| temperature | 0 | 0.7 | 3-seed spec; T=0 would make the seed repeats meaningless |
| tool rounds | unbounded | 5 | bounded cost; **27–34% of tool-arm questions hit the cap** |
| `max_new_tokens` | 4096 (49152 for tool tasks) | 1024 + 32 finalizer | see truncation finding below |
| scoring | rule → math-verify → LLM judge | exact-letter | their judge (Qwen3-235B-A22B) is not reproducible here |
| crop tool | fps 1, ≤128 frames, 224² | identical | — |
| skim budget | fps 1, `max_pixels` 50176 | identical | — |
| prompt | LongVT verbatim | identical | `config.longvt_user_text` |

### THE 2x2 IS CLOSED — compression does not blind the model, it breaks the agent loop (2026-08-13)

The missing cell (compressed skim, **no** tool) was run as `lvbench_c3_notool_seed0`.
All four cells n=1,549, seed 0, **0 errors**, video modality, LongVT-verbatim prompt, T=0.7.

| skim | **no crop** | **+ crop** | crop is worth |
|---|---|---|---|
| **64f uncompressed** | run1 **38.99** | run2 **39.57** | **+0.58** |
| **256f VidCom2 r=0.25** | c3_notool **37.96** | armB **33.59** | **−4.37** |
| *compression costs* | **−1.03** | **−5.98** | interaction **−4.95** |

**Read the row, not the corner.** Compression at matched token budget costs almost nothing
when the model answers in one shot (−1.03 pp). The entire −5.98 pp appears only when the
crop tool is in the loop. Compression and the tool *interact* by −4.95 pp; neither alone
explains the result.

**The mechanism is termination failure, not perception loss.** Same three counters:

| arm | finalizer used | output hit 1024-token cap |
|---|---|---|
| c3_notool — compressed, no tool | **5.6%** | **1.8%** |
| run2 — uncompressed, tool | 29.3% | 15.8% |
| armB — compressed, tool | **70.5%** | **40.1%** |

A compressed skim inside a multi-round loop makes the model ramble past the token cap and
never emit `<answer>`, so the forced finalizer guesses for it on 7 questions in 10. Given
one turn and no tool, the same compressed input is nearly free. This is exactly the failure
mode LongVT's own README warns about (`max_new_tokens` too low → truncated before
`<answer>` → `acc_score = 0`) — and it means **our 1024-token cap is load-bearing on the
compressed arm and not on the others**, i.e. it is a confound, not a neutral setting.

**Before this result is written up, re-run armB with `max_tokens` 4096** (LongVT's own tool
value). If the gap shrinks, part of the published-style negative was our cap; if it holds,
the rambling is genuinely caused by the compressed skim and the finding stands as stated.

### Frame count vs pruning — the attribution, on 400 matched questions

`run5` (256f, crop, **no** pruning, n=400) separates the two changes armB makes at once:

| arm | acc | hit% | finalizer% | trunc% | acc \| self-terminated |
|---|---|---|---|---|---|
| 64f no-prune, no-crop | 39.25 | — | 4.5 | 2.5 | 39.27 |
| 64f no-prune, +crop | 40.25 | 12.50 | 30.2 | 15.8 | 41.22 |
| 256f no-prune, +crop | **42.75** | 11.75 | 35.8 | 31.0 | 42.02 |
| 256f VidCom2, +crop | **31.75** | 10.75 | 69.0 | 43.0 | 36.29 |

- 4x the frames **unpruned** is worth **+2.5 pp** (40.25 → 42.75).
- Pruning back to the 64-frame budget costs **−11.0 pp** (42.75 → 31.75).
- **The negative belongs to the pruning, not the frame count.**

**And compression never bought localization at any setting**: hit rate 12.50 → 11.75 →
10.75, monotonically *down*. That is the chapter's premise tested directly, and it fails.

### L0 — the LongVT reproduction check. HARNESS CLEARED; their number is the outlier (2026-08-13)

The one arm in this chapter with a published counterpart to hit. Stock **Qwen2.5-VL-7B**
(LongVT's own base), 64 **images** @224², upstream lmms-eval `lvbench.yaml` prompt verbatim,
`T=0`, `max_new_tokens=16`, PNG frames, scored with lmms-eval's own `extract_mcq_answer`.
Runner `run_plain.py --prompt-style lmms_eval`; server `fast_agent/serve_qwen25vl.sh`.
Run `outputs/lvbench_plain/lvbench_plain_L0_seed0/`, 1,549 questions, 27 min, **0 errors**.

| | acc | over blind |
|---|---|---|
| blind — no video at all (same model, same loader) | **24.60** | — |
| **L0 — 64 image frames** | **37.06** | **+12.5 pp** |
| LongVT paper Table 2, same cell | **30.7** | +6.1 pp |
| Qwen team's own full-budget number (2 fps, ≤768 frames, Qwen2.5-VL issue #715) | 45.3 | — |

**We are 6.4 pp ABOVE the target, not below.** That failed the ±1.5 pp gate, so it was
diagnosed rather than waved through. Five checks, all clean:

1. **Both scorers agree on 1,549/1,549** — ours (`data.extract_answer`) and lmms-eval's
   (`extract_mcq_answer`). The extractor is not inventing points.
2. **Blind = 24.60%**, i.e. exactly 4-way chance. No answer leakage, no text-only shortcut.
3. `prompt_tokens` median **4,034** ≈ 64 frames × 64 tok — the correct Qwen2.5-VL cost at
   224² (patch 14, merge 2 → 28×28 px/token). The frames really are arriving.
4. **0 no-answers**, mean completion length **2.02 tokens** — the model emits a bare letter,
   as the prompt asks.
5. Prediction distribution spread (A 306 / B 391 / C 409 / D 443) against gold
   (383/374/380/412) — not degenerate.

**Conclusion: the harness is sound and 30.7 is the low outlier.** Our 37.06 sits sensibly
between chance (24.6) and the full-budget official number (45.3); 30.7 would mean 64 frames
buy only 6.1 pp, half what we measure on the same frames.

Corroborating evidence that Table 2 has depressed cells: it reports **VideoRFT-7B at 18.7**
on LVBench dense — *below 25% chance on a 4-way MCQ*, which can only be a format/scoring
failure on their side — and **LongVT-7B-SFT at 12.5** on VideoMME sparse. Both are the same
failure mode LongVT's own README warns about (truncation before `<answer>` → `acc_score=0`).

**How to use this:** L0 validates the shared plumbing (LVBench loader, option parsing, frame
sampling, extraction), which is what it was for. It does **not** license quoting our numbers
against the paper's — the ladder stays internally comparable only.

### D1 — the informed prompt. The null is NOT a prompt artifact (2026-08-13)

**Why it was run.** run2/run4 were operating far below the tool's capacity: median crop
width **10 s** against a **128 s** capacity, on videos of median **3,666 s**, so ~2.2 calls
inspected **~0.6%** of the timeline and genuine-search hit rate was **2–3%**. At a 2% floor
there is no localization signal for compression to move, which would make run2-vs-run4
unreadable. Diagnosis of *why*: the model emits a whole **linear scan plan in one turn**
(98% of multi-call turns, monotonically increasing, e.g. `(0,10) (10,20) (20,30)…`) until
the 1024-token cap cuts it mid-word. The syntax is fine (95%/91% of blocks parse) and the
calls are **not** repeats (0 of 350 turns had identical arguments) — the model simply does
not know the protocol is turn-based or what the tool can return. LongVT's prompt can omit
this because LongVT-RFT was *trained* on the pattern; zero-shot Qwen3-VL was not.

**The intervention.** `--prompt-style informed` (`config.informed_user_text` +
`config.INFORMED_TOOL_SCHEMAS`): LongVT's format contract plus **environment facts only** —
video duration, that `crop_video` returns up to 128 frames at 1 fps, that longer spans are
sampled down to fit, the round budget, and that one call per turn returns frames before the
next decision. **No search strategy.** "Return up to 128 frames" is a fact; "start wide,
then narrow" is the behaviour being measured, and writing it in would manufacture the
result. `--prompt-style longvt` is untouched, so banked runs stay reproducible; with no
tools the informed style falls through to the same prompt, so **run1/run3 need no re-run**.

**Result — 150 questions, all four cells paired, 0 errors.**

| arm | acc | landed (trivial) | landed (genuine) | calls/q | crop width | finalizer % | hit 1024 cap % | video seen % |
|---|---|---|---|---|---|---|---|---|
| run2 longvt | 37.33 | 16/19 | 7/131 | 2.25 | 10 s | 30.7 | 12.0 | 1.77 |
| run2 **informed** | **40.67** | 11/19 | 3/131 | **1.05** | **30 s** | **19.3** | **3.3** | 1.82 |
| run4 longvt | 31.33 | 12/19 | 4/131 | 2.07 | 10 s | 70.0 | 43.3 | 2.22 |
| run4 **informed** | **34.67** | 10/19 | 6/131 | **1.21** | **20 s** | **56.0** | **6.7** | 1.18 |

```
compression penalty, longvt prompt   : 37.33 - 31.33 = -6.00
compression penalty, informed prompt : 40.67 - 34.67 = -6.00
```

**The gap is identical to two decimal places.** The informed prompt shifts both arms up by
+3.34 pp (exactly +5 questions each) and leaves the compression penalty untouched. So the
null result is **not** an artifact of giving a zero-shot model LongVT's trained-model prompt.

### D1 CORRECTS AN EARLIER CLAIM — termination failure was a correlate, not the cause

The 2x2 section above attributes the compressed arm's loss to termination failure (forced
finalizer 70.5%, truncation 40.1%). **D1 falsifies that as a cause.** The informed prompt
cuts run4's truncation from **43.3% to 6.7%** — a 6x reduction — and drops the finalizer
rate 14 pp, and **the accuracy gap does not move at all**. Fix the termination pathology
almost entirely and the penalty is still exactly −6.00 pp.

Termination failure is real and worth reporting as a *behaviour*, but it does not explain
the compression penalty. Do not present it as the mechanism.

### D1's other finding: the floor is a POLICY gap, not an information gap

`video seen %` barely moves (1.77 → 1.82 on run2; it *falls* 2.22 → 1.18 on run4). The model
trades ~2 narrow calls for ~1 wider call and inspects the same sliver of the timeline. Told
explicitly that 128 s per call is available, it asks for 20–30 s.

Genuine-search landings are **7 / 3 / 4 / 6 out of 131** across the four cells — all four
are noise, and neither prompt produces a measurable localization signal. **Handing a
zero-shot agent the tool's manual does not give it a temporal search policy.** That is a
result in its own right, and it fits the north star: the *action space* is useful, but an
untrained policy will not exploit the bandwidth it is offered.

**Sample-size discipline:** at n=150, +3.34 pp is 5 questions and −6.00 pp is 9. No single
cell is well resolved. What is informative is the **stability of the gap under a large
behavioural change** — calls/q halved, truncation down 6x, crop width doubled-to-tripled,
gap unchanged.

### D2 — the compressed oracle. Compression hurts the READING, not just the aiming (2026-08-13)

D1 showed the penalty is not a prompt artifact. D2 asks the next question: when the crop is
*handed* to the model, does compression still cost anything? Both arms get the GT evidence
span forced as the crop, so aiming is removed from the experiment entirely.

| arm | skim | crop | n | acc |
|---|---|---|---|---|
| run5 | 64f, no pruning | forced GT span | 150 | **53.33** |
| D2 | 256f, VidCom2 r=0.25 | forced GT span | 150 | **39.33** |

**Δ_oracle = −14.00 pp**, flips 44 harmed / 23 helped, |z| = 2.44. Not a termination
artifact: acc | self-terminated 52.94 (n=136) vs 32.50 (n=40), acc | forced 57.14 vs 41.82 —
**both slices drop**. Truncation is low in both (1.3% / 3.3%).

Why −14 here but only −6 in run2/run4: run2/run4 land on the evidence only 10–15% of the
time, so the interpretation penalty is diluted by the 85% of questions where the crop was
never on target anyway. The oracle removes that dilution.

**Reading:** compression at r=0.25 degrades the model's ability to *use* the frames it is
given, not merely its ability to choose them. That is a stronger and more damaging claim
than the 2x2 alone supported.

### THE OPERATING-POINT CONFOUND — why the compression result may not be about compression

Everything above (arms a–d) runs at `max_pixels=50176` (224²), inherited **verbatim** from
`LongVT/examples/eval/run_eval.sh:48` and `LongVT/examples/video_tools/mcp_server.py:116`.
LongVT needs it that low because it ships up to **768 frames as images**. That constraint is
inherited, not chosen — it does not bind us (run1 uses 7% of the 40,960 context).

At 224² a 16:9 frame is ~91 tokens, so **VidCom2 at r=0.25 leaves ~23 tokens for an entire
frame**. VidCom2's own validated setting is 32 frames at ~720 tok/frame → ~180 after
pruning. **We have been running it ~8x past its validated regime.**

| | VidCom2's setting | ours (arms a–d) |
|---|---|---|
| frames | 32 | 256 |
| tok/frame pre-prune | ~720 | ~91 |
| tok/frame post-prune | ~180 | ~23 |

Supporting evidence that this is a real confound, not an excuse:

- The official reproduction already passed (2026-08-11, VidCom2's own LMMs-Eval harness,
  VideoMME-long n=900): 57.33 vs README 57.0, and 55.78 vs 56.1 — within 0.35 pp. **The
  implementation is not broken.**
- The paper never evaluates Qwen3-VL at all.
- Their own README claims 57.0 → 56.1 = **−0.9 pp**, i.e. compression is "roughly free",
  never a gain. Our "the paper claims a gain" framing was wrong.
- Matched n=253: 64f unpruned 39.53 / 256f unpruned 44.66 / 256f VidCom2 36.76 →
  paper-style claim −7.91 pp.

**Proxy gotcha:** the existing proxies are **448x250** (~112 tok/frame max). Raising the
server's `max_pixels` alone does nothing — the pixels are gone at proxy-build time.
`make_skim_proxy.py` now takes `--max-side`; hi-res proxies must go in their **own**
`--out-dir` because the filename (`{vid}_n{frames}.mp4`) carries no resolution.

### D3 — VidCom2 at its own operating point (running 2026-08-14)

32 frames, `max_pixels=786432` (~768 tok/frame), LVBench, **no tools**, sequential on GPU 3.

- **arm e** (`serve_arm.sh e`, :8036) — 32f hi-res, NO pruning. The control.
- **arm f** (`serve_arm.sh f`, :8037) — 32f hi-res, VidCom2 r=0.25.
- proxies: `.../lvbench_agent/skim_proxies_hires`, 103/103 built at 1280x720, 0 failures.

**Operating point VERIFIED before the run, not assumed** (n=24 probe on arm e, 0 errors):
prompt_tokens median **11,802** over 16 grid steps = **737.6 tok/grid-frame**, vs ~91 in
arms a–d. This is the check that would have caught a silent bypass — do it for arm f too.

`--gpu-memory-utilization` is now overridable via `GPUUTIL` (`serve_arm.sh`). 0.88 assumes
an empty card and vLLM refuses to start if the *free* fraction is lower; another user held
5.6 GB on GPU 3, so D3 runs at **0.80 for both arms**. It sizes only the KV cache, never
the arithmetic, but it must be identical across a compared pair.

This is the only experiment that cleanly separates the two live hypotheses:

- loss shrinks to <1 pp → **operating point**. The compressor is fine; arms a–d pushed it
  8x past its regime and the −6 / −14 numbers describe that, not compression in general.
- loss stays ~7 pp → the penalty survives at the paper's own setting, and LVBench-scale
  hour-long video is where VidCom2's assumptions break.

### D3 RESULT — halved, not eliminated (2026-08-14)

| arm | 32f skim | n | acc |
|---|---|---|---|
| e | no pruning (737.6 tok/grid-frame) | 1,549 | **39.70** |
| f | VidCom2 r=0.25 (197.6 tok/grid-frame) | 1,549 | **35.70** |

**Δ = −4.00 pp**, 0 errors either arm. Paired **by `question_id`** — an earlier pass paired
by line order and got 351/289; results.jsonl is written in completion order under 8
workers, so line order is NOT aligned across runs. Correct figures: **157 harmed / 95
helped**, discordant 252 (16.3%), McNemar **|z| = 3.84**. Accuracy itself is unaffected by
the pairing bug (it is per-run).

**Not an artifact — checked directly, not argued:**

| check | e | f |
|---|---|---|
| `finish_reason == "length"` | 22 | **9** (compression truncates *less*) |
| `pred is None` | 0 | 1 |
| strict==loose extraction | 1524/1549 | **1524/1549** (identical) |
| completion_tokens median | 10 | 10 |

Raw generations are clean `<answer>X. text</answer>`. The model answers well-formed and
*wrong*: `D. Chicken`→`C. Bird`, `D. Heather Conley`→`C. Fiona Hill`. Fine-grained
recognition, exactly what removing 75% of a frame's tokens should break.

### THE REAL CONFOUND — accuracy tracks TOTAL VISUAL TOKENS (2026-08-14)

| run | prompt_tok (median) | acc |
|---|---|---|
| f 32f hi-res VidCom2 | **3,167** | **35.70** |
| run1 64f lo-res no-prune | 4,091 | 38.99 |
| c3 256f lo-res VidCom2 | 4,445 | 37.96 |
| e 32f hi-res no-prune | 11,807 | 39.70 |

**arm f has the smallest token budget of any arm ever run here.** So "e vs f" does not
isolate compression: pruning changes *both* which tokens are kept *and* the budget (3.73x).
Only a **matched-budget** comparison attributes anything to the compressor.

One matched pair already exists: run1 (4,091 tok, 64f, no prune) **38.99** vs c3 (4,445
tok — 9% *more* — 256f VidCom2) **37.96**. At equal budget VidCom2 is **1.03 pp worse**
than naive uniform frame sampling.

Also note 32f hi-res (39.70) **beats** 64f lo-res (38.99). Halving frames for 8x the
per-frame resolution is a slight win, not a loss.

### VidCom2 HAS ALMOST NO TEMPORAL SELECTIVITY — read from the code (2026-08-14)

`vidcom2_vllm/retention.py:156-158`:
```python
probs  = F.softmax((frame_importance - max) / 0.01, dim=0)   # temp 0.01 -> near one-hot
scales = base * (1.0 + probs - probs.mean())                 # base = 1 - q = 0.25
```
Temp 0.01 makes `probs` essentially one-hot, so the per-frame budget is:

| | tokens kept |
|---|---|
| the single highest-scoring frame | 48.4% (T=16) / 49.8% (T=128) |
| **every other frame** | 23.4% / 24.8% |

**The temporal allocation is uniform except for one frame.** All of VidCom2's intelligence
is *within* a frame — `retention.py:173` keeps the tokens FURTHEST from two centroids (the
whole-video centroid and the frame centroid).

Two consequences, both load-bearing for this chapter:

1. **A compressor with a uniform temporal budget cannot help localization by
   construction.** It does not know where the evidence is and does not try to. The 2x2, D1,
   D2 and D3 were testing something the source code rules out.
2. **The whole-video centroid is meaningless at LVBench scale.** VidCom2 scores redundancy
   as distance from the mean of all tokens in the video. At 256 frames over ~1 hour that is
   one frame per ~14 s — consecutive frames are different scenes, not near-duplicates. "Far
   from the centroid" stops meaning "non-redundant" and starts meaning "visually unusual".
   The assumption holds on the short clips VidCom2 was validated on; it does not hold here.

So the finding is **not** "VidCom2 is weak" — it is "VidCom2 is structurally the wrong tool
for a localization claim". Any replacement must be query-aware or temporally non-uniform
(FastV-style attention pruning is the candidate class). EVS is vLLM-native
(`vllm.multimodal.evs`) and swappable via `FA_PRUNE_METHOD`, so it is cheap to test, but it
prunes temporally-static tokens — at 14 s spacing almost nothing is static.

**Second bottleneck, independent of the compressor:** D1 measured genuine-search hit rate
at 2–5% with `looked_frac` unmoved. Even a perfect query-aware compressor will not produce
a localization story while the zero-shot policy never searches. Swapping compressors fixes
one of the two bottlenecks, not both.

### THE MATCHED-BUDGET LADDER — running 2026-08-14 (arms g/h/i)

Same visual-token budget, different frame counts, so whatever separates them is the
compressor and not the budget.

| arm | frames | pruning | visual tokens | port |
|---|---|---|---|---|
| g | 64 | none | **23,603** | 8038 |
| h | 64 | VidCom2 r=0.25 | 5,901 | 8039 |
| i | 256 | VidCom2 r=0.25 | **23,603** | 8040 |

**g vs i is the experiment**: equal budget, 4x the temporal coverage. It does not depend on
the agent using a tool, so it is answerable regardless of the policy bottleneck. Feasible
without config changes — 256f is 94,413 tok through the ViT but only 23,603 reach the LM.

Proxies: `skim_proxies_hires_64` and `skim_proxies_hires_256`
(`make_skim_proxy --max-side 1280`, own `--out-dir` each).

### THE MATCHED-BUDGET LADDER — COMPLETE (2026-08-14/16). This supersedes D3's reading.

All no-tool, all seed 0, all n=1,549 (arm l is a paired n=500 subset), **0 transport errors
in every arm**. High-res proxies, `max_pixels=786432`.

| arm | frames | how the budget is spent | visual tok | acc |
|---|---|---|---|---|
| e | 32 | uniform, no pruning (738/grid-step) | 11,807 | 39.70 |
| g | 64 | uniform, no pruning (378/grid-step) | 12,106 | 42.16 |
| i | 256 | **VidCom2** r=0.25 from 384/step | 13,133 | 42.54 |
| k | 256 | **EVS** r=0.25 from 384/step | 13,132 | 43.32 |
| j | 256 | **uniform downscale** to 96/step | 13,136 | **44.16** |
| l | 256 | uniform, **2x budget** (192/step) | 24,527 | 45.80 (n=500) |
| — | 64 | oracle: crop forced to GT evidence | 6,586 | **55.39** |

#### Finding 1 — two unrelated query-agnostic selectors both LOSE to uniform downscaling

Identical frames, identical budget (all three within 12 tokens of each other), identical
prompt. The only difference is which tokens survive.

| selector | criterion | Δ vs uniform | flips | \|z\| |
|---|---|---|---|---|
| EVS | temporal change | −0.84 | 144/131 | 0.72 |
| VidCom2 | centroid uniqueness | −1.61 | 137/112 | 1.52 |

Neither gap is significant on its own — the honest claim is **neither heuristic shows any
advantage, and both trend negative**. Two mechanically unrelated criteria failing the same
way is stronger evidence than one method failing.

#### Finding 2 — more frames DOES help; compression was eating the gain

The uniform ladder is monotonic: 32f **39.70** → 64f **42.16** → 256f **44.16**
(64→256 is +2.00 pp, |z|=1.74).

#### Finding 3 — the gain is NOT from hitting the evidence more often

Coverage improves enormously with frame count, and accuracy does not follow it:

| arm | spacing | hit% | frames in evidence | nearest | IoU |
|---|---|---|---|---|---|
| e 32f | 115 s | 36.1 | 3.09 | 25 s | 0.089 |
| g 64f | 57 s | 47.0 | 6.16 | 12 s | 0.142 |
| j 256f | 14 s | 67.2 | 24.58 | 3 s | 0.204 |

Paired 64f→256f (uniform, no compressor confound), split by whether coverage changed:

| stratum | n | g | j | Δ |
|---|---|---|---|---|
| 256f hits, 64f misses (**coverage gained**) | 312 | 43.27 | 43.91 | **+0.64** |
| both hit | 727 | 44.98 | 47.87 | +2.89 |
| neither hits | 508 | 37.60 | 39.17 | +1.57 |

**The gain is absent exactly where coverage improved.** It is largest where both arms
already hit — i.e. where 256f puts ~24 samples inside the evidence instead of ~6. The
mechanism is sampling *density within* the evidence, not probability of touching it.
(Subgroup |z| = 0.13 / 1.68 / 0.71 — none individually significant; read as direction.)

#### Finding 4 — the token-on-evidence invariant

At a fixed budget, uniform sampling spends a **constant** number of tokens inside the
evidence span regardless of frame count — frames go up 8x, tokens/frame go down 8x:

| arm | tok/frame | frames in evidence | **tokens on evidence** |
|---|---|---|---|
| e 32f | 368.9 | 3.09 | 1,139 |
| g 64f | 189.2 | 6.16 | 1,165 |
| j 256f | 48.0 | 24.58 | 1,180 |

**Scope of this claim (corrected after mentor review):** it constrains only the *temporal*
distribution of budget. It does NOT imply that spatial token selection inside evidence
frames is futile — that is what arms i/j/k test, and they answer it at ~96 tok/frame only.

#### Finding 5 — buying more budget is weak; buying better allocation is strong

Doubling the budget at 256 frames: 12,288 → 24,527 tokens buys **+1.60 pp** (n=500,
|z|=0.95, ns). Meanwhile the oracle spends **half** of arm g's tokens and gains
**+13.23 pp**, purely by putting them on the right 17 seconds.

Extrapolating one more doubling puts unpruned 256f @ 49,152 near ~47%. So compression at
r=0.25 costs ~1.6 pp against the best same-budget alternative, and the *entire* remaining
uniform-sampling headroom is ~3 pp — against ~13 pp available from allocation.

> **The scarce resource is not visual tokens and not temporal coverage. It is knowing where
> to spend the tokens.** Compression is the mechanism that makes bandwidth reallocatable;
> it does not decide where the bandwidth should go.

#### THREE CLAIMS FROM EARLIER IN THIS CHAPTER ARE WITHDRAWN

1. **"Compression costs 4 pp" (D3, e vs f) — WITHDRAWN.** e vs f cut the budget 3.73x at
   the same time as compressing. arm f simply had the smallest budget of any arm run here
   (3,167 tokens). At matched budget the cost is 1.61 pp, not 4.00.
2. **"Compression is free at matched budget" (g vs i) — WITHDRAWN.** g vs i tied (+0.39),
   but g is the wrong control: it has 4x fewer frames. Against the right control (arm j,
   same frames, same budget) VidCom2 is −1.61 pp.
3. **"Frames show diminishing returns after 64" — WITHDRAWN.** That was measured as
   g→i (+0.39), which carried VidCom2's penalty. Uniform g→j is +2.00 pp.

Each was a case of a comparison changing two things at once. **Every claim in this chapter
must name its control explicitly.**

#### INFRASTRUCTURE GOTCHAS FOUND BUILDING THIS (all cost a run)

- **Qwen3-VL caps the whole video, not the frame.** `video_preprocessor_config.json`
  `size.longest_edge = 25,165,824` → `/2/1024` = **12,288 visual tokens regardless of frame
  count**. `max_pixels` only binds when it is below that share; at 32 frames 786,432 px ==
  768 tok == exactly the cap's share, which is why arm e looked like it honoured
  `max_pixels`. Override with `size.longest_edge` in `--mm-processor-kwargs`.
- **`encoder_cache_size` is not separately configurable** — `config/scheduler.py:235`
  hardwires it to `max_num_batched_tokens`, and its auto-grow
  (`encoder_cache_manager.py:312`) computes the max item from the *default* video size, so
  it under-estimates when `size` is overridden. Unpruned high-budget arms 400 with "video
  item with length N exceeds the pre-allocated encoder cache size" until `BATCHTOK` is
  raised by hand. Pruned arms slip under it because the cache holds post-prune embeddings.
- **The unpruned 256f @ 49,152 arm does not fit this box.** weights 16.4 + ~49k-batch
  profiling ~12 + KV 6.9 + transient ViT ~8 + neighbour 5.9 = ~49 GiB on a 47.3 GiB card.
  Startup succeeds and the first request OOMs. Ran `VIDCAP=2` (24,576 tok) instead.
- **Results are written in completion order under N workers.** Pair on `question_id`,
  never on line order.
- **`GPUUTIL` deviations to record:** arms i, k ran at 0.55 (ViT OOM at 0.80); arm l at
  0.70. Affects KV cache size only, never the arithmetic — but it is an unmatched field.

#### Faithfulness diagnostic — `fast_agent/diagnose_runs.py`

Five checks, all passing: proxy timeline drift < 0.02% on all 103 videos (the check that
matters most — a drifted proxy relabels every `<t seconds>` marker silently); identical
question_id / gold / videoID / evidence across arms with 0 duplicates; measured token
budgets matching intent; 0 transport errors; 17 pinned hyperparameters identical. The
no-tool arms were confirmed to receive the no-tool prompt (tool schema `null`) via
`--print-prompt`.

One dataset property, not a bug: **17 of 103 videos are below 720p** (204 questions), where
source resolution binds before the token cap. Stratified, the e→g gain is +2.45 pp in
*both* strata, so it is not driven by this.

### IS THE HARNESS BROKEN? No — measured 2026-08-13

38.99% looks alarming next to the published 58.0, so it was tested rather than argued.

| condition | visual tokens | acc |
|---|---|---|
| **blind** — question + options, **no video at all** | 0 | **24.66** |
| run1 — 64-frame video skim | 2,688 | **38.99** |
| run5 — 256-frame video skim (running) | 10,752 | ~42 (n=97) |
| published Qwen3-VL-8B-Instruct | ≤224,000 | 58.0 |

`fast_agent/diag_blind.py`, sharing the same loader, question rendering, scoring and
sampling params as the real runs — the only change is that the frames are removed.

- **Blind is 24.66%, i.e. exactly 4-way chance.** The harness is not leaking answers, the
  option parsing is not degenerate, and the model has no text-only shortcut on LVBench.
- **The 64-frame skim is worth +14.3 pp over blind**, and 4× the frames adds ~3 pp more.
  Video reaches the model, is used, and scales in the right direction.
- The remaining gap to 58.0 is **budget**: we run at ~1.2% of the published visual token
  allowance (2,688 vs ≤224,000; 84 tok/frame vs 640; 64 frames vs up to 2,048).
  Against a 24.66% floor and a 58.0 ceiling, run1 captures **43% of the achievable headroom
  on 1.2% of the tokens.**

So: the implementation is sound and the absolute numbers are simply not leaderboard
numbers. **They must never be quoted as Qwen3-VL's LVBench score** — they are internally
comparable only, which is all the compression question needs.

**Reading run1 → run2 → run3 (seed 0, pending run4 and the seed repeats):**

- **The crop tool is worth +0.58 pp** (38.99 → 39.57). Essentially nothing, and for the
  reason run3 already showed: the model reaches the evidence on 11.7% of questions. Where
  it does reach, accuracy is 53.04% vs 37.85% where it does not — a **+15.2 pp** payoff on
  a 12% slice.
- **Compression did not buy localization — it cost accuracy.** run3 sees 4× the frames at
  the same visual-token budget (2,688 = 2,688, verified) and localizes *worse*: hit 9.96%
  vs 11.70%, `acc | landed` 45.45% vs 53.04%, overall 33.59% vs 39.57%, a **−6.0 pp** drop.
  This is the cleanest test of the chapter's premise so far and it comes out negative:
  4× temporal coverage at matched budget does not help the model decide where to look, and
  the per-frame degradation hurts what it can read once there.
- Both tool arms are floored by the two behaviours in the run3 notes below (10-second crop
  windows, and localization that only works when the question states the time), which are
  shared and therefore do not explain the run2 vs run3 gap.

### OPERATIONAL — two failure modes that silently corrupt a run (both fixed 2026-08-12)

**1. A dead question is scored as a wrong answer.** `run_agent.work()` catches any exception
and writes `pred=None`, which counts as incorrect — and the resume logic then *skips* that
row forever, so a re-run cannot repair it. Measured on the first attempt: **2.8% of run2
and 13.5% of ctrl** were `APITimeoutError`, i.e. those arms were being biased downward by
transport failures. Fixes: `_call` now retries transport errors with backoff to ~5 min
(long enough to survive a server restart), the client timeout dropped 1800 s → 300 s so a
stall fails fast enough to retry inside the question, and `overnight.sh::drop_errors`
strips error rows before every retry so they are actually re-run.
**Always check `sum('error' in r)` over results.jsonl before reporting an accuracy.**

**2. vLLM wedges while still looking healthy — ROOT CAUSE FOUND, and it is a config fix.**
Both arm servers repeatedly stopped serving `/v1/chat/completions` — requests accepted,
never returned, engine reporting `Running: 0 reqs, Waiting: 0` — while `/v1/models` kept
answering 200. Six watchdog restarts in one night before the cause was located in the
engine log:

```
AssertionError: Expected a cached item for mm_hash='793b9497...'
  vllm/v1/engine/core.py -> mm_receiver_cache.get_and_update_features
```

vLLM caches processed multimodal items in the **API process** and ships only a hash to
**EngineCore**, which resolves it against a bounded receiver cache. When the two desync,
EngineCore raises and **dies**; the API process survives and keeps answering `/v1/models`.
This workload provokes it hard: one skim proxy shared by ~15 questions, resent on every
tool round, plus 128-image crops at 8-way concurrency.

**Fix: `--mm-processor-cache-gb 0`** (default is 4) in `serve_arm.sh`. That removes the
sender/receiver split entirely — full MM data travels with each request. Slightly more IPC,
no more assertion.

`watchdog.sh` stays as the safety net, and its probe is a **real completion**, never
`/v1/models` — which would have reported both servers healthy the whole time.

Related, and the reason the restart is not just `kill`: vLLM runs the model in a separate
`VLLM::EngineCore` process. Terminating the API server left both EngineCores alive holding
**47,978 and 47,786 MiB**. Per CLAUDE.md they were released with SIGTERM (never SIGKILL) —
but a restart script must terminate the orphan *and wait for `nvidia-smi` to show the memory
back* before relaunching.

### run3 seed0 — the two findings
1. **The model can only "localize" when the question already states the time.**

   | | n | hit rate | acc |
   |---|---|---|---|
   | stem states the timestamp | 203 | **61.6%** | 38.4% |
   | stem does not | 1,343 | **2.2%** | 32.9% |

   The headline 10% hit rate is almost entirely copied timestamps. Genuine search is ~2%.
   By-task confirms it: temporal grounding 67.2% hit (those questions report times), every
   other category 5–11%.

2. **The model asks for 10-second crops.** Tool cap is 128 frames at 1 fps = 128 s; measured
   width median **10 s** (p25 5, p75 10). Five rounds inspects a median of **0.3%** of the
   video. LongVT's schema text is just "Crop a video to a specified duration" — no guidance
   on width — so the model defaults to tiny windows. 130 spans were rejected outright by
   `clamp_span` (<1 s or unparseable).
   This is a **shared floor across run2 and run3** (identical schema, rounds, clamping), so
   it does not confound the compression comparison — but it caps what any arm can achieve.
   Changing it would require re-running the whole ladder; deliberately not changed.

Also: `finalizer_used` is **70.4%** in run3 (strict accuracy, i.e. answers the model
produced unprompted, is 11.81%). Most of the reported accuracy is a forced choice after an
inconclusive search — which is exactly why run1 (no tools, same input) is the control that
matters, and why run4 is needed for a ceiling.

### 2026-08-12 pilot ladder — all four complete, n=1,549 each, seed 0
These were run **before** the video-modality decision, so runs A/A+ use an image skim and
are NOT the canonical run2. Kept because they isolate the timing mechanism.

| run | skim | acc | hit% | mean cov | acc \| landed | n landed | acc \| miss |
|---|---|---|---|---|---|---|---|
| vanilla, no tool | 64 img | **41.64** | – | – | – | – | – |
| A — crop, LongVT verbatim | 64 img | 40.28 | 10.15 | 0.118 | **58.60** | 157 | 38.27 |
| A+ — crop, + time legend | 64 img | 40.87 | **23.59** | 0.280 | 49.59 | 365 | 38.24 |
| B — crop, VidCom2 q=0.75 | 256 vid | 33.57 | 9.96 | 0.116 | 45.45 | 154 | 32.33 |

Three things to read off this, carefully:

1. **The tool never pays for itself on aggregate** (40–41% vs 41.64% with no tool), but
   landing is worth +20pp. The lever is localization, not perception.
2. **The time legend more than doubles hit rate (10.15 → 23.59) and barely moves accuracy**
   (40.28 → 40.87), while `acc | landed` *falls* (58.60 → 49.59). That is a selection
   effect, not a regression: the 208 extra landings are the questions it previously could
   not even find, which are harder. Reporting `acc | landed` without `n landed` would have
   made a win look like a loss.
3. **Run B is not interpretable as "compression hurts."** It changes three things at once
   against A — modality (video vs images), frame count (256 vs 64), and pruning — so its
   33.57% cannot be attributed. This is precisely why the canonical ladder holds modality
   and budget fixed and varies only frames×retention. **Do not cite B as a compression
   result.**

### The 10% is partly OUR harness, not the model (fixed)
The LongVT prompt gives an image skim **no time anchor at all** — 64 JPEGs, no timestamps,
no duration. LongVT-RFT was *trained* to map frame position to time; zero-shot Qwen3-VL was
not, so it scans linearly from t=0 in ~50 s steps (on a 3,167 s video with evidence at
2,618 s it never got past 275 s). Re-adding `config.initial_view_text`'s exact per-frame
time legend, everything else identical:

| n=29 probe | LongVT verbatim | + timestamp legend |
|---|---|---|
| mean coverage | 0.118 (n=1549) | **0.340** |
| hit rate | 10.15% | **27.59%** |
| accuracy | 40.28% | 44.83% |

Full re-run under way as `lvbench_armA_ts_seed0` (at n=43: 53.5% acc, 32.6% hit).
**The LongVT-verbatim run is kept, not discarded** — it is the faithful reproduction and
the reason the legend's effect is quantifiable.

### The legend must NOT be added to arm B
Video modality already carries per-frame markers, but vLLM computes them as
`frame_index / fps` and then averages them **in pairs** (`temporal_patch_size=2`) → 128
values (14.7, 73.7, 132.7 …). A legend computed client-side says 15, 44, 74 … — two
conflicting labellings of the same frames in one prompt, worse than either alone. Caught
before launch; `run_agent.py` now hard-gates the legend to `--skim-mode images`.
Arm B's markers are **inline per frame**, so it is not the less-salient condition.

### Prompt: now LongVT-verbatim
`config.LONGVT_TOOL_SCHEMAS` + `config.longvt_user_text` reproduce
`LongVT/examples/eval/single_inference.py` exactly, including the `video_path` parameter
our loop does not need (it is part of the text the model conditions on). Deviations from
that reference, all deliberate: 64 frames not 512 (user spec), T=0.7 not 0 (user spec),
`max_tokens` 1024 not 4096, and `data.UNIFY_TIME_FORMAT` still rewrites "17:16" →
"17:16 (= 1036 s)" in the question. `--prompt-style fastagent` keeps the old timestamp-legend
prompt so the legend's effect stays measurable.

## Prior numbers — arm A under DIFFERENT conditions (reference only, do not quote as the baseline)

`max_tokens` is the one hyperparameter that could not be held at the baseline's value —
32 tokens cannot hold a tool call. Set to 1024 for tool turns, **32 for the finalizer**
(so the finalizer is byte-comparable with the vanilla run). Loop risk at pp=0.0 was 7–10%
under *greedy* decoding; T=0.7/top_p=0.8/top_k=20 should suppress it — to be confirmed on
the probe, not assumed.

## Prior numbers — arm A under DIFFERENT conditions (reference only, do not quote as the baseline)

From `lvbench_{baseline,crop,oracle_crop_cot}_full`, n=200 (183 with a parseable evidence span), **greedy decoding, single run, FlashVID-era harness, HF backend**. Re-measure under the conditions above before comparing anything to arm B.

| arm | accuracy |
|---|---|
| direct (no tool) | 31.7% |
| autonomous crop | 33.3% |
| oracle crop (forced onto GT) | 61.2% |

| crop landed on evidence? | n | share | accuracy |
|---|---|---|---|
| landed | 39 | 21% | **46.2%** |
| missed | 98 | 54% | **27.6%** |
| never called the tool | 46 | 25% | 34.8% |

Supporting: crop window median 54s vs GT evidence median 14s; median 1 call/question (46 never call, 45 burn all 5); **median 1.6% of the video ever inspected**; landing correlation r=0.95 on the questions it hits vs r=0.21 on the ones it misses — **bimodal: it either knows where to look or it is guessing, with nothing in between.**

Report: `longvt_compression/fast_agent/PRESENT_lvbench_localization.ipynb`, built by `present_lvbench.py`.

**Interpretation that motivates arm B:** the bottleneck is search *reach*, not perception — 5 calls × ~54s cannot cover a 60-minute video, and the model inspects 1.6% of it. A compressed global view is the natural intervention, and hit rate is the metric that should move.

---

## Current Chapter — Compression on the trained LongVT policy (started 2026-08-06)

**Thesis:** LongVT-RFT is a policy that was actually *trained* to skim globally then `crop_video` into detail. Add compression to its ViT so the global skim covers far more of the video at the same token cost, and measure whether the trained tool-use policy gets better at locating evidence. Then, if it does, turn the compressor into a second tool and train the model to alternate between the two.

**Why this chapter exists:** every compression result before it was measured on a *stock* model (VideoARM's Qwen3.5-9B, then zero-shot Qwen3-VL-8B). A stock model has no learned policy that reads the skim and decides where to crop, so its indifference to skim content was never evidence that a trained one would be indifferent too. That is the open question, and LongVT is the first place it can be asked.

### Baseline: what LongVT actually runs (verified 2026-08-06, no code written)

| Fact | Value | Source |
|---|---|---|
| Checkpoint | LongVT-RFT = Qwen2.5-VL-7B fine-tune, at `/local1/cfyang/LongVT-RFT` | `config.json` |
| Serving | vLLM OpenAI server, `--tool-call-parser hermes`, custom chat template | `examples/eval/run_eval.sh` |
| Global skim | `fps=1`, `max_frames=768` (512 for RFT), `max_pixels=50176` (224²) | `run_eval.sh`, README |
| **Modality** | **image** — every frame is a separate base64 PNG `image_url` | `single_inference.py::encode_video_frames` |
| Tool return | also base64 images (`crop_video`, ≤128 frames) | `single_inference.py::crop_video_local` |

Token cost of the skim (Qwen2.5-VL: patch 14, spatial_merge 2, temporal_patch 2 → 64 merged tokens per 224² frame):

| Send as | 512 frames | note |
|---|---|---|
| image (LongVT today) | **32,768 tok** | one grid-t per frame |
| video | **16,384 tok** | temporal_patch=2 pairs adjacent frames — 2× free |
| video + pruning q=0.75 | **4,096 tok** | |

### Three ways to add ViT-side compression

- **Path A — vLLM built-in EVS.** vLLM 0.16 already implements Qwen2.5-VL video-token pruning: `Qwen2_5_VLForConditionalGeneration` declares `SupportsMultiModalPruning` (`model_executor/models/qwen2_5_vl.py:1011`), enabled by `--video-pruning-rate q`. Algorithm (`multimodal/evs.py:37`): cosine similarity between temporally adjacent tokens at the same spatial position → keep top-`(1-q)·N` by dissimilarity, first frame always fully retained. Pure selection, no merge, query-agnostic, no CLS attention needed. It also handles the two things that break hand-rolled pruning — mrope recomputation (`recompute_mrope_positions`) and matching the prompt's `<|video_pad|>` count to the ViT output (`qwen2_5_vl.py:968-984`). **Zero code.** Requires sending the skim as `video_url`.
- **Path B — HF + FlashVID's Qwen2.5-VL vision patch.** `FlashVID/flashvid/modeling_qwen2_5_vl.py` already exists and splits the same way as the Qwen3-VL version: ViT-only forwards at `:530` / `:268` / `:192`, and the FastV LLM pruning isolated in `Qwen2_5_VLTextModel_forward` (`:31`) which you simply do not patch. Signature-compatible with installed transformers 4.57.6 (verified). Its "cls_attention" is a proxy — last block's manually-computed attention matrix, `mean(head).mean(query)` — since Qwen2.5-VL's ViT has no CLS token. Costs: leaves vLLM, must re-implement tool-call parsing, ~8× slower than the old 64-frame harness.
- **Path C — port FlashVID's selection rule into vLLM's ViT**, reusing EVS's hook points. Keeps vLLM speed and the hermes parser. This is where compression should live if it becomes a tool.

**Path D — VisionZip: evaluated 2026-08-06 and REJECTED as the primary compressor.** Repo cloned to `/home/cfyang/hanklin/VisionZip` (CVPR 2025, dvlab-research). Mechanism: last vision block's attention logits, mean over heads and *summed over queries*, gives a per-token "received attention" score (Qwen2.5-VL's ViT has no CLS token, so this is the same proxy trick FlashVID uses); top-k of that = **dominant tokens**; the leftovers are merged into `contextual_num` evenly-spaced anchors by key-vector cosine similarity. Selection happens after PatchMerger, in merged-token space. Three reasons it does not fit:
  1. **Wrong axis.** VisionZip removes *spatial* redundancy inside a frame. At 224² a frame is already only 64 merged tokens post-PatchMerger. The redundancy in a 512-frame skim is *temporal*, and VisionZip does not model time at all. The authors say as much in `Qwen2_5_VL/README.md`: "Qwen2.5VL already uses PatchMerger … the performance gain from VisionZip is less striking compared to LLaVA."
  2. **Their own Qwen2.5-VL numbers do not show a gain.** Retain 70%: MME 2316→2334, everything else flat-to-down. Retain 50%: MME →2209, **OCRBench 81.5→70.5**, MMVet 61.6→57.0. And we need ~12–25% retention, well past that.
  3. **Not actually easy to integrate.** The pip package `visionzip` is LLaVA-only (imports `llava`). Qwen support is a standalone 2,224-line *fork* of the modeling file. It does import cleanly under transformers 4.57.6 and the LongVT-RFT checkpoint key layout (`visual.*`, `model.layers.*`, `lm_head.*`) matches it — but (a) the **video path is broken**: `qwen2_5vl_visionzip.py:1865` calls `self.visual(...)` expecting one return value while the patched forward returns three, so `video_embeds.shape[0]` raises; and `select_pixel` is only set for `pixel_values`, so compression never applies to video anyway; (b) it assumes **one contiguous image span** — `img_mask[first:last+1] = ~select_mask` — which breaks for LongVT's N-separate-images skim. Measured with the real processor: 3 images = 192 image tokens but a span of 196 (2 delimiter tokens per gap) → shape mismatch. At 512 frames that is 32,768 vs 33,790; (c) `batch_size=1` only, no vLLM path.

  **Keep it for two narrower uses:** a spatial-vs-temporal ablation axis, and possibly compressing `crop_video` returns (≤128 full-detail frames), where within-frame redundancy is the relevant kind.

**Order: A first (same-day numbers), C later. B only to reproduce FlashVID-specific behaviour.**

### Known confound to control for

LongVT was trained end-to-end (SFT 247.9K → RL 1.6K → RFT 15.4K) on **image-modality** frames only. Switching the skim to video modality changes `<|image_pad|>`→`<|video_pad|>`, the mrope temporal index, and pairs adjacent frames. So any "compression helped/hurt" result is confounded with "modality changed" unless a **video-modality-without-compression control arm** is run. This is not optional.

### Status

Verification complete 2026-08-06. No code written. Experiment design not yet fixed; the open questions are which benchmark (must have GT evidence spans — LVBench or VideoSIAH-Eval, *not* VideoMME-long) and whether the primary metric is accuracy or tool-use behaviour (crop-call rate, crop-window coverage of GT evidence, rounds used).

---

## VidCom2 integration check — OFF-SPEC, not the ACTIVE TASK (2026-08-10/11)

> ⚠️ **This whole section violates the ACTIVE TASK's fixed decisions** and must not be read as an experimental result. It was run on **HF in-process inference** (forbidden), **temperature 0** (spec: 0.7), **1 run** (spec: 3), on **VideoMME** (spec: LVBench), with **no trajectories** (spec: must be stored), and reports **accuracy** (spec: localization). Cause: the ACTIVE TASK section was added to this file mid-session and the stale copy was never re-read.
>
> **What survives and transfers** to the real task: the paper↔code audit (§ below), the proof that VidCom2's per-frame budget cannot concentrate, and the measured retention accuracy. **What does not:** every accuracy number and the flip analysis. The VideoMME numbers retain one narrow use — they show the VidCom2 integration reproduces the published README table to within 0.35 pp, i.e. the compressor is wired up correctly.

**Why VidCom2:** vLLM upstreamed it ([PR #47750](https://github.com/vllm-project/vllm/pull/47750), merged 2026-07-28), so it is the closest thing to a community-standard compression baseline. EMNLP 2025, [arXiv 2505.14454](https://arxiv.org/abs/2505.14454). Repo already on disk at `/local1/cfyang/VidCom2` (branch `qwen`, Qwen3-VL supported).

### Where it can run (verified 2026-08-10)

| | status |
|---|---|
| vLLM | `VideoPruningMethod = Literal["evs","vidcom2"]` exists **only on `main`**. Merged 2026-07-28, one day *after* v0.26.0 was cut (2026-07-27). Checked v0.19/0.20/0.22/0.24/0.26 — none have it. Installed here: **0.19.0** (`vllm_src` is a symlink to site-packages). Using it in vLLM = build main from source. |
| HF reference impl | **Runs today, zero install.** `PYTHONPATH=/local1/cfyang/VidCom2` shadows the `flashvid` env's FlashVID lmms-eval; `token_compressor` + `lmms_eval` + Qwen3-VL modeling all import cleanly under transformers 4.57.3 / torch 2.5.1. Runner: `~/hanklin/run_vidcom2_smoke.sh`. |

### IMPLEMENTATION AUDIT — four findings, all load-bearing

**(0) THE PAPER NEVER EVALUATES QWEN3-VL.** Full-text search of arXiv 2505.14454: **"Qwen3" appears 0 times.** The paper's models are LLaVA-OV-7B, LLaVA-Video-7B, and **Qwen2-VL**. The Qwen3-VL-8B table quoted below (64.5 → 62.4 overall, 57.0 → 56.1 long) exists **only in the repo README**, added 2026-01-22, never peer-reviewed. **There is no published number for our configuration to reproduce.** The README table is the sole reference point and should be cited as such.

**(1) The paper and the official code disagree.** Paper Eq. 4 / Algorithm 1 scores tokens by **cosine similarity** to the global video representation. The released code adds two steps the paper never mentions (full-text search: "channel" 0 hits, "variance" 0, "Gaussian" 0, "kernel" 0):
- `select_low_var_channels` — scores only the lowest-variance 50% of channels;
- a sum of 5 **Gaussian kernels** (α = 2⁻³…2¹) on L2-normalised features instead of cosine;
- centres `g^v` / `g^{f,t}` are means of **normalised** tokens, not raw tokens as in Alg. 1;
- `k_t` uses `round()` where Alg. 1 line 32 specifies `⌈r_t × M⌉`.

Structure matches exactly (two stages, same score composition, same Eq. 6). Arithmetic does not. A single Gaussian is monotone in `x·c` for unit-norm `x`, so `v_score` and `f_score` each preserve the paper's ranking in isolation — but they are then **summed**, and monotone(a)+monotone(b) ≠ monotone(a+b), so the relative weighting of video-level vs frame-level uniqueness differs from the paper and final rankings genuinely diverge.

vLLM reimplemented from the **code**, not the paper (PR: "follows the authors' released reference implementation, which is what produced the published accuracy numbers"). So the code is the reproduction target; the paper under-describes the method.

**(2) The "dynamic frame budget" — the paper's headline claim — is bounded to ~one frame's worth of reallocation, by construction.** From the paper's own Eq. 6, `r_t = R(1 + σ_t − 1/T)` with σ a softmax at **τ = 0.01**:
- `Σ r_t = R·T` exactly → average retention is exactly R;
- `r_t ∈ [R(1−1/T), R(2−1/T)]` → **a single frame can get at most 2R. Hard ceiling, data-independent**;
- total displacement `R·Σ|σ_t − 1/T| ≤ 2R` → **at most ~2 frames' worth of budget moves, regardless of T**.

τ = 0.01 makes the softmax near one-hot, so in practice *one* frame gets ~2R and the other T−1 get ~R. Verified numerically with the repo's own functions (`/local1/cfyang/VidCom2/probe_allocation.py`): at T=128, **127 frames get 45 tokens and 1 frame gets 90** — 0.7% of the budget displaced. Frame scoring itself is accurate (injected distinctive frames do rank first); the *magnitude* is what the formula caps.

**Consequence:** VidCom2 does **not** meaningfully address the archive's "the lever is WHERE the tokens go" finding **at the frame level**. Its real difference from FlashVID is the **within-frame** selection rule (outlierness w.r.t. both the video centre and the frame centre). Any concentration measurement must therefore be done **within frames**, not across them.

**(3) Footguns.**
- `batch_size != 1` → compression **silently disabled** (`token_compressor/vidcom2/models/qwen3_vl.py:130-132`). Same class of bug as the pre-07-28 FlashVID r=1.0 bypass that flattered every compression number. Always pin `--batch_size 1` and verify token counts.
- Reference impl has **no exact-count reconcile** (vLLM added one because it sizes placeholders before the mask exists) → nominal R drifts. **Measured on real VideoMME videos: 0.25 → 0.2497/0.2499/0.2517.** Far tighter than FlashVID's 0.10 → 0.117.
- mrope `position_ids` are **sliced, not recomputed** (`:190`), unlike vLLM EVS's `recompute_mrope_positions`. Defensible, but a difference if HF and vLLM numbers are ever compared.
- The checkout carries an **uncommitted required fix** (recompute `cache_position` after pruning); without it the forward crashes.

### What the authors' own numbers predict

Qwen3-VL-8B, 32 frames, R=25%: VideoMME overall 64.5 → 62.4 (−2.1), **VideoMME-long 57.0 → 56.1 (−0.9)**. Same shape as the three FlashVID measurements — roughly free, not helpful. Accuracy is therefore not the informative readout; within-frame concentration is.

### HARNESS BUG FOUND — the shipped `videomme_long` task leaks subtitles

`videomme_long.yaml` includes `_default_template_yaml`, whose `doc_to_text` is **`videomme_doc_to_text_subtitle`** — it pastes the full `.srt` into the prompt. The paper's "VideoMME (Long)" column comes from the plain `videomme` task (`videomme_doc_to_text`, no subtitles). Running the shipped long task therefore measures the setting *least* sensitive to visual compression.

Measured both ways at n=50, and the leak is visible in the churn rate:

| setting | baseline | VidCom2 R=0.25 | Δ | churn |
|---|---:|---:|---:|---:|
| `videomme_long` (**subtitles leaked**) | 56.0% | 60.0% | +4.0pp (p=0.625) | 4/50 (8%) |
| `videomme_long_nosub` (**correct**) | 64.0% | 62.0% | −2.0pp (p=1.000) | 5/50 (10%) |

Fix: new task `lmms_eval/tasks/videomme/videomme_long_nosub.yaml` — verified by diff to be `videomme.yaml` plus the long filter, same `doc_to_text`, same qwen3_vl prompt block. **Use `videomme_long_nosub`, never `videomme_long`.**

### Reproduction fidelity (verified 2026-08-10)

| | paper / README | ours | |
|---|---|---|---|
| eval framework | LMMs-Eval | LMMs-Eval (repo's own, via PYTHONPATH) | ✓ |
| frames | 32 | 32 (→ grid t=16, `temporal_patch_size=2`) | ✓ |
| subtitles | none | none | ✓ |
| decoding | — | temperature 0, `do_sample=false`, `max_new_tokens=16` | ✓ |
| R | 25% | 25% (**measured 0.2497–0.2503**) | ✓ |
| batch_size | 1 | 1 | ✓ |

Repo modifications, all verified non-algorithmic: required `cache_position` crash fix (pre-existing in checkout), one env-gated `VIDCOM2_LOG` print, `token: True→False` (HF auth), the new no-subtitle task.

**No thinking/CoT anywhere:** `reasoning_prompt=None`, `max_new_tokens=16`, and every stored response is exactly 1 character (a bare option letter). This path yields an accuracy number and nothing else — no trajectories to analyse, unlike the archive's agent runs.

Visual token budget: 720 tokens per grid-frame × 16 = **11,520 per question** (low-res videos: 220 × 16 = 3,520). `max_pixels` 1,605,632 requested but clamped to 786,432 by the video preprocessor.

### RESULT — VideoMME-long, n=900, paired (2026-08-11)

Three arms, same 900 questions, temperature 0 (deterministic → every comparison exactly paired).

| arm | surviving tok | acc | README ref | Δ vs README |
|---|---:|---:|---:|---:|
| baseline | 11,520 | **57.33%** | 57.0 | **+0.33 pp** |
| VidCom2 R=0.25 | 2,880 | **55.78%** | 56.1 | **−0.32 pp** |
| VidCom2 R=0.10 | 1,155 | **52.44%** | — | — |

**The reproduction is tight: both arms land within 0.35 pp of the README table.** Given that the README is the only reference (the paper never ran Qwen3-VL), this is as good a validation of the setup as exists.

| comparison | Δ | broke/fixed | McNemar p | 95% CI | churn |
|---|---:|---|---:|---|---:|
| R=0.25 − baseline | −1.56 pp | 48 / 34 | **0.151** | [−3.56, +0.44] | 9.1% |
| R=0.10 − baseline | −4.89 pp | 92 / 48 | **0.00025** | [−7.56, −2.33] | 15.6% |
| R=0.10 − R=0.25 | −3.33 pp | 57 / 27 | **0.0014** | — | — |

### THE FINDING — the archive's absolute-budget law reproduces on a different compressor

The archive (2026-07-28/29, FlashVID, same model) concluded: **the governing variable is the absolute surviving token count (~1.5–2k break point), not the retention ratio** — r=0.25 cost −3.06 pp at a 1,440-token base (391 surviving) but ±0.00 pp at a 14,400-token base (3,963 surviving).

VidCom2's two points straddle that same break point and behave exactly as the law predicts:

- **2,880 surviving → −1.56 pp, p=0.151, CI includes 0** (above the break point → effectively free)
- **1,155 surviving → −4.89 pp, p=0.00025, CI excludes 0** (below it → real damage)

Two algorithmically unrelated compressors — FlashVID (DySeg+ADTS+TSTM merge, flat allocation) and VidCom2 (Gaussian outlierness on low-variance channels, near-flat dynamic budget) — hit the same wall at the same absolute budget. **This is the strongest cross-method evidence yet that the break point is a property of the model's token budget, not of any particular selection rule.** It also means the choice of compressor barely matters in this regime, which is consistent with, and strengthens, the archive's "uniform reduction is nearly free precisely because it preserves the average, which is the same reason it cannot help."

Per-task deltas (categories with n≥50 only; Spatial Perception n=3 and Temporal Perception n=6 are pure noise and must not be quoted):

| task | n | base | R=0.25 | R=0.10 |
|---|---:|---:|---:|---:|
| Object Reasoning | 240 | 57.9% | −1.7 | −2.5 |
| Action Reasoning | 180 | 58.3% | −3.3 | **−8.9** |
| Information Synopsis | 163 | 72.4% | −0.6 | −3.7 |
| Temporal Reasoning | 91 | 44.0% | +2.2 | −3.3 |
| Action Recognition | 63 | 46.0% | +0.0 | −1.6 |
| Object Recognition | 54 | 50.0% | −7.4 | −5.6 |
| Counting Problem | 48 | 41.7% | +4.2 | **−10.4** |

At R=0.10 the biggest losses are Counting (−10.4) and Action Reasoning (−8.9) while Information Synopsis holds (−3.7) — the same "gist survives, within-frame configuration dies" split the archive found at n=882.

### Artifacts

Runner `~/hanklin/run_vidcom2_smoke.sh` (`TASK`, `R_RATIO` env-overridable), analysis `~/hanklin/analyze_vidcom2_full.py`, allocation probe `/local1/cfyang/VidCom2/probe_allocation.py`. Per-arm sample logs copied to unambiguous names at `/local1/cfyang/hanklin/outputs/vidcom2/ARM_{baseline,vidcom2_r0.25,vidcom2_r0.10}_n900.jsonl` (the runner wrote both VidCom2 arms into one dir; they were disambiguated by accuracy).

### FLIP ANALYSIS (2026-08-11) — the damage is systematic, and so is the *help*

Decoding is temperature 0 / `do_sample=False`, so arms are deterministic and exactly paired: **every flip is caused by the compressor, with zero sampling noise.** This is a much cleaner instrument than the archive's temp-0.7 agent runs, where ±20pp self-noise swamped small effects.

**Churn dwarfs the net effect.** At R=0.10, **200/900 (22.2%)** of answers change while net accuracy moves only −4.89 pp. Even among the 336 questions *both* arms get wrong, 17.9% change answer. The aggregate is a small residual of a large two-way flow — same structure as the archive's "16.2% churn for a −1.0pp net."

**Flips are reproducible across retention levels, not jitter:**

| | R=0.25 | R=0.10 | overlap | expected if independent | obs/exp |
|---|---:|---:|---:|---:|---:|
| broke | 48 | 92 | **40** | 8.6 | **4.67×** |
| fixed | 34 | 48 | **29** | 4.2 | **6.82×** |

83.3% of what breaks at R=0.25 also breaks at R=0.10 — damage is monotone in budget, the same questions dying in order.

**The surprise: a stable set of ~29 questions the baseline gets WRONG and compression reliably gets RIGHT at both retention levels** — concentration 6.82× over chance, *higher* than the broke set. This is a reproducible, n=900 measurement of the **dilution effect** the archive only ever saw anecdotally ("full1280 at 28,800 tok loses 703-3 that full256 at 5,760 answers"). Removing tokens is not purely destructive; there is a real population of questions where the full token set actively misleads.

**No question-level feature explains which questions flip.** Tested and rejected:

| hypothesis | result |
|---|---|
| negation questions ("not"/"incorrect"/"except") get rescued | 13.8% of fixed vs 10.7% of not-fixed, **Fisher p=0.54 — rejected** |
| damage concentrates in a category | break rate 8.3–13.3% across all six categories, flat |
| damage concentrates in particular videos | 29 fixed span 28 videos, 40 broke span 39 videos; max 2 per video |
| question length | median 85 (fixed) vs 80 (broke) vs 75 (all) — negligible |
| compressor degrades toward a default option letter | letter distribution essentially unchanged (A 192→183, B 269→285, C 249→245, D 190→187) |

Keyword break-rates at R=0.10 (overall 10.2%) do show the archive's gist-vs-configuration split: "sequence" 20.0%, "where" 18.8%, "last" 18.5%, "how many" 16.7%, "after" 15.1% break *above* average, while "mainly about" 0/10 and "before" 0/20 survive entirely. Underpowered individually but directionally consistent with per-task (Counting −10.4, Action Reasoning −8.9, Information Synopsis only −3.7).

**Interpretation:** the flip set is real and reproducible but is *not* predictable from question text. That points at a visual mechanism, which text-side analysis cannot reach — it needs the archive's token→pixel renderer on the surviving tokens, not more lexical slicing.

Artifacts: `~/hanklin/flip_analysis.py`.

### What this does NOT answer

Accuracy was never the informative readout. Still open, and the reason this path was taken: **does VidCom2's within-frame selection rule concentrate on evidence?** FlashVID measured 1.00–1.03× uniform at every retention. VidCom2's frame-level budget provably cannot concentrate (finding 2 above), but its token-level rule is genuinely different and unmeasured. That needs the archive's concentration instrument on GT-evidence-span data (LVBench), not VideoMME.

---

## Archive

Closed chapters (2026-05-14 → 2026-07-31): [`docs/RESEARCH_ARCHIVE_2026-05_to_2026-07.md`](docs/RESEARCH_ARCHIVE_2026-05_to_2026-07.md). Provenance only — do not carry their conclusions forward without re-measuring on the trained policy.
