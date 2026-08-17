# Does ViT-side compression help a video agent find evidence in long video?

> Reading this to review it? Start with [ARCHITECTURE.md](ARCHITECTURE.md) — it explains
> the design, the invariants that silently break, and where the author is least confident.

Answer, on LVBench with Qwen3-VL-8B-Instruct: **no.** At a matched visual-token budget,
two mechanically unrelated token-selection heuristics (VidCom2 and EVS) both land *below*
simply downscaling every frame, at two different operating points. Meanwhile an oracle that
crops to the ground-truth evidence span scores **+13.23 pp using half the tokens**.

The scarce resource is not visual tokens and not temporal coverage. It is knowing where to
spend the tokens.

This repository holds the harness, the compression plugin, the diagnostics that were used
to check the harness was not lying, and a checked-in extract of every run's settings and
score so the numbers can be audited without re-running ~30 GPU-hours.

---

## Headline numbers

All arms below are **no-tool** (a single video skim, one answer), seed 0, T=0.7, on the
same LVBench question set. "Visual tok" is the number of embeddings entering the LLM after
the PatchMerger and after retention — **not** `prompt_tokens` (see [Measuring the
budget](#measuring-the-budget-do-not-use-prompt_tokens)).

### Frames × budget, uniform downscaling, no compression (n = 500 paired)

|            | ~11.6k visual | ~23.2k visual |
| ---------- | ------------: | ------------: |
| 64 frames  |    40.60%     |       —       |
| 128 frames |    44.20%     |    44.80%     |
| 256 frames |    44.20%     |    45.80%     |

| Comparison            |     Δ | 95% CI (bootstrap) |  \|z\| |
| --------------------- | ----: | ------------------ | -----: |
| 64 → 128 @ 1×         | +3.60 | [−0.40, +7.60]     |   1.65 |
| 128 → 256 @ 1×        | +0.00 | [−3.80, +3.80]     |   0.10 |
| 128 → 256 @ 2×        | +1.00 | [−2.60, +4.80]     |   0.41 |
| 1× → 2× @ 128 frames  | +0.60 | [−2.00, +3.20]     |   0.30 |
| 1× → 2× @ 256 frames  | +1.60 | [−1.40, +4.60]     |   0.95 |

Temporal density saturates at **128 frames**. Doubling the budget buys under 2 pp.

### Compression at matched final budget

| Operating point       | tok/grid-step | VidCom2 vs uniform | EVS vs uniform |
| --------------------- | ------------: | -----------------: | -------------: |
| 256 frames, r = 0.25  |            45 |             −1.61  |         −0.84  |
| 128 frames, r = 0.50  |           182 |             −1.60  |         −1.40  |

Relaxing retention 2× and per-frame density 4× moves VidCom2's penalty from −1.61 to −1.60.
No individual gap is significant (|z| 0.74–1.52); the claim supported is **"neither
selector shows any advantage,"** across four measurements, two operating points, and two
unrelated mechanisms.

### What the budget cannot buy

| Lever                                | Effect |
| ------------------------------------ | -----: |
| 64 → 128 frames                      | +3.60  |
| 128 → 256 frames                     | +0.00  |
| Doubling the budget                  | +0.6 to +1.6 |
| Best query-agnostic selector         | −0.84  |
| **Oracle allocation (GT crop)**      | **+13.23** |

---

## Repository layout

```
fast_agent/            the harness
  data.py              LVBench loading, option normalisation
  run_agent.py         the agent loop (skim -> optional crop rounds -> answer)
  run_plain.py         single-shot eval, incl. an lmms-eval-compatible prompt/scorer
  config.py            prompts and tool schemas (LongVT-verbatim + an "informed" variant)
  tools.py             crop_video
  oracle.py            forced-crop arms (GT span, and a wrong-location control)
  make_skim_proxy.py   pre-builds one small N-frame proxy mp4 per video
  serve_arm.sh         every vLLM server configuration used, arms a..p
  probe_budget.py      measures the REAL visual-token budget of a config
  diagnose_runs.py     faithfulness checks across arms
  analyze_pairs.py     paired deltas, McNemar, bootstrap CI, stratified tables
  longvt_scoring.py    lmms-eval's answer extractor, vendored verbatim
  preflight.py         pre-launch checks: server flags match the intended arm, proxies
                       present and on-timeline, plugin actually activated
  tests/               unit tests

vidcom2_vllm/          VidCom2 as an out-of-tree vLLM plugin
  retention.py         the retention mask (see notes below)
  patch.py             installs it over vllm.multimodal.evs.compute_retention_mask
  serve.py             entry point that applies the patch before vllm serve

results/
  all_runs.csv         every completed run: settings, n, duplicates, errors, accuracy
  manifests/           the run_manifest.json each run wrote at launch
```

This tree is deliberately narrow: it holds what is needed to re-run the experiments in the
report and audit their numbers, and nothing else. Exploratory notebooks, debug traces,
per-case visualisations, judge servers, and design notes for compressors that never entered
the results have all been dropped rather than shipped as clutter.

---

## Reproducing

### Requirements

- vLLM **0.19** (the plugin patches `vllm.multimodal.evs`; the import site matters)
- Qwen3-VL-8B-Instruct weights
- One 48 GB GPU for most arms (a few need more headroom — see below)
- `ffmpeg`/`ffprobe`, `decord`, `transformers`

### 1. Build skim proxies

LVBench is 103 videos / 62 GB, median one hour. The agent re-sends the same video every
tool round, which measured ~39 s per request — over 60 hours for one arm. A proxy is
decoded once per *video* instead of once per *request*.

```bash
python -m longvt_compression.fast_agent.make_skim_proxy \
    --frames 256 --max-side 1280 --workers 10 \
    --out-dir /path/to/skim_proxies_hires_256
```

**The timeline must survive, not just the pixels.** Qwen3-VL derives each frame's
`<t seconds>` marker as `frame_index / fps` read off the container. A proxy written at, say,
8 fps would label a 3,664-second video as 32 seconds and every crop the model asks for
would be wrong by two orders of magnitude — silently. The proxy is therefore written at
`fps = n_frames / duration`, and `--verify` re-opens each file to confirm the round trip.

Each frame count needs its **own** `--out-dir`: the filename is `{video_id}_n{frames}.mp4`
and carries no resolution, so mixing resolutions in one directory silently serves the wrong
pixels.

### 2. Start a server

Every configuration used in the paper-style tables is a named arm in `serve_arm.sh`:

```bash
GPUUTIL=0.80 ./fast_agent/serve_arm.sh m            # 128f, uniform, 1x budget
MAXLEN=32768 BATCHTOK=26624 GPUUTIL=0.70 \
  ./fast_agent/serve_arm.sh n                       # 128f, uniform, 2x budget
GPUUTIL=0.70 ./fast_agent/serve_arm.sh o            # 128f, VidCom2 r=0.50
GPUUTIL=0.70 ./fast_agent/serve_arm.sh p            # 128f, EVS     r=0.50
```

`FA_PRUNE_METHOD=vidcom2` installs the plugin; anything else leaves vLLM's stock EVS in
place, which is how the EVS arms are served with zero code difference.

### 3. Verify the budget *before* spending GPU-hours

```bash
python -m longvt_compression.fast_agent.probe_budget --frames 128 --vidcap 2 --rate 0.5
```

Prints, per video: decoded frames, `grid_thw`, proxy resolution, processor resolution,
visual tokens before and after compression, the realised retention ratio, and tokens per
grid-step. Two arms are only comparable if their **post** columns match.

### 4. Run an arm

```bash
python -m longvt_compression.fast_agent.run_agent \
    --dataset lvbench --skim-mode video --frames 128 --tools \
    --proxy-dir /path/to/skim_proxies_hires_128 --require-proxy \
    --num 500 --seed 0 --tag my_arm \
    --base-url http://localhost:8044/v1 --workers 8
```

`--tools` with no values is the **no-tool** arm and gets the no-tool prompt. Confirm with
`--print-prompt`, which prints the content blocks and the tool schema (`null` when there
are no tools). A no-tool arm that receives the tool prompt is not a vanilla baseline.

`--num N --seed S` is deterministic, so every arm run with the same pair covers the same
question IDs and can be paired.

### 5. Check the run before believing it

```bash
python -m longvt_compression.fast_agent.diagnose_runs
python -m longvt_compression.fast_agent.analyze_pairs
```

`diagnose_runs` checks, in order of how badly a failure would corrupt the result: proxy
timeline drift, cross-arm question/gold/evidence identity and duplicates, measured token
budgets, transport errors and termination, and that every pinned hyperparameter is
identical across arms.

---

## Fixed hyperparameters

Identical across every arm unless explicitly noted:

| | |
| --- | --- |
| model | `Qwen/Qwen3-VL-8B-Instruct` |
| serving | vLLM 0.19, `--enable-auto-tool-choice --tool-call-parser hermes` |
| sampling | T = 0.7, top-p 0.8, top-k 20, presence/frequency 0, repetition 1.0 |
| max tokens | 1024 per turn, 32 for the forced finalizer |
| tool rounds | ≤ 5 |
| skim modality | **video** (vLLM prunes video only; images pass through untouched) |
| crop tool | ≤ 128 frames, fps 1, returned as **images** so they are never pruned |
| prompt | LongVT-verbatim (`--prompt-style longvt`) |
| seed | 0 |
| scoring | rule-based `<answer>` extraction, no LLM judge |

**Deliberate deviations from LongVT**, recorded rather than hidden: LongVT feeds the skim
as images and we feed video (VidCom2 requires video); LongVT evaluates at temperature 0
with an unbounded tool loop; LongVT's scorer ends in an LLM judge (Qwen3-235B-A22B) that is
not reproducible here.

---

## Measuring the budget: do not use `prompt_tokens`

Qwen3-VL's processor interleaves a real `<t seconds>` marker before **every grid step**, so
the text contribution grows with frame count — 128 markers at 256 frames, 32 at 64 frames,
a difference of roughly 1,000 tokens. That is the same order as the effects being measured.

Two arms measured on `prompt_tokens` looked 8.5 % apart in budget (12,106 vs 13,136) when
their actual visual budgets were **identical** (11,648 vs 11,648). Within a fixed frame
count the tax is constant and `prompt_tokens` differences are safe; across frame counts
they are not. `probe_budget.py` reports the real number.

---

## Things that silently broke a run

Each of these produced a plausible-looking result rather than an error.

**Qwen3-VL caps the whole video, not the frame.** `video_preprocessor_config.json` sets
`size.longest_edge = 25,165,824`; divide by `temporal_patch_size` and by 1,024 px per output
token and you get a hard ceiling of **12,288 visual tokens regardless of frame count**.
Adding frames does not add tokens, it thins them. Override with `size.longest_edge` in
`--mm-processor-kwargs`.

**`max_pixels` is inert for video.** `Qwen3VLVideoProcessor` reports it as an unrecognized
kwarg and the measured budget is byte-identical with and without it. It is an image-processor
knob. The only pixel control for video is the cap above. It is still passed in `serve_arm.sh`
so the launch strings match the earlier arms verbatim.

**`do_sample_frames` must be False when replicating the processor offline.** vLLM presamples
via `--media-io-kwargs num_frames` and then calls the processor with `do_sample_frames=False`.
Omit it and the processor resamples to its own default, reporting `grid_t = 11` instead of
`N/2`, and every offline budget estimate is wrong.

**The encoder cache is not separately configurable.** `config/scheduler.py` hardwires
`encoder_cache_size = max_num_batched_tokens`, and its auto-grow computes the largest
multimodal item from the *default* video size — so it under-estimates whenever `size` is
overridden. Unpruned high-budget arms return HTTP 400 until `BATCHTOK` is raised by hand.
Pruned arms slip under it because the cache holds post-prune embeddings.

**Results are written in completion order.** Under N workers, line order is not aligned
across runs. Pairing positionally once produced 351/289 flips where the truth was 157/95.
Pair on `question_id`.

**`run_agent` resumes by appending, not de-duplicating.** An interrupted run that is
restarted leaves duplicate `question_id`s — 777 rows for 500 questions in one case, 3 of
them with disagreeing predictions under T=0.7. `diagnose_runs` counts duplicates; a run with
any should be re-run clean.

**A dead arm looks like a bad arm.** One run was launched against a port with no server;
1,362 `APIConnectionError`s were written as `pred=None, correct=False` and it scored 4.71 %,
far below the 24.6 % chance floor. Anything below chance is a transport failure until proven
otherwise. That run is in `all_runs.csv` marked as such and must not be cited.

**Proxy resolution is fixed at build time.** The original proxies were 448 px wide, capping
every arm at ~112 tok/frame. Raising the server's pixel budget does nothing — the pixels are
already gone.

---

## Why the compressors cannot help here, mechanically

`vidcom2_vllm/retention.py` computes each frame's share of the budget as

```python
probs  = F.softmax((frame_importance - frame_importance.max()) / 0.01, dim=0)
scales = base * (1.0 + probs - probs.mean())          # base = 1 - q
```

A softmax at temperature 0.01 is essentially one-hot, so exactly one frame gets roughly
double share and every other frame gets the flat base: **48.4 % for one frame, 23.4 % for
all the rest** at T=16. The temporal allocation is uniform except for a single frame. All of
VidCom2's selectivity is *within* a frame — it keeps the tokens furthest from the whole-video
and per-frame centroids.

Two consequences. First, a compressor with a uniform temporal budget cannot concentrate
bandwidth on the evidence, by construction. Second, the whole-video centroid is not a
meaningful reference at LVBench scale: at 256 frames over an hour, consecutive samples are
14 s apart and are different scenes, not near-duplicates, so "far from the centroid" stops
meaning "non-redundant."

This is a statement about fit, not quality. The same plugin reproduces VidCom2's published
VideoMME-long numbers within 0.35 pp.

---

## Auditing the numbers without a GPU

`results/all_runs.csv` has one row per completed run with `rows`, `unique_q`, `dupes`,
`errors`, `no_answer`, `accuracy`, the median `prompt_tok`, and the settings that define the
arm. `results/manifests/` has the manifest each run wrote at launch, including sampling
parameters, proxy directory, prompt style, and the server port.

Any accuracy in the report should be checkable against that CSV, and any arm with non-zero
`dupes` or `errors` should be treated as suspect.
