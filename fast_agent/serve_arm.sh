#!/usr/bin/env bash
# Servers for the two tool arms. Identical except for the pruning block, so any
# accuracy gap is attributable to compression and not to a serving difference.
#
#   ./serve_arm.sh a   # arm A -- no pruning,      GPU 4, :8030
#   ./serve_arm.sh b   # arm B -- VidCom2 q=0.75,  GPU 7, :8031
#
# Sizing (both arms):
#   64-frame image skim  64 x 47                    = 3,008 tok
#   one crop_video call  128 x 47                   = 6,016 tok
#   MAX_ROUNDS=5 crops   5 x 6,016                  = 30,080 tok
#   -> max-model-len 40960 and image limit 704+64 leave headroom for the worst case
#      instead of failing the long trajectories, which are exactly the ones that
#      carry the localization signal.
set -euo pipefail
PY=/local1/cfyang/miniconda3/envs/vllm/bin
MODEL=/local1/cfyang/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
LOG=/local1/cfyang/hanklin/outputs/lvbench_agent
mkdir -p "$LOG"

COMMON=(
  --served-model-name qwen3vl
  # 40960 fits every arm whose video budget is the stock 12,288-token cap. Arm l ships
  # 49,152 visual tokens and MUST raise it; KV is 144 KiB/token, so the cost is real.
  # --max-model-len and --max-num-batched-tokens moved OUT of COMMON: they are a
  # PROPERTY OF THE ARM (they scale with the visual-token count), and defaulting them
  # here meant an arm could launch with sizing that silently did not fit its own video.
  # They are set per-arm in the case block below and assembled into SIZING afterwards.
  # == encoder cache size, and it is a HARD per-item limit: vLLM 400s with "video item
  # with length N exceeds the pre-allocated encoder cache size" when one multimodal item
  # is bigger. The pruned arms slip under it because the cache holds POST-prune
  # embeddings (arm i's 4x-cap video is 49k pre-prune but 12k after), so only the
  # UNPRUNED high-budget arm (l) has to raise it. 16384 fits a 128-image crop (6,016).

  # 0.88 assumes an empty card. Other users' processes come and go on this box, and vLLM
  # refuses to start if the *free* fraction is below this -- so it is overridable. It only
  # sizes the KV cache (throughput), never the arithmetic, but keep it IDENTICAL across the
  # two arms of a comparison so the pair differs by one thing.
  --gpu-memory-utilization "${GPUUTIL:-0.88}"
  --allowed-local-media-path /local1/cfyang   # file:// video for the arm-B skim
  # THE FIX FOR THE RECURRING SERVER WEDGE (2026-08-13). vLLM caches processed
  # multimodal items in the API process and ships only a hash to EngineCore, which looks
  # it up in a bounded receiver cache. When the two desync -- which this workload provokes
  # hard: one proxy shared by ~15 questions, resent every tool round, plus 128-image crops
  # at 8-way concurrency -- EngineCore hits
  #     AssertionError: Expected a cached item for mm_hash=...
  # and DIES, while the API process stays up answering /v1/models with 200. That is the
  # "healthy but serving nothing" state the watchdog kept restarting. Setting the cache to
  # 0 removes the sender/receiver split entirely: full MM data travels with every request.
  --mm-processor-cache-gb 0
  # LongVT's own eval (examples/eval/single_inference.py) drives the loop with
  # tool_choice="auto" and role:"tool" replies, which vLLM only serves with these two.
  --enable-auto-tool-choice
  --tool-call-parser hermes
)

# Per-frame-count cap = (frames/2) * 880 * 2048, i.e. exactly the native-resolution token
# count for that many frames. All three still render at 704x1280 (verified), so resolution
# is identical across arms -- but the cap is now the SMALLEST value that achieves it.
#
# It cannot be one big saturating number: vLLM sizes its startup PROFILING dummy video from
# the processor ceiling (longest_edge/2/1024 tokens) and ignores --media-io-kwargs
# num_frames, so a 230,686,720 cap profiles a 112,640-token video for every arm. That
# overruns max-num-batched-tokens, and on the PRUNED arms the retained-count then disagrees
# with the placeholder count and vLLM dies in masked_scatter. Confirmed not to be our
# plugin: vLLM's own EVS selector fails identically at that cap.
# The cap a grid arm needs depends on whether it PRUNES, because vLLM merges the visual
# embeddings by two different routes:
#
#   PRUNED arms take _postprocess_video_embeds_evs, which recomputes the retained count
#   from the real grid. A tight cap -- exactly (frames/2)*880*2048, the native-resolution
#   token count -- is required there, because vLLM sizes its startup PROFILING dummy video
#   from the processor ceiling (cap/2/1024) and ignores --media-io-kwargs num_frames. A
#   generous cap makes that dummy overrun max-num-batched-tokens and the pruned path dies
#   in masked_scatter before serving anything.  (Verified: v64_50 and v128 run clean here.)
#
#   UNPRUNED arms take _compute_deepstack_embeds, and there a tight cap fails at INFERENCE
#   time -- the same masked_scatter error, on every full-resolution video, while the 16
#   smaller-source videos still succeed. A generous cap fixes it.  (Verified: u32 fails at
#   ceiling 14,080 and at 15,488, and is clean at 112,640.)
#
# Both routes end up at the SAME geometry -- 704x1280, 880 tokens per grid step -- which
# preflight_arm.py re-measures over all 103 videos for every arm. The cap is therefore a
# serving parameter here, not an experimental one; the resolution it produces is checked
# independently and is identical across the grid.
CAP_GENEROUS=230686720
cap_for () { if [ -z "${RATE:-}" ]; then echo $CAP_GENEROUS; else echo $(( ($1/2) * 880 * 2048 )); fi; }
case "${1:?usage: serve_arm.sh a|b|c|d}" in
  a) GPU=4; PORT=8030
     # run1 + run2: 64-frame video skim, NO pruning. num_frames is what makes the
     # budget comparison exact -- 64 frames pair to grid_t 32, and run3's 256 frames
     # pair to grid_t 128 which q=0.75 cuts back to 32 frames' worth. Same visual
     # tokens, 4x the temporal coverage: that is the only difference between them.
     EXTRA=(--media-io-kwargs '{"video":{"num_frames":64}}')
     export FA_PRUNE_METHOD=evs ;;
  c) GPU=7; PORT=8031
     # run4 / ctrl: identical to arm a (64-frame video, no pruning) on a second GPU, so
     # the oracle ceiling and its wrong-location control can run beside run2 rather than
     # queue behind it.
     EXTRA=(--media-io-kwargs '{"video":{"num_frames":64}}')
     export FA_PRUNE_METHOD=evs ;;
  d) GPU=7; PORT=8031
     # diagnostic: 256-frame video skim, NO pruning -- the uncompressed twin of run3,
     # used to separate "compression hurt" from "more frames at lower per-frame quality".
     EXTRA=(--media-io-kwargs '{"video":{"num_frames":256}}')
     export FA_PRUNE_METHOD=evs ;;
  b) GPU=4; PORT=8032
     # 256 raw frames -> grid_t 128 -> 128 x 49 = 6,272 visual tokens, x0.25 kept.
     # num_frames is NOT optional: without it vLLM resamples to its own default and
     # a 256-frame request silently becomes the same token count as a 64-frame one.
     EXTRA=(--media-io-kwargs '{"video":{"num_frames":256}}' --video-pruning-rate 0.75)
     export FA_PRUNE_METHOD=vidcom2 ;;
  # --- VidCom2 AT ITS OWN OPERATING POINT (added 2026-08-14) ---------------------
  # Arms a-d run at max_pixels 50176 (224^2), inherited from LongVT's run_eval.sh --
  # LongVT needs it that low because it ships up to 768 frames as images. At 224^2 a
  # 16:9 frame is ~91 tokens, so VidCom2 at r=0.25 leaves ~23 tokens for a whole
  # frame. VidCom2's own validated setting is 32 frames at ~720 tokens/frame, leaving
  # ~180 after pruning -- 8x more. These two arms reproduce THAT point on LVBench, so
  # "the compressor is broken" can be told apart from "we ran it 8x past its regime".
  #   e = 32f high-res, NO pruning  (the control)
  #   f = 32f high-res, VidCom2 r=0.25
  # Needs the high-res proxies: make_skim_proxy --frames 32 --max-side 1280
  #   --out-dir .../skim_proxies_hires   (448-wide proxies cap you at ~112 tok/frame,
  #   so raising max_pixels alone does nothing -- the pixels are already gone).
  e) GPU=3; PORT=8036
     EXTRA=(--media-io-kwargs '{"video":{"num_frames":32}}')
     MMKW='{"max_pixels":786432,"min_pixels":3136}'
     export FA_PRUNE_METHOD=evs ;;
  f) GPU=3; PORT=8037
     EXTRA=(--media-io-kwargs '{"video":{"num_frames":32}}' --video-pruning-rate 0.75)
     MMKW='{"max_pixels":786432,"min_pixels":3136}'
     export FA_PRUNE_METHOD=vidcom2 ;;
  # --- MATCHED-BUDGET LADDER (added 2026-08-14) ----------------------------------
  # MEASURED 2026-08-14, and it invalidates the naive reading of `max_pixels`:
  # Qwen3-VL caps the WHOLE video, not each frame. video_preprocessor_config.json has
  # size.longest_edge = 25,165,824 px; divide by temporal_patch_size 2 and by 1024 px
  # per output token -> a hard ceiling of 12,288 visual tokens NO MATTER THE FRAMES.
  #     32f -> 16 grid steps -> 768 tok each (measured 737.6)
  #     64f -> 32 grid steps -> 384 tok each (measured 378.1)
  # So --mm-processor-kwargs max_pixels only binds when it is BELOW that share; at 32f
  # 786,432 px == 768 tok, exactly the cap's share, which is why arm e looked like it
  # was honouring max_pixels. Adding frames does not add tokens, it thins them.
  #
  # To hold the FINAL budget equal while varying temporal coverage, arm i raises the
  # video cap 4x so that pruning brings it back to arm g's number:
  #   g = 64f,  no pruning,      default cap -> 32 grid x 384          = 12,288 tok
  #   i = 256f, VidCom2 r=0.25,  4x cap      -> 128 grid x 384 x 0.25  = 12,288 tok
  # g and i match on BOTH total budget and per-frame resolution (384 tok/grid step).
  # The only difference is 4x the temporal coverage, paid for by pruning.
  #   h = 64f, VidCom2 r=0.25, default cap -> 3,072 tok. Same frames, quarter budget.
  #   j = 256f, NO pruning, default cap    -> 12,288 tok at 96 tok/grid step. The
  #       PUREST compressor test: identical frames and identical final budget to arm i,
  #       but tokens chosen by uniform downscaling instead of by VidCom2. Any gap
  #       between i and j is VidCom2's token SELECTION and nothing else.
  # Needs proxies from `make_skim_proxy --max-side 1280` at the matching --frames, each
  # in its OWN --out-dir (filenames carry frames but not resolution).
  g) GPU=3; PORT=8038
     EXTRA=(--media-io-kwargs '{"video":{"num_frames":64}}')
     MMKW='{"max_pixels":786432,"min_pixels":3136}'
     export FA_PRUNE_METHOD=evs ;;
  h) GPU=3; PORT=8039
     EXTRA=(--media-io-kwargs '{"video":{"num_frames":64}}' --video-pruning-rate 0.75)
     MMKW='{"max_pixels":786432,"min_pixels":3136}'
     export FA_PRUNE_METHOD=vidcom2 ;;
  i) GPU=3; PORT=8040
     # 4x video cap: 25,165,824 * 4 = 100,663,296 -> 49,152 tok pre-prune, 12,288 after.
     # 49k tokens go through the ViT but only 12k reach the LM, so max-model-len 40960
     # still holds and vLLM grows the encoder cache to fit (encoder_cache_manager.py:312).
     EXTRA=(--media-io-kwargs '{"video":{"num_frames":256}}' --video-pruning-rate 0.75)
     MMKW='{"max_pixels":786432,"min_pixels":3136,"size":{"longest_edge":100663296,"shortest_edge":4096}}'
     export FA_PRUNE_METHOD=vidcom2 ;;
  j) GPU=3; PORT=8041
     EXTRA=(--media-io-kwargs '{"video":{"num_frames":256}}')
     MMKW='{"max_pixels":786432,"min_pixels":3136}'
     export FA_PRUNE_METHOD=evs ;;
  # --- SWAP THE COMPRESSOR, HOLD EVERYTHING ELSE (added 2026-08-14) ---------------
  # arm k is arm i with ONE character changed: the retention heuristic. The plugin
  # no-ops unless FA_PRUNE_METHOD=vidcom2 ("leaving stock EVS in place", patch.py:50),
  # so `evs` + --video-pruning-rate gives vLLM's NATIVE EVS at the same retention.
  # The two heuristics answer different questions about a token:
  #     VidCom2 -- how unlike the video/frame centroid is it?  (uniqueness)
  #     EVS     -- how much did it change from the previous frame?  (temporal change)
  # i vs k vs j therefore tests a CLASS of query-agnostic selectors against uniform
  # downscaling, not one method. Same frames, same final budget, same prompt.
  k) GPU=3; PORT=8042
     EXTRA=(--media-io-kwargs '{"video":{"num_frames":256}}' --video-pruning-rate 0.75)
     MMKW='{"max_pixels":786432,"min_pixels":3136,"size":{"longest_edge":100663296,"shortest_edge":4096}}'
     export FA_PRUNE_METHOD=evs ;;
  # --- THE UNCOMPRESSED TWIN OF ARM i (added 2026-08-14) -------------------------
  # 256 frames at arm g's per-frame resolution, NO pruning: 128 grid x 384 = 49,152
  # visual tokens. This does NOT fit the standard 40,960 context -- it is the one
  # configuration compression exists to make possible -- so it needs MAXLEN raised and
  # is meant for a PAIRED SUBSET, not all 1,549. It answers the question arm i cannot:
  # how much accuracy is there in 256 full-resolution frames before anything is thrown
  # away? Without it, "compression is lossless here" and "there was nothing to lose"
  # are indistinguishable.
  #   run as:  MAXLEN=57344 GPUUTIL=0.66 ./serve_arm.sh l      (workers 2-3; ~6.75 GB
  #   of KV per sequence at 49k tokens, plus ~7 GB of ViT activations)
  # MEASURED 2026-08-16: the 4x (49,152-token) version DOES NOT FIT this box. The
  # encoder cache is not separately configurable -- config/scheduler.py:235 hardwires
  # encoder_cache_size = max_num_batched_tokens -- so BATCHTOK must exceed the largest
  # mm item, and a ~49k prefill batch's profiling activation then starves the KV cache;
  # raising GPUUTIL to compensate leaves nothing for the transient ViT forward, which
  # OOMs at request time even though startup succeeded. Budget on a 47.3 GiB card with
  # another user holding 5.9:
  #     weights 16.4 + profiling ~12 + KV 6.9 + ViT ~8 + neighbour 5.9 = ~49 GiB.
  # VIDCAP=2 (24,576 tokens, 192 tok/grid-step) fits and answers the same question --
  # it is 2x arm j's budget at the same 256 frames, so it bounds the headroom.
  l) GPU=3; PORT=8043
     _mult=${VIDCAP:-2}
     EXTRA=(--media-io-kwargs '{"video":{"num_frames":256}}')
     MMKW="{\"max_pixels\":786432,\"min_pixels\":3136,\"size\":{\"longest_edge\":$((25165824*_mult)),\"shortest_edge\":4096}}"
     export FA_PRUNE_METHOD=evs ;;
  # --- 2x2 FRAMES x BUDGET, PLUS A MILDER COMPRESSION POINT (added 2026-08-16) ------
  # MEASURED with `probe_budget.py` -- these are ACTUAL visual embeddings entering the
  # LLM on a 720p source, not prompt_tokens (which carries a timestamp tax that scales
  # with frame count: 128 markers at 256f vs 32 at 64f, ~1,000 tokens of difference).
  #
  #                     ~11.6k visual            ~23.2k visual
  #     128 frames      m  cap1x -> 11,520       n  cap2x -> 23,296
  #     256 frames      j  cap1x -> 11,648       l  cap2x -> 23,040
  #
  # The milder compression triple, all 128 frames, all ~11.6k FINAL visual tokens:
  #     m  uniform downscale            -> 11,520
  #     o  VidCom2 r=0.50 from cap2x    -> 11,648
  #     p  EVS     r=0.50 from cap2x    -> 11,648
  # i/k tested r=0.25 at 256f (45 tok/grid-step); this tests r=0.50 at 128f
  # (182 tok/grid-step) -- 4x less aggressive, to separate "selection is a bad idea"
  # from "0.25 at 256 frames was too aggressive".
  #
  # NOTE: `max_pixels` in MMKW is INERT for video -- Qwen3VLVideoProcessor reports it as
  # an unrecognized kwarg and the measured budget is identical with and without it. It is
  # kept only so these launch strings match the earlier arms verbatim. The only pixel
  # control for video is size.longest_edge.
  m) GPU=3; PORT=8044
     EXTRA=(--media-io-kwargs '{"video":{"num_frames":128}}')
     MMKW='{"max_pixels":786432,"min_pixels":3136}'
     export FA_PRUNE_METHOD=evs ;;
  n) GPU=3; PORT=8045
     # 23,296 visual tokens reach the LM unpruned -> needs MAXLEN and BATCHTOK raised:
     #   MAXLEN=32768 BATCHTOK=26624 GPUUTIL=0.70 ./serve_arm.sh n
     EXTRA=(--media-io-kwargs '{"video":{"num_frames":128}}')
     MMKW='{"max_pixels":786432,"min_pixels":3136,"size":{"longest_edge":50331648,"shortest_edge":4096}}'
     export FA_PRUNE_METHOD=evs ;;
  o) GPU=3; PORT=8046
     EXTRA=(--media-io-kwargs '{"video":{"num_frames":128}}' --video-pruning-rate 0.50)
     MMKW='{"max_pixels":786432,"min_pixels":3136,"size":{"longest_edge":50331648,"shortest_edge":4096}}'
     export FA_PRUNE_METHOD=vidcom2 ;;
  p) GPU=3; PORT=8047
     EXTRA=(--media-io-kwargs '{"video":{"num_frames":128}}' --video-pruning-rate 0.50)
     MMKW='{"max_pixels":786432,"min_pixels":3136,"size":{"longest_edge":50331648,"shortest_edge":4096}}'
     export FA_PRUNE_METHOD=evs ;;
  # ---- THE FRAMES x RETENTION GRID (frozen 2026-08-19) --------------------------
  # Six primary arms + one iso-budget enhancement. TWO variables only: frame count and
  # VidCom2 retention. Everything else -- including spatial resolution -- is identical.
  #
  #                    no compression        VidCom2 25%
  #     32 frames           u32                  v32
  #     64 frames           u64                  v64
  #    128 frames          u128                 v128
  #     64 frames                            v64_50  (50%, the iso-budget midpoint)
  #
  # WHY ONE CAP FOR EVERY ARM
  #   Qwen3-VL's video processor spends `size.longest_edge` across the WHOLE clip, so at
  #   the stock cap more frames silently means blurrier frames -- which confounds "more
  #   observations" with "less spatial fidelity" and is exactly the flaw in the earlier
  #   iso-token grid. 230,686,720 is past the point where the LVBench sources become the
  #   binding constraint (measured: raising it further does not change grid_thw), so every
  #   arm renders at the source's native resolution -- 704x1280, 880 tokens per grid step,
  #   for the 86 of 103 videos that are 1280x720. Resolution is therefore fixed BY
  #   SATURATION rather than by a per-arm cap that has to track the frame count.
  #
  # GRID STEPS, NOT FRAMES. temporal_patch_size=2, so N frames -> N/2 grid steps, and
  # VidCom2 allocates over the grid steps:
  #     32f -> 16 steps -> 14,080 visual     64f -> 32 -> 28,160     128f -> 64 -> 56,320
  #
  # --video-pruning-rate is the DROP fraction: retention 0.25 -> rate 0.75.
  #
  # SIZING. max-num-batched-tokens is the encoder-cache size and is a hard per-item
  # limit, so it must exceed the arm's PRE-prune visual count even when pruning is on
  # (the compressed arms still push the full clip through the vision tower).
  u32)    GPU=3; PORT=8061; FRAMES=32;  RATE=""     ; MAXLEN=${MAXLEN:-16384}; BATCHTOK=${BATCHTOK:-16384} ;;
  v32)    GPU=3; PORT=8062; FRAMES=32;  RATE=0.75   ; MAXLEN=${MAXLEN:-16384}; BATCHTOK=${BATCHTOK:-16384} ;;
  u64)    GPU=3; PORT=8063; FRAMES=64;  RATE=""     ; MAXLEN=${MAXLEN:-32768}; BATCHTOK=${BATCHTOK:-32768} ;;
  v64)    GPU=3; PORT=8064; FRAMES=64;  RATE=0.75   ; MAXLEN=${MAXLEN:-32768}; BATCHTOK=${BATCHTOK:-30720} ;;
  v64_50) GPU=3; PORT=8065; FRAMES=64;  RATE=0.50   ; MAXLEN=${MAXLEN:-32768}; BATCHTOK=${BATCHTOK:-30720} ;;
  u128)   GPU=3; PORT=8066; FRAMES=128; RATE=""     ; MAXLEN=${MAXLEN:-63488}; BATCHTOK=${BATCHTOK:-63488} ;;
  v128)   GPU=3; PORT=8067; FRAMES=128; RATE=0.75   ; MAXLEN=${MAXLEN:-61440}; BATCHTOK=${BATCHTOK:-57344} ;;
  *) echo "usage: serve_arm.sh a..p | u32|v32|u64|v64|v64_50|u128|v128" >&2; exit 1 ;;
esac

# The grid arms declare FRAMES/RATE instead of EXTRA/MMKW, so assemble them here -- one
# place, so the seven arms cannot drift apart in anything but frames and retention.
if [ -n "${FRAMES:-}" ]; then
  EXTRA=(--media-io-kwargs "{\"video\":{\"num_frames\":$FRAMES}}")
  [ -n "${RATE:-}" ] && EXTRA+=(--video-pruning-rate "$RATE")
  MMKW="{\"size\":{\"longest_edge\":$(cap_for $FRAMES),\"shortest_edge\":4096}}"
  MMLIMIT='{"image":0,"video":1}'
  # PRUNE_FORCE lets a diagnostic swap in vLLM's built-in EVS selector to tell a plugin
  # bug apart from a vLLM one, without touching the arm definition.
  export FA_PRUNE_METHOD=${PRUNE_FORCE:-vidcom2}
fi

# Sizing is per-arm (see the grid block); the legacy arms keep the old defaults.
# --limit-mm-per-prompt belongs here too: vLLM sizes its startup PROFILING run from these
# limits, so a no-tool arm left at the tool arms' 768-image worst case profiles for 768
# dummy images it will never receive -- which OOMs a native-resolution arm before it
# serves a single request. 768/2 is the five-crop-round ceiling the TOOL arms genuinely
# need; the grid arms send one video and no images, ever.
SIZING=(--max-model-len "${MAXLEN:-40960}"
        --max-num-batched-tokens "${BATCHTOK:-16384}"
        --limit-mm-per-prompt "${MMLIMIT:-{\"image\":768,\"video\":2\}}")

# Per-frame pixel budget. Default is LongVT's 224^2; arms e/f raise it (see above).
MMKW=${MMKW:-'{"max_pixels":50176,"min_pixels":3136}'}
MMKW=${MAXPIX_FORCE:-$MMKW}

# The GPU/port pairs above are the ones the banked runs used, so they are the default.
# They are overridable because which card is free changes hour to hour on this box, and
# an arm should not be blocked on a specific card when the *config* is what defines it:
#   GPU=7 PORT=8031 ./serve_arm.sh b
GPU=${GPU_FORCE:-$GPU}
PORT=${PORT_FORCE:-$PORT}

echo "arm=$1 gpu=$GPU port=$PORT prune=$FA_PRUNE_METHOD mm=$MMKW"
# GPU may be a comma list for tensor parallelism -- u128 needs it. Its profiling dummy is
# 56,320+ visual tokens, and the vision-tower forward on that peaks around 17 GiB; with
# 16.4 GiB of weights there is not enough left on one 48 GiB card for the 8.7 GiB of KV a
# 57k-token context needs. vLLM subtracts the measured activation peak from the
# gpu-memory-utilization budget, so raising util does not help -- only splitting does.
TP=$(awk -F, '{print NF}' <<< "$GPU")
[ "$TP" -gt 1 ] && SIZING+=(--tensor-parallel-size "$TP")
CUDA_VISIBLE_DEVICES=$GPU nohup "$PY/vllm" serve "$MODEL" \
  --port "$PORT" "${COMMON[@]}" "${SIZING[@]}" --mm-processor-kwargs "$MMKW" "${EXTRA[@]}" \
  >> "$LOG/server_arm_$1.log" 2>&1 &
echo "pid $! -> $LOG/server_arm_$1.log"
