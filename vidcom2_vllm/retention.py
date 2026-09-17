"""VidCom2 token selection, expressed as a vLLM retention mask.

Drop-in replacement for `vllm.multimodal.evs.compute_retention_mask` with an
identical signature, so it plugs into the hook vLLM 0.19 already calls for
Qwen3-VL (`qwen3_vl.py::_postprocess_video_embeds_evs`).

Algorithm is lifted from the authors' released reference implementation
(`/local1/cfyang/VidCom2/token_compressor/vidcom2/vidcom2.py`), NOT from the
paper -- the two disagree, and vLLM upstream also reimplemented from the code
because that is what produced the published numbers. Specifically the code adds,
and the paper never mentions: a low-variance channel subset, a sum of 5 Gaussian
kernels instead of cosine similarity, and centres computed on L2-normalised
tokens.

THE ONE THING THAT MUST NOT DRIFT
---------------------------------
vLLM sizes the `<|video_pad|>` placeholder run in the *processor*, before the
model ever runs, via `compute_retained_tokens_count(tokens_per_frame, num_frames, q)`.
The mask produced here runs *later*, in the model forward. If the number of True
values disagrees with that prediction by even one, the embeddings no longer match
the placeholders and generation is corrupted.

The reference implementation has no such reconciliation -- it rounds a per-frame
budget independently per frame, so its total drifts off any target. This module
therefore keeps EVS's own count function as the single source of truth and
apportions that exact total across frames (largest-remainder), preserving
VidCom2's per-frame allocation *shape* while guaranteeing the sum.
"""

import json
import os

import torch
import torch.nn.functional as F

# Reference-implementation constants (vidcom2.py).
CHANNEL_KEEP_RATIO = 0.5           # select_low_var_channels(ratio=0.5)
GAUSSIAN_ALPHAS = [2.0 ** k for k in range(-3, 2)]   # alphas = [2**k for k in range(-3,2)]
SOFTMAX_TEMP = 0.01                # compute_scales(temp=0.01)

_ANNOUNCED = False                 # set on first mask computation (see below)


def _apportion(ideal: torch.Tensor, target: int, lo: int, hi: int) -> torch.Tensor:
    """Integer per-frame budgets summing to EXACTLY `target`, each within [lo, hi].

    Largest-remainder (Hamilton) apportionment. Every frame gets floor(ideal), then the
    remaining units are handed out ONE AT A TIME in descending-remainder order, skipping
    frames already at their cap and wrapping to further rounds if caps block the first
    pass. Deterministic: `stable=True` breaks ties by frame index.

    BUG FIXED 2026-08-17 -- the previous implementation "water-filled", giving each
    frame in remainder order its ENTIRE headroom rather than one unit. It satisfied the
    sum, so every budget check passed, but the distribution was badly wrong:

        ideal = [1.9, 1.8, 1.7], target = 5  ->  gave [3, 1, 1], correct is [2, 2, 1]

    On a real 256-frame shape (T=128, tpf=220, target=7,040) one arbitrary frame -- the
    one with the largest fractional remainder, not the one VidCom2 judged important --
    received 127 tokens against an ideal of 54.6, and 73 of 128 frames differed from
    correct allocation. Since `sum == target` held throughout, `probe_budget.py` and
    `diagnose_runs.py` both reported the arms as budget-matched. They verify the TOTAL;
    this bug was in the SPLIT.
    """
    T = ideal.numel()
    lo = min(lo, hi)
    target = int(max(T * lo, min(T * hi, target)))

    ideal = ideal.clamp(min=float(lo), max=float(hi))
    k = ideal.floor().long().clamp(min=lo, max=hi)
    diff = target - int(k.sum())
    if diff == 0:
        return k

    remainder = ideal - k.to(ideal.dtype)
    up = diff > 0
    # Descending remainder when adding, ascending when removing: the frames that lost
    # the most to flooring get the first extra unit.
    order = torch.argsort(remainder, descending=up, stable=True).tolist()
    step = 1 if up else -1
    diff = abs(diff)

    while diff:
        moved = 0
        for i in order:
            if not diff:
                break
            if (up and k[i] >= hi) or (not up and k[i] <= lo):
                continue          # this frame is capped; skip it, do not overfill
            k[i] += step
            diff -= 1
            moved += 1
        if not moved:             # every frame is capped -- target is unreachable
            break
    return k


_MAIN_WIDTH = None                 # resolved once, from the live vLLM config
_WIDTH_WARNED = False


def main_embed_width(total_width: int) -> int:
    """Width of the FINAL visual embedding inside vLLM's concatenated video tensor.

    Qwen3-VL's vision tower returns `cat([merger_out] + deepstack_features, dim=1)`
    (`qwen3_vl.py`, Qwen3_VisionModel.forward), i.e. `hidden_size * (1 + n_deepstack)`
    -- 4,096 * 4 = 16,384 for the 8B model. That whole tensor is what reaches
    `compute_retention_mask`.

    BUG FIXED 2026-08-17 -- this module used to score all 16,384 channels. The released
    VidCom2 wrapper scores only the final `hidden_size` block and then applies the
    resulting token indices to the DeepStack blocks separately. Scoring the concatenation
    makes `select_low_var_channels` pick 8,192 channels spread across four feature levels
    with different variance scales, so the low-variance half is dominated by whichever
    level happens to be flattest -- a different selection criterion from the reference.

    The mask is per TOKEN, so it still applies correctly to every block; only the scoring
    input needed narrowing.

    Resolution order, because `get_current_vllm_config()` is only valid during engine
    init and this runs in the model forward, in the EngineCore process:
      1. `VIDCOM2_MAIN_WIDTH` if set explicitly,
      2. the served model's own config.json, located from the `vllm serve` argv,
      3. give up loudly and score everything (the pre-fix behaviour).
    """
    global _MAIN_WIDTH, _WIDTH_WARNED
    if _MAIN_WIDTH is not None:
        return _MAIN_WIDTH if total_width % _MAIN_WIDTH == 0 else total_width

    import json as _json
    import os as _os
    import sys as _sys

    env = _os.environ.get("VIDCOM2_MAIN_WIDTH")
    if env and env.isdigit() and total_width % int(env) == 0:
        _MAIN_WIDTH = int(env)
        print(f"[vidcom2_vllm] scoring the final {_MAIN_WIDTH} of {total_width} channels "
              f"(VIDCOM2_MAIN_WIDTH)", flush=True)
        return _MAIN_WIDTH

    for cand in _sys.argv[1:]:
        cfg = _os.path.join(cand, "config.json")
        if not (_os.sep in cand and _os.path.exists(cfg)):
            continue
        try:
            c = _json.load(open(cfg))
            h = (c.get("text_config") or c).get("hidden_size")
            if h and total_width % int(h) == 0:
                _MAIN_WIDTH = int(h)
                n_blocks = total_width // _MAIN_WIDTH
                print(f"[vidcom2_vllm] scoring the final {_MAIN_WIDTH} of {total_width} "
                      f"channels ({n_blocks} blocks: 1 merger + {n_blocks - 1} DeepStack), "
                      f"read from {cfg}", flush=True)
                return _MAIN_WIDTH
        except Exception:
            continue

    if not _WIDTH_WARNED:
        _WIDTH_WARNED = True
        print(f"[vidcom2_vllm] WARNING: could not resolve hidden_size from the vLLM "
              f"config; scoring all {total_width} channels, which includes DeepStack "
              f"blocks and diverges from the reference implementation.", flush=True)
    return total_width


def vidcom2_scores(video_embeds: torch.Tensor, T: int, tpf: int):
    """(frame_importance[T], token_score[T, tpf]) -- lower token_score = keep.

    Mirrors vidcom2.py: select_low_var_channels -> compute_gaussian_scores,
    then score fusion `vid_score + frame_score`.

    Scores the FINAL embedding block only (see `main_embed_width`), in the tensor's own
    dtype. Upcasting to fp32 was unconditional here until 2026-08-17: on a 28,160-token
    clip at width 16,384 the fp32 copy, the selected-channel tensor and the normalised
    tensor cost ~3.4 GiB between them before any Gaussian temporaries, and it diverges
    from the reference, which scores in the model dtype. Set VIDCOM2_FP32=1 to restore
    the old behaviour for a controlled comparison.
    """
    import os
    x = video_embeds.reshape(T * tpf, -1)
    x = x[:, : main_embed_width(x.shape[-1])]
    if os.environ.get("VIDCOM2_FP32") == "1":
        x = x.float()

    # select_low_var_channels: keep the LOWEST-variance half of the channels
    var = x.var(dim=0, unbiased=False)
    n_keep = max(1, int(x.shape[-1] * CHANNEL_KEEP_RATIO))
    chan = torch.topk(var, k=n_keep, largest=False).indices
    sel = x[:, chan].reshape(T, tpf, n_keep)

    # compute_gaussian_scores: centres are means of L2-NORMALISED tokens
    fr = F.normalize(sel, dim=-1)
    vid_center = fr.mean(dim=(0, 1), keepdim=True)     # (1, 1, C)
    frame_center = fr.mean(dim=1, keepdim=True)        # (T, 1, C)

    def _multi_scale_gaussian(centre):
        d2 = ((fr - centre) ** 2).sum(dim=-1)
        return sum(torch.exp(-d2 / (2 * a)) for a in GAUSSIAN_ALPHAS)

    v_score = _multi_scale_gaussian(vid_center)        # (T, tpf)
    f_score = _multi_scale_gaussian(frame_center)      # (T, tpf)
    # compute_scales is driven by -v_score.mean(-1); selection by v+f.
    return -v_score.mean(dim=-1), v_score + f_score


def compute_retention_mask(
    video_embeds: torch.Tensor,
    video_size_thw: "torch.LongTensor | tuple[int, int, int]",
    spatial_merge_size: int,
    q: float,
) -> torch.Tensor:
    """VidCom2 retention mask. Signature matches vllm.multimodal.evs.

    Args:
        video_embeds: `(T * H * W // spatial_merge_size**2, hidden)` post-merger.
        video_size_thw: (T, H, W) grid dims.
        spatial_merge_size: ViT spatial merge factor.
        q: PRUNING rate in [0, 1). Retention base = 1 - q.

    Returns:
        Bool mask `(T * H * W // spatial_merge_size**2,)` whose True count equals
        `evs.compute_retained_tokens_count(tokens_per_frame, T, q)` exactly.
    """
    from vllm.multimodal.evs import compute_retained_tokens_count

    # Proof-of-life: vLLM runs the model in a separate EngineCore process, so a
    # patch applied in the API process is not evidence that the model uses it.
    # This fires once, from inside whichever process actually computes the mask.
    global _ANNOUNCED
    if not _ANNOUNCED:
        _ANNOUNCED = True
        import os as _os
        print(f"[vidcom2_vllm] ACTIVE in pid={_os.getpid()} "
              f"(first mask: thw={tuple(int(v) for v in video_size_thw)}, q={q})",
              flush=True)

    T, H, W = (int(v) for v in video_size_thw)
    h, w = H // spatial_merge_size, W // spatial_merge_size
    tpf = h * w
    n = T * tpf

    if q is None or q <= 0.0:
        return torch.ones(n, dtype=torch.bool, device=video_embeds.device)

    target = compute_retained_tokens_count(tokens_per_frame=tpf, num_frames=T, q=q)
    if target >= n:
        return torch.ones(n, dtype=torch.bool, device=video_embeds.device)

    frame_importance, token_score = vidcom2_scores(video_embeds, T, tpf)

    # compute_scales: near-one-hot softmax (temp 0.01) -> base*(1 + p - mean(p))
    probs = F.softmax((frame_importance - frame_importance.max()) / SOFTMAX_TEMP, dim=0)
    base = 1.0 - float(q)
    scales = (base * (1.0 + probs - probs.mean())).clamp(max=1.0)

    # The reference clamps every frame to >=1 token (`clamp(min=1)`), which is
    # impossible when the budget is smaller than the frame count. vLLM's exact
    # count is not negotiable -- the placeholder run is already sized -- so the
    # floor is relaxed to 0 in that regime and kept at 1 everywhere else.
    # For our grids (tpf=45) this only triggers at q>0.978, i.e. never in practice.
    ks = _apportion(scales * tpf, target=target, lo=1 if target >= T else 0, hi=tpf)

    # Optional forensic dump: VIDCOM2_DUMP=<path.jsonl> records what the allocator
    # decided, so its budget can be correlated against the ground-truth evidence window
    # offline. Keyed by a content hash of the embeddings so repeated requests for the
    # same video (extra turns, the finalizer) collapse to one entry.
    _dump = os.environ.get("VIDCOM2_DUMP")
    if _dump:
        with open(_dump, "a") as fh:
            fh.write(json.dumps({
                "thw": [int(v) for v in video_size_thw],
                "tpf": int(tpf),
                "q": float(q),
                "hash": int(video_embeds.reshape(-1)[::9973].float().sum().item() * 1e3),
                "frame_importance": [round(float(v), 5) for v in frame_importance],
                "ks": [int(v) for v in ks],
            }) + "\n")

    mask = torch.zeros(T, tpf, dtype=torch.bool, device=video_embeds.device)
    for t in range(T):
        k = int(ks[t])
        if k <= 0:
            continue
        # largest=False -> retain the tokens LEAST similar to the two centres
        idx = torch.topk(token_score[t], k=k, largest=False, sorted=False).indices
        mask[t, idx] = True
    return mask.reshape(-1)
