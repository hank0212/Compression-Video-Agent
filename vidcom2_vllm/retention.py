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

import torch
import torch.nn.functional as F

# Reference-implementation constants (vidcom2.py).
CHANNEL_KEEP_RATIO = 0.5           # select_low_var_channels(ratio=0.5)
GAUSSIAN_ALPHAS = [2.0 ** k for k in range(-3, 2)]   # alphas = [2**k for k in range(-3,2)]
SOFTMAX_TEMP = 0.01                # compute_scales(temp=0.01)

_ANNOUNCED = False                 # set on first mask computation (see below)


def _apportion(ideal: torch.Tensor, target: int, lo: int, hi: int) -> torch.Tensor:
    """Integer per-frame budgets summing to EXACTLY `target`, each within [lo, hi].

    Largest-remainder (Hamilton) apportionment of `ideal`. Deterministic: ties are
    broken by frame index, so the same input always yields the same mask.
    """
    T = ideal.numel()
    lo = min(lo, hi)
    target = int(max(T * lo, min(T * hi, target)))

    ideal = ideal.clamp(min=float(lo), max=float(hi))
    k = ideal.floor().long().clamp(min=lo, max=hi)
    remainder = ideal - k.to(ideal.dtype)
    diff = target - int(k.sum())
    if diff == 0:
        return k

    # Water-fill in remainder-priority order, honouring each frame's headroom.
    # O(T log T); a naive per-unit loop is both slower and easy to get wrong
    # (an earlier version stopped after one pass and silently under-filled).
    if diff > 0:
        order = torch.argsort(remainder, descending=True, stable=True)
        room = (hi - k)[order]
    else:
        order = torch.argsort(remainder, descending=False, stable=True)
        room = (k - lo)[order]
        diff = -diff

    csum = torch.cumsum(room, dim=0)
    n_full = int((csum <= diff).sum())            # frames that absorb their whole room
    step = torch.zeros_like(k)
    if n_full:
        step[order[:n_full]] = room[:n_full]
    used = int(csum[n_full - 1]) if n_full else 0
    if n_full < T and diff - used > 0:
        step[order[n_full]] = diff - used

    return k + step if target > int(k.sum()) else k - step


def vidcom2_scores(video_embeds: torch.Tensor, T: int, tpf: int):
    """(frame_importance[T], token_score[T, tpf]) -- lower token_score = keep.

    Mirrors vidcom2.py: select_low_var_channels -> compute_gaussian_scores,
    then score fusion `vid_score + frame_score`.
    """
    x = video_embeds.reshape(T * tpf, -1).float()

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

    mask = torch.zeros(T, tpf, dtype=torch.bool, device=video_embeds.device)
    for t in range(T):
        k = int(ks[t])
        if k <= 0:
            continue
        # largest=False -> retain the tokens LEAST similar to the two centres
        idx = torch.topk(token_score[t], k=k, largest=False, sorted=False).indices
        mask[t, idx] = True
    return mask.reshape(-1)
