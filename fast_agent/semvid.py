"""SemVID query-aware token selection, lifted as pure tensor functions.

Source: official SemVID repo (github.com/JiaqiLi404/SemVID, Apache-2.0),
`modeling_qwen3_vl_semvid.py` — `_semantic_prune_video_region` + helpers
(`_mmr_select`, `_allocate_integer_budget[_with_cap]`, `_compute_query_embedding`).
Paper: "Keeping the Evidence Chain" (arXiv:2603.05663, ECCV 2026).

Why lift instead of importing: upstream ships full modeling-file copies (their
selector lives on the ForConditionalGeneration class and runs at prefill on the
scattered sequence). Our harness already owns the exact tensors it needs at
tool-encode time (visual() output = post-projector video tokens), so the
selector reduces to standalone functions on (T, P, D) — same seam FlashVID uses,
returning keep-indices the existing Clip/assemble path consumes unchanged.

Deviations from upstream (each deliberate, none algorithmic):
- retention_ratio is a per-call argument (matched-budget harness), not a fixed
  model attr; ablation branches (uniform alloc / "fastvid" selector) dropped;
  keep_coords pairs replaced by scalar counts for trajectory metadata.
- scoring in float32 (inputs are bf16; cosine/argmax there is needlessly noisy).

Selection semantics vs FlashVID: SELECTS tokens (unique sorted indices), never
merges — so keep_indices carry no duplicate anchors and deepstack gathering is
exact. Query-blind fallback (no/empty query): frame-mean pseudo-query, exactly
upstream's `q_vec = fg_norm.mean(0)` branch.
"""

from dataclasses import dataclass

import torch


@dataclass
class SemVidConfig:
    """Mirrors upstream `semantic_*` hyperparams (defaults = released VideoQA
    regime: qwen2_5_videoqa preset direction — obj_ratio 0.4, mmr_lambda 0.3,
    soft segment allocation — not the Charades grounding preset)."""

    retention_ratio: float          # kept/base token ratio (harness-derived)
    dyseg_c: int = 0                # extra top-k cuts (0 = threshold cuts only)
    dyseg_tau: float = 0.0          # cut where adjacent cosine < tau (released
                                    # presets use 0.0 -> segmentation ~off)
    stage1_topk_segments: int = 0   # 0 = soft allocation over all segments
    stage1_smooth_win: int = 1      # odd window; 1 = no smoothing
    frame_weight_alpha: float = 0.7 # alpha*query_rel + (1-alpha)*motion_energy
    obj_ratio: float = 0.4          # per-frame extra budget: object vs motion
    mmr_lambda: float = 0.3         # MMR: 1 -> relevance, 0 -> diversity
    min_tokens_per_frame: int = 1   # coherence floor (>=1 recommended upstream)
    motion_query_beta: float = 0.5  # blend query sim into motion scores


# ---------------------------------------------------------------------------
# helpers (faithful lifts)

def _allocate_integer_budget(weights: torch.Tensor, total: int) -> torch.Tensor:
    """Largest-remainder integer allocation: alloc.sum() == total, alloc >= 0."""
    if total <= 0:
        return torch.zeros_like(weights, dtype=torch.long)
    if weights.numel() == 0:
        return torch.empty((0,), device=weights.device, dtype=torch.long)
    w = weights.float().clamp_min(0)
    if w.sum() < 1e-8:
        w = torch.ones_like(w)
    raw = w / w.sum() * float(total)
    flo = torch.floor(raw).to(torch.long)
    rem = int(total - flo.sum().item())
    if rem > 0:
        top = torch.topk(raw - flo.float(), k=min(rem, raw.numel())).indices
        flo[top] += 1
    return flo


def _allocate_integer_budget_with_cap(
    weights: torch.Tensor, total: int, cap: torch.Tensor
) -> torch.Tensor:
    """Capped variant (rounds of allocate-then-clamp) for leftover redistribution."""
    if total <= 0 or weights.numel() == 0:
        return torch.zeros_like(cap, dtype=torch.long)
    cap = cap.to(weights.device)
    total = int(min(total, int(cap.sum().item())))
    if total <= 0:
        return torch.zeros_like(cap, dtype=torch.long)
    alloc = torch.zeros_like(cap, dtype=torch.long)
    active = cap > 0
    remaining = total
    for _ in range(16):
        if remaining <= 0 or not bool(active.any()):
            break
        w = weights.float().clamp_min(0) * active.float()
        if w.sum() < 1e-8:
            w = active.float()
        raw = w / w.sum() * float(remaining)
        flo = torch.floor(raw).to(torch.long)
        rem = int(remaining - flo.sum().item())
        if rem > 0:
            frac = (raw - flo.float()).masked_fill(~active, -1.0)
            top = torch.topk(frac, k=min(rem, int(active.sum().item()))).indices
            flo[top] += 1
        flo = torch.minimum(flo, cap - alloc)
        used = int(flo.sum().item())
        alloc = alloc + flo
        remaining -= used
        active = (cap - alloc) > 0
        if used == 0:
            break
    return alloc


def _mmr_select(
    rel_scores: torch.Tensor, cand_embeds_norm: torch.Tensor, k: int, lambda_mmr: float
) -> torch.Tensor:
    """Maximal Marginal Relevance: argmax lambda*rel - (1-lambda)*max_sim_to_selected."""
    nc = int(rel_scores.numel())
    if k <= 0 or nc <= 0:
        return rel_scores.new_empty((0,), dtype=torch.long)
    k = min(int(k), nc)
    first = int(torch.argmax(rel_scores).item())
    selected = [first]
    if k == 1:
        return torch.tensor(selected, device=rel_scores.device, dtype=torch.long)
    available = torch.ones(nc, device=rel_scores.device, dtype=torch.bool)
    available[first] = False
    redundancy = (cand_embeds_norm @ cand_embeds_norm[first].unsqueeze(-1)).squeeze(-1)
    for _ in range(k - 1):
        if not bool(available.any()):
            break
        mmr = lambda_mmr * rel_scores - (1.0 - lambda_mmr) * redundancy
        nxt = int(torch.argmax(mmr.masked_fill(~available, -1e9)).item())
        selected.append(nxt)
        available[nxt] = False
        sim_n = (cand_embeds_norm @ cand_embeds_norm[nxt].unsqueeze(-1)).squeeze(-1)
        redundancy = torch.maximum(redundancy, sim_n)
    return torch.tensor(selected, device=rel_scores.device, dtype=torch.long)


# ---------------------------------------------------------------------------
def embed_query(tokenizer, embed_layer, text: str, device, token_max: int = 64):
    """Upstream `_compute_query_embedding`, query_ids path: plain-text tokens ->
    embedding-layer lookup (no decoder forward). Returns (L, D) row-normalized
    per-token embeddings if L <= token_max (multi-token coverage path), else a
    single mean-pooled L2-normalized (D,) vector."""
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    ids = ids.to(device)
    if ids.numel() == 0:
        return None
    with torch.inference_mode():
        tok = embed_layer(ids).float()                     # (L, D)
    if tok.shape[0] <= token_max:
        return tok / tok.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    q = tok.mean(dim=0)
    return q / q.norm(dim=-1).clamp_min(1e-6)


# ---------------------------------------------------------------------------
@torch.inference_mode()
def semvid_select(
    video_features: torch.Tensor,        # (T, P, D) post-projector video tokens
    frame_token_scores: torch.Tensor,    # (T, P) saliency proxy (feature norm)
    query_embed: torch.Tensor | None,    # (L, D) or (D,) from embed_query; None ok
    cfg: SemVidConfig,
) -> tuple[torch.Tensor, dict]:
    """Two-stage query-aware selection. Returns (keep_idx, stats):
    keep_idx — unique SORTED indices into [0, T*P); stats — counts for metadata.

    Stage-1: frame relevance = cosine(query, frame mean feature); DySeg scene
    cuts; token budget allocated to segments by relevance, to frames inside by
    alpha*relevance + (1-alpha)*motion_energy, floored at min_tokens_per_frame.
    Stage-2 per frame: context proto token (closest to frame mean) + query-aware
    object tokens (MMR, per-query-token budget when multi-token) + motion tokens
    (temporal-difference norm, query-blended), saliency fill to exact budget.
    """
    T, P, D = video_features.shape
    device = video_features.device
    vf = video_features.float()

    # ---- 0) normalize query + frame globals ----
    fg = vf.mean(dim=1)                                    # (T, D) frame means
    fg_norm = fg / fg.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    q_tokens = None
    q_vec = None
    if query_embed is not None and query_embed.ndim == 2 and query_embed.shape[-1] == D:
        q_tokens = query_embed.to(device, torch.float32)
        q_vec = q_tokens.mean(dim=0)
    elif query_embed is not None and query_embed.ndim == 1 and query_embed.shape[0] == D:
        q_vec = query_embed.to(device, torch.float32)
    else:                                                  # query-blind fallback
        q_vec = fg_norm.mean(dim=0)
    q_vec = q_vec / q_vec.norm(dim=-1).clamp_min(1e-6)

    if q_tokens is None:
        frame_rel = (fg_norm * q_vec.unsqueeze(0)).sum(dim=-1)          # (T,)
    else:
        rel_mat = fg_norm @ q_tokens.t()                                # (T, L)
        if rel_mat.shape[1] == 1:
            frame_rel = rel_mat.squeeze(1)
        else:  # top-2 mean over query tokens: sharper than mean, stabler than max
            k_red = min(2, rel_mat.shape[1])
            frame_rel = torch.topk(rel_mat, k=k_red, dim=1).values.mean(dim=1)

    frame_rel_s = frame_rel
    win = cfg.stage1_smooth_win
    if win > 1 and T >= 3:
        if win % 2 == 0:
            win += 1
        win = min(win, T if (T % 2 == 1) else max(T - 1, 1))
        if win >= 3:
            x = torch.nn.functional.pad(frame_rel.view(1, 1, T), (win // 2, win // 2),
                                        mode="replicate")
            kernel = torch.ones((1, 1, win), device=device, dtype=x.dtype) / float(win)
            frame_rel_s = torch.nn.functional.conv1d(x, kernel).view(-1)

    # ---- 1) DySeg scene cuts (query-agnostic) ----
    if T <= 1:
        seg_sizes = [T]
    else:
        adj_sim = (fg_norm[:-1] * fg_norm[1:]).sum(dim=-1)              # (T-1,)
        k = min(max(int(cfg.dyseg_c) - 1, 0), int(adj_sim.numel()))
        cut_topk = (torch.topk(adj_sim, k, largest=False).indices if k > 0
                    else adj_sim.new_empty((0,), dtype=torch.long))
        cut_tau = torch.nonzero(adj_sim < float(cfg.dyseg_tau)).squeeze(1)
        cut = torch.unique(torch.cat([cut_topk, cut_tau])).sort().values
        if cut.numel() == 0:
            seg_sizes = [T]
        else:
            seg_sizes = [int(cut[0].item()) + 1]
            for i in range(1, cut.numel()):
                seg_sizes.append(int(cut[i].item() - cut[i - 1].item()))
            seg_sizes.append(int(T - cut[-1].item() - 1))
            seg_sizes = [s for s in seg_sizes if s > 0] or [T]

    # ---- 2) budget across segments -> frames (coherence floor) ----
    base_k = max(min(cfg.min_tokens_per_frame, P), 1)
    ratio = max(min(cfg.retention_ratio, 1.0), 0.0)
    total_tokens = int(T * P)
    target_keep = min(max(int(round(total_tokens * ratio)), int(T * base_k)), total_tokens)

    seg_scores, st = [], 0
    for sz in seg_sizes:
        seg_scores.append(frame_rel_s[st: st + sz].mean())
        st += sz
    seg_scores = torch.stack(seg_scores, dim=0)                         # (Nseg,)

    remaining = int(target_keep - T * base_k)
    w_seg = torch.relu(seg_scores)
    if w_seg.sum() < 1e-8:
        w_seg = (seg_scores - seg_scores.min()).clamp_min(0) + 1e-6
    topk = cfg.stage1_topk_segments
    if 0 < topk < int(w_seg.numel()):
        top_idx = torch.topk(w_seg, k=topk).indices
        w_mask = torch.zeros_like(w_seg)
        w_mask[top_idx] = w_seg[top_idx]
        w_seg = w_mask
        if w_seg.sum() < 1e-8:
            w_seg[top_idx] = 1.0

    motion_energy = torch.zeros((T,), device=device, dtype=frame_rel_s.dtype)
    if T > 1:
        motion_energy[1:] = (fg_norm[1:] - fg_norm[:-1]).norm(dim=-1)
        motion_energy[0] = motion_energy[1]

    frame_keep = torch.full((T,), base_k, device=device, dtype=torch.long)
    alpha = cfg.frame_weight_alpha
    if remaining > 0:
        seg_extra = _allocate_integer_budget(w_seg, remaining)
        st = 0
        for i, sz in enumerate(seg_sizes):
            extra_i = int(seg_extra[i].item())
            if extra_i > 0:
                w_frm = (alpha * torch.relu(frame_rel_s[st: st + sz])
                         + (1.0 - alpha) * motion_energy[st: st + sz].clamp_min(0))
                if w_frm.sum() < 1e-8:
                    w_frm = torch.ones_like(w_frm)
                frame_keep[st: st + sz] += _allocate_integer_budget(w_frm, extra_i)
            st += sz
        frame_keep = torch.clamp(frame_keep, max=P)
        leftover = int(target_keep - int(frame_keep.sum().item()))
        slack = (P - frame_keep).clamp_min(0)
        if leftover > 0 and int(slack.sum().item()) > 0:
            w_all = (alpha * torch.relu(frame_rel_s)
                     + (1.0 - alpha) * motion_energy.clamp_min(0)) * slack.float()
            if w_all.sum() < 1e-8:
                w_all = slack.float()
            frame_keep = frame_keep + _allocate_integer_budget_with_cap(w_all, leftover, slack)

    # ---- 3) within-frame structured selection ----
    scores_all = frame_token_scores.to(device, torch.float32)           # (T, P)
    keep_local, n_ctx, n_obj, n_mot = [], 0, 0, 0
    for t in range(T):
        k_total = min(int(frame_keep[t].item()), P)
        if k_total <= 0:
            continue
        tokens = vf[t]                                                  # (P, D)
        scores = scores_all[t]
        tok_norm = tokens / tokens.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        sim_tok_q = tok_norm @ q_tokens.t() if q_tokens is not None else None

        # context proto: token closest to the frame-global mean
        proto = int(torch.argmax(tok_norm @ fg_norm[t]).item())
        sel_mask = torch.zeros((P,), device=device, dtype=torch.bool)
        sel_mask[proto] = True
        n_ctx += 1

        if base_k > 1:
            sc = scores.clone()
            sc[sel_mask] = -1e9
            k_ctx = min(int(base_k - 1), int((~sel_mask).sum().item()))
            if k_ctx > 0:
                sel_mask[torch.topk(sc, k=k_ctx).indices] = True
                n_ctx += k_ctx

        extra = int(k_total - int(sel_mask.sum().item()))
        if extra > 0:
            k_obj = max(min(int(round(extra * cfg.obj_ratio)), extra), 0)
            k_mot = extra - k_obj

            # object tokens (query-aware, MMR diversity)
            if k_obj > 0:
                cand_idx = torch.arange(P, device=device)[~sel_mask]
                if cand_idx.numel() > 0:
                    if q_tokens is None:
                        cand_emb = tok_norm[cand_idx]
                        sel_rel = _mmr_select(cand_emb @ q_vec, cand_emb, k_obj,
                                              cfg.mmr_lambda)
                        sel_mask[cand_idx[sel_rel]] = True
                        n_obj += int(sel_rel.numel())
                    else:  # allocate k_obj across query tokens by per-frame presence
                        w_q = sim_tok_q.max(dim=0).values.clamp_min(0)  # (L,)
                        alloc_q = _allocate_integer_budget(w_q, k_obj)
                        for qi in torch.argsort(w_q, descending=True).tolist():
                            ki = int(alloc_q[qi].item())
                            if ki <= 0:
                                continue
                            cand2 = torch.arange(P, device=device)[~sel_mask]
                            if cand2.numel() == 0:
                                break
                            sel2 = _mmr_select(sim_tok_q[cand2, qi], tok_norm[cand2],
                                               ki, cfg.mmr_lambda)
                            sel_mask[cand2[sel2]] = True
                            n_obj += int(sel2.numel())

            # motion tokens (temporal-difference norm, query-blended)
            if k_mot > 0:
                cand_idx = torch.arange(P, device=device)[~sel_mask]
                if cand_idx.numel() > 0:
                    if T == 1:
                        mot_score = torch.zeros((P,), device=device)
                    elif t == 0:
                        mot_score = (vf[1] - vf[0]).norm(dim=-1)
                    elif t == T - 1:
                        mot_score = (vf[t] - vf[t - 1]).norm(dim=-1)
                    else:
                        mot_score = 0.5 * ((vf[t] - vf[t - 1]).norm(dim=-1)
                                           + (vf[t + 1] - vf[t]).norm(dim=-1))
                    ms = mot_score[cand_idx]
                    beta = cfg.motion_query_beta
                    if q_tokens is not None and beta > 1e-6:
                        mot_n = ms / ms.max().clamp_min(1e-6)
                        sim_n = sim_tok_q[cand_idx].max(dim=1).values.clamp_min(0)
                        sim_n = sim_n / sim_n.max().clamp_min(1e-6)
                        ms = (1.0 - beta) * mot_n + beta * sim_n
                    k_m = min(k_mot, int(cand_idx.numel()))
                    if k_m > 0:
                        sel_mask[cand_idx[torch.topk(ms, k=k_m).indices]] = True
                        n_mot += k_m

            # saliency fill to exact per-frame budget
            deficit = int(k_total - int(sel_mask.sum().item()))
            if deficit > 0:
                cand_idx = torch.arange(P, device=device)[~sel_mask]
                if cand_idx.numel() > 0:
                    k_fill = min(deficit, int(cand_idx.numel()))
                    sel_mask[cand_idx[torch.topk(scores[cand_idx], k=k_fill).indices]] = True
                    n_ctx += k_fill

        if int(sel_mask.sum().item()) == 0:                 # never-empty guard
            sel_mask[0] = True
            n_ctx += 1
        keep_local.append(
            torch.nonzero(sel_mask, as_tuple=False).squeeze(1).sort().values + t * P
        )

    keep_idx = (torch.cat(keep_local).sort().values if keep_local
                else torch.arange(T * P, device=device, dtype=torch.long))
    stats = {
        "n_segments": len(seg_sizes),
        "n_context": n_ctx,
        "n_object": n_obj,
        "n_motion": n_mot,
        "query_tokens": (0 if query_embed is None
                         else (int(q_tokens.shape[0]) if q_tokens is not None else 1)),
        "frame_keep_min": int(frame_keep.min().item()),
        "frame_keep_max": int(frame_keep.max().item()),
    }
    return keep_idx, stats
