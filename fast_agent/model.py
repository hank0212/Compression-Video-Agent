"""Engine: Qwen3-VL-8B + vision-side FlashVID patches + VisionService.

Design (verified against FlashVID/flashvid/modeling_qwen3_vl.py):
- We patch ONLY the vision tower (attention/block/model forwards) so `visual()`
  returns (merged_tokens, deepstack_list, cls_attention). The text model stays
  100% stock — no fastv, no flashvid_config plumbing on the LM side.
- The patched vision attention asserts flash_attention_2 → we load with it.
- Compression calls flashvid.utils.flashvid_compression DIRECTLY (same call the
  patched Qwen3VLModel_forward makes at line 307), so DySeg+ADTS+TSTM and real
  last-block CLS attention are exactly FlashVID-faithful.
- Each clip is ViT-encoded ONCE and cached (embeds + deepstack + token segment);
  the agent loop re-assembles embeddings across rounds without re-encoding.
"""

import dataclasses
import sys
from dataclasses import dataclass, field

import torch

from . import config
from . import semvid as semvid_lib

sys.path.insert(0, config.FLASHVID_REPO)

from flashvid.configuration_flashvid import FlashVidConfig  # noqa: E402
from flashvid.utils import flashvid_compression  # noqa: E402


# ---------------------------------------------------------------------------
@dataclass
class Clip:
    """One encoded visual block, ready to splice into any round's sequence."""

    kind: str                       # "images" | "video"
    segment_ids: torch.Tensor       # (S,) FULL token segment incl. vision_start/end
    embeds: torch.Tensor            # (K, D) tokens to scatter (post-compression for video)
    deepstack: list                 # per-level (K, D)
    grid_thw: torch.Tensor          # (n, 3) grids for get_rope_index (n images or 1 video)
    keep_indices: torch.Tensor | None = None  # video only: kept anchors into full pad run
    meta: dict = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        return self.embeds.shape[0]


def clip_tokens(grid_thw, merge_size: int = 2, retention: float = 1.0) -> int:
    """Exact merged-token count for a grid (derive, don't hardcode: patch_size=16)."""
    total = 0
    for t, h, w in grid_thw.tolist():
        total += int(t) * (int(h) // merge_size) * (int(w) // merge_size)
    return int(round(total * retention))


def retention_for_budget(base_tokens: int, target_tokens: int, r_min: float = 0.02) -> float:
    return max(r_min, min(1.0, target_tokens / max(base_tokens, 1)))


# ---------------------------------------------------------------------------
class Engine:
    def __init__(self, device: str = "cuda:0"):
        from transformers import AutoProcessor, AutoTokenizer
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLForConditionalGeneration,
        )

        self.device = device
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.MODEL_SNAPSHOT,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(
            config.MODEL_SNAPSHOT,
            max_pixels=config.MAX_PIXELS,
            min_pixels=config.MIN_PIXELS,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_SNAPSHOT)

        cfg = self.model.config
        self.image_token_id = cfg.image_token_id
        self.video_token_id = cfg.video_token_id
        self.vision_start_id = cfg.vision_start_token_id
        self.vision_end_id = cfg.vision_end_token_id
        self.eos_ids = {self.tokenizer.eos_token_id}
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None:
            self.eos_ids.add(im_end)

        self._vision_patched = False
        self._orig_forwards = {}
        self._fast_patch_embed()

    def _fast_patch_embed(self):
        """Replace patch_embed's Conv3d with the mathematically exact matmul.
        kernel==stride makes the conv a linear map over flattened patches; cuDNN
        picks a catastrophic kernel for this shape on the A6000 (34.3s for 64
        frames vs 0.005s as a matmul — 6900x; max abs diff 0.016 = bf16 noise).
        Instance-level override; all arms share it, so A/Bs stay fair."""
        pe = self.model.model.visual.patch_embed
        w = pe.proj.weight.data
        b = pe.proj.bias.data if pe.proj.bias is not None else None
        w2 = w.view(w.shape[0], -1)

        def fast_forward(hidden_states):
            out = hidden_states.view(hidden_states.shape[0], -1) @ w2.T
            return out + b if b is not None else out

        pe.forward = fast_forward

    # -- vision-side FlashVID patch (reversible) ----------------------------
    def apply_vision_patches(self):
        if self._vision_patched:
            return
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionAttention,
            Qwen3VLVisionBlock,
            Qwen3VLVisionModel,
        )
        from flashvid.modeling_qwen3_vl import (
            Qwen3VLVisionAttention_forward,
            Qwen3VLVisionBlock_forward,
            Qwen3VLVisionModel_forward,
        )

        self._orig_forwards = {
            "attn": Qwen3VLVisionAttention.forward,
            "block": Qwen3VLVisionBlock.forward,
            "model": Qwen3VLVisionModel.forward,
        }
        Qwen3VLVisionAttention.forward = Qwen3VLVisionAttention_forward
        Qwen3VLVisionBlock.forward = Qwen3VLVisionBlock_forward
        Qwen3VLVisionModel.forward = Qwen3VLVisionModel_forward
        self._vision_patched = True

    def restore_vision(self):
        if not self._vision_patched:
            return
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionAttention,
            Qwen3VLVisionBlock,
            Qwen3VLVisionModel,
        )

        Qwen3VLVisionAttention.forward = self._orig_forwards["attn"]
        Qwen3VLVisionBlock.forward = self._orig_forwards["block"]
        Qwen3VLVisionModel.forward = self._orig_forwards["model"]
        self._vision_patched = False

    # -- encoding ------------------------------------------------------------
    @torch.inference_mode()
    def encode_images(self, pils: list, meta: dict | None = None) -> Clip:
        """Encode PIL frames as IMAGE modality (full detail; never compressed).
        ViT runs exactly once; result cached in the returned Clip."""
        self.apply_vision_patches()
        proc = self.processor.image_processor(images=pils, return_tensors="pt")
        pixel_values = proc["pixel_values"].to(self.device, torch.bfloat16)
        grid_thw = proc["image_grid_thw"].to(self.device)
        hidden, deepstack, _attn = self.model.model.visual(pixel_values, grid_thw)

        # segment ids: per image <|vision_start|> pads <|vision_end|>
        seg = []
        for t, h, w in grid_thw.tolist():
            n = t * (h // 2) * (w // 2)
            seg += [self.vision_start_id] + [self.image_token_id] * n + [self.vision_end_id]
        return Clip(
            kind="images",
            segment_ids=torch.tensor(seg, dtype=torch.long, device=self.device),
            embeds=hidden,
            deepstack=list(deepstack),
            grid_thw=grid_thw,
            meta=meta or {},
        )

    @torch.inference_mode()
    def encode_video_compressed(
        self,
        video_tensor: torch.Tensor,
        target_tokens: int,
        frame_times: list[float],
        meta: dict | None = None,
        query_text: str | None = None,
    ) -> Clip:
        """Encode a (T,C,H,W) uint8 clip as VIDEO modality and compress to
        ~target_tokens via config.COMPRESSOR. ViT runs exactly once either way.
        - flashvid (default): patched ViT last-block CLS attention → DySeg +
          ADTS + TSTM merge (alpha=0.7). Query-agnostic; query_text ignored.
        - semvid: query-aware selection (semvid.py) on the same ViT output —
          query_text drives frame budgets + object-token picks; keep-indices
          are unique (selection, no merge anchors)."""
        import time

        self.apply_vision_patches()
        # do_sample_frames=False: the processor's default (True, fps=2) would
        # RESAMPLE our already-sampled frames and break frame_times alignment.
        #
        # size override: Qwen3VLVideoProcessor has its OWN global token ceiling
        # (video_preprocessor_config.json's size.longest_edge, 25,165,824 for this
        # checkpoint) checked as t_bar*h_bar*w_bar > longest_edge -- a budget on
        # the WHOLE clip's frames*h*w, independent of our own per-frame MAX_PIXELS
        # in tools.py. Once a clip's frame count crosses ~560 (at our 224px-derived
        # per-frame size) it silently downscales resolution to claw back under that
        # ceiling -- confirmed empirically: 512 frames -> untouched 10x18 grid,
        # 640 -> 8x16, 768 -> 8x14. That's a SECOND, hidden compression happening
        # before FlashVID even runs, invisible in the reported retention (which
        # only reflects FlashVID's own ratio against the already-shrunk base).
        # Override longest_edge per-call so OUR chosen per-frame resolution
        # (already fixed by tools.py's smart_resize) is what actually gets used,
        # and FlashVID's retention_ratio is the only compression in effect.
        T, _, H, W = video_tensor.shape
        size_override = {"shortest_edge": 4096, "longest_edge": T * H * W * 2}
        t0 = time.time()
        vp = self.processor.video_processor(
            videos=[video_tensor], do_sample_frames=False, size=size_override,
            return_tensors="pt"
        )
        t_proc = time.time() - t0
        pixel_values = vp["pixel_values_videos"].to(self.device, torch.bfloat16)
        grid_thw = vp["video_grid_thw"].to(self.device)
        t, h, w = grid_thw[0].tolist()
        tpf = (h // 2) * (w // 2)
        base_tokens = t * tpf

        t0 = time.time()
        hidden, deepstack, cls_attention = self.model.model.visual(pixel_values, grid_thw)
        torch.cuda.synchronize(self.device)
        t_vit = time.time() - t0

        auto_r = retention_for_budget(base_tokens, target_tokens)
        if config.COMPRESSOR == "semvid":
            # SemVID keeps EXACTLY round(base*ratio) tokens (integer allocation,
            # no segment-floor overshoot), so matched budget needs no calibrated
            # fixed retention — the FlashVID FIXED_RETENTION/2128-frame-cap pair
            # exists only to correct FlashVID's kept/base drift and would
            # under-budget an exact selector by ~24% at r=0.1.
            retention = auto_r
        elif config.FIXED_RETENTION > 0:
            # On short spans, fixed retention would produce fewer tokens than
            # crop_video's budget and become a blurry crop. Keep the nominal
            # matched-budget ratio there; use calibrated fixed retention for
            # long spans where it preserves the intended coverage advantage.
            retention = (
                max(config.FIXED_RETENTION, auto_r)
                if T < config.FLOOR_ENGAGE_FRAMES
                else config.FIXED_RETENTION
            )
        else:
            retention = auto_r
        video_features = hidden.view(t, tpf, -1)
        # Small scalar/shape diagnostics only.  Keeping these in metadata makes
        # the compression boundary inspectable without retaining duplicate
        # feature tensors or changing FlashVID's inputs.
        diagnostics = {
            "input_video_tensor": list(video_tensor.shape),
            "processor_pixel_values": list(pixel_values.shape),
            "grid_thw": list(grid_thw.shape),
            "vision_hidden": list(hidden.shape),
            "video_features_before_flashvid": list(video_features.shape),
            "cls_attention_before_flashvid": list(cls_attention.shape),
            "deepstack_before_flashvid": [list(d.shape) for d in deepstack],
            "finite": {
                "pixel_values": bool(torch.isfinite(pixel_values).all().item()),
                "vision_hidden": bool(torch.isfinite(hidden).all().item()),
                "video_features": bool(torch.isfinite(video_features).all().item()),
                "cls_attention": bool(torch.isfinite(cls_attention).all().item()),
                "deepstack": all(bool(torch.isfinite(d).all().item()) for d in deepstack),
            },
        }
        t0 = time.time()
        sel_stats = None
        if config.COMPRESSOR == "semvid":
            # Query-aware selection on the SAME ViT output tensors. Saliency proxy
            # is the feature norm (upstream line 883) — cls_attention stays unused.
            q = (semvid_lib.embed_query(
                    self.tokenizer, self.model.get_input_embeddings(), query_text,
                    self.device, config.SEMVID_QUERY_TOKEN_MAX)
                 if query_text else None)
            s_cfg = semvid_lib.SemVidConfig(retention_ratio=retention, **config.SEMVID_KW)
            keep_idx, sel_stats = semvid_lib.semvid_select(
                video_features,
                hidden.float().norm(dim=-1).view(t, tpf),
                q,
                s_cfg,
            )
            keep_idx = keep_idx.to(self.device).long().reshape(-1)
            compressed = hidden[keep_idx]
        else:
            fv_kw = dict(config.FLASHVID_KW)
            fv_kw["retention_ratio"] = retention
            valid = {f.name for f in dataclasses.fields(FlashVidConfig)}
            fv_cfg = FlashVidConfig(**{k: v for k, v in fv_kw.items() if k in valid})
            fv_cfg.H, fv_cfg.W = h // 2, w // 2
            compressed, keep_idx = flashvid_compression(
                video_features=video_features,
                cls_attention=cls_attention,
                flashvid_config=fv_cfg,
            )
            compressed = compressed.reshape(-1, hidden.shape[-1])
            keep_idx = keep_idx.to(self.device).long().reshape(-1)
        t_fv = time.time() - t0
        deepstack_kept = [d[keep_idx] for d in deepstack]  # anchors' rows (FlashVID line 324)

        # Native Qwen3-VL video layout (processing_qwen3_vl.py:218-227): per
        # temporal group `<{t:.1f} seconds><|vision_start|>pads<|vision_end|>`.
        # get_rope_index depends on this (splits the grid per-frame, t=1 each;
        # temporal position is carried by the timestamp TEXT).
        import numpy as np

        gt = np.asarray(frame_times, dtype=float)
        n_groups_have = len(gt) // 2
        group_times = gt[: n_groups_have * 2].reshape(-1, 2).mean(1)
        if len(group_times) < t:  # video processor may pad the last group
            pad = group_times[-1] if len(group_times) else 0.0
            group_times = np.concatenate([group_times, [pad] * (t - len(group_times))])
        seg = []
        for k in range(t):
            seg += self.tokenizer.encode(
                f"<{group_times[k]:.1f} seconds>", add_special_tokens=False
            )
            seg += [self.vision_start_id] + [self.video_token_id] * tpf + [self.vision_end_id]
        m = dict(meta or {})
        m.update(
            base_tokens=base_tokens,
            kept_tokens=int(compressed.shape[0]),
            retention=retention,
            grid=(t, h, w),
            t_proc=round(t_proc, 2),
            t_vit=round(t_vit, 2),
            t_flashvid=round(t_fv, 2),
            target_tokens=int(target_tokens),
            compressor=config.COMPRESSOR,
            diagnostics=diagnostics,
        )
        if sel_stats is not None:
            m["semvid"] = sel_stats
        return Clip(
            kind="video",
            segment_ids=torch.tensor(seg, dtype=torch.long, device=self.device),
            embeds=compressed,
            deepstack=deepstack_kept,
            grid_thw=grid_thw,
            keep_indices=keep_idx,
            meta=m,
        )
