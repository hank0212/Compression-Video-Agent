"""SemVID compressor smoke — machinery + selection-behavior checks, NOT accuracy.

Three arms on ONE real LVBench question (evidence span known), same clip tensor,
same matched-budget target:
  A. flashvid            — incumbent, query-agnostic (run2 machinery, unchanged)
  B. semvid + query      — the new path, question+options drive selection
  C. semvid, no query    — upstream's frame-mean fallback (isolates the query's
                           effect: B vs C differ ONLY in the query embedding)

Checks:
  1. budget adherence (kept ~= target) per arm
  2. semvid keep-indices are unique+sorted; deepstack rows match embeds rows
  3. keep-set overlaps (A/B, B/C) — how much the query actually moves selection
  4. evidence concentration: share of kept tokens whose temporal group falls in
     the GT evidence span, vs the span's share of groups (uniform = 1.0x)
  5. end-to-end: assemble + decode the MCQ from the semvid clip (injection works)

  CUDA_VISIBLE_DEVICES=0 python -m fast_agent.smoke_semvid
"""

import os
import time

import cv2
import numpy as np
import torch

from . import config, data, tools
from .agent_loop import assemble, crop_budget_tokens, decode
from .model import Engine


# Wider than the oracle's +/-300s: at +/-300 around a narrow ref the span is so
# short that retention lands ~0.8 and every selector keeps almost everything —
# the smoke must exercise the LOW-retention regime where selection matters.
SMOKE_MARGIN = 900.0
MIN_EV_WIDTH = 30.0   # need a real span for the concentration readout
                      # (263/1549 LVBench refs are zero-width)
MAX_EV_WIDTH = 180.0  # ... but a NARROW one: an hour-wide "evidence" span covers
                      # ~70% of the clip and caps measurable concentration at ~1x


def pick_row():
    """First WIDE-evidence row that is cheap to decode (cached proxy or
    cv2-decodable original) — avoids a 30-min AV1 transcode inside a smoke."""
    for row in data.load_lvbench():
        ev = row["evidence"]
        if not ev or not (MIN_EV_WIDTH <= ev[1] - ev[0] <= MAX_EV_WIDTH):
            continue
        proxy = os.path.join(tools.PROXY_DIR, os.path.basename(row["video_path"]))
        if os.path.exists(proxy) and os.path.getsize(proxy) > 0:
            return row
        cap = cv2.VideoCapture(row["video_path"])
        try:
            n_v = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(n_v // 2 - 1, 0))
            ok, _ = cap.read()
        finally:
            cap.release()
        if ok:
            return row
    raise RuntimeError("no decodable evidence-bearing LVBench row found")


def frame_share(keep_idx: torch.Tensor, tpf: int, group_times: np.ndarray,
                ev: tuple[float, float]) -> tuple[float, float]:
    """(kept-token share in evidence groups, evidence groups' share of all groups)."""
    groups = (keep_idx.cpu().numpy() // tpf)
    in_ev_group = (group_times >= ev[0]) & (group_times <= ev[1])
    if len(keep_idx) == 0 or not in_ev_group.any():
        return 0.0, float(in_ev_group.mean())
    kept_in_ev = in_ev_group[groups].mean()
    return float(kept_in_ev), float(in_ev_group.mean())


def main():
    row = pick_row()
    ev = row["evidence"]
    print(f"[pick] qid={row['question_id']} video={row['videoID']} "
          f"evidence={ev[0]:.0f}-{ev[1]:.0f}s ({row['time_reference']!r})")
    print(f"[pick] Q: {row['question'][:120]}")

    dur = data.video_duration(row["video_path"])
    s = max(0.0, ev[0] - SMOKE_MARGIN)
    e = min(dur, ev[1] + SMOKE_MARGIN)
    vt, times = tools.compress_tensor(row["video_path"], s, e)
    print(f"[clip] span {s:.0f}-{e:.0f}s of {dur:.0f}s -> {vt.shape[0]} frames")

    engine = Engine()
    target = crop_budget_tokens(engine)
    query = data.format_question(row)

    arms = {}
    for name, (comp, q) in {
        "flashvid": ("flashvid", None),
        "semvid_q": ("semvid", query),
        "semvid_nq": ("semvid", None),
    }.items():
        config.COMPRESSOR = comp
        t0 = time.time()
        clip = engine.encode_video_compressed(
            vt.clone(), target, list(times), meta={"role": f"smoke_{name}"},
            query_text=q,
        )
        dt = time.time() - t0
        m = clip.meta
        arms[name] = clip
        print(f"[{name}] kept={m['kept_tokens']} target={target} "
              f"(ratio {m['kept_tokens']/target:.2f}) base={m['base_tokens']} "
              f"retention={m['retention']:.3f} t_compress={m['t_flashvid']}s "
              f"total={dt:.1f}s" + (f" stats={m.get('semvid')}" if comp == "semvid" else ""))
        assert clip.embeds.shape[0] == clip.keep_indices.shape[0]
        assert all(d.shape[0] == clip.embeds.shape[0] for d in clip.deepstack)

    # 2) semvid invariants: unique + sorted (selection semantics, no merge anchors)
    for name in ("semvid_q", "semvid_nq"):
        ki = arms[name].keep_indices
        assert ki.unique().numel() == ki.numel(), f"{name}: duplicate keep indices!"
        assert bool((ki[1:] > ki[:-1]).all()), f"{name}: keep indices not sorted!"
    print("[ok] semvid keep-indices unique + sorted; deepstack rows aligned")

    # 3) keep-set overlaps
    sets = {n: set(c.keep_indices.cpu().tolist()) for n, c in arms.items()}
    for a, b in (("flashvid", "semvid_q"), ("semvid_q", "semvid_nq")):
        inter = len(sets[a] & sets[b])
        print(f"[overlap] {a} vs {b}: |∩|={inter} "
              f"({inter/max(len(sets[b]),1):.0%} of {b})")

    # 4) evidence concentration per arm
    t_groups = arms["flashvid"].grid_thw[0, 0].item()
    tpf = arms["flashvid"].meta["base_tokens"] // t_groups
    gt = np.asarray(times, dtype=float)
    group_times = gt[: (len(gt) // 2) * 2].reshape(-1, 2).mean(1)[:t_groups]
    for name, clip in arms.items():
        kept_share, ev_share = frame_share(clip.keep_indices, tpf, group_times, ev)
        conc = kept_share / max(ev_share, 1e-9)
        print(f"[evidence] {name}: {kept_share:.1%} of kept tokens in evidence "
              f"(span holds {ev_share:.1%} of groups) -> concentration {conc:.2f}x")

    # 5) end-to-end: answer the MCQ from the semvid clip alone
    prompt = (query + "\n\nBased on the compressed video above, answer with "
              "<answer>X</answer> where X is one of A, B, C, or D.")
    a = assemble(engine, [{"role": "user", "parts": [arms["semvid_q"], prompt]}])
    out = decode(engine, a, max_new_tokens=256)
    print(f"[e2e] ctx={a.embeds.shape[1]} tok; gold={row['answer']}; "
          f"output tail: {out[-200:]!r}")
    print("[smoke_semvid] DONE")


if __name__ == "__main__":
    main()
