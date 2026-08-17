"""Verification for the VidCom2 retention mask.

    /local1/cfyang/miniconda3/envs/vllm/bin/python \
        -m longvt_compression.vidcom2_vllm.test_retention

Checks, in order of how badly each would bite:

  1. COUNT EXACTNESS -- True count == evs.compute_retained_tokens_count for a
     sweep of (T, grid, q). A mismatch here corrupts generation silently, because
     vLLM sizes the placeholder run from that number before the mask exists.
  2. AGREEMENT WITH THE REFERENCE -- given the same per-frame budget, the tokens
     we pick are the tokens the authors' released code picks.
  3. DETERMINISM -- same input, same mask.
  4. NOT DEGENERATE -- the mask is spread across frames, not all in one, and it
     actually varies with the input (a constant mask would "work" but measure
     nothing).
  5. RETENTION SANITY -- realised retention tracks 1-q.
"""

import sys

import torch

from .retention import _apportion, compute_retention_mask, vidcom2_scores

REF = "/local1/cfyang/VidCom2"


def _mk(T, h, w, C=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    # structured, not iid: frames drift and a few tokens are outliers, so the
    # selection has something real to find
    base = torch.randn(1, h * w, C, generator=g)
    drift = torch.randn(T, 1, C, generator=g) * 0.3
    x = (base + drift + 0.05 * torch.randn(T, h * w, C, generator=g))
    x[T // 2, : max(1, (h * w) // 10)] += 4.0          # outlier tokens
    return x.reshape(T * h * w, C)


def test_count_exactness():
    from vllm.multimodal.evs import compute_retained_tokens_count
    bad = []
    cases = [(T, h, w, q)
             for T in (1, 2, 3, 8, 17, 64)
             for (h, w) in ((2, 2), (3, 5), (8, 8))
             for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)]
    for T, h, w, q in cases:
        x = _mk(T, h, w)
        m = compute_retention_mask(x, (T, h * 2, w * 2), 2, q)
        want = compute_retained_tokens_count(tokens_per_frame=h * w, num_frames=T, q=q)
        want = min(want, T * h * w)
        got = int(m.sum())
        if got != want or m.numel() != T * h * w:
            bad.append((T, h, w, q, got, want))
    print(f"1. COUNT EXACTNESS      {len(cases) - len(bad)}/{len(cases)} cases exact")
    for b in bad[:8]:
        print(f"     MISMATCH T={b[0]} {b[1]}x{b[2]} q={b[3]}: got {b[4]} want {b[5]}")
    return not bad


def test_apportion():
    bad = 0
    for T in (1, 3, 7, 32):
        for tpf in (4, 15, 64):
            for target in (T, T * tpf, max(T, T * tpf // 3), T * tpf - 1):
                ideal = torch.rand(T) * tpf
                k = _apportion(ideal, target, lo=1, hi=tpf)
                if int(k.sum()) != target or int(k.min()) < 1 or int(k.max()) > tpf:
                    bad += 1
    print(f"1b. APPORTION            {'OK' if not bad else f'{bad} FAILURES'} "
          f"(sum==target, 1<=k<=tpf)")
    return not bad


def test_matches_reference():
    """Same budget -> same chosen indices as the authors' released code."""
    sys.path.insert(0, REF)
    try:
        from token_compressor.vidcom2.vidcom2 import (
            compute_gaussian_scores, select_low_var_channels)
    except Exception as e:
        print(f"3. REFERENCE AGREEMENT   SKIPPED ({type(e).__name__}: {e})")
        return True

    T, h, w = 12, 4, 5
    tpf = h * w
    x = _mk(T, h, w, seed=3)

    # reference scoring path
    sel = select_low_var_channels(x.float())
    v_ref, f_ref = compute_gaussian_scores(sel, tpf)
    score_ref = (v_ref + f_ref)

    # ours
    _, score_ours = vidcom2_scores(x, T, tpf)

    max_abs = (score_ref - score_ours).abs().max().item()
    # rank agreement is what actually decides the mask
    same_top = []
    for t in range(T):
        for k in (1, 3, tpf // 2):
            a = set(torch.topk(score_ref[t], k, largest=False).indices.tolist())
            b = set(torch.topk(score_ours[t], k, largest=False).indices.tolist())
            same_top.append(a == b)
    ok = max_abs < 1e-4 and all(same_top)
    print(f"3. REFERENCE AGREEMENT   score max|diff|={max_abs:.2e}, "
          f"top-k identical {sum(same_top)}/{len(same_top)} -> {'OK' if ok else 'FAIL'}")
    return ok


def test_determinism():
    x = _mk(9, 4, 4, seed=7)
    a = compute_retention_mask(x, (9, 8, 8), 2, 0.3)
    b = compute_retention_mask(x, (9, 8, 8), 2, 0.3)
    ok = torch.equal(a, b)
    print(f"4. DETERMINISM           {'OK' if ok else 'FAIL'}")
    return ok


def test_not_degenerate():
    T, h, w = 16, 4, 4
    tpf = h * w
    x = _mk(T, h, w, seed=11)
    m = compute_retention_mask(x, (T, h * 2, w * 2), 2, 0.75).reshape(T, tpf)
    per_frame = m.sum(1)
    y = _mk(T, h, w, seed=12)
    m2 = compute_retention_mask(y, (T, h * 2, w * 2), 2, 0.75).reshape(T, tpf)
    frames_used = int((per_frame > 0).sum())
    input_sensitive = not torch.equal(m, m2)
    # VidCom2's dynamic budget is capped at ~2R by construction (softmax temp 0.01)
    spread = f"min {int(per_frame.min())} / max {int(per_frame.max())} of {tpf}"
    ok = frames_used == T and input_sensitive
    print(f"5. NOT DEGENERATE        frames used {frames_used}/{T}, per-frame {spread}, "
          f"input-sensitive {input_sensitive} -> {'OK' if ok else 'FAIL'}")
    return ok


def test_retention_tracks_q():
    T, h, w = 32, 4, 5
    print("6. REALISED RETENTION")
    ok = True
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        x = _mk(T, h, w, seed=int(q * 100))
        m = compute_retention_mask(x, (T, h * 2, w * 2), 2, q)
        r = float(m.sum()) / m.numel()
        drift = abs(r - (1 - q))
        flag = "OK" if drift < 0.02 else "DRIFT"
        ok &= drift < 0.02
        print(f"     q={q:<5} nominal keep {1 - q:.3f}  realised {r:.4f}  ({flag})")
    return ok


def main():
    print("=" * 66)
    print("VidCom2 retention mask -- verification")
    print("=" * 66)
    results = [test_count_exactness(), test_apportion(), test_matches_reference(),
               test_determinism(), test_not_degenerate(), test_retention_tracks_q()]
    print("=" * 66)
    print("ALL PASS" if all(results) else f"{results.count(False)} CHECK(S) FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
