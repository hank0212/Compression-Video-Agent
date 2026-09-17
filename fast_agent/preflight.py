"""Runtime invariants for a fast_agent arm. Run this BEFORE launching an expensive run.

    /local1/cfyang/miniconda3/envs/vllm/bin/python -m longvt_compression.fast_agent.preflight \
        --base-url http://localhost:8031/v1 --frames 256 --expect-retention 0.25 \
        --proxy-dir /local1/cfyang/hanklin/outputs/lvbench_agent/skim_proxies \
        --out-dir /local1/cfyang/hanklin/outputs/lvbench_agent/lvbench_r3_seed0

Design rule: a check is a HARD FAIL only when its violation makes the run's numbers
meaningless. Everything else is a WARN. Variable-resolution behaviour is legitimate --
tokens-per-frame differs per video because Qwen resizes by aspect ratio -- so nothing
here asserts a fixed token count. What it does assert is that the number the server
actually produced matches the number the *same* server predicted for the *same* input.

Every check reports the value it measured, not just pass/fail: the point is to make you
look at the numbers, not to give you a green light to stop looking.
"""

import argparse
import json
import os
import subprocess
import sys

OK, WARN, FAIL = "ok", "warn", "FAIL"
_RESULTS = []


def _rec(level, name, detail):
    _RESULTS.append((level, name, detail))
    tag = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[level]
    print(f"[{tag}] {name}: {detail}", flush=True)


# --------------------------------------------------------------------------------
# environment

def check_env(expect_vllm="0.19.0"):
    """conda env + vLLM version. `conda run -n vllm` resolves to the WRONG env on this
    box (dvd_tool, python 3.10, vLLM 0.16) -- that confusion produced several notes in
    RESEARCH.md that were written under the wrong version."""
    _rec(OK if sys.version_info[:2] == (3, 12) else FAIL,
         "python", f"{sys.version.split()[0]} (want 3.12 = the `vllm` env)")
    _rec(OK if "envs/vllm/" in sys.executable else FAIL,
         "interpreter", sys.executable)
    try:
        import vllm
        v = vllm.__version__
        _rec(OK if v == expect_vllm else FAIL, "vllm", f"{v} (expect {expect_vllm})")
    except ImportError as e:
        _rec(FAIL, "vllm", f"not importable: {e}")
    try:
        import transformers
        _rec(OK, "transformers", transformers.__version__)
    except ImportError:
        _rec(WARN, "transformers", "not importable")


# --------------------------------------------------------------------------------
# server

def check_server(base_url, expect_model=None, expect_frames=None,
                 expect_pruning=None, expect_max_pixels=None):
    """The server is the only authority on what it was started with. `curl /v1/models`
    returning 200 proves the API process is alive -- it does NOT prove EngineCore is
    (the 2026-08-13 wedge: API 200, EngineCore dead, every request hanging)."""
    import urllib.request
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=5) as r:
            models = json.load(r)
        ids = [m["id"] for m in models.get("data", [])]
        _rec(OK, "server /v1/models", f"{ids}")
        if expect_model and expect_model not in ids:
            _rec(FAIL, "served model name", f"{expect_model!r} not in {ids}")
    except Exception as e:
        _rec(FAIL, "server /v1/models", f"{type(e).__name__}: {e}")
        return

    # A real generation is the only proof EngineCore is alive.
    try:
        from openai import OpenAI
        c = OpenAI(base_url=base_url, api_key="EMPTY", timeout=60, max_retries=0)
        r = c.chat.completions.create(model=expect_model or ids[0],
                                      messages=[{"role": "user", "content": "say ok"}],
                                      max_tokens=4, temperature=0.0)
        _rec(OK, "server generates", f"{r.choices[0].message.content!r} "
                                     f"({r.usage.prompt_tokens} prompt tok)")
    except Exception as e:
        _rec(FAIL, "server generates", f"{type(e).__name__}: {e} "
                                       f"-- API process up but EngineCore may be dead")


def check_server_launch_flags(log_path, expect_frames=None, expect_pruning=None,
                              expect_max_pixels=None):
    """Read the flags off the server's own log. serve_arm.sh is a script that has been
    edited; the log is what actually ran."""
    if not os.path.exists(log_path):
        _rec(WARN, "server log", f"{log_path} not found -- cannot verify launch flags")
        return {}
    line = None
    with open(log_path, errors="replace") as f:
        for ln in f:
            if "non-default args:" in ln:
                line = ln
    if line is None:
        _rec(WARN, "server log", "no 'non-default args' line found")
        return {}
    blob = line.split("non-default args:", 1)[1].strip()
    _rec(OK, "server launch flags", blob[:400])
    got = {}
    for key, want, hard in (("media_io_kwargs", expect_frames, True),
                            ("video_pruning_rate", expect_pruning, True),
                            ("mm_processor_kwargs", expect_max_pixels, False)):
        got[key] = key in blob
    if expect_frames is not None:
        need = f"'num_frames': {expect_frames}"
        _rec(OK if need in blob else FAIL, "media_io_kwargs num_frames",
             f"expect {need!r} -- {'present' if need in blob else 'ABSENT: vLLM will use its own default sampling'}")
    if expect_pruning is not None:
        need = f"'video_pruning_rate': {expect_pruning}"
        _rec(OK if need in blob else FAIL, "video_pruning_rate",
             f"expect {need!r} -- {'present' if need in blob else 'ABSENT'}")
    if expect_max_pixels is not None:
        _rec(WARN, "mm_processor_kwargs max_pixels",
             "NOTE: max_pixels does NOT affect the VIDEO grid in vLLM 0.19 "
             "(verified: any value gives the same grid_thw). Only `size.longest_edge` "
             "does, and it is a WHOLE-VIDEO budget. Do not treat this flag as controlling "
             "video resolution.")
    return got


def check_vidcom2_active(log_path, expect_method="vidcom2"):
    """The plugin prints from inside the process that computes the mask. vLLM runs the
    model in a separate EngineCore process, so a patch applied in the API process is not
    evidence the model uses it -- that print is."""
    if not os.path.exists(log_path):
        _rec(WARN, "vidcom2 proof-of-life", f"{log_path} not found")
        return
    txt = open(log_path, errors="replace").read()
    active = [l for l in txt.splitlines() if "[vidcom2_vllm] ACTIVE in pid=" in l]
    stock = "leaving stock EVS in place" in txt
    if expect_method == "vidcom2":
        _rec(OK if active else FAIL, "vidcom2 proof-of-life",
             active[-1].strip() if active else
             "NO 'ACTIVE in pid=' line -- the mask was computed by stock EVS, not VidCom2")
    else:
        _rec(OK if stock and not active else WARN, "compressor",
             "stock EVS" if stock else "unexpected: VidCom2 lines present")


# --------------------------------------------------------------------------------
# proxies

def check_proxies(proxy_dir, frames, dataset="lvbench", sample=8, tol_frac=0.02):
    """A proxy whose duration drifts from the source's relabels every `<t seconds>`
    marker the model reads -- and the crop tool, which opens the REAL file, obeys the
    model's wrong number silently. This is the check that catches that."""
    from . import data, make_skim_proxy
    rows = data.load_dataset(dataset, n=None, seed=0)
    vids = {}
    for r in rows:
        vids.setdefault(r["videoID"], r["video_path"])
    missing = [v for v in vids if not os.path.exists(make_skim_proxy.proxy_path(proxy_dir, v, frames))]
    _rec(OK if not missing else FAIL, f"proxies present @ {frames}f",
         f"{len(vids)-len(missing)}/{len(vids)} in {proxy_dir}"
         + (f" -- MISSING {missing[:5]}" if missing else ""))

    import random
    rng = random.Random(0)
    checked = rng.sample([v for v in vids if v not in missing], min(sample, len(vids) - len(missing)))
    worst = None
    for v in checked:
        p = make_skim_proxy.proxy_path(proxy_dir, v, frames)
        try:
            src = make_skim_proxy._probe(vids[v])
            pr = make_skim_proxy._probe(p)
        except Exception as e:
            _rec(FAIL, "proxy probe", f"{v}: {e}")
            continue
        drift = abs(pr["duration"] - src["duration"]) / max(src["duration"], 1e-6)
        nf_ok = abs(pr["nb_frames"] - frames) <= 1
        lvl = OK if (drift <= tol_frac and nf_ok) else FAIL
        if worst is None or drift > worst[1]:
            worst = (v, drift, pr, src)
        if lvl == FAIL:
            _rec(FAIL, f"proxy {v}",
                 f"frames={pr['nb_frames']} (want {frames}), duration {pr['duration']:.0f}s "
                 f"vs source {src['duration']:.0f}s -- drift {drift:.1%}")
    if worst:
        v, drift, pr, src = worst
        _rec(OK if drift <= tol_frac else FAIL, f"proxy timeline (worst of {len(checked)})",
             f"{v}: proxy {pr['duration']:.0f}s vs source {src['duration']:.0f}s "
             f"(drift {drift:.2%}), proxy {pr['w']}x{pr['h']}")
        _rec(WARN if max(pr["w"], pr["h"]) <= 448 else OK, "proxy resolution",
             f"{pr['w']}x{pr['h']} -- this is a HARD CEILING on visual detail; "
             f"no server-side setting can recover pixels the proxy already discarded")


# --------------------------------------------------------------------------------
# the measurement that matters: grid + token counts, measured not assumed

def measure_tokens(proxy_path, num_frames, video_pruning_rate=None,
                   model=None, mm_processor_kwargs=None):
    """Run vLLM's REAL multimodal processor on one proxy and return what it produced.

    Returns dict(T, H, W, tpf, pre, kept, ratio, per_frame, prompt_tokens).
    `kept` is counted from the actual <|video_pad|> run in the token ids -- i.e. what the
    LLM will reserve -- not from any formula.
    """
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.video import OpenCVVideoBackend
    from vllm.multimodal.processing.inputs import ProcessorInputs
    from vllm.multimodal.processing.context import TimingContext
    from . import config as fa_config

    model = model or fa_config.MODEL_SNAPSHOT
    mc = ModelConfig(model=model, tokenizer=model, dtype="bfloat16", max_model_len=40960,
                     limit_mm_per_prompt={"image": 768, "video": 2},
                     mm_processor_kwargs=mm_processor_kwargs or {},
                     media_io_kwargs={"video": {"num_frames": num_frames}},
                     video_pruning_rate=video_pruning_rate, enforce_eager=True)
    proc = MULTIMODAL_REGISTRY.create_processor(mc)
    frames, meta = OpenCVVideoBackend.load_bytes(open(proxy_path, "rb").read(),
                                                 num_frames=num_frames)
    items = proc.data_parser.parse_mm_data({"video": [(frames, meta)]})
    tok = proc.info.get_tokenizer()
    # MUST pass token ids, not a str: the str branch of _apply_hf_processor_main lets the
    # HF processor expand the placeholder itself, and vLLM 0.19 then cannot locate the
    # pruning-shortened run. The OpenAI server tokenizes first, so ids is the real path.
    ids = tok.encode("<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
                     "q<|im_end|>\n<|im_start|>assistant\n")
    out = proc.apply(ProcessorInputs(prompt=ids, mm_data_items=items),
                     TimingContext(enabled=False))
    pids = out["prompt_token_ids"]
    g = out["mm_kwargs"]["video"][0]["video_grid_thw"].data
    T, H, W = [int(x) for x in (g[0] if g.ndim == 2 else g)]
    tpf = (H // 2) * (W // 2)
    kept = sum(1 for i in pids if i == 151656)          # <|video_pad|>
    return {"proxy": os.path.basename(proxy_path), "num_frames": num_frames,
            "T": T, "H": H, "W": W, "tpf": tpf, "pre": T * tpf, "kept": kept,
            "ratio": kept / (T * tpf), "per_frame": kept / T,
            "px": (H * 16, W * 16), "prompt_tokens": len(pids),
            "loader_fps": meta["fps"], "loader_duration": meta["duration"],
            "n_sampled": len(meta["frames_indices"])}


def check_tokens(proxy_path, num_frames, expect_retention=None, tol=0.01, **kw):
    m = measure_tokens(proxy_path, num_frames,
                       video_pruning_rate=(None if expect_retention in (None, 1.0)
                                           else 1.0 - expect_retention), **kw)
    _rec(OK if m["T"] == num_frames // 2 else WARN, "temporal grid",
         f"requested {num_frames} raw frames -> grid T={m['T']} "
         f"(expected {num_frames // 2}: temporal_patch_size=2 pairs raw frames)")
    _rec(OK if m["n_sampled"] == num_frames else FAIL, "frames actually sampled",
         f"{m['n_sampled']} (requested {num_frames})")
    _rec(OK, "grid_thw / tokens", f"({m['T']},{m['H']},{m['W']})  "
         f"resized {m['px'][0]}x{m['px'][1]}px  tokens/grid-frame={m['tpf']}  "
         f"pre-compression total={m['pre']:,}")
    if expect_retention is not None and expect_retention < 1.0:
        good = abs(m["ratio"] - expect_retention) <= tol
        _rec(OK if good else FAIL, "actual retention",
             f"{m['ratio']:.4f} (requested {expect_retention}) -- {m['kept']:,} of {m['pre']:,} "
             f"survive = {m['per_frame']:.1f} tokens per grid-frame")
        _rec(WARN if m["per_frame"] < 60 else OK, "surviving tokens/grid-frame",
             f"{m['per_frame']:.1f}. VidCom2's validated operating point is ~180 "
             f"(32 frames x 720 tok/frame x 0.25). Below ~60 you are testing the "
             f"compressor far outside the regime its numbers were measured in.")
    else:
        _rec(OK if m["ratio"] == 1.0 else FAIL, "no compression",
             f"retention {m['ratio']:.4f} (want 1.0) -- "
             f"{m['per_frame']:.1f} tokens per grid-frame")
    _rec(OK, "skim prompt tokens", f"~{m['prompt_tokens']:,} for the video alone "
         f"(add the question, the tool schema block, and 6,016 per crop round)")
    return m


# --------------------------------------------------------------------------------
# prompt + output dir

def check_prompt(argv_extra):
    """Print the exact turn-1 prompt. The 2026-08-13 run1 was invalid for one reason:
    a no-tool arm was handed the tool prompt. `--print-prompt` would have shown it."""
    cmd = [sys.executable, "-m", "longvt_compression.fast_agent.run_agent",
           "--print-prompt"] + list(argv_extra)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/cfyang/hanklin")
    if r.returncode != 0:
        _rec(FAIL, "print-prompt", r.stderr.strip()[-400:])
        return
    print("\n" + "-" * 78)
    print(r.stdout)
    print("-" * 78)
    has_tool_text = "crop_video" in r.stdout and "Think first" in r.stdout
    has_schema = '"function"' in r.stdout
    if has_tool_text and not has_schema:
        _rec(FAIL, "prompt/tool agreement",
             "prompt tells the model to call crop_video but NO tool schema is attached "
             "-- this is exactly the run1 bug. A no-tool arm must get the no-tool prompt.")
    else:
        _rec(OK, "prompt/tool agreement",
             f"tool prompt={has_tool_text}, schemas attached={has_schema}")


def check_out_dir(out_dir, expect_resume=False):
    p = os.path.join(out_dir, "results.jsonl")
    if not os.path.exists(p):
        _rec(OK, "output dir", f"{out_dir} is fresh (no results.jsonl)")
        return
    n = err = 0
    with open(p) as f:
        for ln in f:
            if not ln.strip():
                continue
            n += 1
            try:
                err += "error" in json.loads(ln)
            except json.JSONDecodeError:
                pass
    lvl = OK if expect_resume else FAIL
    _rec(lvl, "output dir NOT clean",
         f"{out_dir} already holds {n} rows ({err} with an 'error' key). "
         f"run_agent SKIPS these. {'Resuming as intended.' if expect_resume else 'On a fresh run this contaminates the arm -- purge it.'}")


def check_results_errors(out_dir):
    """The failure that has now bitten this chapter twice: rows written as
    pred=None, correct=False because the server was unreachable, then quoted as accuracy."""
    p = os.path.join(out_dir, "results.jsonl")
    if not os.path.exists(p):
        return
    rows = [json.loads(l) for l in open(p) if l.strip()]
    err = [r for r in rows if "error" in r]
    none_pred = [r for r in rows if r.get("pred") is None]
    lvl = OK if not err else FAIL
    _rec(lvl, "errors in results.jsonl",
         f"{len(err)}/{len(rows)} rows carry an 'error' key, {len(none_pred)} have pred=None. "
         + (f"Example: {err[0].get('error','')[:120]}" if err else "Accuracy is quotable."))


# --------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default="qwen3vl")
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--expect-retention", type=float, default=None,
                    help="1.0 for an uncompressed arm, 0.25 for VidCom2 r=0.25")
    ap.add_argument("--expect-method", default="evs", choices=["evs", "vidcom2"])
    ap.add_argument("--proxy-dir", default=None)
    ap.add_argument("--proxy-sample", type=int, default=6)
    ap.add_argument("--measure-video", default=None,
                    help="one proxy mp4 to push through the real processor")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--expect-resume", action="store_true")
    ap.add_argument("--prompt-args", nargs=argparse.REMAINDER, default=[],
                    help="everything after this is passed to run_agent --print-prompt")
    a = ap.parse_args()

    print("=" * 78)
    print("PREFLIGHT")
    print("=" * 78)
    check_env()
    if a.server_log:
        check_server_launch_flags(a.server_log, a.frames,
                                  None if a.expect_retention in (None, 1.0)
                                  else 1.0 - a.expect_retention, True)
        check_vidcom2_active(a.server_log, a.expect_method)
    if a.base_url:
        check_server(a.base_url, a.model)
    if a.proxy_dir and a.frames:
        check_proxies(a.proxy_dir, a.frames, sample=a.proxy_sample)
    if a.measure_video and a.frames:
        check_tokens(a.measure_video, a.frames, a.expect_retention)
    if a.prompt_args:
        check_prompt(a.prompt_args)
    if a.out_dir:
        check_out_dir(a.out_dir, a.expect_resume)
        check_results_errors(a.out_dir)

    print("=" * 78)
    n_fail = sum(1 for l, _, _ in _RESULTS if l == FAIL)
    n_warn = sum(1 for l, _, _ in _RESULTS if l == WARN)
    print(f"{len(_RESULTS)} checks | {n_fail} FAIL | {n_warn} warn")
    if n_fail:
        print("\nFAILURES:")
        for l, n, d in _RESULTS:
            if l == FAIL:
                print(f"  - {n}: {d}")
    print("=" * 78)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
