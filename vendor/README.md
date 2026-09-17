# vendor/ — work that lives outside this repo and cannot be pushed upstream

Everything here was sitting **uncommitted or untracked inside someone else's clone** on the
original box (`kraken`, `/home/cfyang/hanklin/`). None of it can be pushed to its own origin,
because we do not own those remotes. It is checked in here so a fresh machine can rebuild the
serving stack without the original filesystem.

Apply each patch onto the exact upstream commit named below. These are the commits the
measurements in `RESEARCH.md` were taken against; a different upstream base is untested.

| what | upstream | base commit |
|---|---|---|
| `patches/FlashVID.diff` | `github.com/Fanziyang-v/FlashVID` | `07fd9421dac04f5946239760140948d2f972ba8c` |
| `patches/flashvid_plugin-worktree.diff` | *(no remote — see `flashvid_plugin.bundle`)* | `982add5e20d82f7fd2332de68c60b93f6aa6b8a0` |
| `longvt_demo/` | `github.com/EvolvingLMMs-Lab/LongVT` | `08d755b973e4ad990ac5cdd64fb992c804d840db` |

Two more clones are dirty on the original box but are **not** needed by anything in this
chapter, so they are deliberately not vendored: `videoarm`
(`671a8420d0bfc92e4e16b8498e7af06bdd28f18e`, untracked analysis scripts from the previous
chapter) and `DVD` (`github.com/joeyy5588/DVD`). The one thing this chapter *does* use from
`videoarm` is `videoarm/video/aks_select.py::_CLIPScorer` — see `fast_agent/signal_auc.py`,
which adds the repo to `sys.path` at import time.

---

## `flashvid_plugin.bundle` — read this one first

This is the **only copy that exists anywhere.** `flashvid_plugin/` was a local git repo with
**no remote configured**, so it was one disk failure away from being gone. The bundle carries
its full history (2 commits), not just a snapshot.

```bash
git clone /path/to/vendor/flashvid_plugin.bundle flashvid_plugin
cd flashvid_plugin && git apply /path/to/vendor/patches/flashvid_plugin-worktree.diff
```

It is the FlashVID-for-vLLM plugin: a reversible, EVS-compatible mask swap. Same trick as
`vidcom2_vllm/` and it predates it. **The current chapter does not use it** — VidCom2
replaced FlashVID on 2026-08-06 and every FlashVID compression log before that date is
archived (see `RESEARCH.md § Archive`). It is preserved because the mask-swap mechanism is
reusable, not because anything pending depends on it.

## `patches/FlashVID.diff`

Four files, 196 insertions. The two that matter:

- `flashvid/modeling_qwen3_vl.py` + `flashvid/utils.py` — the Qwen3-VL port of the compressor.
- `lmms-eval/lmms_eval/models/chat/async_openai.py` (+167 lines) — the async OpenAI client
  used to drive an lmms-eval task against a vLLM endpoint instead of in-process HF.

## `longvt_demo/` — LongVT-RFT serving

Untracked files from `LongVT/demo/`. `serve_longvt.sh` is the launcher referenced by
`CLAUDE.md` for the **LongVT-RFT** checkpoint (Qwen2.5-VL-7B fine-tune, `/local1/cfyang/LongVT-RFT`,
16 GB) — the *trained* tool-use policy used as a comparison arm. See
`RESEARCH.md § LongVT-RFT as an extra arm`, and note the confound recorded there: LongVT-RFT
is Qwen2.5-VL-7B while every other arm is Qwen3-VL-8B, so it is a different backbone, not an
ablation.

`longvt_interleaved_demo.ipynb` (990 KB, with outputs) and the two `serve*.log` files were
left behind on purpose — they are large and reproduce nothing.
