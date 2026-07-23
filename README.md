# longvt_compression

Compression-as-agent-action experiments on zero-shot Qwen3-VL-8B (VideoMME-long).
Research state: see `/home/cfyang/hanklin/RESEARCH.md`.

## What lives here (all ours)

- `fast_agent/` — the self-contained eval harness: local-HF Qwen3-VL agent loop
  with `crop_video` / `compress_video` (in-ViT FlashVID) tools, trajectory
  recording, notebook viz. This replaced the lmms-eval+vLLM+MCP stack (2026-07-12).
  - `analyze_run1.ipynb`, `analyze_fixed20.ipynb` — executed result notebooks
- `eval_logs/` — legacy lmms-eval full-900 run results (2026-07-11/12)
- `examples/eval/` — vLLM server + legacy lmms-eval launchers (port-8010 baseline server)
- `examples/video_tools/mcp_server_flashvid.py` — legacy MCP compression tool
- `lmms_eval_tasks/videomme/videomme_long_tool.yaml` — our custom long-split tool task
- `FLASHVID_TOOL_PLAN.md`, `FLASHVID_INTEGRATION.md` — design docs

## Where the stock LongVT scaffolding went (2026-07-14 cleanup)

Everything byte-identical to the original clone (`/home/cfyang/hanklin/LongVT`)
was moved to `/home/cfyang/hanklin/.trash/longvt_compression_20260714/`
(verl/, recipe/, tests/, assets/, scripts/, data/, custom_*/, requirements*, ...).

To re-import a piece, copy from either location, e.g.:

    cp -r /home/cfyang/hanklin/LongVT/verl .

Note the legacy `examples/eval/run_qwen3vl_*.sh` scripts additionally need
`examples/video_tools/mcp_server.py` and the stock `lmms_eval_tasks/` content
re-imported to run (their lmms-eval fork stack was superseded by `fast_agent/`).
