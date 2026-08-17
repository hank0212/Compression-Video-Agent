"""Minimal OpenAI-compatible client for the local Qwen3-VL reader server.

Used by `case_notes.py` to have a VLM read agent trajectories (montages +
reasoning) and write case notes. Start the server with `fast_agent/serve_judge.sh`.

Extracted from the retired `flip_judge.py` (2026-07-15 run2 audit tooling) during
the 2026-07-31 cleanup — these five helpers were its only surviving consumers.
"""

import base64
import os

import requests

DEFAULT_ENDPOINT = os.environ.get("OPENAI_BASE_URL", "http://localhost:8010/v1")


def _trunc(s: str, head: int = 700, tail: int = 500) -> str:
    s = (s or "").strip()
    if len(s) <= head + tail + 20:
        return s
    return s[:head] + f"\n …[{len(s) - head - tail} chars truncated]… \n" + s[-tail:]


def _image_part(path: str | None) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def discover_model(endpoint: str) -> str:
    m = os.environ.get("FA_JUDGE_MODEL")
    if m:
        return m
    r = requests.get(f"{endpoint}/models", timeout=10)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def _chat(endpoint: str, model: str, messages: list, max_tokens: int = 600) -> str:
    r = requests.post(f"{endpoint}/chat/completions", json={
        "model": model, "messages": messages,
        "temperature": 0.0, "max_tokens": max_tokens,
    }, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
