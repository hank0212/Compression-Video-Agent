"""Dataset loading + scoring helpers (VideoMME long split, LVBench)."""

import json
import os
import re

import cv2
import pandas as pd

from . import config


def load_long_split(n: int | None = None, seed: int = 0) -> list[dict]:
    """Rows of the long split whose video exists locally. Stable order; optional
    stratified-ish head sample (shuffle with fixed seed, then take n)."""
    df = pd.read_parquet(config.VIDEOMME_PARQUET)
    df = df[df["duration"] == "long"].copy()
    have = {f[:-4] for f in os.listdir(config.VIDEO_DIR) if f.endswith(".mp4")}
    df = df[df["videoID"].isin(have)]
    if n is not None:
        df = df.sample(frac=1.0, random_state=seed).head(n)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "question_id": r["question_id"],
                "videoID": r["videoID"],
                "video_path": os.path.join(config.VIDEO_DIR, r["videoID"] + ".mp4"),
                "question": r["question"],
                "options": list(r["options"]),
                "answer": r["answer"],
                "task_type": r["task_type"],
            }
        )
    return rows


def video_duration(path: str) -> float:
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return n / fps if fps > 0 else 0.0
    finally:
        cap.release()


def format_question(row: dict) -> str:
    opts = "\n".join(row["options"])
    return f"{row['question']}\n{opts}"


_ANS_RE = re.compile(r"<answer>\s*\(?([A-D])\)?", re.IGNORECASE)
_FALLBACK_RE = re.compile(r"\b([A-D])\b")


def extract_answer(text: str) -> str | None:
    m = _ANS_RE.search(text)
    if m:
        return m.group(1).upper()
    # fallback: last bare option letter in the final line(s)
    tail = text.strip()[-200:]
    hits = _FALLBACK_RE.findall(tail)
    return hits[-1].upper() if hits else None


# ---------------------------------------------------------------------------
# LVBench (long-video QA with per-question GT evidence spans)

_TIME_TOKEN_BAD = {"", "none"}
# Strict option marker: "(A) text" (LVBench) OR "A. text" / "A) text" (VideoMME style).
# NOT a bare leading letter — otherwise a stem like "After.." / "An.." matches as "A".
_OPT_RE = re.compile(r"^\s*(?:\(([A-D])\)|([A-D])[.)])\s*(.*)$")


def parse_time_reference(tr: str) -> tuple[float, float] | None:
    """Parse LVBench `time_reference` ("MM:SS-MM:SS", tolerates HH:MM:SS) into
    (start_sec, end_sec). Returns None for malformed refs (e.g. "08:17-None",
    "None-None"). Zero-/negative-width spans pass through unchanged — callers apply a
    min-width floor. (LVBench data: all refs are MM:SS, 2 malformed, 263 zero-width.)"""
    if not isinstance(tr, str) or "-" not in tr:
        return None
    a, b = tr.split("-", 1)

    def _to_s(tok: str):
        parts = [p.strip() for p in tok.strip().split(":")]
        if any(p.lower() in _TIME_TOKEN_BAD for p in parts):
            return None
        try:
            xs = [int(p) for p in parts]
        except ValueError:
            return None
        if len(xs) == 1:
            return float(xs[0])
        if len(xs) == 2:
            return float(xs[0] * 60 + xs[1])
        if len(xs) == 3:
            return float(xs[0] * 3600 + xs[1] * 60 + xs[2])
        return None

    s, e = _to_s(a), _to_s(b)
    if s is None or e is None:
        return None
    return (s, e)


def _opt_match(ln: str) -> tuple[str, str] | None:
    """(letter, text) if the line is an option marker, else None."""
    m = _OPT_RE.match(ln)
    if not m:
        return None
    return (m.group(1) or m.group(2), m.group(3).strip())


def _split_lvbench_question(text: str) -> tuple[str, list[str]]:
    """LVBench packs the options into the question as `\\n(A) ..\\n(B) ..`. Split into
    (stem, ["A. ..", "B. ..", "C. ..", "D. .."]) normalized to VideoMME's "LETTER. text"
    style so format_question renders identically. Falls back to (full_text, []) if the
    trailing A-D block is not cleanly present."""
    lines = text.split("\n")
    start = next((i for i, ln in enumerate(lines)
                  if (om := _opt_match(ln)) and om[0] == "A"), None)
    if start is None:
        return text.strip(), []
    stem = "\n".join(lines[:start]).strip()
    opts = [f"{om[0]}. {om[1]}" for ln in lines[start:] if (om := _opt_match(ln))]
    if [o[0] for o in opts] != ["A", "B", "C", "D"]:
        return text.strip(), []
    return stem, opts


def load_lvbench(n: int | None = None, seed: int = 0) -> list[dict]:
    """LVBench rows whose video exists locally, in the same schema load_long_split
    returns PLUS `evidence` ([start_s, end_s] GT span, or None if the ref is malformed)
    and the raw `time_reference` string. Stable order (by uid); optional seeded head
    sample identical across arms."""
    with open(config.LVBENCH_META) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    have = {f[:-4] for f in os.listdir(config.LVBENCH_VIDEO_DIR) if f.endswith(".mp4")}
    rows = []
    for r in recs:
        key = r["key"]
        if key not in have:
            continue
        vp = os.path.join(config.LVBENCH_VIDEO_DIR, key + ".mp4")
        for qa in r["qa"]:
            stem, opts = _split_lvbench_question(qa["question"])
            ev = parse_time_reference(qa.get("time_reference", ""))
            qt = qa.get("question_type") or []
            rows.append({
                "question_id": str(qa["uid"]),
                "videoID": key,
                "video_path": vp,
                "question": stem,
                "options": opts,
                "answer": str(qa["answer"]).strip().upper(),
                "task_type": qt[0] if qt else "unknown",
                "evidence": list(ev) if ev else None,
                "time_reference": qa.get("time_reference"),
            })
    rows.sort(key=lambda x: int(x["question_id"]))
    if n is not None:
        import random
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:n]
    return rows


def load_dataset(name: str, n: int | None = None, seed: int = 0) -> list[dict]:
    """Dispatch to a per-benchmark loader. All loaders return the same row schema
    ({question_id, videoID, video_path, question, options, answer, task_type}); LVBench
    additionally carries `evidence`/`time_reference` for the oracle arm."""
    if name == "videomme":
        return load_long_split(n=n, seed=seed)
    if name == "lvbench":
        return load_lvbench(n=n, seed=seed)
    raise ValueError(f"unknown dataset: {name!r} (expected 'videomme' or 'lvbench')")
