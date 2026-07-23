"""VideoMME long-split loading + scoring helpers."""

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
