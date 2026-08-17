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


# LVBench phrases times as M:SS/H:MM:SS inside the question ("What happens from
# 17:16-17:40?") while the tools take SECONDS. Zero-shot Qwen read "17:16" as ~17
# seconds and called crop_video(17,17) six times (qid 2367, 2026-07-25). Annotating
# each clock time with its second-value unifies the two formats without removing any
# information. NOTE for analysis: questions that embed their own timestamp are
# localization-TRIVIAL by construction -- score them separately.
_CLOCK_RE = re.compile(r"\b(\d{1,2}):([0-5]\d)(?::([0-5]\d))?\b")


def _clock_to_seconds(m: re.Match) -> float:
    a, b, c = m.group(1), m.group(2), m.group(3)
    return (int(a) * 3600 + int(b) * 60 + int(c)) if c else (int(a) * 60 + int(b))


def annotate_clock_times(text: str) -> str:
    """Append '(= N s)' after each M:SS / H:MM:SS so clock text and tool seconds agree."""
    def sub(m):
        return f"{m.group(0)} (= {_clock_to_seconds(m):.0f} s)"
    return _CLOCK_RE.sub(sub, text)


UNIFY_TIME_FORMAT = os.environ.get("FA_UNIFY_TIME_FORMAT", "1") == "1"


def format_question(row: dict) -> str:
    opts = "\n".join(row["options"])
    q = row["question"]
    if UNIFY_TIME_FORMAT:
        q = annotate_clock_times(q)
    return f"{q}\n{opts}"


_ANS_RE = re.compile(r"<answer>\s*\(?([A-D])\)?", re.IGNORECASE)
_FALLBACK_RE = re.compile(r"\b([A-D])\b")


# A refusal that enumerates the options ("...I cannot pick one of A, B, C, or D.")
# ends in a bare 'D', which the fallback scan would return as a confident answer --
# deterministically scoring refusals as D. Strip enumerations before the fallback.
_ENUM_RE = re.compile(r"\b[A-D]\s*(?:,\s*[A-D]\s*)+(?:,?\s*(?:or|and)\s*[A-D])?\b",
                      re.IGNORECASE)
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def extract_answer(text: str) -> str | None:
    """Final option letter, or None for a refusal / no answer.

    Callers pass the CONCATENATION of every round's text, so two details matter:
      - the LAST <answer> tag wins, not the first: the finalizer turn appends its
        answer after the earlier rounds, and the model's final word is the answer.
      - enumerations are stripped from the WHOLE text before the tail window is
        taken. Stripping after windowing lets a 200-char cut land inside
        "...between A, B, C, or D", leaving a bare "D" that the fallback scores as
        a confident answer -- i.e. booking a refusal as a wrong letter.
    """
    hits = _ANS_RE.findall(text)
    if hits:
        return hits[-1].upper()
    # An explicit <answer> tag whose content is not an option letter (e.g. "Unknown",
    # "None of the above") is a REFUSAL, not a parse failure -- never guess past it.
    if _ANSWER_TAG_RE.search(text):
        return None
    # fallback: last bare option letter in the final line(s), ignoring enumerations
    hits = _FALLBACK_RE.findall(_ENUM_RE.sub(" ", text.strip())[-200:])
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
                "task_types": list(qt),
                "evidence": list(ev) if ev else None,
                "time_reference": qa.get("time_reference"),
                # 13.1% of LVBench questions state their own evidence timestamp
                # ("What happens from 17:16-17:40?"). Those are localization-TRIVIAL:
                # any arm can hit the right span without searching, which inflates the
                # autonomous arm's recall. Flag so analysis can stratify or exclude.
                "localization_trivial": bool(_CLOCK_RE.search(stem)),
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
