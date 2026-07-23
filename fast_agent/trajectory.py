"""Trajectory recording: montages of what the model saw + structured per-round JSON.

A trajectory JSON captures everything needed to replay a sample by eye:
  question/options/gold/pred, the initial skim montage, then per round the
  thinking text, the action taken, and (for tool calls) a montage of the frames
  that tool fed back. viz.py renders these in a notebook.
"""

import json
import math
import os

import numpy as np
import torch
from PIL import Image


def _to_pils(items, max_n: int) -> list:
    """Accept a list of PIL frames OR a (T,C,H,W) uint8 tensor; return <=max_n
    uniformly-spaced PIL frames."""
    if isinstance(items, torch.Tensor):
        T = items.shape[0]
        idx = np.linspace(0, T - 1, min(max_n, T)).astype(int)
        return [
            Image.fromarray(items[i].permute(1, 2, 0).cpu().numpy().astype("uint8"))
            for i in idx
        ]
    items = list(items)
    if len(items) > max_n:
        idx = np.linspace(0, len(items) - 1, max_n).astype(int)
        items = [items[i] for i in idx]
    return items


def save_montage(items, path: str, max_thumbs: int = 48, cols: int = 8,
                 thumb_w: int = 160) -> str | None:
    """Tile frames into a single PNG grid (downsampled). Returns path or None."""
    pils = _to_pils(items, max_thumbs)
    if not pils:
        return None
    w0, h0 = pils[0].size
    th = max(1, int(thumb_w * h0 / w0))
    rows = math.ceil(len(pils) / cols)
    canvas = Image.new("RGB", (cols * thumb_w, rows * th), (18, 18, 18))
    for i, im in enumerate(pils):
        r, c = divmod(i, cols)
        canvas.paste(im.convert("RGB").resize((thumb_w, th)), (c * thumb_w, r * th))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)
    return path


def save_trajectory(traj: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(traj, f, indent=1)
    return path
