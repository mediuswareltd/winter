"""Keyframe extraction: one frame per visual scene, plus a frame at least every N seconds."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Keyframe:
    t: float
    path: Path


def _signature(frame: np.ndarray) -> np.ndarray:
    """A small normalised HSV histogram; cheap to compute and robust to noise and small motion."""
    hsv = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def extract_keyframes(
    video: Path,
    out_dir: Path,
    scene_threshold: float = 0.35,
    max_gap: float = 20.0,
    sample_fps: float = 2.0,
    max_width: int = 768,
    on_progress: Callable[[float], None] | None = None,
) -> list[Keyframe]:
    """Sample the video at `sample_fps` and keep a frame whenever the scene changes
    (Bhattacharyya histogram distance > scene_threshold) or `max_gap` seconds have passed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, int(round(fps / sample_fps)))

    frames: list[Keyframe] = []
    last_sig: np.ndarray | None = None
    last_t = -1e9
    idx = 0
    while True:
        ok = cap.grab()  # grab() skips decoding frames we do not sample
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = idx / fps
            sig = _signature(frame)
            changed = last_sig is None or cv2.compareHist(last_sig, sig, cv2.HISTCMP_BHATTACHARYYA) > scene_threshold
            if changed or t - last_t >= max_gap:
                h, w = frame.shape[:2]
                if w > max_width:
                    frame = cv2.resize(frame, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
                path = out_dir / f"{t:010.2f}.jpg"
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frames.append(Keyframe(t, path))
                last_sig, last_t = sig, t
            if on_progress and total:
                on_progress(min(1.0, idx / total))
        idx += 1
    cap.release()
    return frames
