"""Speech-to-text with timestamps, using faster-whisper (CTranslate2) so it runs well on CPU."""

from __future__ import annotations

import wave
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


def load_wav(path: Path) -> np.ndarray:
    """Read the 16 kHz mono 16-bit WAV written by media.extract_audio as float32 in [-1, 1].

    Decoding it ourselves avoids faster-whisper's PyAV dependency, whose API changes between versions.
    """
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path} must be 16 kHz mono 16-bit PCM")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


@lru_cache(maxsize=2)
def _load_model(name: str, device: str):
    from faster_whisper import WhisperModel

    compute_type = "int8" if device in ("cpu", "auto") else "float16"
    return WhisperModel(name, device=device, compute_type=compute_type)


def transcribe(
    audio: Path,
    model_name: str = "base",
    device: str = "auto",
    on_progress: Callable[[float], None] | None = None,
) -> tuple[list[Segment], str]:
    """Return (segments, detected_language). on_progress receives 0..1 as audio is consumed."""
    model = _load_model(model_name, device)
    segments, info = model.transcribe(load_wav(audio), vad_filter=True, beam_size=5)
    out: list[Segment] = []
    for s in segments:  # a lazy generator: transcription happens while we iterate
        text = s.text.strip()
        if text:
            out.append(Segment(float(s.start), float(s.end), text))
        if on_progress and info.duration:
            on_progress(min(1.0, s.end / info.duration))
    return out, info.language
