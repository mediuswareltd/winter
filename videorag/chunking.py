"""Turn Whisper segments into overlapping, time-bounded chunks for retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from videorag.transcribe import Segment


@dataclass(frozen=True)
class Chunk:
    start: float
    end: float
    text: str


def chunk_segments(segments: list[Segment], window: float = 45.0, overlap: float = 10.0) -> list[Chunk]:
    """Group consecutive segments into ~`window`-second chunks.

    Each new chunk starts with the segments from the last `overlap` seconds of the previous one,
    so an answer that straddles a boundary is still retrievable from a single chunk. Segment
    boundaries are never split, so every chunk starts and ends on a real spoken-sentence edge.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if not 0 <= overlap < window:
        raise ValueError("overlap must be in [0, window)")

    chunks: list[Chunk] = []
    i, n = 0, len(segments)
    while i < n:
        j = i
        while j < n and (j == i or segments[j].end - segments[i].start <= window):
            j += 1
        group = segments[i:j]
        chunks.append(Chunk(group[0].start, group[-1].end, " ".join(s.text for s in group)))
        if j >= n:
            break
        # Rewind to the first segment that starts inside the overlap tail, but always move forward.
        tail_start = group[-1].end - overlap
        k = next((x for x in range(i + 1, j) if segments[x].start >= tail_start), j)
        i = k
    return chunks


def format_ts(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
