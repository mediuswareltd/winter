"""Persistence: SQLite for metadata and text, float32 BLOBs for vectors, an in-memory matrix for search.

Exact (brute-force) cosine search over a NumPy matrix is fast up to a few hundred thousand chunks,
needs no extra service, and gives exact results. `VectorIndex` is the single place to swap in
pgvector, Qdrant or LanceDB if a deployment outgrows that.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    path TEXT,
    duration REAL,
    language TEXT,
    status TEXT NOT NULL,          -- queued | processing | ready | failed
    stage TEXT,
    progress REAL DEFAULT 0,
    error TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    start REAL NOT NULL,
    end REAL NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    t REAL NOT NULL,
    path TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_video ON chunks(video_id);
CREATE INDEX IF NOT EXISTS idx_frames_video ON frames(video_id);
"""


@dataclass
class Video:
    id: str
    title: str
    source: str
    path: str | None
    duration: float | None
    language: str | None
    status: str
    stage: str | None
    progress: float
    error: str | None
    created_at: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class ChunkRow:
    id: int
    video_id: str
    start: float
    end: float
    text: str


@dataclass
class FrameRow:
    id: int
    video_id: str
    t: float
    path: str


class VectorIndex:
    """Normalised row vectors + parallel id list; dot product == cosine similarity."""

    def __init__(self, ids: list[int], video_ids: list[str], matrix: np.ndarray):
        self.ids = np.asarray(ids, dtype=np.int64)
        self.video_ids = np.asarray(video_ids, dtype=object)
        self.matrix = matrix

    def search(self, query: np.ndarray, k: int, video_filter: set[str] | None = None) -> list[tuple[int, float]]:
        if len(self.ids) == 0:
            return []
        scores = self.matrix @ query
        if video_filter:
            mask = np.isin(self.video_ids, list(video_filter))
            scores = np.where(mask, scores, -np.inf)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(self.ids[i]), float(scores[i])) for i in top if np.isfinite(scores[i])]


def _blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


class Store:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._version = 0  # bumped on every write that changes the searchable corpus
        self._cache: dict[str, tuple[int, object]] = {}
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        # One short-lived connection per operation keeps this safe across the API's worker threads.
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- videos -------------------------------------------------------------------------------

    def create_video(self, title: str, source: str) -> Video:
        vid = uuid.uuid4().hex[:12]
        with self._conn() as c:
            c.execute(
                "INSERT INTO videos (id, title, source, status, stage, progress, created_at) VALUES (?,?,?,?,?,?,?)",
                (vid, title, source, "queued", "queued", 0.0, time.time()),
            )
        return self.get_video(vid)  # type: ignore[return-value]

    def update_video(self, vid: str, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE videos SET {cols} WHERE id = ?", (*fields.values(), vid))

    def get_video(self, vid: str) -> Video | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM videos WHERE id = ?", (vid,)).fetchone()
        return Video(**dict(row)) if row else None

    def list_videos(self) -> list[Video]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM videos ORDER BY created_at DESC").fetchall()
        return [Video(**dict(r)) for r in rows]

    def delete_video(self, vid: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM videos WHERE id = ?", (vid,))
        self._bump()

    # ---- content ------------------------------------------------------------------------------

    def add_chunks(self, vid: str, rows: list[tuple[float, float, str]], embeddings: np.ndarray) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM chunks WHERE video_id = ?", (vid,))
            c.executemany(
                "INSERT INTO chunks (video_id, start, end, text, embedding) VALUES (?,?,?,?,?)",
                [(vid, s, e, t, _blob(v)) for (s, e, t), v in zip(rows, embeddings, strict=True)],
            )
        self._bump()

    def add_frames(self, vid: str, rows: list[tuple[float, str]], embeddings: np.ndarray) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM frames WHERE video_id = ?", (vid,))
            c.executemany(
                "INSERT INTO frames (video_id, t, path, embedding) VALUES (?,?,?,?)",
                [(vid, t, p, _blob(v)) for (t, p), v in zip(rows, embeddings, strict=True)],
            )
        self._bump()

    def chunks_by_id(self, ids: list[int]) -> dict[int, ChunkRow]:
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        with self._conn() as c:
            rows = c.execute(f"SELECT id, video_id, start, end, text FROM chunks WHERE id IN ({q})", ids).fetchall()
        return {r["id"]: ChunkRow(**dict(r)) for r in rows}

    def frames_by_id(self, ids: list[int]) -> dict[int, FrameRow]:
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        with self._conn() as c:
            rows = c.execute(f"SELECT id, video_id, t, path FROM frames WHERE id IN ({q})", ids).fetchall()
        return {r["id"]: FrameRow(**dict(r)) for r in rows}

    def nearest_frame(self, vid: str, t: float) -> FrameRow | None:
        with self._conn() as c:
            r = c.execute(
                "SELECT id, video_id, t, path FROM frames WHERE video_id = ? ORDER BY ABS(t - ?) LIMIT 1", (vid, t)
            ).fetchone()
        return FrameRow(**dict(r)) if r else None

    def transcript(self, vid: str) -> list[ChunkRow]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, video_id, start, end, text FROM chunks WHERE video_id = ? ORDER BY start", (vid,)
            ).fetchall()
        return [ChunkRow(**dict(r)) for r in rows]

    # ---- indexes (cached until the corpus changes) --------------------------------------------

    def _bump(self) -> None:
        with self._lock:
            self._version += 1

    def _cached(self, key: str, build):
        with self._lock:
            hit = self._cache.get(key)
            if hit and hit[0] == self._version:
                return hit[1]
            version = self._version
        value = build()
        with self._lock:
            self._cache[key] = (version, value)
        return value

    def _index(self, table: str) -> VectorIndex:
        def build() -> VectorIndex:
            with self._conn() as c:
                rows = c.execute(
                    f"SELECT {table}.id, {table}.video_id, {table}.embedding FROM {table} "
                    f"JOIN videos ON videos.id = {table}.video_id WHERE videos.status = 'ready'"
                ).fetchall()
            if not rows:
                return VectorIndex([], [], np.zeros((0, 0), dtype=np.float32))
            matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            return VectorIndex([r["id"] for r in rows], [r["video_id"] for r in rows], matrix)

        return self._cached(table, build)

    def chunk_index(self) -> VectorIndex:
        return self._index("chunks")

    def frame_index(self) -> VectorIndex:
        return self._index("frames")

    def chunk_texts(self) -> tuple[list[int], list[str], list[str]]:
        """(ids, video_ids, texts) for every ready chunk, used to build the BM25 index."""

        def build():
            with self._conn() as c:
                rows = c.execute(
                    "SELECT chunks.id, chunks.video_id, chunks.text FROM chunks "
                    "JOIN videos ON videos.id = chunks.video_id WHERE videos.status = 'ready' ORDER BY chunks.id"
                ).fetchall()
            return [r["id"] for r in rows], [r["video_id"] for r in rows], [r["text"] for r in rows]

        return self._cached("chunk_texts", build)

    def mark_ready(self, vid: str) -> None:
        self.update_video(vid, status="ready", stage="ready", progress=1.0, error=None)
        self._bump()
