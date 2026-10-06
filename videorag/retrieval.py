"""Hybrid retrieval over transcript chunks (dense + BM25, fused with RRF) plus CLIP search over keyframes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from videorag import embeddings
from videorag.config import Settings
from videorag.store import ChunkRow, FrameRow, Store

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Merge several ranked id lists. RRF uses ranks, not raw scores, so BM25 and cosine
    scores never need to be put on a common scale."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


@dataclass
class Source:
    """One piece of evidence handed to the LLM and shown to the user as a citation."""

    n: int
    video_id: str
    video_title: str
    start: float
    end: float
    text: str
    score: float
    kind: str = "transcript"  # transcript | frame
    frame_path: str | None = None
    frame_t: float | None = None

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["frame_url"] = f"/api/frames/{self.video_id}/{self.frame_path.split('/')[-1]}" if self.frame_path else None
        d.pop("frame_path")
        return d


@dataclass
class RetrievalResult:
    sources: list[Source] = field(default_factory=list)


class Retriever:
    def __init__(self, store: Store, settings: Settings):
        self.store = store
        self.settings = settings

    def _bm25(self):
        from rank_bm25 import BM25Okapi

        def build():
            ids, vids, texts = self.store.chunk_texts()
            model = BM25Okapi([tokenize(t) for t in texts]) if texts else None
            return ids, np.asarray(vids, dtype=object), model

        return self.store._cached("bm25", build)

    def _bm25_search(self, query: str, k: int, video_filter: set[str] | None) -> list[int]:
        ids, vids, model = self._bm25()
        if model is None:
            return []
        scores = model.get_scores(tokenize(query))
        if video_filter:
            scores = np.where(np.isin(vids, list(video_filter)), scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [ids[i] for i in order if scores[i] > 0]

    def search(
        self,
        query: str,
        k: int = 6,
        video_ids: list[str] | None = None,
        visual_k: int = 3,
        visual_min_score: float = 0.22,
    ) -> RetrievalResult:
        vf = set(video_ids) if video_ids else None
        pool = max(k * 4, 20)

        # 1) Transcript: dense + lexical, fused.
        q_vec = embeddings.embed_query(query, self.settings.text_model)
        dense = [cid for cid, _ in self.store.chunk_index().search(q_vec, pool, vf)]
        lexical = self._bm25_search(query, pool, vf)
        fused = reciprocal_rank_fusion([dense, lexical])[:k]
        chunks = self.store.chunks_by_id([cid for cid, _ in fused])

        # 2) Visual: CLIP text->image similarity over keyframes. Catches things that are shown, not said.
        frames: list[tuple[FrameRow, float]] = []
        if visual_k > 0:
            clip_q = embeddings.embed_clip_text(query, self.settings.clip_model)
            hits = self.store.frame_index().search(clip_q, visual_k, vf)
            rows = self.store.frames_by_id([fid for fid, _ in hits])
            frames = [(rows[fid], s) for fid, s in hits if s >= visual_min_score and fid in rows]

        titles = {v.id: v.title for v in self.store.list_videos()}
        sources: list[Source] = []
        for cid, score in fused:
            ch: ChunkRow | None = chunks.get(cid)
            if ch is None:
                continue
            mid = (ch.start + ch.end) / 2
            fr = self.store.nearest_frame(ch.video_id, mid)
            sources.append(
                Source(
                    n=len(sources) + 1,
                    video_id=ch.video_id,
                    video_title=titles.get(ch.video_id, ch.video_id),
                    start=ch.start,
                    end=ch.end,
                    text=ch.text,
                    score=round(score, 5),
                    frame_path=fr.path.replace("\\", "/") if fr else None,
                    frame_t=fr.t if fr else None,
                )
            )
        for fr, score in frames:
            # Skip a frame already covered by a transcript source from the same moment.
            if any(s.video_id == fr.video_id and s.start - 5 <= fr.t <= s.end + 5 for s in sources):
                continue
            sources.append(
                Source(
                    n=len(sources) + 1,
                    video_id=fr.video_id,
                    video_title=titles.get(fr.video_id, fr.video_id),
                    start=fr.t,
                    end=fr.t,
                    text="(visual match: keyframe only, no speech at this moment)",
                    score=round(score, 5),
                    kind="frame",
                    frame_path=fr.path.replace("\\", "/"),
                    frame_t=fr.t,
                )
            )
        return RetrievalResult(sources)
