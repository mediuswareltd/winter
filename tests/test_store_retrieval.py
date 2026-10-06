"""Store + retrieval tests with stub embeddings, so they run without downloading any model."""

import numpy as np
import pytest

from videorag import embeddings
from videorag.config import Settings
from videorag.retrieval import Retriever, reciprocal_rank_fusion, tokenize
from videorag.store import Store, VectorIndex

VOCAB = ["pricing", "database", "deploy", "kubernetes", "invoice", "camera"]


def bow(text: str) -> np.ndarray:
    """Bag-of-words over a tiny vocabulary: a deterministic stand-in for a real embedding model."""
    v = np.array([text.lower().count(w) for w in VOCAB], dtype=np.float32) + 1e-3
    return v / np.linalg.norm(v)


@pytest.fixture
def stub_embeddings(monkeypatch):
    monkeypatch.setattr(embeddings, "embed_query", lambda t, m: bow(t))
    monkeypatch.setattr(embeddings, "embed_clip_text", lambda t, m: bow(t))


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "db.sqlite3")
    a = s.create_video("Billing talk", "a.mp4")
    b = s.create_video("Infra talk", "b.mp4")
    a_rows = [(0, 40, "Today we cover pricing tiers"), (40, 80, "Each invoice is generated monthly")]
    b_rows = [(0, 40, "We deploy to kubernetes"), (40, 80, "The database runs on managed postgres")]
    s.add_chunks(a.id, a_rows, np.stack([bow(t) for *_, t in a_rows]))
    s.add_chunks(b.id, b_rows, np.stack([bow(t) for *_, t in b_rows]))
    s.add_frames(b.id, [(200.0, str(tmp_path / "f.jpg"))], np.stack([bow("camera")]))
    s.mark_ready(a.id)
    s.mark_ready(b.id)
    s.ids = (a.id, b.id)
    return s


def test_rrf_prefers_items_ranked_well_in_both_lists():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]])
    assert [i for i, _ in fused][:2] == [1, 3]


def test_tokenize():
    assert tokenize("Hello, GPT-4o world!") == ["hello", "gpt", "4o", "world"]


def test_vector_index_filter_and_order():
    m = np.eye(3, dtype=np.float32)
    idx = VectorIndex([10, 20, 30], ["a", "b", "a"], m)
    assert idx.search(np.array([0, 0, 1], dtype=np.float32), 2)[0][0] == 30
    hits = idx.search(np.array([0, 1, 0], dtype=np.float32), 3, {"a"})
    assert 20 not in [i for i, _ in hits]


def test_hybrid_search_finds_right_chunk(store, stub_embeddings):
    r = Retriever(store, Settings())
    res = r.search("how does the invoice work", k=2, visual_k=0)
    assert res.sources[0].text.startswith("Each invoice")
    assert res.sources[0].video_title == "Billing talk"
    assert [s.n for s in res.sources] == [1, 2]


def test_search_scoped_to_one_video(store, stub_embeddings):
    _, infra = store.ids
    res = Retriever(store, Settings()).search("pricing", k=4, video_ids=[infra], visual_k=0)
    assert res.sources and all(s.video_id == infra for s in res.sources)


def test_visual_hit_is_added_as_frame_source(store, stub_embeddings):
    res = Retriever(store, Settings()).search("camera", k=1, visual_k=1, visual_min_score=0.5)
    kinds = [s.kind for s in res.sources]
    assert "frame" in kinds


def test_videos_not_ready_are_not_searchable(tmp_path, stub_embeddings):
    s = Store(tmp_path / "db.sqlite3")
    v = s.create_video("pending", "x.mp4")
    s.add_chunks(v.id, [(0, 10, "database")], np.stack([bow("database")]))
    assert Retriever(s, Settings()).search("database", visual_k=0).sources == []
    s.mark_ready(v.id)
    assert len(Retriever(s, Settings()).search("database", visual_k=0).sources) == 1


def test_delete_cascades(store):
    a, _ = store.ids
    store.delete_video(a)
    assert store.get_video(a) is None
    assert store.transcript(a) == []
