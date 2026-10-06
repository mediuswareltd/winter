import pytest

from videorag.chunking import chunk_segments, format_ts
from videorag.transcribe import Segment


def segs(n: int, length: float = 5.0) -> list[Segment]:
    return [Segment(i * length, (i + 1) * length, f"s{i}") for i in range(n)]


def test_empty():
    assert chunk_segments([]) == []


def test_single_chunk_when_short():
    chunks = chunk_segments(segs(3), window=45, overlap=10)
    assert len(chunks) == 1
    assert (chunks[0].start, chunks[0].end, chunks[0].text) == (0, 15, "s0 s1 s2")


def test_windows_respect_size_and_overlap():
    chunks = chunk_segments(segs(30), window=30, overlap=10)  # 150 s of speech
    for c in chunks:
        assert c.end - c.start <= 30
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert b.start < a.end, "consecutive chunks should overlap"
        assert b.start > a.start, "chunking must always move forward"
    assert chunks[-1].end == 150
    # every segment is covered
    covered = {w for c in chunks for w in c.text.split()}
    assert covered == {f"s{i}" for i in range(30)}


def test_long_segment_is_never_dropped():
    s = [Segment(0, 100, "long monologue"), Segment(100, 105, "tail")]
    chunks = chunk_segments(s, window=30, overlap=5)
    assert chunks[0].text == "long monologue"
    assert chunks[-1].text.endswith("tail")


def test_zero_overlap():
    chunks = chunk_segments(segs(12), window=20, overlap=0)
    assert [c.text for c in chunks] == ["s0 s1 s2 s3", "s4 s5 s6 s7", "s8 s9 s10 s11"]


@pytest.mark.parametrize("window,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_params(window, overlap):
    with pytest.raises(ValueError):
        chunk_segments(segs(2), window=window, overlap=overlap)


def test_format_ts():
    assert format_ts(0) == "0:00"
    assert format_ts(75.9) == "1:15"
    assert format_ts(3725) == "1:02:05"
