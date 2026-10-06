from videorag.llm import build_messages, extractive_answer
from videorag.retrieval import Source


def src(n, **kw):
    base = dict(n=n, video_id="v", video_title="Demo", start=60.0 * n, end=60.0 * n + 30, text=f"text {n}", score=1.0)
    return Source(**{**base, **kw})


def test_prompt_contains_numbered_sources_and_timestamps():
    msgs = build_messages("what?", [src(1), src(2)], images=0)
    text = msgs[-1]["content"][-1]["text"]
    assert '[1] "Demo" @ 1:00-1:30' in text
    assert "[2]" in text and text.endswith("Question: what?")


def test_images_are_attached_and_labelled(tmp_path):
    img = tmp_path / "f.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    msgs = build_messages("q", [src(1, frame_path=str(img), frame_t=61.0), src(2, frame_path=str(img))], images=4)
    content = msgs[-1]["content"]
    assert [b["type"] for b in content] == ["text", "image", "text"]  # duplicate frame sent once
    assert "source [1]" in content[0]["text"]


def test_history_is_prepended():
    hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    msgs = build_messages("q", [src(1)], images=0, history=hist)
    assert msgs[:2] == hist and msgs[2]["role"] == "user"


def test_extractive_answer():
    assert "[1]" in extractive_answer([src(1)])
    assert "No matching" in extractive_answer([])
