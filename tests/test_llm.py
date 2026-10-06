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


def _ollama_settings(**kw):
    from videorag.config import Settings

    return Settings(**{"llm_provider": "ollama", "ollama_url": "127.0.0.1:11434", "ollama_model": "tiny:1b", **kw})


def test_provider_selection(monkeypatch):
    import videorag.llm as llm

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert llm.active_llm(_ollama_settings(llm_provider="auto"))[0] == "claude"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setattr(llm, "_ollama_ready", lambda s: False)
    assert llm.active_llm(_ollama_settings(llm_provider="auto")) == ("none", None)
    monkeypatch.setattr(llm, "_ollama_ready", lambda s: True)
    assert llm.active_llm(_ollama_settings(llm_provider="auto")) == ("ollama", "tiny:1b")


def test_ollama_streams_text_and_sends_sources(monkeypatch):
    import json

    import httpx

    import videorag.llm as llm

    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        lines = [{"message": {"content": "Latency fell "}}, {"message": {"content": "[1]."}}, {"done": True}]
        return httpx.Response(200, text="\n".join(json.dumps(x) for x in lines))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(llm.httpx, "stream", lambda method, url, **kw: client.stream(method, url, json=kw["json"]))
    out = "".join(llm.stream_answer("how fast?", [src(1)], _ollama_settings()))
    assert out == "Latency fell [1]."
    assert seen["url"] == "http://127.0.0.1:11434/api/chat"
    assert seen["body"]["model"] == "tiny:1b" and seen["body"]["messages"][0]["role"] == "system"
    assert '[1] "Demo"' in seen["body"]["messages"][-1]["content"]


def test_ollama_down_falls_back_to_passages(monkeypatch):
    import httpx

    import videorag.llm as llm

    def boom(*a, **kw):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(llm.httpx, "stream", boom)
    out = "".join(llm.stream_answer("q", [src(1)], _ollama_settings()))
    assert "Local model error" in out and "[1] text 1" in out
