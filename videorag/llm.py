"""Grounded answering: transcript excerpts (+ keyframe images for Claude) in, cited answer out.

Answers come from Claude when an API key is set, otherwise from a local Ollama model if one is running.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path

import anthropic
import httpx

from videorag.chunking import format_ts
from videorag.config import Settings
from videorag.retrieval import Source

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You answer questions about a library of videos, using only the evidence provided.

The evidence is a numbered list of sources. Each one is a transcript excerpt (what was said, with \
timestamps) and, for some, a keyframe image of what was on screen at that moment. Images are labelled \
with the source number they belong to.

How to answer:
- Ground every factual statement in the sources and cite them inline as [n], e.g. "The speaker \
recommends batching requests [2][5]." Cite the specific source, not a range.
- Use the images when the question is about what is shown (slides, code, diagrams, people, objects). \
Say so when a detail comes from an image rather than speech.
- If the sources do not contain the answer, say that plainly and mention what the videos do cover. \
Do not fill gaps from general knowledge without labelling it as such.
- Transcripts come from automatic speech recognition and may misspell names or jargon; read through \
obvious recognition errors.
- Be concise: lead with the direct answer, then supporting detail. Use short paragraphs or bullets."""


def _sources_block(sources: list[Source]) -> str:
    lines = []
    for s in sources:
        span = format_ts(s.start) if s.kind == "frame" else f"{format_ts(s.start)}-{format_ts(s.end)}"
        lines.append(f'[{s.n}] "{s.video_title}" @ {span}\n{s.text}')
    return "\n\n".join(lines)


def _image_blocks(sources: list[Source], limit: int) -> list[dict]:
    blocks: list[dict] = []
    seen: set[str] = set()
    for s in sources:
        if len(seen) >= limit:
            break
        if not s.frame_path or s.frame_path in seen or not Path(s.frame_path).exists():
            continue
        seen.add(s.frame_path)
        data = base64.standard_b64encode(Path(s.frame_path).read_bytes()).decode()
        blocks.append({"type": "text", "text": f"Keyframe for source [{s.n}] at {format_ts(s.frame_t or s.start)}:"})
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
    return blocks


def build_messages(question: str, sources: list[Source], images: int, history: list[dict] | None = None) -> list[dict]:
    content: list[dict] = _image_blocks(sources, images)
    content.append(
        {"type": "text", "text": f"<sources>\n{_sources_block(sources)}\n</sources>\n\nQuestion: {question}"}
    )
    return [*(history or []), {"role": "user", "content": content}]


def build_text_messages(question: str, sources: list[Source], history: list[dict] | None = None) -> list[dict]:
    """Plain-text chat messages for local models, which get the transcripts but not the keyframe images."""
    user = f"<sources>\n{_sources_block(sources)}\n</sources>\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, *(history or []), {"role": "user", "content": user}]


def _ollama_base(settings: Settings) -> str:
    url = settings.ollama_url.rstrip("/")
    return url if "://" in url else f"http://{url}"


_ollama_probe: tuple[float, bool] = (0.0, False)


def _ollama_ready(settings: Settings) -> bool:
    """Whether Ollama is running and has the configured model pulled. Cached for a few seconds."""
    global _ollama_probe
    checked, ok = _ollama_probe
    if time.monotonic() - checked < 5:
        return ok
    try:
        tags = httpx.get(f"{_ollama_base(settings)}/api/tags", timeout=0.5).json().get("models", [])
        wanted = settings.ollama_model if ":" in settings.ollama_model else f"{settings.ollama_model}:latest"
        ok = any(m.get("name") == wanted for m in tags)
    except (httpx.HTTPError, ValueError):
        ok = False
    _ollama_probe = (time.monotonic(), ok)
    return ok


def active_llm(settings: Settings) -> tuple[str, str | None]:
    """(provider, model) that will write the next answer; provider is "claude", "ollama" or "none"."""
    choice = settings.llm_provider.lower()
    if choice == "claude" or (choice == "auto" and os.environ.get("ANTHROPIC_API_KEY")):
        return "claude", settings.claude_model
    if choice == "ollama" or (choice == "auto" and _ollama_ready(settings)):
        return "ollama", settings.ollama_model
    return "none", None


def _stream_ollama(messages: list[dict], settings: Settings) -> Iterator[str]:
    body = {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.2, "num_ctx": 8192},
    }
    timeout = httpx.Timeout(10.0, read=300.0)  # CPU inference can pause a while before the first token
    with httpx.stream("POST", f"{_ollama_base(settings)}/api/chat", json=body, timeout=timeout) as r:
        if r.status_code != 200:
            r.read()
            raise RuntimeError(f"Ollama returned {r.status_code}: {r.text[:200]}")
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                raise RuntimeError(f"Ollama: {chunk['error']}")
            if text := chunk.get("message", {}).get("content"):
                yield text
            if chunk.get("done"):
                if chunk.get("done_reason") == "length":
                    yield "\n\n_(answer truncated)_"
                return


def extractive_answer(sources: list[Source]) -> str:
    """Used when no model is available: show the best-matching moments instead of a generated answer."""
    if not sources:
        return "No matching moments found in the indexed videos."
    out = ["Most relevant moments (no LLM configured, showing retrieved passages):\n"]
    for s in sources[:4]:
        out.append(f"- [{s.n}] {s.text[:280]}{'...' if len(s.text) > 280 else ''}")
    return "\n".join(out)


def stream_answer(
    question: str, sources: list[Source], settings: Settings, history: list[dict] | None = None
) -> Iterator[str]:
    """Yield answer text as it is generated. Falls back to an extractive answer if the API is unreachable."""
    if not sources:
        yield "I could not find anything relevant to that in the indexed videos."
        return
    provider, _ = active_llm(settings)
    if provider == "none":
        yield extractive_answer(sources)
        return
    if provider == "ollama":
        try:
            yield from _stream_ollama(build_text_messages(question, sources, history), settings)
        except (httpx.HTTPError, RuntimeError) as e:
            log.warning("Local model unavailable (%s); returning extractive answer", e)
            yield f"\n\n_Local model error: {e}. Showing retrieved passages instead._\n\n"
            yield extractive_answer(sources)
        return
    try:
        client = anthropic.Anthropic()
        with client.beta.messages.stream(
            model=settings.claude_model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=build_messages(question, sources, settings.images_per_answer, history),
            output_config={"effort": settings.claude_effort},
            # If a safety classifier declines, the API re-runs the request on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
        if final.stop_reason == "refusal":
            yield "\n\n_The model declined to answer this question._"
        elif final.stop_reason == "max_tokens":
            yield "\n\n_(answer truncated)_"
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, TypeError) as e:
        # TypeError: the SDK found no credentials at all.
        log.warning("Claude unavailable (%s); returning extractive answer", e)
        yield extractive_answer(sources)
    except anthropic.APIConnectionError as e:
        log.warning("Cannot reach the Claude API (%s); returning extractive answer", e)
        yield extractive_answer(sources)
    except anthropic.APIStatusError as e:
        log.error("Claude API error %s (request id %s): %s", e.status_code, e.request_id, e.message)
        yield f"\n\n_Claude API error ({e.status_code}). Showing retrieved passages instead._\n\n"
        yield extractive_answer(sources)
