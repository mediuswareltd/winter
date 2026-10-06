"""Grounded answering with Claude: transcript excerpts + keyframe images in, cited answer out."""

from __future__ import annotations

import base64
import logging
from collections.abc import Iterator
from pathlib import Path

import anthropic

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


def extractive_answer(sources: list[Source]) -> str:
    """Used when Claude is unavailable: show the best-matching moments instead of a generated answer."""
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
