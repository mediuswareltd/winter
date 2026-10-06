"""HTTP API + web UI."""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from videorag.config import settings
from videorag.llm import active_llm, stream_answer
from videorag.pipeline import Ingestor
from videorag.retrieval import Retriever
from videorag.store import Store

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).parent / "web"
ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mp3", ".wav", ".m4a"}

store = Store(settings.db_path)
ingestor = Ingestor(store, settings)
retriever = Retriever(store, settings)
# One ingest at a time: transcription and embedding already use every CPU core.
jobs = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")


@asynccontextmanager
async def lifespan(_: FastAPI):
    for v in store.list_videos():  # resume work interrupted by a restart
        if v.status in ("queued", "processing"):
            jobs.submit(ingestor.run, v.id)
    yield
    jobs.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Winter", version="0.1.0", lifespan=lifespan)


class UrlIn(BaseModel):
    url: str = Field(..., examples=["https://www.youtube.com/watch?v=..."])


class AskIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    video_ids: list[str] | None = None
    k: int = Field(6, ge=1, le=20)
    history: list[dict] | None = Field(None, description="Prior turns as [{role, content}] with string content")


class SearchIn(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    video_ids: list[str] | None = None
    k: int = Field(8, ge=1, le=50)


def _video_or_404(vid: str):
    v = store.get_video(vid)
    if v is None:
        raise HTTPException(404, "video not found")
    return v


def _public(v) -> dict:
    """Video as returned to clients: no server filesystem paths."""
    d = v.to_dict()
    d["has_media"] = bool(d.pop("path"))
    return d


@app.get("/api/health")
def health():
    provider, model = active_llm(settings)
    return {
        "status": "ok",
        "claude_model": settings.claude_model,
        "whisper_model": settings.whisper_model,
        "llm_provider": provider,
        "llm_model": model,
        "llm_configured": provider != "none",
    }


@app.get("/api/videos")
def list_videos():
    return [_public(v) for v in store.list_videos()]


@app.post("/api/videos", status_code=202)
def upload_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(400, f"unsupported file type {suffix!r}; allowed: {', '.join(sorted(ALLOWED_EXT))}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
    name = Path(file.filename or "video").name
    vid = ingestor.register_file(Path(tmp.name), title=Path(name).stem, move=True, source=name)
    jobs.submit(ingestor.run, vid)
    return _public(store.get_video(vid))


@app.post("/api/videos/url", status_code=202)
def add_url(body: UrlIn):
    if not re.match(r"^https?://", body.url):
        raise HTTPException(400, "url must start with http:// or https://")
    vid = ingestor.register_url(body.url)
    jobs.submit(ingestor.run, vid)
    return _public(store.get_video(vid))


@app.get("/api/videos/{vid}")
def get_video(vid: str):
    return _public(_video_or_404(vid))


@app.delete("/api/videos/{vid}", status_code=204)
def delete_video(vid: str):
    v = _video_or_404(vid)
    if v.status == "processing":
        raise HTTPException(409, "video is still processing")
    store.delete_video(vid)
    if v.path:
        Path(v.path).unlink(missing_ok=True)
    shutil.rmtree(settings.frames_dir / vid, ignore_errors=True)


@app.get("/api/videos/{vid}/transcript")
def transcript(vid: str):
    _video_or_404(vid)
    return [c.__dict__ for c in store.transcript(vid)]


@app.get("/api/videos/{vid}/media")
def video_media(vid: str):
    v = _video_or_404(vid)
    if not v.path or not Path(v.path).exists():
        raise HTTPException(404, "media not available yet")
    return FileResponse(v.path)  # supports HTTP Range, so the player can seek


@app.get("/api/videos/{vid}/thumb")
def video_thumb(vid: str):
    v = _video_or_404(vid)
    f = store.nearest_frame(vid, (v.duration or 0) * 0.15)
    path = settings.frames_dir / vid / Path(f.path).name if f else None
    if path is None or not path.exists():
        raise HTTPException(404, "no thumbnail yet")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/frames/{vid}/{name}")
def frame(vid: str, name: str):
    if not re.fullmatch(r"[0-9a-f]{12}", vid) or not re.fullmatch(r"[\d.]+\.jpg", name):
        raise HTTPException(400, "bad frame path")
    path = settings.frames_dir / vid / name
    if not path.exists():
        raise HTTPException(404, "frame not found")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/search")
def search(body: SearchIn):
    result = retriever.search(body.query, k=body.k, video_ids=body.video_ids)
    return [s.to_dict() for s in result.sources]


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/ask")
def ask(body: AskIn):
    """Server-sent events: one `sources` event, many `delta` events, then `done`."""
    result = retriever.search(body.question, k=body.k, video_ids=body.video_ids)
    history = [
        {"role": h["role"], "content": str(h["content"])}
        for h in (body.history or [])[-6:]
        if h.get("role") in ("user", "assistant") and h.get("content")
    ]

    def events():
        yield _sse("sources", [s.to_dict() for s in result.sources])
        try:
            for text in stream_answer(body.question, result.sources, settings, history):
                yield _sse("delta", text)
        except Exception as e:
            log.exception("answer failed")
            yield _sse("error", str(e))
        yield _sse("done", {})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
