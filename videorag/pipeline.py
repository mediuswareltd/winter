"""Ingestion: video -> audio -> transcript -> chunks -> embeddings, and video -> keyframes -> CLIP embeddings."""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path

from videorag import embeddings, media
from videorag.chunking import chunk_segments
from videorag.config import Settings
from videorag.frames import extract_keyframes
from videorag.store import Store
from videorag.transcribe import transcribe

log = logging.getLogger(__name__)

# Share of the progress bar given to each stage (rough wall-clock proportions on CPU).
_STAGES = {"download": (0.00, 0.10), "transcribe": (0.10, 0.65), "frames": (0.65, 0.85), "embed": (0.85, 1.00)}


class Ingestor:
    def __init__(self, store: Store, settings: Settings):
        self.store = store
        self.settings = settings
        settings.ensure_dirs()

    def _progress(self, vid: str, stage: str, frac: float = 0.0) -> None:
        lo, hi = _STAGES[stage]
        self.store.update_video(vid, status="processing", stage=stage, progress=round(lo + (hi - lo) * frac, 3))

    def register_file(self, src: Path, title: str | None = None, move: bool = False, source: str | None = None) -> str:
        """Copy (or move) a local file into the library and return its new video id."""
        video = self.store.create_video(title or src.stem, source or src.name)
        dest = self.settings.videos_dir / f"{video.id}{src.suffix.lower() or '.mp4'}"
        (shutil.move if move else shutil.copy2)(str(src), dest)
        self.store.update_video(video.id, path=str(dest))
        return video.id

    def register_url(self, url: str) -> str:
        return self.store.create_video(url, url).id

    def run(self, vid: str) -> None:
        """Process a registered video end to end. Errors are recorded on the video row, not raised."""
        t0 = time.time()
        try:
            media.require_ffmpeg()
            video = self.store.get_video(vid)
            if video is None:
                raise ValueError(f"unknown video {vid}")

            if not video.path:
                self._progress(vid, "download")
                path, title = media.download_url(video.source, self.settings.videos_dir, vid)
                self.store.update_video(vid, path=str(path), title=title)
            path = Path(self.store.get_video(vid).path)  # type: ignore[union-attr]
            duration = media.probe_duration(path)
            self.store.update_video(vid, duration=duration)

            # Speech
            self._progress(vid, "transcribe")
            segments, language = [], None
            if media.has_audio(path):
                with tempfile.TemporaryDirectory() as tmp:
                    wav = media.extract_audio(path, Path(tmp) / "audio.wav")
                    segments, language = transcribe(
                        wav,
                        self.settings.whisper_model,
                        self.settings.whisper_device,
                        on_progress=lambda f: self._progress(vid, "transcribe", f),
                    )
            chunks = chunk_segments(segments, self.settings.chunk_seconds, self.settings.chunk_overlap_seconds)
            self.store.update_video(vid, language=language)

            # Vision
            self._progress(vid, "frames")
            frame_dir = self.settings.frames_dir / vid
            if frame_dir.exists():
                shutil.rmtree(frame_dir)
            keyframes = extract_keyframes(
                path,
                frame_dir,
                self.settings.scene_threshold,
                self.settings.max_frame_gap_seconds,
                on_progress=lambda f: self._progress(vid, "frames", f),
            )

            # Embeddings
            self._progress(vid, "embed")
            chunk_vecs = embeddings.embed_passages([c.text for c in chunks], self.settings.text_model)
            self.store.add_chunks(vid, [(c.start, c.end, c.text) for c in chunks], chunk_vecs)
            self._progress(vid, "embed", 0.5)
            frame_vecs = embeddings.embed_images([k.path for k in keyframes], self.settings.clip_model)
            self.store.add_frames(vid, [(k.t, str(k.path)) for k in keyframes], frame_vecs)

            if not chunks and not keyframes:
                raise RuntimeError("no speech and no frames could be extracted from this file")
            self.store.mark_ready(vid)
            log.info(
                "ingested %s: %.0fs video, %d segments, %d chunks, %d keyframes in %.1fs",
                vid,
                duration,
                len(segments),
                len(chunks),
                len(keyframes),
                time.time() - t0,
            )
        except Exception as e:
            log.exception("ingest failed for %s", vid)
            self.store.update_video(vid, status="failed", error=f"{type(e).__name__}: {e}")
