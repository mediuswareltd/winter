"""Runtime settings, read from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass


os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("VIDEORAG_DATA_DIR", "data")).resolve())

    # Speech-to-text (faster-whisper). "base" is a good CPU default; use "small"/"medium" for accuracy.
    whisper_model: str = _env("VIDEORAG_WHISPER_MODEL", "base")
    whisper_device: str = _env("VIDEORAG_WHISPER_DEVICE", "auto")

    # Embedding models (sentence-transformers).
    text_model: str = _env("VIDEORAG_TEXT_MODEL", "BAAI/bge-small-en-v1.5")
    clip_model: str = _env("VIDEORAG_CLIP_MODEL", "clip-ViT-B-32")

    # Chunking: transcript windows of ~N seconds, overlapping by M seconds.
    chunk_seconds: float = float(_env("VIDEORAG_CHUNK_SECONDS", "45"))
    chunk_overlap_seconds: float = float(_env("VIDEORAG_CHUNK_OVERLAP", "10"))

    # Keyframes: scene-change threshold (0-1) and a hard cap on the gap between frames.
    scene_threshold: float = float(_env("VIDEORAG_SCENE_THRESHOLD", "0.35"))
    max_frame_gap_seconds: float = float(_env("VIDEORAG_MAX_FRAME_GAP", "20"))

    # Answering with Claude.
    claude_model: str = _env("VIDEORAG_CLAUDE_MODEL", "claude-opus-5-5")
    claude_effort: str = _env("VIDEORAG_CLAUDE_EFFORT", "medium")
    images_per_answer: int = int(_env("VIDEORAG_IMAGES_PER_ANSWER", "4"))

    # Which model writes answers: "auto" uses Claude when ANTHROPIC_API_KEY is set, otherwise a local
    # Ollama model if one is running, otherwise returns the retrieved passages. Or force "claude",
    # "ollama" or "none".
    llm_provider: str = _env("VIDEORAG_LLM", "auto")
    ollama_url: str = _env("OLLAMA_HOST", "http://127.0.0.1:11434")
    ollama_model: str = _env("VIDEORAG_OLLAMA_MODEL", "qwen2.5:3b")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "videorag.sqlite3"

    @property
    def videos_dir(self) -> Path:
        return self.data_dir / "videos"

    @property
    def frames_dir(self) -> Path:
        return self.data_dir / "frames"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.videos_dir, self.frames_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
