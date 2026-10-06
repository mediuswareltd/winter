"""Text and image embeddings. Models load lazily and are cached for the life of the process."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

# BGE models retrieve better when the *query* (not the passage) carries this instruction.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=4)
def _model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed_passages(texts: list[str], model: str) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    return (
        _model(model)
        .encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        .astype(np.float32)
    )


def embed_query(text: str, model: str) -> np.ndarray:
    prefix = _BGE_QUERY_PREFIX if "bge" in model.lower() else ""
    return (
        _model(model).encode([prefix + text], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)
    )


def embed_images(paths: list[Path], model: str) -> np.ndarray:
    """CLIP image embeddings, in the same space as `embed_clip_text`."""
    from PIL import Image

    if not paths:
        return np.zeros((0, 0), dtype=np.float32)
    out = []
    for i in range(0, len(paths), 32):
        imgs = [Image.open(p).convert("RGB") for p in paths[i : i + 32]]
        out.append(_model(model).encode(imgs, normalize_embeddings=True, show_progress_bar=False))
    return np.concatenate(out).astype(np.float32)


def embed_clip_text(text: str, model: str) -> np.ndarray:
    return _model(model).encode([text], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)
