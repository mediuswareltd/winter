"""Getting video onto disk and pulling the audio track out of it."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class MediaError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise MediaError("ffmpeg and ffprobe must be installed and on PATH (https://ffmpeg.org/download.html)")


def probe_duration(path: Path) -> float:
    """Duration of a media file in seconds."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def extract_audio(video: Path, out_wav: Path) -> Path:
    """16 kHz mono WAV: the format Whisper expects, so it does no resampling of its own."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(out_wav)],
        check=True,
    )
    return out_wav


def has_audio(path: Path) -> bool:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    return bool(out.stdout.strip())


def download_url(url: str, out_dir: Path, stem: str) -> tuple[Path, str]:
    """Download a YouTube (or any yt-dlp supported) URL as MP4, capped at 720p. Returns (path, title)."""
    import yt_dlp

    opts = {
        "format": "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b[height<=720]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / f"{stem}.%(ext)s"),
        "quiet": True,
        "noprogress": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    files = sorted(out_dir.glob(f"{stem}.*"))
    if not files:
        raise MediaError(f"Download produced no file for {url}")
    return files[0], info.get("title") or url
