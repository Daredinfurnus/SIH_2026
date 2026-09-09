"""
Audio service — validates uploads, inspects metadata, manages temporary storage.

Responsibilities:
- validate file extension, size, emptiness
- sanitize filenames
- obtain duration where practical
- temporary storage with cleanup

If optional audio-processing tools are unavailable the service still runs
in DEMO MODE and returns a placeholder duration.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from app.config import settings
from app.schemas import is_allowed_extension


# ---------------------------------------------------------------------------
# Supported MIME broad-brush set (never trust MIME alone, used as hint only)
# ---------------------------------------------------------------------------
_ALLOWED_MAGIC_PREFIXES = (
    b"\x52\x49\x46\x46",  # RIFF...WAVE (wav)
    b"\xff\xfb",           # MP3 sync
    b"\xff\xf3",           # MP3 sync
    b"\xff\xf2",           # MP3 sync
    b"\x00\x00\x00\x1c\x66\x74\x79\x70",  # MP4 ftyp
    b"\x80\x00\x00\x00\x66\x74\x79\x70",  # MP4 ftyp (alternatively)
    b"\x4f\x67\x67\x53",  # OggS
    b"\x1a\x45\xdf\xa3",  # WebM/EBML
)


def _read_magic(path: str, n: int = 16) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read(n)
    except OSError:
        return b""


def is_probably_audio(path: str) -> bool:
    """Heuristic magic-byte check — a hint, not authoritative."""
    magic = _read_magic(path)
    if not magic:
        return False
    return magic.startswith(_ALLOWED_MAGIC_PREFIXES)


# ---------------------------------------------------------------------------
# Public audio service API
# ---------------------------------------------------------------------------

class AudioService:
    """Stateless audio validation + temporary storage helper."""

    def __init__(self) -> None:
        self.upload_dir = Path(settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    # ---- validation --------------------------------------------------------

    def validate(self, filename: str, file_size: int) -> str | None:
        """Return an error message string if invalid, else None."""
        if not filename or not filename.strip():
            return "No file selected."

        if not is_allowed_extension(filename):
            return (
                f"Unsupported file format. Supported: "
                + ", ".join(sorted({"wav", "mp3", "m4a", "ogg", "webm"}))
            )

        if file_size <= 0:
            return "The uploaded file appears to be empty."

        if file_size > settings.max_upload_size_bytes:
            return (
                f"File too large. Maximum allowed size is "
                f"{settings.max_upload_size_mb} MB."
            )

        return None

    # ---- filenames ---------------------------------------------------------

    @staticmethod
    def sanitized_name(original: str) -> str:
        """Return a safe, unique storage name preserving the original extension."""
        ext = os.path.splitext(original)[1].lower()
        unique = uuid.uuid4().hex[:10]
        return f"upload_{unique}{ext}"

    # ---- storage -----------------------------------------------------------

    def store_temp(self, original_filename: str, file_bytes: bytes) -> str:
        """Write file_bytes to a temporary location and return the path."""
        safe_name = self.sanitized_name(original_filename)
        dest = self.upload_dir / safe_name
        dest.write_bytes(file_bytes)
        return str(dest)

    def release(self, path: str) -> None:
        """Delete a temporary uploaded file when no longer needed."""
        try:
            os.remove(path)
        except OSError:
            pass

    # ---- metadata ---------------------------------------------------------

    def inspect(self, path: str) -> dict[str, Any]:
        """Return available metadata for the uploaded file.

        duration_seconds is a best-effort value. When no audio library is
        available we return 0.0 and the caller should treat it as unknown
        rather than as a real measurement.
        """
        size = os.path.getsize(path)
        duration = self._guess_duration(path)
        return {
            "size_bytes": size,
            "duration_seconds": duration,
        }

    # ---- internal ----------------------------------------------------------

    def _guess_duration(self, path: str) -> float:
        """Try to read duration; fall back to 0.0 when tools are absent."""
        # Prefer standard library / lightweight probes first.
        try:
            return _duration_whisper(path)
        except (_AudioProbeError, Exception):
            pass

        try:
            return _duration_ffprobe(path)
        except (_AudioProbeError, Exception):
            pass

        return 0.0


class _AudioProbeError(RuntimeError):
    """Raised when an audio probe fails for any reason."""


def _duration_whisper(path: str) -> float:
    """Probe duration via the `wave` stdlib module (WAV only)."""
    import wave

    try:
        with wave.open(path, "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            if rate <= 0:
                raise _AudioProbeError("Invalid sample rate")
            return frames / rate
    except Exception as exc:
        raise _AudioProbeError(str(exc)) from exc


def _duration_ffprobe(path: str) -> float:
    """Probe duration via ffprobe if available on the system."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise _AudioProbeError(result.stderr.strip() or "ffprobe failed")
        raw = result.stdout.strip()
        if not raw:
            raise _AudioProbeError("No duration returned")
        return float(raw)
    except FileNotFoundError:
        raise _AudioProbeError("ffprobe not found")
    except (subprocess.TimeoutExpired, ValueError) as exc:
        raise _AudioProbeError(str(exc)) from exc
