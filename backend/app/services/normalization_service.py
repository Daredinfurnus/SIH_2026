"""
Audio normalization service — converts uploaded audio to a standard internal
representation usable by both Faster-Whisper and Wav2Vec2.

Uses FFmpeg as the normalization layer. When FFmpeg is unavailable, the raw
upload is used as-is and a clear warning is logged.

Internal format: mono PCM WAV, 16 kHz, 16-bit — appropriate for both models.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import os
from typing import Any

logger = logging.getLogger(__name__)

# Fixed internal audio spec — suitable for both Faster-Whisper and Wav2Vec2
_INTERNAL_SR = 16000       # sample rate
_INTERNAL_CHANNELS = 1    # mono
_INTERNAL_FORMAT = "pcm_s16le"  # 16-bit PCM


def _ffmpeg_available() -> bool:
    """Check whether ffmpeg is on the system PATH."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _ffprobe_available() -> bool:
    try:
        result = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def probe_audio(path: str) -> dict[str, Any]:
    """
    Probe an audio file's container/codec info via ffprobe.

    Returns dict with:
        codec_name, codec_type, format_name, duration, sample_rate,
        channels, has_audio, is_audio, error

    When ffprobe is unavailable, returns a best-effort dict with is_audio=True
    and an error note.
    """
    if not _ffprobe_available():
        return {
            "codec_name": "unknown",
            "codec_type": "audio",
            "format_name": "unknown",
            "duration": None,
            "sample_rate": None,
            "channels": None,
            "has_audio": True,
            "is_audio": True,
            "error": "ffprobe not available — cannot probe codec",
        }

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries",
                "stream=codec_name,codec_type,sample_rate,channels:"
                "format=format_name,duration:",
                "-of", "json",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return {
                "codec_name": "unknown",
                "codec_type": "unknown",
                "format_name": "unknown",
                "duration": None,
                "sample_rate": None,
                "channels": None,
                "has_audio": False,
                "is_audio": False,
                "error": f"ffprobe failed: {result.stderr.strip() or 'non-zero exit'}",
            }

        import json
        data = json.loads(result.stdout)

        streams = data.get("streams", [])
        format_info = data.get("format", {})

        audio_stream = None
        for s in streams:
            if s.get("codec_type") == "audio":
                audio_stream = s
                break

        if audio_stream is None and streams:
            # No audio stream found — could be video-only
            return {
                "codec_name": streams[0].get("codec_name", "unknown"),
                "codec_type": streams[0].get("codec_type", "unknown"),
                "format_name": format_info.get("format_name", "unknown"),
                "duration": float(format_info.get("duration", 0)) if format_info.get("duration") else None,
                "sample_rate": None,
                "channels": None,
                "has_audio": False,
                "is_audio": False,
                "error": None,
            }

        return {
            "codec_name": audio_stream.get("codec_name", "unknown"),
            "codec_type": "audio",
            "format_name": format_info.get("format_name", "unknown"),
            "duration": float(format_info.get("duration", 0)) if format_info.get("duration") else None,
            "sample_rate": int(audio_stream.get("sample_rate", 0)) if audio_stream.get("sample_rate") else None,
            "channels": int(audio_stream.get("channels", 0)) if audio_stream.get("channels") else None,
            "has_audio": True,
            "is_audio": True,
            "error": None,
        }
    except Exception as e:
        return {
            "codec_name": "unknown",
            "codec_type": "unknown",
            "format_name": "unknown",
            "duration": None,
            "sample_rate": None,
            "channels": None,
            "has_audio": False,
            "is_audio": False,
            "error": f"ffprobe exception: {e}",
        }


def normalize_audio(
    input_path: str,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """
    Convert input audio to the internal standard format.

    Output: mono PCM WAV, 16 kHz, 16-bit — written to a temporary file.

    Returns dict with:
        normalized_path    — path to the normalized WAV
        original_path      — the original input path
        original_format    — format/codec description
        duration_seconds   — duration of the audio
        ffmpeg_used       — whether FFmpeg was used for conversion
        ffmpeg_available  — whether FFmpeg is available on this system
        error             — error string if normalization failed

    When FFmpeg is unavailable, the original file is returned as-is with
    ffmpeg_used=False and a warning in the error field.
    """
    if output_dir is None:
        output_dir = tempfile.gettempdir()

    ffmpeg_ok = _ffmpeg_available()
    probe = probe_audio(input_path) if ffmpeg_ok else {
        "codec_name": "unknown",
        "format_name": "unknown",
        "duration": None,
        "error": "ffprobe not available",
    }

    original_format = (
        f"{probe.get('format_name', 'unknown')}/"
        f"{probe.get('codec_name', 'unknown')}"
    )

    # If FFmpeg is unavailable, return original path with warning
    if not ffmpeg_ok:
        logger.warning(
            "FFmpeg not available — audio normalization skipped. "
            "Using original file as-is: %s (%s). "
            "Wav2Vec2 acoustic analysis may fail on non-PCM formats.",
            input_path, original_format,
        )
        return {
            "normalized_path": input_path,
            "original_path": input_path,
            "original_format": original_format,
            "duration_seconds": probe.get("duration"),
            "ffmpeg_used": False,
            "ffmpeg_available": False,
            "error": None,
        }

    # Build normalized output path
    fd, tmp_path = tempfile.mkstemp(
        suffix=".wav",
        prefix="traumasense_norm_",
        dir=output_dir,
    )
    os.close(fd)

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",           # overwrite output
                "-i", input_path,
                "-ac", "1",     # mono
                "-ar", str(_INTERNAL_SR),  # 16 kHz
                "-c:a", _INTERNAL_FORMAT,
                tmp_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            # Clean up failed output
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            stderr_tail = result.stderr[-500:] if result.stderr else "(no stderr)"
            logger.error(
                "FFmpeg normalization failed for %s (%s): %s",
                input_path, original_format, stderr_tail,
            )
            return {
                "normalized_path": None,
                "original_path": input_path,
                "original_format": original_format,
                "duration_seconds": probe.get("duration"),
                "ffmpeg_used": True,
                "ffmpeg_available": True,
                "error": f"FFmpeg normalization failed: {stderr_tail}",
            }

        # Probe normalized file duration
        norm_probe = probe_audio(tmp_path) if _ffprobe_available() else {}
        duration = norm_probe.get("duration") or probe.get("duration")

        logger.info(
            "Normalized %s (%s) → %s (mono 16kHz PCM WAV, %.1fs)",
            os.path.basename(input_path), original_format,
            os.path.basename(tmp_path), duration or 0,
        )

        return {
            "normalized_path": tmp_path,
            "original_path": input_path,
            "original_format": original_format,
            "duration_seconds": duration,
            "ffmpeg_used": True,
            "ffmpeg_available": True,
            "error": None,
        }
    except subprocess.TimeoutExpired:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return {
            "normalized_path": None,
            "original_path": input_path,
            "original_format": original_format,
            "duration_seconds": probe.get("duration"),
            "ffmpeg_used": True,
            "ffmpeg_available": True,
            "error": "FFmpeg normalization timed out (>120s)",
        }
    except Exception as e:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return {
            "normalized_path": None,
            "original_path": input_path,
            "original_format": original_format,
            "duration_seconds": probe.get("duration"),
            "ffmpeg_used": True,
            "ffmpeg_available": True,
            "error": f"FFmpeg normalization exception: {e}",
        }


def cleanup_normalized(path: str) -> None:
    """Remove a normalized temp file if it exists and is not the original."""
    if path and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass
