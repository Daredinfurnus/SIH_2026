"""
Validation utilities for uploads and request parameters.
"""
from __future__ import annotations

from app.schemas import is_allowed_extension


def validate_upload(filename: str | None, file_size: int, max_bytes: int) -> str | None:
    """Return an error string if the upload is invalid, else None."""
    if not filename:
        return "No file selected."

    if not is_allowed_extension(filename):
        return (
            "Unsupported file format. Supported formats: "
            "wav, mp3, m4a, ogg, webm."
        )

    if file_size <= 0:
        return "The uploaded file appears to be empty."

    if file_size > max_bytes:
        mb = max_bytes // (1024 * 1024)
        return f"File too large. Maximum upload size is {mb} MB."

    return None
