"""
TraumaSense configuration.
Reads from environment variables with sensible defaults so the prototype
runs offline without any .env file present.
"""
import os
from functools import lru_cache

MAX_UPLOAD_SIZE_DEFAULT_MB = 50


@lru_cache
def _get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


class Settings:
    """Runtime settings for TraumaSense backend."""

    # ---- Providers ----------------------------------------------------------
    stt_provider: str = _get_env("STT_PROVIDER", "whisper")
    ai_provider: str = _get_env("AI_PROVIDER", "indicbert")

    # ---- Upload constraints -------------------------------------------------
    max_upload_size_mb: int = int(
        _get_env("MAX_UPLOAD_SIZE_MB", str(MAX_UPLOAD_SIZE_DEFAULT_MB))
    )
    max_upload_size_bytes: int = max_upload_size_mb * 1024 * 1024

    # ---- Storage ------------------------------------------------------------
    upload_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads"
    )

    # ---- Language ------------------------------------------------------------
    default_language: str = _get_env("DEFAULT_LANGUAGE", "en")


settings = Settings()
