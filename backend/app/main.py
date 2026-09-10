"""TraumaSense — AI-assisted stress & trauma-related conversational assessment
SIH 2026 · Problem Statement 26093
Team ESPADA-X · KCC Institute of Technology & Management

Backend application root.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Environment — load .env (if present) BEFORE importing settings so that
# FIREBASE_KEY_PATH and other env vars are available at import time.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv, find_dotenv

    _env_path = find_dotenv(raise_if_not_found=False)
    if _env_path:
        load_dotenv(dotenv_path=_env_path)
        import logging

        logging.getLogger(__name__).info("Loaded environment from %s", _env_path)
except ImportError:
    pass  # python-dotenv not installed; rely on the real environment

from app.config import settings
from app.api.routes import api_router

app = FastAPI(
    title="TraumaSense API",
    description=(
        "AI-assisted stress & trauma-related conversational assessment for "
        "NHAA 14566 helpline calls. This is an assistive decision-support system, "
        "not a clinical diagnostic tool. Trained human operators retain decision authority."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — only the development origins we actually use
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "http://localhost:5177",
        "http://localhost:5178",
        "http://localhost:5179",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.get("/")
def root():
    return {
        "name": "TraumaSense",
        "version": "1.0.0",
        "ps": "26093",
        "message": (
            "AI-assisted stress & trauma-related conversational assessment. "
            "Assistive decision-support only — not a clinical diagnosis."
        ),
    }
