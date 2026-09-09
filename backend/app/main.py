"""
TraumaSense — AI-assisted stress & trauma-related conversational assessment
SIH 2026 · Problem Statement 26093
Team ESPADA-X · KCC Institute of Technology & Management

Backend application root.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
