"""FastAPI application entry point for the RecursiveMAS benchmark."""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

from backend.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-32s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = FastAPI(
    title="RecursiveMAS Benchmark API",
    description=(
        "Benchmarks Traditional (flat) MAS against RecursiveMAS across three "
        "recursion rounds, measuring inference time, tokens, cost, and quality."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["Health"])
async def root() -> dict:
    return {
        "service": "RecursiveMAS Benchmark API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health() -> dict:
    has_key = bool(os.environ.get("GEMINI_API_KEY"))
    return {
        "status": "healthy",
        "llm_mode": "live" if has_key else "mock",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
