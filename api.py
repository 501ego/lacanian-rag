"""ASGI entrypoint for FastAPI CLI and Uvicorn."""

from app.interfaces.api import app

__all__ = ["app"]
