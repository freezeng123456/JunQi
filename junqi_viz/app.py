"""FastAPI application serving one explicitly selected replay file."""

from __future__ import annotations

from pathlib import Path

from .replay_data import ReplayData


def create_app(replay_path: str):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            'Replay UI dependencies are missing; install with pip install -e ".[viz]"'
        ) from exc

    data = ReplayData(replay_path)
    static_dir = Path(__file__).with_name("static")
    app = FastAPI(
        title="JunQi Replay",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.mount("/assets", StaticFiles(directory=static_dir), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/api/meta", include_in_schema=False)
    def metadata():
        return data.metadata()

    @app.get("/api/frame/{step}", include_in_schema=False)
    def frame(step: int):
        try:
            return data.frame(step)
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


__all__ = ["create_app"]
