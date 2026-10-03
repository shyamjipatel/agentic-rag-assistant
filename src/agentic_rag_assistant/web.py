"""Serve the same-origin chat client; the JSON API remains the source of truth."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

STATIC_DIRECTORY = Path(__file__).parent / "static"
router = APIRouter()


@router.get("/", include_in_schema=False)
async def workspace() -> FileResponse:
    return FileResponse(
        STATIC_DIRECTORY / "index.html",
        headers={
            "Cache-Control": "no-cache",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": (
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                "connect-src 'self'; img-src 'self'; object-src 'none'; "
                "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
            ),
        },
    )
