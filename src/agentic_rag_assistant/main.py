"""FastAPI application entry point."""

from fastapi import FastAPI

app = FastAPI(title="Agentic RAG Assistant", version="0.1.0")


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    """Report that the API is responding."""
    return {"status": "ok"}
