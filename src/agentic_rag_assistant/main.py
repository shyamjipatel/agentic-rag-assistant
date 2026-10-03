"""FastAPI application entry point."""

from fastapi import FastAPI

from agentic_rag_assistant.documents import router as documents_router
from agentic_rag_assistant.search import router as search_router

app = FastAPI(title="Agentic RAG Assistant", version="0.1.0")
app.include_router(documents_router)
app.include_router(search_router)


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    """Report that the API is responding."""
    return {"status": "ok"}
