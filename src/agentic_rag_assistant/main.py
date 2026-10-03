"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from agentic_rag_assistant.answers import router as answers_router
from agentic_rag_assistant.conversations import router as conversations_router
from agentic_rag_assistant.documents import router as documents_router
from agentic_rag_assistant.search import router as search_router
from agentic_rag_assistant.web import STATIC_DIRECTORY, router as web_router

app = FastAPI(title="Agentic RAG Assistant", version="0.1.0")
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(answers_router)
app.include_router(conversations_router)
app.include_router(web_router)
app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    """Report that the API is responding."""
    return {"status": "ok"}
