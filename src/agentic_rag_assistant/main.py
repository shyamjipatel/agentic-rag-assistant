"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles

from agentic_rag_assistant.answers import router as answers_router
from agentic_rag_assistant.conversations import router as conversations_router
from agentic_rag_assistant.documents import router as documents_router
from agentic_rag_assistant.search import router as search_router
from agentic_rag_assistant.web import STATIC_DIRECTORY, router as web_router
from agentic_rag_assistant.operations import (
    RequestDiagnosticsMiddleware, configure_request_logging, validation_error,
)
from agentic_rag_assistant.readiness import router as readiness_router

configure_request_logging()
app = FastAPI(title="Agentic RAG Assistant", version="0.1.0")
app.add_middleware(RequestDiagnosticsMiddleware)
app.add_exception_handler(RequestValidationError, validation_error)
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(answers_router)
app.include_router(conversations_router)
app.include_router(readiness_router)
app.include_router(web_router)
app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    """Report that the API is responding."""
    return {"status": "ok"}
