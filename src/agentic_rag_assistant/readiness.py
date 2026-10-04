"""Readiness checks local dependencies without contacting or billing an LLM."""

from functools import lru_cache
from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from agentic_rag_assistant.database import StorageUnavailableError, check_configuration, connect
from agentic_rag_assistant.embeddings import EmbeddingUnavailableError, LocalEmbedder
from agentic_rag_assistant.retrieval import get_retrieval_service
from agentic_rag_assistant.settings import get_settings

router = APIRouter(tags=["health"])


class ReadinessService:
    def __init__(self, database_url: str, embedder: LocalEmbedder):
        self.database_url, self.embedder = database_url, embedder

    def check(self) -> None:
        with connect(self.database_url) as connection:
            check_configuration(connection)
            connection.execute("SELECT title, updated_at FROM conversations LIMIT 0")
            connection.execute("SELECT response FROM conversation_turns LIMIT 0")
            connection.execute("SELECT document_id FROM documents LIMIT 0")
            connection.execute("SELECT embedding FROM document_chunks LIMIT 0")
        # Exercises the cached model, tokenizer, and vector checks; downloads stay disabled.
        self.embedder.embed_query("Check local embedding readiness.")


@lru_cache
def get_readiness_service() -> ReadinessService:
    return ReadinessService(
        get_settings().database_url.get_secret_value(), get_retrieval_service().embedder,
    )


@router.get("/ready", responses={503: {"description": "Local dependencies are not ready"}})
async def ready(
    service: Annotated[ReadinessService, Depends(get_readiness_service)],
) -> dict[str, str]:
    try:
        await run_in_threadpool(service.check)
    except (psycopg.Error, StorageUnavailableError, EmbeddingUnavailableError) as exc:
        raise HTTPException(503, "PostgreSQL or cached embeddings are not ready.") from exc
    return {"status": "ready"}
