"""HTTP endpoint for retrieving source chunks by semantic similarity."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, StringConstraints

from agentic_rag_assistant.database import StorageUnavailableError
from agentic_rag_assistant.embeddings import EmbeddingInputError, EmbeddingUnavailableError
from agentic_rag_assistant.models import SearchResponse
from agentic_rag_assistant.retrieval import RetrievalService, get_retrieval_service

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000)]
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    min_score: float | None = Field(default=None, ge=-1, le=1)


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={503: {"description": "Embedding model or vector storage unavailable"}},
)
async def search_documents(
    body: SearchRequest,
    service: Annotated[RetrievalService, Depends(get_retrieval_service)],
) -> SearchResponse:
    """Return matching source passages and cosine scores, without generating an answer."""
    try:
        results = await run_in_threadpool(
            service.search, body.query, top_k=body.top_k,
            document_id=body.document_id, min_score=body.min_score,
        )
    except EmbeddingInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (EmbeddingUnavailableError, StorageUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SearchResponse(query=body.query, results=results)
