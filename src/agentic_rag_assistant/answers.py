"""HTTP boundary and dependency wiring for retrieval-augmented answers."""

from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, StringConstraints

from agentic_rag_assistant.agent import RAGAgent
from agentic_rag_assistant.answering import (
    MAX_ANSWER_SOURCES,
    GenerationTimeoutError,
    GenerationUnavailableError,
    InvalidGenerationError,
)
from agentic_rag_assistant.database import StorageUnavailableError
from agentic_rag_assistant.embeddings import EmbeddingInputError, EmbeddingUnavailableError
from agentic_rag_assistant.llm.factory import create_llm_provider
from agentic_rag_assistant.models import AnswerResponse
from agentic_rag_assistant.retrieval import get_retrieval_service
from agentic_rag_assistant.settings import get_settings

router = APIRouter(tags=["answers"])


class AskRequest(BaseModel):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000)
    ]
    top_k: int = Field(default=MAX_ANSWER_SOURCES, ge=1, le=MAX_ANSWER_SOURCES)
    document_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    min_score: float | None = Field(default=None, ge=-1, le=1)


@lru_cache
def get_answer_service() -> RAGAgent:
    return RAGAgent(get_retrieval_service(), create_llm_provider(get_settings()))


@router.post(
    "/ask",
    response_model=AnswerResponse,
    responses={
        502: {"description": "Invalid model output or source references"},
        503: {"description": "Embedding, storage, or LLM provider unavailable"},
        504: {"description": "LLM provider timed out"},
    },
)
async def ask_question(
    body: AskRequest,
    service: Annotated[RAGAgent, Depends(get_answer_service)],
) -> AnswerResponse:
    """Answer from retrieved evidence, with server-resolved source citations."""
    try:
        return await run_in_threadpool(
            service.ask, body.question, top_k=body.top_k,
            document_id=body.document_id, min_score=body.min_score,
        )
    except EmbeddingInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InvalidGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except GenerationTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except (
        EmbeddingUnavailableError, StorageUnavailableError, GenerationUnavailableError,
    ) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
