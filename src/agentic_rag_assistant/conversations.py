"""Create, inspect, and delete persisted conversations."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.concurrency import run_in_threadpool

from agentic_rag_assistant.conversation_store import (
    ConversationNotFoundError, PostgresConversationStore, get_conversation_store,
)
from agentic_rag_assistant.database import StorageUnavailableError
from agentic_rag_assistant.models import ConversationHistory

router = APIRouter(prefix="/conversations", tags=["conversations"])
StoreDependency = Annotated[PostgresConversationStore, Depends(get_conversation_store)]


async def storage_call(operation, *args, **kwargs):
    try:
        return await run_in_threadpool(operation, *args, **kwargs)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except StorageUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("", response_model=ConversationHistory, status_code=201)
async def create_conversation(store: StoreDependency) -> ConversationHistory:
    return await storage_call(store.create)


@router.get("/{conversation_id}", response_model=ConversationHistory)
async def get_conversation(
    conversation_id: UUID, store: StoreDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> ConversationHistory:
    """Return the latest requested turns in chronological order and the total count."""
    return await storage_call(store.load, conversation_id, limit=limit)


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: UUID, store: StoreDependency) -> Response:
    await storage_call(store.delete, conversation_id)
    return Response(status_code=204)
