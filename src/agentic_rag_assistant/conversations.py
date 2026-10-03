"""Create, inspect, and delete persisted conversations."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, StringConstraints, field_validator

from agentic_rag_assistant.conversation_store import (
    ConversationNotFoundError, PostgresConversationStore, get_conversation_store,
)
from agentic_rag_assistant.database import StorageUnavailableError
from agentic_rag_assistant.models import ConversationHistory, ConversationList, ConversationSummary

router = APIRouter(prefix="/conversations", tags=["conversations"])
StoreDependency = Annotated[PostgresConversationStore, Depends(get_conversation_store)]


class RenameConversation(BaseModel):
    title: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)
    ]

    @field_validator("title")
    @classmethod
    def reject_null_character(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("Titles cannot contain null characters.")
        return value


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


@router.get("", response_model=ConversationList)
async def list_conversations(
    store: StoreDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ConversationList:
    """List saved sessions, with the most recently changed session first."""
    return await storage_call(store.list, limit=limit, offset=offset)


@router.patch("/{conversation_id}", response_model=ConversationSummary)
async def rename_conversation(
    conversation_id: UUID, body: RenameConversation, store: StoreDependency,
) -> ConversationSummary:
    return await storage_call(store.rename, conversation_id, body.title)


@router.get("/{conversation_id}", response_model=ConversationHistory)
async def get_conversation(
    conversation_id: UUID, store: StoreDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    before_turn: Annotated[int | None, Query(ge=1)] = None,
) -> ConversationHistory:
    """Return the latest requested turns in chronological order and the total count."""
    options = {"limit": limit}
    if before_turn is not None:
        options["before_turn"] = before_turn
    return await storage_call(store.load, conversation_id, **options)


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: UUID, store: StoreDependency) -> Response:
    await storage_call(store.delete, conversation_id)
    return Response(status_code=204)
