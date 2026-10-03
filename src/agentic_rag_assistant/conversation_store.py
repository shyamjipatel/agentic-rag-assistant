"""Persist completed turns with optimistic concurrency and bounded reads."""

from dataclasses import replace
from functools import lru_cache
from uuid import UUID, uuid4

import psycopg
from psycopg.types.json import Jsonb
from pydantic import TypeAdapter, ValidationError

from agentic_rag_assistant.database import StorageUnavailableError, connect
from agentic_rag_assistant.models import AnswerResponse, ConversationHistory, ConversationTurn
from agentic_rag_assistant.settings import get_settings

RESPONSE_ADAPTER = TypeAdapter(AnswerResponse)


class ConversationNotFoundError(RuntimeError):
    """The requested conversation does not exist."""


class ConversationConflictError(RuntimeError):
    """Another request saved a turn after this request loaded its history."""


class PostgresConversationStore:
    def __init__(self, database_url: str):
        self.database_url = database_url

    def create(self) -> ConversationHistory:
        identifier = uuid4()
        try:
            with connect(self.database_url) as connection:
                connection.execute(
                    "INSERT INTO conversations (conversation_id) VALUES (%s)", (identifier,),
                )
            return ConversationHistory(identifier, 0, [])
        except psycopg.Error as exc:
            raise StorageUnavailableError(
                "Conversation storage is unavailable. Check database initialization."
            ) from exc

    def load(self, identifier: UUID, *, limit: int = 3) -> ConversationHistory:
        if not 1 <= limit <= 100:
            raise ValueError("Conversation history limits must be between 1 and 100.")
        try:
            with connect(self.database_url) as connection:
                # A single snapshot keeps the revision and recent turns consistent.
                rows = connection.execute(
                    """SELECT c.turn_count, t.turn_number, t.response, t.created_at
                    FROM conversations c
                    LEFT JOIN LATERAL (
                        SELECT turn_number, response, created_at FROM conversation_turns
                        WHERE conversation_id = c.conversation_id
                        ORDER BY turn_number DESC LIMIT %s
                    ) t ON TRUE
                    WHERE c.conversation_id = %s ORDER BY t.turn_number""",
                    (limit, identifier),
                ).fetchall()
            if not rows:
                raise ConversationNotFoundError("Conversation not found.")
            turns = [
                ConversationTurn(number, RESPONSE_ADAPTER.validate_python(response), created)
                for _, number, response, created in rows if number is not None
            ]
            return ConversationHistory(identifier, rows[0][0], turns)
        except (psycopg.Error, ValidationError) as exc:
            raise StorageUnavailableError("Conversation history could not be loaded.") from exc

    def append(
        self, identifier: UUID, *, expected_turn_count: int, response: AnswerResponse,
    ) -> AnswerResponse:
        saved = replace(response, conversation_id=identifier)
        payload = RESPONSE_ADAPTER.dump_python(saved, mode="json")
        try:
            with connect(self.database_url) as connection:
                row = connection.execute(
                    """UPDATE conversations SET turn_count = turn_count + 1
                    WHERE conversation_id = %s AND turn_count = %s RETURNING turn_count""",
                    (identifier, expected_turn_count),
                ).fetchone()
                if row is None:
                    exists = connection.execute(
                        "SELECT 1 FROM conversations WHERE conversation_id = %s", (identifier,),
                    ).fetchone()
                    if exists is None:
                        raise ConversationNotFoundError("Conversation not found.")
                    raise ConversationConflictError(
                        "Conversation changed during this request. Retry after loading its history."
                    )
                connection.execute(
                    """INSERT INTO conversation_turns (conversation_id, turn_number, response)
                    VALUES (%s, %s, %s)""", (identifier, row[0], Jsonb(payload)),
                )
            return saved
        except psycopg.Error as exc:
            raise StorageUnavailableError("The conversation turn could not be saved.") from exc

    def delete(self, identifier: UUID) -> None:
        try:
            with connect(self.database_url) as connection:
                row = connection.execute(
                    "DELETE FROM conversations WHERE conversation_id = %s "
                    "RETURNING conversation_id",
                    (identifier,),
                ).fetchone()
                if row is None:
                    raise ConversationNotFoundError("Conversation not found.")
        except psycopg.Error as exc:
            raise StorageUnavailableError("The conversation could not be deleted.") from exc


@lru_cache
def get_conversation_store() -> PostgresConversationStore:
    return PostgresConversationStore(get_settings().database_url.get_secret_value())
