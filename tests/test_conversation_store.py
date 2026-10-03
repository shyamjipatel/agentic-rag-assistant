import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from uuid import uuid4

import pytest
from psycopg.conninfo import conninfo_to_dict

from agentic_rag_assistant.conversation_store import (
    ConversationConflictError, ConversationNotFoundError, PostgresConversationStore,
)
from agentic_rag_assistant.database import StorageUnavailableError, initialize_database
from agentic_rag_assistant.models import AnswerResponse

pytestmark = pytest.mark.integration


@pytest.fixture
def saved_conversation():
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to run PostgreSQL integration tests.")
    if not conninfo_to_dict(url).get("dbname", "").endswith("_test"):
        pytest.fail("Integration tests require a separate database ending in _test.")
    initialize_database(url)
    store = PostgresConversationStore(url)
    conversation = store.create()
    yield store, conversation.conversation_id
    try:
        store.delete(conversation.conversation_id)
    except ConversationNotFoundError:
        pass


def response(question="Annual leave?"):
    return AnswerResponse(question, "24 days.", True, [])


def test_history_survives_a_new_store_and_is_bounded_and_ordered(saved_conversation):
    store, identifier = saved_conversation
    assert store.load(identifier).turns == []
    for number in range(4):
        store.append(identifier, expected_turn_count=number, response=response(f"Q{number}"))
    reopened = PostgresConversationStore(store.database_url)
    loaded = reopened.load(identifier, limit=2)
    assert loaded.turn_count == 4
    assert [turn.turn_number for turn in loaded.turns] == [3, 4]
    assert all(turn.response.conversation_id == identifier for turn in loaded.turns)
    assert all(turn.created_at.tzinfo is not None for turn in loaded.turns)


def test_concurrent_appends_only_commit_one_turn(saved_conversation):
    store, identifier = saved_conversation

    def append(question):
        try:
            return store.append(identifier, expected_turn_count=0, response=response(question))
        except ConversationConflictError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(append, ["First", "Second"]))
    assert sum(result is not None for result in results) == 1
    loaded = store.load(identifier)
    assert loaded.turn_count == 1 and len(loaded.turns) == 1


def test_failed_insert_rolls_back_revision_and_preserves_previous_turn(saved_conversation):
    store, identifier = saved_conversation
    store.append(identifier, expected_turn_count=0, response=response())
    invalid = replace(response(), answer="PostgreSQL jsonb cannot store \x00")
    with pytest.raises(StorageUnavailableError):
        store.append(identifier, expected_turn_count=1, response=invalid)
    loaded = store.load(identifier)
    assert loaded.turn_count == 1 and len(loaded.turns) == 1
    assert loaded.turns[0].response.answer == "24 days."


def test_delete_removes_turns_and_unknown_ids_are_not_implicitly_created(saved_conversation):
    store, identifier = saved_conversation
    store.append(identifier, expected_turn_count=0, response=response())
    store.delete(identifier)
    for missing in (identifier, uuid4()):
        with pytest.raises(ConversationNotFoundError):
            store.load(missing)
        with pytest.raises(ConversationNotFoundError):
            store.append(missing, expected_turn_count=0, response=response())
