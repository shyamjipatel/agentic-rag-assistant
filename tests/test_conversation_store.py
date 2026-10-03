import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from uuid import uuid4

import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict
from psycopg.types.json import Jsonb
from pydantic import TypeAdapter

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


def test_titles_directory_pagination_and_older_turns_survive_restart(saved_conversation):
    store, identifier = saved_conversation
    for number in range(6):
        store.append(identifier, expected_turn_count=number, response=response(f"Question {number}"))
    reopened = PostgresConversationStore(store.database_url)
    summary = next(item for item in reopened.list().conversations if item.conversation_id == identifier)
    assert summary.title == "Question 0" and summary.turn_count == 6
    renamed = reopened.rename(identifier, "  Policy research  ")
    assert renamed.title == "Policy research" and renamed.updated_at >= summary.updated_at
    older = reopened.load(identifier, limit=2, before_turn=4)
    assert older.turn_count == 6
    assert [turn.turn_number for turn in older.turns] == [2, 3]
    assert reopened.load(identifier, before_turn=1).turns == []
    page = reopened.list(limit=1)
    assert page.conversations[0].conversation_id == identifier
    beyond_end = reopened.list(offset=page.total + 1)
    assert beyond_end.total == page.total and beyond_end.conversations == []


def test_custom_title_is_preserved_on_first_turn_and_reinitialization(saved_conversation):
    store, identifier = saved_conversation
    store.rename(identifier, "Manual name")
    store.append(identifier, expected_turn_count=0, response=response("Generated name"))
    initialize_database(store.database_url)
    summary = next(item for item in store.list().conversations if item.conversation_id == identifier)
    assert summary.title == "Manual name"
    store.rename(identifier, "New conversation")
    initialize_database(store.database_url)
    summary = next(item for item in store.list().conversations if item.conversation_id == identifier)
    assert summary.title == "New conversation"


def test_additive_migration_preserves_legacy_turns_and_backfills_titles(
    saved_conversation, monkeypatch,
):
    from agentic_rag_assistant import database

    store, _ = saved_conversation
    original_connect = database.connect
    schema = sql.Identifier(f"workspace_migration_{uuid4().hex}")
    legacy_id = uuid4()
    with original_connect(store.database_url) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(schema))
        connection.execute(sql.SQL("SET search_path TO {}, public").format(schema))
        connection.execute("""CREATE TABLE conversations (
            conversation_id uuid PRIMARY KEY, turn_count integer NOT NULL DEFAULT 0,
            created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        connection.execute("""CREATE TABLE conversation_turns (
            conversation_id uuid REFERENCES conversations ON DELETE CASCADE,
            turn_number integer NOT NULL, response jsonb NOT NULL,
            created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (conversation_id, turn_number))""")
        connection.execute(
            "INSERT INTO conversations (conversation_id, turn_count) VALUES (%s, 1)",
            (legacy_id,),
        )
        payload = TypeAdapter(AnswerResponse).dump_python(response("Legacy question"), mode="json")
        connection.execute(
            "INSERT INTO conversation_turns (conversation_id, turn_number, response) "
            "VALUES (%s, 1, %s)", (legacy_id, Jsonb(payload)),
        )

    def isolated_connection(url):
        connection = original_connect(url)
        connection.execute(sql.SQL("SET search_path TO {}, public").format(schema))
        return connection

    try:
        monkeypatch.setattr(database, "connect", isolated_connection)
        initialize_database(store.database_url)
        with isolated_connection(store.database_url) as connection:
            row = connection.execute(
                "SELECT title, turn_count FROM conversations WHERE conversation_id = %s",
                (legacy_id,),
            ).fetchone()
            assert row == ("Legacy question", 1)
            assert connection.execute("SELECT count(*) FROM conversation_turns").fetchone()[0] == 1
            connection.execute("UPDATE conversations SET title = 'Renamed legacy chat'")
        initialize_database(store.database_url)
        with isolated_connection(store.database_url) as connection:
            assert connection.execute("SELECT title FROM conversations").fetchone()[0] == (
                "Renamed legacy chat"
            )
    finally:
        with original_connect(store.database_url) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(schema))
