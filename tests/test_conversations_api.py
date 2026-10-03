from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from agentic_rag_assistant.agent import RAGAgent
from agentic_rag_assistant.answering import (
    GeneratedAnswer, GenerationTimeoutError, InvalidGenerationError, SupportedStatement,
)
from agentic_rag_assistant.answers import get_answer_service
from agentic_rag_assistant.conversation_store import (
    ConversationConflictError, ConversationNotFoundError, get_conversation_store,
)
from agentic_rag_assistant.database import StorageUnavailableError
from agentic_rag_assistant.main import app
from agentic_rag_assistant.models import (
    ConversationHistory, ConversationTurn, DocumentChunk, SearchHit,
)


class Store:
    def __init__(self):
        self.records = {}
        self.loads = []
        self.append_failure = None

    def create(self):
        identifier = uuid4()
        self.records[identifier] = ConversationHistory(identifier, 0, [])
        return self.records[identifier]

    def load(self, identifier, *, limit):
        self.loads.append((identifier, limit))
        if identifier not in self.records:
            raise ConversationNotFoundError("Conversation not found.")
        history = self.records[identifier]
        return replace(history, turns=history.turns[-limit:])

    def append(self, identifier, *, expected_turn_count, response):
        if self.append_failure:
            raise self.append_failure
        old = self.load(identifier, limit=100)
        assert old.turn_count == expected_turn_count
        saved = replace(response, conversation_id=identifier)
        new = ConversationTurn(old.turn_count + 1, saved, datetime.now(timezone.utc))
        self.records[identifier] = replace(
            old, turn_count=old.turn_count + 1, turns=[*old.turns, new],
        )
        return saved

    def delete(self, identifier):
        self.load(identifier, limit=1)
        del self.records[identifier]


class Provider:
    def __init__(self):
        self.rewrites = []
        self.generated = []
        self.failure = None
        self.rewrite_failure = None

    def rewrite_question(self, question, history):
        self.rewrites.append((question, history))
        if self.rewrite_failure:
            raise self.rewrite_failure
        return "Annual leave over three years?"

    def generate(self, question, sources):
        self.generated.append(question)
        if self.failure:
            raise self.failure
        return GeneratedAnswer(supported=True, statements=[
            SupportedStatement(text="24 days annually.", source_ids=[1]),
        ])


@pytest.fixture
def client_state():
    class Retriever:
        def search(self, query, **options):
            text = "24 days of annual leave."
            return [SearchHit("a" * 64, "leave.txt", 0.8,
                              DocumentChunk("stored", 0, text, None, 0, len(text)))]

    store, provider = Store(), Provider()
    app.dependency_overrides[get_conversation_store] = lambda: store
    app.dependency_overrides[get_answer_service] = lambda: RAGAgent(Retriever(), provider)
    try:
        with TestClient(app) as client:
            yield client, store, provider
    finally:
        app.dependency_overrides.pop(get_conversation_store, None)
        app.dependency_overrides.pop(get_answer_service, None)


def test_create_ask_followup_read_and_delete_a_conversation(client_state):
    client, store, provider = client_state
    created = client.post("/conversations")
    assert created.status_code == 201
    identifier = created.json()["conversation_id"]
    assert created.json()["turn_count"] == 0 and created.json()["turns"] == []
    first = client.post("/ask", json={"question": "Annual leave?", "conversation_id": identifier})
    assert first.status_code == 200 and first.json()["conversation_id"] == identifier
    assert provider.rewrites == []
    second = client.post("/ask", json={
        "question": "And over three years?", "conversation_id": identifier,
    })
    assert second.status_code == 200
    assert second.json()["question"] == "And over three years?"
    assert provider.rewrites[0][1][0].response.question == "Annual leave?"
    history = client.get(f"/conversations/{identifier}").json()
    assert history["turn_count"] == 2
    assert [turn["turn_number"] for turn in history["turns"]] == [1, 2]
    assert history["turns"][0]["response"]["citations"][0]["filename"] == "leave.txt"
    recent = client.get(f"/conversations/{identifier}?limit=1").json()
    assert recent["turn_count"] == 2 and recent["turns"][0]["turn_number"] == 2
    assert client.delete(f"/conversations/{identifier}").status_code == 204
    assert client.get(f"/conversations/{identifier}").status_code == 404
    assert not store.records


def test_stateless_request_does_not_access_conversation_storage(client_state):
    client, store, provider = client_state
    response = client.post("/ask", json={"question": "Annual leave?"})
    assert response.status_code == 200 and response.json()["conversation_id"] is None
    assert store.loads == [] and store.records == {} and provider.rewrites == []


def test_other_conversation_and_stateless_requests_do_not_receive_prior_history(client_state):
    client, _, provider = client_state
    first, second = [client.post("/conversations").json()["conversation_id"] for _ in range(2)]
    client.post("/ask", json={"question": "First topic", "conversation_id": first})
    client.post("/ask", json={"question": "Second topic", "conversation_id": second})
    client.post("/ask", json={"question": "Stateless topic"})
    assert provider.rewrites == []
    client.post("/ask", json={"question": "Followup", "conversation_id": first})
    assert [turn.response.question for turn in provider.rewrites[0][1]] == ["First topic"]


@pytest.mark.parametrize("failure,status", [
    (InvalidGenerationError("Invalid output"), 502),
    (StorageUnavailableError("Cannot save"), 503),
    (ConversationConflictError("Changed during generation"), 409),
])
def test_failed_generation_or_save_does_not_append_a_turn(client_state, failure, status):
    client, store, provider = client_state
    identifier = client.post("/conversations").json()["conversation_id"]
    if isinstance(failure, InvalidGenerationError):
        provider.failure = failure
    else:
        store.append_failure = failure
    response = client.post("/ask", json={
        "question": "Annual leave?", "conversation_id": identifier,
    })
    assert response.status_code == status
    assert client.get(f"/conversations/{identifier}").json()["turn_count"] == 0


def test_unknown_or_invalid_conversation_is_rejected_before_generation(client_state):
    client, _, provider = client_state
    assert client.post("/ask", json={
        "question": "Q", "conversation_id": str(uuid4()),
    }).status_code == 404
    assert client.post("/ask", json={"question": "Q", "conversation_id": "bad"}).status_code == 422
    assert client.get("/conversations/bad").status_code == 422
    assert client.delete("/conversations/bad").status_code == 422
    assert provider.generated == []


@pytest.mark.parametrize("limit", [0, 101])
def test_history_endpoint_rejects_unbounded_or_empty_limits(client_state, limit):
    client, _, _ = client_state
    identifier = client.post("/conversations").json()["conversation_id"]
    assert client.get(f"/conversations/{identifier}?limit={limit}").status_code == 422


@pytest.mark.parametrize("failure,status", [
    (InvalidGenerationError("Bad rewrite"), 502), (GenerationTimeoutError("Timeout"), 504),
])
def test_failed_rewrite_does_not_change_committed_history(client_state, failure, status):
    client, _, provider = client_state
    identifier = client.post("/conversations").json()["conversation_id"]
    assert client.post("/ask", json={
        "question": "Annual leave?", "conversation_id": identifier,
    }).status_code == 200
    provider.rewrite_failure = failure
    result = client.post("/ask", json={"question": "And later?", "conversation_id": identifier})
    assert result.status_code == status
    assert len(provider.generated) == 1
    assert client.get(f"/conversations/{identifier}").json()["turn_count"] == 1
