from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from agentic_rag_assistant.conversation_store import (
    ConversationNotFoundError, get_conversation_store,
)
from agentic_rag_assistant.database import StorageUnavailableError
from agentic_rag_assistant.main import app
from agentic_rag_assistant.models import (
    ConversationHistory, ConversationList, ConversationSummary, DocumentList, DocumentSummary,
)
from agentic_rag_assistant.vector_store import get_vector_store


@pytest.fixture
def workspace_client():
    now, identifier = datetime.now(timezone.utc), uuid4()

    class Store:
        failure = None
        options = None

        def list(self, **options):
            self.options = options
            if self.failure:
                raise self.failure
            return ConversationList(
                [ConversationSummary(identifier, "My research", 2, now, now)], 1,
            )

        def rename(self, requested_id, title):
            if self.failure:
                raise self.failure
            assert requested_id == identifier
            return ConversationSummary(identifier, title, 2, now, now)

        def load(self, requested_id, **options):
            self.options = options
            return ConversationHistory(requested_id, 150, [])

    class Documents:
        failure = None
        options = None

        def list_documents(self, **options):
            self.options = options
            if self.failure:
                raise self.failure
            return DocumentList([DocumentSummary("a" * 64, "guide.pdf", 900, 2, 3, now)], 1)

    sessions, documents = Store(), Documents()
    app.dependency_overrides[get_conversation_store] = lambda: sessions
    app.dependency_overrides[get_vector_store] = lambda: documents
    try:
        with TestClient(app) as client:
            yield client, sessions, documents, identifier
    finally:
        app.dependency_overrides.pop(get_conversation_store, None)
        app.dependency_overrides.pop(get_vector_store, None)


def test_workspace_static_assets_and_existing_api_are_available():
    with TestClient(app) as client:
        home = client.get("/")
        assert home.status_code == 200 and "text/html" in home.headers["content-type"]
        assert home.headers["cache-control"] == "no-cache"
        assert "script-src 'self'" in home.headers["content-security-policy"]
        assert home.headers["x-content-type-options"] == "nosniff"
        assert '<textarea id="question"' in home.text
        for path, content_type in [
            ("app.css", "text/css"), ("app.js", "javascript"), ("api.js", "javascript"),
            ("render.js", "javascript"), ("favicon.svg", "image/svg+xml"),
        ]:
            asset = client.get(f"/static/{path}")
            assert asset.status_code == 200 and content_type in asset.headers["content-type"]
        assert client.get("/static/../settings.py").status_code == 404
        assert client.get("/health").json() == {"status": "ok"}
        schema = client.get("/openapi.json").json()
        assert "/" not in schema["paths"] and "/ask" in schema["paths"]


def test_session_directory_and_rename(workspace_client):
    client, sessions, _, identifier = workspace_client
    listed = client.get("/conversations?limit=10&offset=5")
    assert listed.status_code == 200
    assert listed.json()["conversations"][0]["title"] == "My research"
    assert listed.json()["total"] == 1
    assert sessions.options == {"limit": 10, "offset": 5}
    renamed = client.patch(f"/conversations/{identifier}", json={"title": "  Leave policy  "})
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Leave policy"
    assert renamed.json()["turn_count"] == 2


@pytest.mark.parametrize("title", ["", "  ", "x" * 121, "a\x00b", None])
def test_invalid_names_are_rejected_before_storage(workspace_client, title):
    client, sessions, _, identifier = workspace_client
    sessions.failure = AssertionError("Invalid title reached storage")
    assert client.patch(f"/conversations/{identifier}", json={"title": title}).status_code == 422


@pytest.mark.parametrize("path", ["/conversations", "/documents"])
@pytest.mark.parametrize("query", ["limit=0", "limit=101", "offset=-1"])
def test_directory_pagination_is_bounded(workspace_client, path, query):
    client, _, _, _ = workspace_client
    assert client.get(f"{path}?{query}").status_code == 422


def test_older_history_passes_a_turn_cursor_without_changing_total(workspace_client):
    client, sessions, _, identifier = workspace_client
    response = client.get(f"/conversations/{identifier}?limit=20&before_turn=131")
    assert response.status_code == 200 and response.json()["turn_count"] == 150
    assert sessions.options == {"limit": 20, "before_turn": 131}
    assert client.get(f"/conversations/{identifier}?before_turn=0").status_code == 422


def test_document_directory_returns_metadata_without_embedding_inference(workspace_client):
    client, _, documents, _ = workspace_client
    response = client.get("/documents?limit=10&offset=2")
    assert response.status_code == 200
    assert documents.options == {"limit": 10, "offset": 2}
    doc = response.json()["documents"][0]
    assert doc["filename"] == "guide.pdf" and doc["chunk_count"] == 3
    assert doc["page_count"] == 2 and "embedding" not in doc


def test_workspace_storage_failures_and_unknown_session_have_safe_statuses(workspace_client):
    client, sessions, documents, identifier = workspace_client
    sessions.failure = StorageUnavailableError("Session storage unavailable")
    documents.failure = StorageUnavailableError("Document storage unavailable")
    assert client.get("/conversations").status_code == 503
    assert client.get("/documents").status_code == 503
    assert client.patch(f"/conversations/{identifier}", json={"title": "Q"}).status_code == 503
    sessions.failure = ConversationNotFoundError("Conversation not found.")
    assert client.patch(f"/conversations/{uuid4()}", json={"title": "Q"}).status_code == 404
