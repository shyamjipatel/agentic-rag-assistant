import pytest
from fastapi.testclient import TestClient

from agentic_rag_assistant.database import StorageUnavailableError
from agentic_rag_assistant.embeddings import EmbeddingInputError, EmbeddingUnavailableError
from agentic_rag_assistant.main import app
from agentic_rag_assistant.models import IndexedDocument, SearchHit
from agentic_rag_assistant.retrieval import get_retrieval_service


class FakeService:
    failure = None

    def index(self, document):
        if self.failure:
            raise self.failure
        self.document = document
        return IndexedDocument(document.document_id, document.filename, document.chunk_count,
                               "test-model", 384)

    def search(self, query, **options):
        if self.failure:
            raise self.failure
        self.options = options
        if not hasattr(self, "document"):
            return []
        return [SearchHit(self.document.document_id, self.document.filename, 0.9,
                          self.document.chunks[0])]


@pytest.fixture
def retrieval_client():
    service = FakeService()
    app.dependency_overrides[get_retrieval_service] = lambda: service
    try:
        with TestClient(app) as client:
            yield client, service
    finally:
        app.dependency_overrides.pop(get_retrieval_service, None)


def test_index_then_search_preserves_source_metadata(retrieval_client, pdf_factory) -> None:
    client, service = retrieval_client
    indexed = client.post(
        "/documents/index",
        files={"file": ("guide.pdf", pdf_factory("A useful source."), "application/pdf")},
    )
    assert indexed.status_code == 200
    assert indexed.json()["chunk_count"] == 1

    response = client.post("/search", json={"query": "  useful sources  ", "top_k": 2})
    assert response.status_code == 200
    result = response.json()
    assert result["query"] == "useful sources"
    hit = result["results"][0]
    assert hit["document_id"] == indexed.json()["document_id"]
    assert hit["filename"] == "guide.pdf"
    assert hit["score"] == 0.9
    assert hit["chunk"]["page_number"] == 1
    assert hit["chunk"]["text"] == "A useful source."
    assert service.options["top_k"] == 2


def test_empty_search_returns_an_empty_result_list(retrieval_client) -> None:
    client, _ = retrieval_client
    response = client.post("/search", json={"query": "nothing indexed yet"})
    assert response.status_code == 200
    assert response.json()["results"] == []


@pytest.mark.parametrize(
    "body",
    [{"query": "  "}, {"query": "x" * 2_001}, {"query": "x", "top_k": 0},
     {"query": "x", "top_k": 21}, {"query": "x", "document_id": "invalid"},
     {"query": "x", "min_score": 1.1}],
)
def test_search_validation(retrieval_client, body) -> None:
    client, service = retrieval_client
    assert client.post("/search", json=body).status_code == 422
    assert not hasattr(service, "options")


@pytest.mark.parametrize(
    ("failure", "status"),
    [(EmbeddingInputError("input too long"), 422),
     (EmbeddingUnavailableError("prepare the model"), 503),
     (StorageUnavailableError("database unavailable"), 503)],
)
def test_retrieval_failures_have_explicit_http_statuses(retrieval_client, failure, status) -> None:
    client, service = retrieval_client
    service.failure = failure
    assert client.post("/search", json={"query": "question"}).status_code == status
    response = client.post(
        "/documents/index", files={"file": ("notes.txt", b"Hello", "text/plain")}
    )
    assert response.status_code == status


def test_invalid_document_is_not_sent_for_indexing(retrieval_client) -> None:
    client, service = retrieval_client
    response = client.post(
        "/documents/index", files={"file": ("notes.csv", b"Hello", "text/csv")}
    )
    assert response.status_code == 415
    assert not hasattr(service, "document")
