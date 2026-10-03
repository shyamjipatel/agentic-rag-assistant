import pytest

from agentic_rag_assistant.embeddings import EmbeddingUnavailableError
from agentic_rag_assistant.ingestion import ingest_document
from agentic_rag_assistant.retrieval import RetrievalService


class FakeEmbedder:
    model_name = "test-model"
    dimensions = 2

    def embed_documents(self, texts):
        self.passages = texts
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, query):
        self.query = query
        return [0.0, 1.0]


class RecordingStore:
    def save_document(self, document, vectors):
        self.saved = (document, vectors)

    def search(self, vector, **options):
        self.searched = (vector, options)
        return []


def test_index_coordinates_embeddings_and_storage() -> None:
    embedder = FakeEmbedder()
    store = RecordingStore()
    document = ingest_document(filename="notes.txt", content=b"x" * 1_200)

    result = RetrievalService(embedder, store).index(document)

    assert embedder.passages == ["x" * 1_000, "x" * 400]
    assert store.saved == (document, [[1.0, 0.0], [1.0, 0.0]])
    assert result.document_id == document.document_id
    assert result.embedding_model == "test-model"
    assert result.embedding_dimensions == 2


def test_search_uses_query_embedding_and_forwards_filters() -> None:
    embedder = FakeEmbedder()
    store = RecordingStore()
    service = RetrievalService(embedder, store)

    assert service.search("my question", top_k=3, document_id="a" * 64, min_score=0.7) == []
    assert embedder.query == "my question"
    assert store.searched == (
        [0.0, 1.0], {"top_k": 3, "document_id": "a" * 64, "min_score": 0.7}
    )


def test_incomplete_embedding_batch_never_reaches_storage(monkeypatch) -> None:
    embedder = FakeEmbedder()
    store = RecordingStore()
    monkeypatch.setattr(embedder, "embed_documents", lambda texts: [])
    document = ingest_document(filename="notes.txt", content=b"Hello")

    with pytest.raises(EmbeddingUnavailableError, match="incomplete batch"):
        RetrievalService(embedder, store).index(document)
    assert not hasattr(store, "saved")
