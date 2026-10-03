"""Coordinate embedding and storage without HTTP dependencies."""

from functools import lru_cache
from typing import Protocol

from agentic_rag_assistant.embeddings import (
    EmbeddingProvider, EmbeddingUnavailableError, LocalEmbedder, validate_vector,
)
from agentic_rag_assistant.models import IngestedDocument, IndexedDocument, SearchHit
from agentic_rag_assistant.settings import get_settings
from agentic_rag_assistant.vector_store import PostgresVectorStore


class DocumentStore(Protocol):
    def save_document(self, document: IngestedDocument, vectors: list[list[float]]) -> None: ...

    def search(
        self, vector: list[float], *, top_k: int,
        document_id: str | None = None, min_score: float | None = None,
    ) -> list[SearchHit]: ...


class RetrievalService:
    def __init__(self, embedder: EmbeddingProvider, store: DocumentStore):
        self.embedder = embedder
        self.store = store

    def index(self, document: IngestedDocument) -> IndexedDocument:
        vectors = self.embedder.embed_documents([chunk.text for chunk in document.chunks])
        if len(vectors) != len(document.chunks):
            raise EmbeddingUnavailableError("The model returned an incomplete batch.")
        for vector in vectors:
            validate_vector(vector, self.embedder.dimensions)
        self.store.save_document(document, vectors)
        return IndexedDocument(
            document_id=document.document_id, filename=document.filename,
            chunk_count=document.chunk_count, embedding_model=self.embedder.model_name,
            embedding_dimensions=self.embedder.dimensions,
        )

    def search(
        self, query: str, *, top_k: int = 5,
        document_id: str | None = None, min_score: float | None = None,
    ) -> list[SearchHit]:
        vector = self.embedder.embed_query(query)
        validate_vector(vector, self.embedder.dimensions)
        return self.store.search(
            vector, top_k=top_k, document_id=document_id, min_score=min_score
        )


@lru_cache
def get_retrieval_service() -> RetrievalService:
    settings = get_settings()
    return RetrievalService(
        LocalEmbedder(settings.model_cache_dir),
        PostgresVectorStore(settings.database_url.get_secret_value()),
    )
