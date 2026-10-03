"""Document data shared by the ingestion service and HTTP response."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    chunk_index: int
    text: str
    page_number: int | None
    start_char: int
    end_char: int


@dataclass(frozen=True)
class IngestedDocument:
    document_id: str
    filename: str
    character_count: int
    page_count: int | None
    skipped_pages: list[int]
    chunk_count: int
    chunks: list[DocumentChunk]


@dataclass(frozen=True)
class IndexedDocument:
    document_id: str
    filename: str
    chunk_count: int
    embedding_model: str
    embedding_dimensions: int


@dataclass(frozen=True)
class SearchHit:
    document_id: str
    filename: str
    score: float
    chunk: DocumentChunk


@dataclass(frozen=True)
class SearchResponse:
    query: str
    results: list[SearchHit]
