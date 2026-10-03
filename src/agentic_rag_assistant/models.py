"""Document data shared by the ingestion service and HTTP response."""

from dataclasses import dataclass, field


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


@dataclass(frozen=True)
class SourceCitation:
    source_id: int
    document_id: str
    filename: str
    score: float
    chunk: DocumentChunk


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    operation: str
    left: str
    right: str
    value: str
    source_ids: list[int]


@dataclass(frozen=True)
class AnswerResponse:
    question: str
    answer: str
    answered: bool
    citations: list[SourceCitation]
    tool_results: list[ToolResult] = field(default_factory=list)
