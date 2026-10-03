"""Document data shared by the ingestion service and HTTP response."""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID


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
    conversation_id: UUID | None = None


@dataclass(frozen=True)
class ConversationTurn:
    turn_number: int
    response: AnswerResponse
    created_at: datetime


@dataclass(frozen=True)
class ConversationHistory:
    conversation_id: UUID
    turn_count: int
    turns: list[ConversationTurn]


@dataclass(frozen=True)
class ConversationSummary:
    conversation_id: UUID
    title: str
    turn_count: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ConversationList:
    conversations: list[ConversationSummary]
    total: int


@dataclass(frozen=True)
class DocumentSummary:
    document_id: str
    filename: str
    character_count: int
    page_count: int | None
    chunk_count: int
    indexed_at: datetime


@dataclass(frozen=True)
class DocumentList:
    documents: list[DocumentSummary]
    total: int
