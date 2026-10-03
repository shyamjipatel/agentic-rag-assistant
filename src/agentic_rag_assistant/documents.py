"""HTTP boundary for document ingestion."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from agentic_rag_assistant.database import StorageUnavailableError
from agentic_rag_assistant.embeddings import EmbeddingInputError, EmbeddingUnavailableError
from agentic_rag_assistant.ingestion import (
    MAX_DOCUMENT_BYTES,
    DocumentLimitError,
    InvalidDocumentError,
    UnsupportedDocumentError,
    ingest_document,
)
from agentic_rag_assistant.models import DocumentList, IngestedDocument, IndexedDocument
from agentic_rag_assistant.retrieval import RetrievalService, get_retrieval_service
from agentic_rag_assistant.vector_store import PostgresVectorStore, get_vector_store

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentList)
async def list_documents(
    store: Annotated[PostgresVectorStore, Depends(get_vector_store)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentList:
    """Browse the shared library without loading the embedding model."""
    try:
        return await run_in_threadpool(store.list_documents, limit=limit, offset=offset)
    except StorageUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/ingest",
    response_model=IngestedDocument,
    responses={
        400: {"description": "Empty, unreadable, encrypted, or textless document"},
        413: {"description": "File, page, or extracted-text limit exceeded"},
        415: {"description": "Unsupported document format"},
    },
)
async def ingest_upload(
    file: Annotated[UploadFile, File(description="A UTF-8 .txt file or text-based PDF")],
) -> IngestedDocument:
    """Return parsed chunks for inspection without storing the document."""
    return await parse_upload(file)


async def parse_upload(file: UploadFile) -> IngestedDocument:
    """Read and close an upload, mapping parsing failures to HTTP errors."""
    try:
        if file.size is not None and file.size > MAX_DOCUMENT_BYTES:
            raise DocumentLimitError("Documents must be at most 5 MiB.")
        content = await file.read(MAX_DOCUMENT_BYTES + 1)
        return await run_in_threadpool(
            ingest_document, filename=file.filename or "", content=content
        )
    except UnsupportedDocumentError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except DocumentLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()


@router.post(
    "/index",
    response_model=IndexedDocument,
    responses={
        400: {"description": "Invalid document"},
        413: {"description": "Document limit exceeded"},
        415: {"description": "Unsupported document format"},
        503: {"description": "Embedding model or vector storage unavailable"},
    },
)
async def index_upload(
    file: Annotated[UploadFile, File(description="A UTF-8 .txt file or text-based PDF")],
    service: Annotated[RetrievalService, Depends(get_retrieval_service)],
) -> IndexedDocument:
    """Parse, embed, and atomically store a document for semantic search."""
    document = await parse_upload(file)
    try:
        return await run_in_threadpool(service.index, document)
    except EmbeddingInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (EmbeddingUnavailableError, StorageUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
