"""HTTP boundary for document ingestion."""

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from agentic_rag_assistant.ingestion import (
    MAX_DOCUMENT_BYTES,
    DocumentLimitError,
    InvalidDocumentError,
    UnsupportedDocumentError,
    ingest_document,
)
from agentic_rag_assistant.models import IngestedDocument

router = APIRouter(prefix="/documents", tags=["documents"])


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
    """Return parsed chunks for inspection; documents are not stored yet."""
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
