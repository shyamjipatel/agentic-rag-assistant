"""Parse supported documents and prepare traceable chunks without persistence."""

from hashlib import sha256
from io import BytesIO
from pathlib import PurePosixPath

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from agentic_rag_assistant.chunking import chunk_text
from agentic_rag_assistant.models import DocumentChunk, IngestedDocument

MAX_DOCUMENT_BYTES = 5 * 1024 * 1024
MAX_PDF_PAGES = 100
MAX_TEXT_CHARACTERS = 500_000


class InvalidDocumentError(ValueError):
    """The supplied document cannot be parsed into usable text."""


class UnsupportedDocumentError(ValueError):
    """The filename identifies a format that is not supported."""


class DocumentLimitError(ValueError):
    """The document exceeds an ingestion limit."""


def _parse_pdf(content: bytes) -> list[tuple[int | None, str]]:
    if not content.startswith(b"%PDF-"):
        raise InvalidDocumentError("The file does not contain a valid PDF header.")
    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise InvalidDocumentError("Encrypted PDFs are not supported.")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise DocumentLimitError(f"The PDF exceeds the {MAX_PDF_PAGES}-page limit.")
        pages = []
        character_count = 0
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if "\x00" in text:
                raise InvalidDocumentError("PDF text must not contain null characters.")
            character_count += len(text)
            if character_count > MAX_TEXT_CHARACTERS:
                raise DocumentLimitError("Extracted text exceeds the character limit.")
            pages.append((number, text))
        return pages
    except (InvalidDocumentError, DocumentLimitError):
        raise
    except (PyPdfError, ValueError, KeyError, TypeError, IndexError, RecursionError) as exc:
        raise InvalidDocumentError("The PDF is malformed or cannot be read.") from exc


def ingest_document(*, filename: str, content: bytes) -> IngestedDocument:
    """Parse UTF-8 text or PDF pages and assign stable source identifiers."""
    # The client filename is metadata, never a filesystem destination.
    source_name = PurePosixPath(filename.replace("\\", "/")).name
    if "\x00" in source_name or len(source_name) > 255:
        raise InvalidDocumentError("Filenames must be at most 255 characters without null bytes.")
    extension = PurePosixPath(source_name).suffix.lower()
    if extension not in {".txt", ".pdf"}:
        raise UnsupportedDocumentError("Supported document formats are .txt and .pdf.")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise DocumentLimitError("Documents must be at most 5 MiB.")
    if not content:
        raise InvalidDocumentError("The uploaded document is empty.")

    if extension == ".txt":
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise InvalidDocumentError("Text documents must use UTF-8 encoding.") from exc
        if "\x00" in text:
            raise InvalidDocumentError("Text documents must not contain null bytes.")
        if len(text) > MAX_TEXT_CHARACTERS:
            raise DocumentLimitError("Extracted text exceeds the character limit.")
        pages = [(None, text)]
    else:
        pages = _parse_pdf(content)

    document_id = sha256(content).hexdigest()
    chunks = []
    skipped_pages = []
    for page_number, text in pages:
        if not text.strip() and page_number is not None:
            skipped_pages.append(page_number)
        for span in chunk_text(text):
            index = len(chunks)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document_id}:v1:{index}",
                    chunk_index=index,
                    text=span.text,
                    page_number=page_number,
                    start_char=span.start_char,
                    end_char=span.end_char,
                )
            )
    if not chunks:
        message = (
            "The PDF contains no extractable text. Scanned PDFs require OCR."
            if extension == ".pdf"
            else "Text documents must contain non-whitespace text."
        )
        raise InvalidDocumentError(message)

    return IngestedDocument(
        document_id=document_id,
        filename=source_name,
        character_count=sum(len(text) for _, text in pages),
        page_count=len(pages) if extension == ".pdf" else None,
        skipped_pages=skipped_pages,
        chunk_count=len(chunks),
        chunks=chunks,
    )
