import pytest

from agentic_rag_assistant.ingestion import (
    DocumentLimitError,
    InvalidDocumentError,
    ingest_document,
)


def test_identifiers_are_stable_across_repeated_uploads() -> None:
    first = ingest_document(filename="notes.txt", content=b"A source document.")
    repeated = ingest_document(filename="renamed.txt", content=b"A source document.")
    changed = ingest_document(filename="notes.txt", content=b"An updated document.")

    assert first.document_id == repeated.document_id
    assert first.chunks == repeated.chunks
    assert first.document_id != changed.document_id
    assert first.chunks[0].chunk_id != changed.chunks[0].chunk_id


def test_utf8_bom_and_filename_paths_are_handled_as_metadata() -> None:
    document = ingest_document(filename="..\\private\\NOTES.TXT", content=b"\xef\xbb\xbfHello")

    assert document.filename == "NOTES.TXT"
    assert document.character_count == 5
    assert document.chunks[0].text == "Hello"
    assert (document.chunks[0].start_char, document.chunks[0].end_char) == (0, 5)


def test_pdf_page_and_text_limits_are_enforced(pdf_factory, monkeypatch) -> None:
    monkeypatch.setattr("agentic_rag_assistant.ingestion.MAX_PDF_PAGES", 1)
    with pytest.raises(DocumentLimitError, match="1-page limit"):
        ingest_document(filename="notes.pdf", content=pdf_factory("First", "Second"))

    monkeypatch.setattr("agentic_rag_assistant.ingestion.MAX_TEXT_CHARACTERS", 4)
    with pytest.raises(DocumentLimitError, match="character limit"):
        ingest_document(filename="notes.pdf", content=pdf_factory("Hello"))


def test_truncated_pdf_returns_a_document_error() -> None:
    with pytest.raises(InvalidDocumentError, match="malformed"):
        ingest_document(filename="broken.pdf", content=b"%PDF-1.7\ntruncated")
