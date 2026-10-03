import pytest
from fastapi.testclient import TestClient

from agentic_rag_assistant.ingestion import MAX_DOCUMENT_BYTES, MAX_TEXT_CHARACTERS
from agentic_rag_assistant.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_upload_text_returns_traceable_overlapping_chunks(client) -> None:
    text = "α" * 1_800 + "tail"
    response = client.post(
        "/documents/ingest", files={"file": ("notes.txt", text.encode("utf-8"), "text/plain")}
    )

    assert response.status_code == 200
    document = response.json()
    assert document["filename"] == "notes.txt"
    assert document["character_count"] == len(text)
    assert document["page_count"] is None
    assert document["skipped_pages"] == []
    assert document["chunk_count"] == 3
    assert len(document["document_id"]) == 64
    chunks = document["chunks"]
    assert [chunk["chunk_index"] for chunk in chunks] == [0, 1, 2]
    assert len({chunk["chunk_id"] for chunk in chunks}) == 3
    assert all(chunk["page_number"] is None for chunk in chunks)
    assert [(chunk["start_char"], chunk["end_char"]) for chunk in chunks] == [
        (0, 1_000), (800, 1_800), (1_600, 1_804)
    ]
    assert all(chunk["text"] == text[chunk["start_char"] : chunk["end_char"]] for chunk in chunks)


def test_pdf_preserves_page_numbers_and_reports_empty_pages(client, pdf_factory) -> None:
    response = client.post(
        "/documents/ingest",
        files={
            "file": (
                "guide.pdf", pdf_factory("First page", None, "Third page"), "application/pdf"
            )
        },
    )

    assert response.status_code == 200
    document = response.json()
    assert document["page_count"] == 3
    assert document["skipped_pages"] == [2]
    assert document["chunk_count"] == 2
    assert [chunk["page_number"] for chunk in document["chunks"]] == [1, 3]
    assert [chunk["text"] for chunk in document["chunks"]] == ["First page", "Third page"]
    assert [chunk["start_char"] for chunk in document["chunks"]] == [0, 0]


def test_pdf_chunks_never_cross_pages(client, pdf_factory) -> None:
    response = client.post(
        "/documents/ingest",
        files={"file": ("guide.pdf", pdf_factory("a" * 1_200, "b" * 25), "application/pdf")},
    )

    assert response.status_code == 200
    document = response.json()
    assert document["character_count"] == 1_225
    assert document["chunk_count"] == 3
    chunks = document["chunks"]
    assert [chunk["page_number"] for chunk in chunks] == [1, 1, 2]
    assert [(chunk["start_char"], chunk["end_char"]) for chunk in chunks] == [
        (0, 1_000), (800, 1_200), (0, 25)
    ]
    assert [chunk["text"] for chunk in chunks] == ["a" * 1_000, "a" * 400, "b" * 25]


@pytest.mark.parametrize(
    ("filename", "content", "expected_status", "detail"),
    [
        ("notes.csv", b"hello", 415, "Supported document formats"),
        ("notes.txt", b"", 400, "empty"),
        ("notes.txt", b" \n\t ", 400, "non-whitespace text"),
        ("notes.txt", b"\xff\xfe", 400, "UTF-8"),
        ("notes.txt", b"hello\x00", 400, "null bytes"),
        ("notes.pdf", b"hello", 400, "valid PDF header"),
        ("notes.txt", b"x" * (MAX_DOCUMENT_BYTES + 1), 413, "5 MiB"),
        ("notes.txt", b"x" * (MAX_TEXT_CHARACTERS + 1), 413, "character limit"),
    ],
)
def test_invalid_uploads_return_actionable_errors(
    client, filename, content, expected_status, detail
) -> None:
    response = client.post(
        "/documents/ingest", files={"file": (filename, content, "application/octet-stream")}
    )

    assert response.status_code == expected_status
    assert detail in response.json()["detail"]


@pytest.mark.parametrize(
    ("pages", "encrypted", "detail"),
    [([None], False, "no extractable text"), (["Private document"], True, "Encrypted")],
)
def test_textless_and_encrypted_pdfs_are_rejected(
    client, pdf_factory, pages, encrypted, detail
) -> None:
    response = client.post(
        "/documents/ingest",
        files={"file": ("notes.pdf", pdf_factory(*pages, encrypted=encrypted), "application/pdf")},
    )

    assert response.status_code == 400
    assert detail in response.json()["detail"]


def test_missing_upload_returns_validation_error(client) -> None:
    assert client.post("/documents/ingest").status_code == 422


def test_openapi_describes_multipart_upload_and_typed_response(client) -> None:
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/documents/ingest"]["post"]

    assert "multipart/form-data" in operation["requestBody"]["content"]
    assert operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/IngestedDocument"
    )
    assert {"400", "413", "415", "422"} <= operation["responses"].keys()
