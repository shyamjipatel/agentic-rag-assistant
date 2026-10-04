import json

import psycopg
import pytest
from fastapi.testclient import TestClient

from agentic_rag_assistant.answers import get_answer_service
from agentic_rag_assistant.database import StorageUnavailableError
from agentic_rag_assistant.embeddings import EmbeddingUnavailableError
from agentic_rag_assistant.main import app
from agentic_rag_assistant.operations import LOGGER, MAX_REQUEST_BYTES, RequestDiagnosticsMiddleware
from agentic_rag_assistant.readiness import get_readiness_service


def test_request_ids_are_generated_for_success_validation_and_not_found():
    with TestClient(app) as client:
        responses = [
            client.get("/health", headers={"X-Request-ID": "untrusted-id"}),
            client.post("/ask", json={"question": " "}), client.get("/missing"),
        ]
    identifiers = [response.headers["x-request-id"] for response in responses]
    assert len(set(identifiers)) == 3
    assert all(len(identifier) == 32 and int(identifier, 16) >= 0 for identifier in identifiers)
    assert [response.status_code for response in responses] == [200, 422, 404]


def test_validation_errors_do_not_echo_private_inputs():
    private = "PRIVATE_DOCUMENT_TEXT_" * 110
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": private})
    assert response.status_code == 422 and private not in response.text
    assert all(set(error) == {"loc", "msg", "type"} for error in response.json()["detail"])


def test_unexpected_failure_is_sanitized_and_logs_only_request_metadata(caplog, monkeypatch):
    secret = "secret-provider-key-and-private-question"

    def failing_service():
        raise RuntimeError(secret)

    monkeypatch.setattr(LOGGER, "propagate", True)
    caplog.set_level("INFO", logger=LOGGER.name)
    app.dependency_overrides[get_answer_service] = failing_service
    try:
        with TestClient(app) as client:
            response = client.post(f"/ask?private={secret}", json={"question": secret},
                                   headers={"Authorization": secret})
        assert response.status_code == 500 and secret not in response.text
        entries = [json.loads(record.message) for record in caplog.records
                   if record.name == LOGGER.name]
        assert entries[-1]["error_type"] == "RuntimeError"
        assert entries[-1]["route"] == "/ask"
        assert entries[-1]["request_id"] == response.headers["x-request-id"]
        assert secret not in caplog.text
    finally:
        app.dependency_overrides.pop(get_answer_service, None)


def test_oversized_declared_body_is_rejected_before_route_dependencies():
    with TestClient(app) as client:
        response = client.post("/ask", content=b"x", headers={
            "Content-Length": str(MAX_REQUEST_BYTES + 1),
        })
    assert response.status_code == 413 and "x-request-id" in response.headers


@pytest.mark.parametrize("length", ["-1", "invalid", "1,1"])
def test_invalid_content_length_returns_bad_request(length):
    with TestClient(app) as client:
        assert client.post("/ask", content=b"{}", headers={
            "Content-Length": length,
        }).status_code == 400


@pytest.mark.anyio
async def test_chunked_body_limit_is_enforced_without_content_length():
    events = []
    chunks = iter([
        {"type": "http.request", "body": b"x" * MAX_REQUEST_BYTES, "more_body": True},
        {"type": "http.request", "body": b"x", "more_body": False},
    ])

    async def receive():
        return next(chunks)

    async def send(message):
        events.append(message)

    # Use the real FastAPI parser and exception handlers for the JSON-body route.
    await RequestDiagnosticsMiddleware(app)({
        "type": "http", "method": "POST", "path": "/ask", "raw_path": b"/ask",
        "query_string": b"", "headers": [(b"content-type", b"application/json")],
        "root_path": "", "scheme": "http", "server": ("test", 80),
        "client": ("test", 123), "http_version": "1.1",
    }, receive, send)
    assert next(event["status"] for event in events if event["type"] == "http.response.start") == 413


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.parametrize("failure", [
    StorageUnavailableError("Private database URL"),
    EmbeddingUnavailableError("Private model path"), psycopg.OperationalError("Password"),
])
def test_readiness_failure_is_safe_and_does_not_change_liveness(failure):
    class Service:
        def check(self):
            raise failure

    app.dependency_overrides[get_readiness_service] = Service
    try:
        with TestClient(app) as client:
            response = client.get("/ready")
            assert response.status_code == 503 and str(failure) not in response.text
            assert client.get("/health").json() == {"status": "ok"}
    finally:
        app.dependency_overrides.pop(get_readiness_service, None)


def test_readiness_reports_ready_when_local_dependencies_pass():
    class Service:
        def check(self):
            pass

    app.dependency_overrides[get_readiness_service] = Service
    try:
        with TestClient(app) as client:
            assert client.get("/ready").json() == {"status": "ready"}
    finally:
        app.dependency_overrides.pop(get_readiness_service, None)
