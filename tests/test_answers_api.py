import json

import httpx
import pytest
from fastapi.testclient import TestClient

from agentic_rag_assistant.answering import (
    INSUFFICIENT_EVIDENCE,
    AnswerService,
    GeneratedAnswer,
    GenerationTimeoutError,
    GenerationUnavailableError,
    InvalidGenerationError,
    SupportedStatement,
)
from agentic_rag_assistant.answers import get_answer_service
from agentic_rag_assistant.database import StorageUnavailableError
from agentic_rag_assistant.embeddings import EmbeddingInputError, EmbeddingUnavailableError
from agentic_rag_assistant.ingestion import ingest_document
from agentic_rag_assistant.llm.factory import create_llm_provider
from agentic_rag_assistant.main import app
from agentic_rag_assistant.models import SearchHit
from agentic_rag_assistant.settings import Settings


class FakeRetriever:
    failure = None

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, query, **options):
        self.calls.append((query, options))
        if self.failure:
            raise self.failure
        return self.hits


class FakeProvider:
    failure = None

    def __init__(self):
        self.calls = []
        self.answer = GeneratedAnswer(
            supported=True,
            statements=[SupportedStatement(text="Employees get 24 days of leave.", source_ids=[1])],
        )

    def generate(self, question, sources):
        self.calls.append((question, sources))
        if self.failure:
            raise self.failure
        return self.answer


@pytest.fixture
def answer_client(pdf_factory):
    document = ingest_document(
        filename="leave.pdf", content=pdf_factory("Employees get 24 days of annual leave.")
    )
    retriever = FakeRetriever([
        SearchHit(document.document_id, document.filename, 0.8, document.chunks[0])
    ])
    provider = FakeProvider()
    service = AnswerService(retriever, provider)
    app.dependency_overrides[get_answer_service] = lambda: service
    try:
        with TestClient(app) as client:
            yield client, service, retriever, provider
    finally:
        app.dependency_overrides.pop(get_answer_service, None)


def test_ask_returns_answer_and_authoritative_pdf_citations(answer_client):
    client, _, retriever, _ = answer_client
    response = client.post("/ask", json={
        "question": "  How much leave?  ", "top_k": 2,
        "document_id": retriever.hits[0].document_id, "min_score": 0.6,
    })
    assert response.status_code == 200
    result = response.json()
    assert result["question"] == "How much leave?"
    assert result["answer"] == "Employees get 24 days of leave. [1]"
    assert result["answered"] is True
    citation = result["citations"][0]
    assert citation["source_id"] == 1
    assert citation["document_id"] == retriever.hits[0].document_id
    assert citation["filename"] == "leave.pdf"
    assert citation["score"] == 0.8
    assert citation["chunk"]["page_number"] == 1
    assert citation["chunk"]["text"] == "Employees get 24 days of annual leave."
    assert citation["chunk"]["end_char"] == len(citation["chunk"]["text"])
    assert retriever.calls == [("How much leave?", {
        "top_k": 2, "document_id": citation["document_id"], "min_score": 0.6,
    })]


def test_no_retrieved_evidence_returns_abstention_without_a_model_call(answer_client):
    client, _, retriever, provider = answer_client
    retriever.hits = []
    provider.failure = GenerationUnavailableError("not configured")
    response = client.post("/ask", json={"question": "Question?"})
    assert response.status_code == 200
    assert response.json() == {
        "question": "Question?", "answer": INSUFFICIENT_EVIDENCE,
        "answered": False, "citations": [],
    }
    assert provider.calls == []


def test_model_can_abstain_when_passages_do_not_answer_the_question(answer_client):
    client, _, _, provider = answer_client
    provider.answer = GeneratedAnswer(supported=False, statements=[])
    response = client.post("/ask", json={"question": "What is the pension policy?"})
    assert response.status_code == 200
    assert response.json()["answered"] is False
    assert response.json()["citations"] == []


def test_invented_source_reference_is_rejected_instead_of_returning_an_answer(answer_client):
    client, _, _, provider = answer_client
    provider.answer = GeneratedAnswer(
        supported=True,
        statements=[SupportedStatement(text="Invented citation.", source_ids=[2])],
    )
    response = client.post("/ask", json={"question": "Question?"})
    assert response.status_code == 502
    assert "retrieved evidence" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [{}, {"question": " "}, {"question": "x" * 2001},
     {"question": "x", "top_k": 0}, {"question": "x", "top_k": 6},
     {"question": "x", "document_id": "invented"},
     {"question": "x", "min_score": -1.1}, {"question": "x", "min_score": 1.1}],
)
def test_invalid_questions_are_rejected_before_retrieval(answer_client, body):
    client, _, retriever, provider = answer_client
    assert client.post("/ask", json=body).status_code == 422
    assert retriever.calls == []
    assert provider.calls == []


@pytest.mark.parametrize(
    "failure,status",
    [(GenerationUnavailableError("not available"), 503),
     (GenerationTimeoutError("timed out"), 504),
     (InvalidGenerationError("invalid structured output"), 502)],
)
def test_provider_failures_have_consistent_http_statuses(answer_client, failure, status):
    client, _, _, provider = answer_client
    provider.failure = failure
    response = client.post("/ask", json={"question": "Question?"})
    assert response.status_code == status
    assert response.json() == {"detail": str(failure)}


@pytest.mark.parametrize(
    "failure,status",
    [(EmbeddingInputError("too many tokens"), 422),
     (EmbeddingUnavailableError("model unavailable"), 503),
     (StorageUnavailableError("storage unavailable"), 503)],
)
def test_retrieval_failures_do_not_call_the_model(answer_client, failure, status):
    client, _, retriever, provider = answer_client
    retriever.failure = failure
    assert client.post("/ask", json={"question": "Question?"}).status_code == status
    assert provider.calls == []


def test_unconfigured_provider_is_an_actionable_service_error(answer_client):
    client, service, _, _ = answer_client
    service.provider = create_llm_provider(Settings(
        _env_file=None, llm_provider=None, llm_model=None,
    ))
    response = client.post("/ask", json={"question": "Question?"})
    assert response.status_code == 503
    assert "LLM_PROVIDER and LLM_MODEL" in response.json()["detail"]


@pytest.mark.parametrize("provider_name", ["openai", "ollama"])
def test_switching_adapters_preserves_the_http_contract_and_citations(answer_client, provider_name):
    client, service, _, _ = answer_client
    calls = []
    content = '{"supported": true, "statements": [{"text": "24 days.", "source_ids": [1]}]}'

    def handle(request):
        calls.append(request)
        if provider_name == "openai":
            data = {"status": "completed", "output": [
                {"type": "message", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": content}]},
            ]}
        else:
            data = {"done": True, "done_reason": "stop",
                    "message": {"role": "assistant", "content": content}}
        return httpx.Response(200, json=data)

    service.provider = create_llm_provider(
        Settings(_env_file=None, llm_provider=provider_name,
                 llm_model="chosen-model", openai_api_key="test-only-key"),
        transport=httpx.MockTransport(handle),
    )
    response = client.post("/ask", json={"question": "How much leave?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "24 days. [1]"
    assert response.json()["citations"][0]["filename"] == "leave.pdf"
    assert len(calls) == 1
    assert json.loads(calls[0].content)["model"] == "chosen-model"


def test_openapi_documents_ask_request_and_response_contracts(answer_client):
    client, _, _, _ = answer_client
    schema = client.get("/openapi.json").json()
    endpoint = schema["paths"]["/ask"]["post"]
    assert {"200", "422", "502", "503", "504"} <= endpoint["responses"].keys()
    assert "AnswerResponse" in schema["components"]["schemas"]
    assert "SourceCitation" in schema["components"]["schemas"]
