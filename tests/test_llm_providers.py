import json

import httpx
import pytest
from pydantic import ValidationError

from agentic_rag_assistant.answering import (
    SYSTEM_PROMPT,
    AnswerService,
    GeneratedAnswer,
    GenerationTimeoutError,
    GenerationUnavailableError,
    InvalidGenerationError,
)
from agentic_rag_assistant.llm.factory import create_llm_provider
from agentic_rag_assistant.llm.ollama import OllamaProvider
from agentic_rag_assistant.llm.openai import OpenAIProvider
from agentic_rag_assistant.models import DocumentChunk, SearchHit
from agentic_rag_assistant.settings import Settings

ANSWER = {"supported": True, "statements": [{"text": "24 days.", "source_ids": [1]}]}
TEST_KEY = "test-only-provider-key"


@pytest.fixture(autouse=True)
def isolate_llm_environment(monkeypatch):
    for variable in (
        "LLM_PROVIDER", "LLM_MODEL", "LLM_TIMEOUT_SECONDS", "LLM_MAX_OUTPUT_TOKENS",
        "OPENAI_API_KEY", "OLLAMA_BASE_URL",
    ):
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture
def sources():
    text = "Employees get 24 days of annual leave."
    return [SearchHit("a" * 64, "leave.pdf", 0.8,
                      DocumentChunk("stored-chunk", 0, text, 3, 0, len(text)))]


def envelope(provider: str, answer=ANSWER):
    content = json.dumps(answer)
    if provider == "openai":
        return {"status": "completed", "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": content}]},
        ]}
    return {"done": True, "done_reason": "stop",
            "message": {"role": "assistant", "content": content}}


def configured(provider: str, handler, **overrides):
    values = {
        "llm_provider": provider, "llm_model": "operator-chosen-model",
        "openai_api_key": TEST_KEY,
    }
    values.update(overrides)
    return create_llm_provider(
        Settings(_env_file=None, **values), transport=httpx.MockTransport(handler)
    )


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_adapters_share_an_answer_contract_and_send_only_retrieved_evidence(provider, sources):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=envelope(provider))

    adapter = configured(provider, handle, llm_timeout_seconds=30, llm_max_output_tokens=512)
    result = adapter.generate("How much leave?", sources)
    assert result == GeneratedAnswer.model_validate(ANSWER)
    assert len(requests) == 1
    request = requests[0]
    payload = json.loads(request.content)
    assert payload["model"] == "operator-chosen-model"
    assert payload["stream"] is False
    assert request.extensions["timeout"]["read"] == 30
    provider_messages = payload["input"] if provider == "openai" else payload["messages"]
    assert provider_messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert json.loads(provider_messages[1]["content"]) == {
        "question": "How much leave?",
        "sources": [{"source_id": 1, "text": sources[0].chunk.text}],
    }
    if provider == "openai":
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert request.headers["Authorization"] == f"Bearer {TEST_KEY}"
        assert payload["text"]["format"]["strict"] is True
        assert payload["text"]["format"]["schema"] == GeneratedAnswer.model_json_schema()
        assert payload["max_output_tokens"] == 512
        assert payload["store"] is False
    else:
        assert str(request.url) == "http://127.0.0.1:11434/api/chat"
        assert "Authorization" not in request.headers
        assert payload["format"] == GeneratedAnswer.model_json_schema()
        assert payload["options"]["num_predict"] == 512
        assert payload["options"]["num_ctx"] == 8192


@pytest.mark.parametrize("provider", ["openai", "ollama"])
@pytest.mark.parametrize(
    ("failure", "expected"),
    [(httpx.ReadTimeout("upstream timeout"), GenerationTimeoutError),
     (httpx.ConnectError("upstream connection error"), GenerationUnavailableError),
     (httpx.RemoteProtocolError("upstream protocol error"), GenerationUnavailableError)],
)
def test_transport_errors_are_normalized(provider, sources, failure, expected):
    def handle(request):
        raise failure

    with pytest.raises(expected):
        configured(provider, handle).generate("Question?", sources)


@pytest.mark.parametrize("provider", ["openai", "ollama"])
@pytest.mark.parametrize("status", [400, 401, 404, 429, 503])
def test_rejected_requests_do_not_expose_upstream_body_or_credentials(provider, sources, status):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(status, text=f"private upstream details: {TEST_KEY}")

    with pytest.raises(GenerationUnavailableError) as error:
        configured(provider, handle).generate("Question?", sources)
    assert TEST_KEY not in str(error.value)
    assert "private upstream details" not in str(error.value)
    assert len(calls) == 1  # No implicit retries or calls to another provider.


@pytest.mark.parametrize("provider", ["openai", "ollama"])
@pytest.mark.parametrize("data", [[], {}, {"unexpected": "body"}])
def test_invalid_provider_envelopes_are_rejected(provider, sources, data):
    with pytest.raises(InvalidGenerationError):
        configured(provider, lambda request: httpx.Response(200, json=data)).generate(
            "Question?", sources
        )


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_non_json_http_response_is_rejected(provider, sources):
    adapter = configured(provider, lambda request: httpx.Response(200, text="not JSON"))
    with pytest.raises(InvalidGenerationError):
        adapter.generate("Question?", sources)


@pytest.mark.parametrize("provider", ["openai", "ollama"])
@pytest.mark.parametrize(
    "answer",
    [{"supported": True}, {"supported": True, "statements": [{"text": "Uncited claim"}]},
     {"supported": "true", "statements": []}],
)
def test_invalid_answer_schema_is_rejected(provider, sources, answer):
    adapter = configured(
        provider, lambda request: httpx.Response(200, json=envelope(provider, answer))
    )
    with pytest.raises(InvalidGenerationError):
        adapter.generate("Question?", sources)


def test_openai_refusal_is_handled_explicitly(sources):
    data = envelope("openai")
    data["output"][1]["content"] = [{"type": "refusal", "refusal": "Declined"}]
    adapter = configured("openai", lambda request: httpx.Response(200, json=data))
    with pytest.raises(InvalidGenerationError, match="declined"):
        adapter.generate("Question?", sources)


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_incomplete_output_is_rejected_even_with_valid_answer_json(provider, sources):
    data = envelope(provider)
    if provider == "openai":
        data["status"] = "incomplete"
    else:
        data["done_reason"] = "length"
    adapter = configured(provider, lambda request: httpx.Response(200, json=data))
    with pytest.raises(InvalidGenerationError, match="complete"):
        adapter.generate("Question?", sources)


def test_missing_openai_key_fails_before_any_request(sources):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=envelope("openai"))

    with pytest.raises(GenerationUnavailableError, match="OPENAI_API_KEY"):
        configured("openai", handle, openai_api_key=None).generate("Question?", sources)
    assert calls == []


def test_ollama_uses_the_operator_configured_endpoint(sources):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=envelope("ollama"))

    adapter = configured("ollama", handle, ollama_base_url="http://localhost:12345")
    adapter.generate("Question?", sources)
    assert str(calls[0].url) == "http://localhost:12345/api/chat"


@pytest.mark.parametrize(
    "provider,expected", [("openai", OpenAIProvider), ("ollama", OllamaProvider)]
)
def test_configuration_selects_provider_without_changing_the_model(provider, expected):
    adapter = create_llm_provider(Settings(
        _env_file=None, llm_provider=provider, llm_model="exact-model-name"
    ))
    assert isinstance(adapter, expected)
    assert adapter.model == "exact-model-name"


@pytest.mark.parametrize("values", [{}, {"llm_provider": "ollama"}, {"llm_model": "model"}])
def test_missing_selection_has_an_actionable_error(values, sources):
    adapter = create_llm_provider(Settings(_env_file=None, **values))
    with pytest.raises(GenerationUnavailableError, match="LLM_PROVIDER and LLM_MODEL"):
        adapter.generate("Question?", sources)


def test_settings_load_provider_and_model_from_environment_without_exposing_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-6-luna")
    monkeypatch.setenv("OPENAI_API_KEY", TEST_KEY)
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-6-luna"
    assert settings.openai_api_key.get_secret_value() == TEST_KEY
    assert TEST_KEY not in repr(settings)


def test_empty_example_configuration_leaves_provider_unselected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "")
    monkeypatch.setenv("LLM_MODEL", "")
    settings = Settings(_env_file=None)
    assert settings.llm_provider is None
    assert settings.llm_model is None


@pytest.mark.parametrize(
    "values",
    [{"llm_provider": "unknown"}, {"llm_model": "   "}, {"llm_timeout_seconds": 0},
     {"llm_max_output_tokens": 0}, {"ollama_base_url": "not a URL"}],
)
def test_invalid_configuration_is_rejected(values):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)


def test_an_unconfigured_provider_does_not_block_empty_retrieval():
    class EmptyRetriever:
        def search(self, query, **options):
            return []

    result = AnswerService(
        EmptyRetriever(), create_llm_provider(Settings(_env_file=None))
    ).ask("Question?")
    assert result.answered is False
    assert result.citations == []
