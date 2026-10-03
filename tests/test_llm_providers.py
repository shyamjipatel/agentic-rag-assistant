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
from agentic_rag_assistant.llm.huggingface import HuggingFaceProvider
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
        "OPENAI_API_KEY", "OLLAMA_BASE_URL", "HF_TOKEN",
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
    if provider == "huggingface":
        return {"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": content,
        }}]}
    return {"done": True, "done_reason": "stop",
            "message": {"role": "assistant", "content": content}}


def configured(provider: str, handler, **overrides):
    values = {
        "llm_provider": provider, "llm_model": "operator-chosen-model",
        "openai_api_key": TEST_KEY, "hf_token": TEST_KEY,
    }
    values.update(overrides)
    return create_llm_provider(
        Settings(_env_file=None, **values), transport=httpx.MockTransport(handler)
    )


@pytest.mark.parametrize("provider", ["openai", "ollama", "huggingface"])
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
    elif provider == "huggingface":
        assert str(request.url) == "https://router.huggingface.co/v1/chat/completions"
        assert request.headers["Authorization"] == f"Bearer {TEST_KEY}"
        assert payload["response_format"]["json_schema"]["strict"] is True
        schema = payload["response_format"]["json_schema"]["schema"]
        assert schema == GeneratedAnswer.model_json_schema()
        assert payload["max_tokens"] == 512
    else:
        assert str(request.url) == "http://127.0.0.1:11434/api/chat"
        assert "Authorization" not in request.headers
        assert payload["format"] == GeneratedAnswer.model_json_schema()
        assert payload["options"]["num_predict"] == 512
        assert payload["options"]["num_ctx"] == 8192


@pytest.mark.parametrize("provider", ["openai", "ollama", "huggingface"])
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


@pytest.mark.parametrize("provider", ["openai", "ollama", "huggingface"])
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


@pytest.mark.parametrize("provider", ["openai", "ollama", "huggingface"])
@pytest.mark.parametrize("data", [[], {}, {"unexpected": "body"}])
def test_invalid_provider_envelopes_are_rejected(provider, sources, data):
    with pytest.raises(InvalidGenerationError):
        configured(provider, lambda request: httpx.Response(200, json=data)).generate(
            "Question?", sources
        )


@pytest.mark.parametrize("provider", ["openai", "ollama", "huggingface"])
def test_non_json_http_response_is_rejected(provider, sources):
    adapter = configured(provider, lambda request: httpx.Response(200, text="not JSON"))
    with pytest.raises(InvalidGenerationError):
        adapter.generate("Question?", sources)


@pytest.mark.parametrize("provider", ["openai", "ollama", "huggingface"])
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


@pytest.mark.parametrize("provider", ["openai", "ollama", "huggingface"])
def test_incomplete_output_is_rejected_even_with_valid_answer_json(provider, sources):
    data = envelope(provider)
    if provider == "openai":
        data["status"] = "incomplete"
    elif provider == "huggingface":
        data["choices"][0]["finish_reason"] = "length"
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
    "provider,expected",
    [("openai", OpenAIProvider), ("ollama", OllamaProvider), ("huggingface", HuggingFaceProvider)],
)
def test_configuration_selects_provider_without_changing_the_model(provider, expected):
    adapter = create_llm_provider(Settings(
        _env_file=None, llm_provider=provider, llm_model="exact-model-name"
    ))
    assert isinstance(adapter, expected)
    assert adapter.model == "exact-model-name"


@pytest.mark.parametrize("provider", ["openai", "huggingface"])
def test_remote_provider_requires_an_explicit_model(provider, sources):
    adapter = create_llm_provider(Settings(_env_file=None, llm_provider=provider))
    with pytest.raises(GenerationUnavailableError, match="LLM_MODEL"):
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


def test_empty_configuration_uses_the_local_default(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "")
    monkeypatch.setenv("LLM_MODEL", "")
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "ollama"
    assert settings.llm_model == "qwen2.5:7b"


def test_defaults_select_local_inference_without_credentials():
    settings = Settings(_env_file=None)
    adapter = create_llm_provider(settings)
    assert isinstance(adapter, OllamaProvider)
    assert adapter.model == "qwen2.5:7b"
    assert settings.hf_token is None
    assert settings.openai_api_key is None


def test_a_custom_local_model_overrides_the_default(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "custom-local-model")
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "ollama"
    assert create_llm_provider(settings).model == "custom-local-model"


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
        EmptyRetriever(), create_llm_provider(Settings(_env_file=None, llm_provider="huggingface"))
    ).ask("Question?")
    assert result.answered is False
    assert result.citations == []


def test_failed_local_requests_do_not_fall_back_to_remote_providers(sources):
    calls = []

    def handle(request):
        calls.append(request)
        raise httpx.ConnectError("local model unavailable")

    adapter = create_llm_provider(
        Settings(_env_file=None, hf_token=TEST_KEY, openai_api_key=TEST_KEY),
        transport=httpx.MockTransport(handle),
    )
    with pytest.raises(GenerationUnavailableError):
        adapter.generate("Question?", sources)
    assert len(calls) == 1
    assert str(calls[0].url) == "http://127.0.0.1:11434/api/chat"
    assert "Authorization" not in calls[0].headers


@pytest.mark.parametrize("token", [None, "   "])
def test_missing_huggingface_token_fails_before_any_request(sources, token):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=envelope("huggingface"))

    with pytest.raises(GenerationUnavailableError, match="HF_TOKEN"):
        configured("huggingface", handle, hf_token=token).generate("Question?", sources)
    assert calls == []


@pytest.mark.parametrize(
    "choice",
    [{"finish_reason": "stop", "message": {"role": "assistant", "refusal": "declined"}},
     {"finish_reason": "stop", "message": {"role": "assistant", "content": None}},
     {"finish_reason": "tool_calls", "message": {"role": "assistant", "content": "{}"}}],
)
def test_huggingface_refusals_missing_content_and_tool_calls_are_rejected(sources, choice):
    data = {"choices": [choice]}
    adapter = configured("huggingface", lambda request: httpx.Response(200, json=data))
    with pytest.raises(InvalidGenerationError):
        adapter.generate("Question?", sources)


def test_remote_provider_loads_token_and_preserves_model_routing_suffix(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "huggingface")
    monkeypatch.setenv("LLM_MODEL", "Qwen/Qwen3-32B:cerebras")
    monkeypatch.setenv("HF_TOKEN", TEST_KEY)
    settings = Settings(_env_file=None)
    adapter = create_llm_provider(settings)
    assert isinstance(adapter, HuggingFaceProvider)
    assert adapter.model == "Qwen/Qwen3-32B:cerebras"
    assert settings.hf_token.get_secret_value() == TEST_KEY
    assert TEST_KEY not in repr(settings)


@pytest.mark.parametrize("choices", [[], [envelope("huggingface")["choices"][0]] * 2])
def test_huggingface_rejects_missing_or_multiple_answers(sources, choices):
    adapter = configured(
        "huggingface", lambda request: httpx.Response(200, json={"choices": choices})
    )
    with pytest.raises(InvalidGenerationError, match="number of answers"):
        adapter.generate("Question?", sources)
