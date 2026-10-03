"""Native provider tool calls through the real graph and HTTP boundary."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from agentic_rag_assistant.agent import RAGAgent
from agentic_rag_assistant.answers import get_answer_service
from agentic_rag_assistant.llm.factory import create_llm_provider
from agentic_rag_assistant.main import app
from agentic_rag_assistant.models import DocumentChunk, SearchHit
from agentic_rag_assistant.settings import Settings

ARGS = {"operation": "multiply", "left": "24", "right": "3", "source_ids": [1]}
ANSWER = {
    "supported": True, "statements": [{"text": "72 days over three years.", "source_ids": [1]}],
}


def tool_envelope(provider, *, name="calculator", args=None, count=1, call=True):
    args = ARGS if args is None else args
    if provider == "openai":
        output = [{"type": "reasoning", "summary": [], "encrypted_content": "test-reasoning"}]
        if call:
            output += [{"type": "function_call", "status": "completed", "call_id": f"call-{i}",
                        "name": name, "arguments": json.dumps(args)} for i in range(count)]
        else:
            output += [{"type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "No calculation needed."}]}]
        return {"status": "completed", "output": output}
    message = {"role": "assistant", "content": "" if call else "No calculation needed."}
    if call:
        message["tool_calls"] = [
            {"type": "function", "id": f"call-{i}", "function": {
                "name": name, "arguments": args if provider == "ollama" else json.dumps(args),
            }} for i in range(count)
        ]
    if provider == "ollama":
        message["thinking"] = "Use the calculator."
        return {"done": True, "done_reason": "stop", "message": message}
    return {"choices": [{"finish_reason": "tool_calls" if call else "stop", "message": message}]}


def answer_envelope(provider, answer=None):
    content = json.dumps(ANSWER if answer is None else answer)
    if provider == "openai":
        return {"status": "completed", "output": [
            {"type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": content}]},
        ]}
    if provider == "ollama":
        return {"done": True, "done_reason": "stop",
                "message": {"role": "assistant", "content": content}}
    return {"choices": [{"finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}]}


class Retriever:
    def __init__(self):
        text = "Employees receive 24 days of annual leave."
        self.hits = [SearchHit("a" * 64, "leave.pdf", 0.8,
                               DocumentChunk("stored", 0, text, 2, 0, len(text)))]
        self.calls = []

    def search(self, query, **options):
        self.calls.append((query, options))
        return self.hits


@pytest.fixture(params=["openai", "ollama", "huggingface"])
def tool_client(request):
    provider = request.param
    requests = []
    replies = [tool_envelope(provider), answer_envelope(provider)]

    def handle(outgoing):
        requests.append(json.loads(outgoing.content))
        response = replies.pop(0)
        if isinstance(response, Exception):
            raise response
        return httpx.Response(200, json=response)

    adapter = create_llm_provider(Settings(
        _env_file=None, llm_provider=provider, llm_model="test-model",
        hf_token="test-token", openai_api_key="test-token",
    ), transport=httpx.MockTransport(handle))
    retriever = Retriever()
    agent = RAGAgent(retriever, adapter)
    app.dependency_overrides[get_answer_service] = lambda: agent
    try:
        with TestClient(app) as client:
            yield client, provider, requests, replies, retriever, agent
    finally:
        app.dependency_overrides.pop(get_answer_service, None)


def test_native_call_executes_and_correlated_result_reaches_final_generation(tool_client):
    client, provider, requests, _, retriever, _ = tool_client
    response = client.post("/ask", json={
        "question": "How much annual leave over three years?", "use_tools": True,
        "top_k": 2, "document_id": "a" * 64, "min_score": 0.7,
    })
    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == "72 days over three years. [1]"
    assert result["citations"][0]["filename"] == "leave.pdf"
    assert result["citations"][0]["chunk"]["page_number"] == 2
    assert result["tool_results"] == [{
        "tool_name": "calculator", **ARGS, "value": "72",
    }]
    assert len(requests) == 2
    assert retriever.calls[0][1] == {"top_k": 2, "document_id": "a" * 64, "min_score": 0.7}
    assert "tools" in requests[0] and "tools" not in requests[1]
    if provider == "openai":
        assert requests[0]["parallel_tool_calls"] is False
        assert requests[0]["tools"][0]["strict"] is True
        conversation = requests[1]["input"]
        output = conversation[-1]
        assert output["type"] == "function_call_output" and output["call_id"] == "call-0"
        assert conversation[2]["encrypted_content"] == "test-reasoning"
        assert requests[0]["store"] is False and requests[1]["store"] is False
        rendered = output["output"]
    else:
        conversation = requests[1]["messages"]
        output = conversation[-1]
        assert output["role"] == "tool"
        if provider == "ollama":
            assert output["tool_name"] == "calculator"
            assert conversation[2]["thinking"] == "Use the calculator."
        else:
            assert output["tool_call_id"] == "call-0"
        rendered = output["content"]
    assert json.loads(rendered)["value"] == "72"
    assert "test-token" not in response.text
    assert "continuation" not in response.text


@pytest.mark.parametrize(
    "name,args,count",
    [("shell", ARGS, 1), ("calculator", {**ARGS, "left": "NaN"}, 1),
     ("calculator", {**ARGS, "source_ids": [2]}, 1),
     ("calculator", {**ARGS, "operation": "divide", "right": "0"}, 1),
     ("calculator", ARGS, 2)],
)
def test_rejected_calls_stop_before_final_generation(tool_client, name, args, count):
    client, provider, requests, replies, _, _ = tool_client
    replies[0] = tool_envelope(provider, name=name, args=args, count=count)
    response = client.post("/ask", json={"question": "Calculate leave.", "use_tools": True})
    assert response.status_code == 502
    assert len(requests) == 1


def test_model_can_decline_the_tool_and_still_answer(tool_client):
    client, provider, requests, replies, _, _ = tool_client
    replies[0] = tool_envelope(provider, call=False)
    response = client.post("/ask", json={"question": "How much leave?", "use_tools": True})
    assert response.status_code == 200
    assert response.json()["tool_results"] == []
    assert len(requests) == 2


def test_empty_evidence_skips_tool_selection_and_generation(tool_client):
    client, _, requests, _, retriever, _ = tool_client
    retriever.hits = []
    response = client.post("/ask", json={"question": "Calculate leave.", "use_tools": True})
    assert response.status_code == 200
    assert response.json()["answered"] is False
    assert response.json()["tool_results"] == []
    assert requests == []


def test_tool_results_do_not_leak_into_a_subsequent_plain_rag_request(tool_client):
    client, provider, requests, replies, _, _ = tool_client
    first = client.post("/ask", json={"question": "Calculate.", "use_tools": True})
    assert first.status_code == 200
    replies.append(answer_envelope(provider))
    response = client.post("/ask", json={"question": "How much leave?"})
    assert response.status_code == 200
    assert response.json()["tool_results"] == []
    assert len(requests) == 3
    conversation = requests[-1]["input" if provider == "openai" else "messages"]
    assert len(conversation) == 2


@pytest.mark.parametrize("failure,status", [
    (httpx.ReadTimeout("private upstream details"), 504),
    (httpx.ConnectError("private upstream details"), 503),
])
def test_tool_selection_errors_use_existing_safe_http_mapping(tool_client, failure, status):
    client, _, requests, replies, _, _ = tool_client
    replies[0] = failure
    response = client.post("/ask", json={"question": "Calculate.", "use_tools": True})
    assert response.status_code == status
    assert "private upstream details" not in response.text
    assert len(requests) == 1


@pytest.mark.parametrize("invalid", ["true", 1, None])
def test_tool_mode_requires_a_json_boolean(tool_client, invalid):
    client, _, requests, _, retriever, _ = tool_client
    response = client.post("/ask", json={"question": "Calculate.", "use_tools": invalid})
    assert response.status_code == 422
    assert requests == [] and retriever.calls == []


def test_final_answer_must_cite_calculator_supporting_sources(tool_client):
    client, provider, requests, replies, retriever, _ = tool_client
    retriever.hits.append(retriever.hits[0])
    replies[1] = answer_envelope(provider, {
        "supported": True, "statements": [{"text": "72 days.", "source_ids": [2]}],
    })
    response = client.post("/ask", json={"question": "Calculate.", "use_tools": True})
    assert response.status_code == 502
    assert "supporting sources" in response.json()["detail"]
    assert len(requests) == 2


@pytest.mark.parametrize("case", ["empty", "incomplete", "refusal"])
def test_invalid_native_decisions_do_not_execute_tools(tool_client, case):
    client, provider, requests, replies, _, _ = tool_client
    invalid = tool_envelope(provider)
    if case == "empty":
        invalid = {}
    elif case == "incomplete":
        if provider == "openai":
            invalid["status"] = "incomplete"
        elif provider == "ollama":
            invalid["done"] = False
        else:
            invalid["choices"][0]["finish_reason"] = "length"
    elif provider == "openai":
        invalid["output"].append({
            "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "refusal", "text": "Refused"}],
        })
    else:
        message = invalid["message"] if provider == "ollama" else invalid["choices"][0]["message"]
        message["refusal"] = "Refused"
    replies[0] = invalid
    response = client.post("/ask", json={"question": "Calculate.", "use_tools": True})
    assert response.status_code == 502
    assert len(requests) == 1


def test_final_generation_cannot_request_another_tool(tool_client):
    client, provider, requests, replies, _, _ = tool_client
    final = answer_envelope(provider)
    extra = tool_envelope(provider)
    if provider == "openai":
        final["output"] += extra["output"]
    elif provider == "ollama":
        final["message"]["tool_calls"] = extra["message"]["tool_calls"]
    else:
        final["choices"][0]["message"]["tool_calls"] = extra["choices"][0]["message"]["tool_calls"]
    replies[1] = final
    response = client.post("/ask", json={"question": "Calculate.", "use_tools": True})
    assert response.status_code == 502
    assert len(requests) == 2
