import json
from datetime import datetime, timezone

import httpx
import pytest

from agentic_rag_assistant.agent import RAGAgent
from agentic_rag_assistant.answering import (
    GeneratedAnswer, InvalidGenerationError, SupportedStatement,
)
from agentic_rag_assistant.llm.factory import create_llm_provider
from agentic_rag_assistant.memory import (
    ANSWER_CONTEXT_CHARS, QUESTION_CONTEXT_CHARS, RECENT_TURNS,
    parse_rewritten_question, rewrite_messages,
)
from agentic_rag_assistant.models import AnswerResponse, ConversationTurn, DocumentChunk, SearchHit
from agentic_rag_assistant.settings import Settings


def turn(number, question="How much annual leave?", answer="24 days. [1]"):
    return ConversationTurn(number, AnswerResponse(question, answer, True, []),
                            datetime.now(timezone.utc))


def test_rewrite_context_is_bounded_and_excludes_source_snapshots_and_tool_state():
    history = [turn(i, "q" * 2_000, "a" * 5_000) for i in range(8)]
    payload = json.loads(rewrite_messages("And over three years?", history)[1]["content"])
    assert payload["current_question"] == "And over three years?"
    assert len(payload["recent_conversation"]) == RECENT_TURNS
    assert all(len(item["question"]) == QUESTION_CONTEXT_CHARS
               and len(item["answer"]) == ANSWER_CONTEXT_CHARS
               for item in payload["recent_conversation"])
    assert all(set(item) == {"question", "answer"} for item in payload["recent_conversation"])


@pytest.mark.parametrize("content", [
    "invalid JSON", '{"question": " "}', '{"question": 1}',
    '{"question": "Valid", "facts": "invented"}',
    json.dumps({"question": "x" * 2_001}),
])
def test_invalid_standalone_questions_are_rejected(content):
    with pytest.raises(InvalidGenerationError, match="standalone question"):
        parse_rewritten_question(content)


@pytest.mark.parametrize("provider", ["openai", "ollama", "huggingface"])
def test_providers_rewrite_followups_using_the_same_strict_schema(provider):
    outgoing = []

    def handle(request):
        outgoing.append(json.loads(request.content))
        content = '{"question": "How much annual leave over three years?"}'
        if provider == "openai":
            data = {"status": "completed", "output": [
                {"type": "message", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": content}]},
            ]}
        elif provider == "ollama":
            data = {"done": True, "message": {"role": "assistant", "content": content}}
        else:
            data = {"choices": [{"finish_reason": "stop", "message": {
                "role": "assistant", "content": content,
            }}]}
        return httpx.Response(200, json=data)

    adapter = create_llm_provider(Settings(
        _env_file=None, llm_provider=provider, llm_model="test-model",
        hf_token="test-token", openai_api_key="test-token",
    ), transport=httpx.MockTransport(handle))
    assert adapter.rewrite_question("And over three years?", [turn(1)]) == (
        "How much annual leave over three years?"
    )
    payload = outgoing[0]
    assert "tools" not in payload
    if provider == "openai":
        schema = payload["text"]["format"]["schema"]
        messages = payload["input"]
    elif provider == "ollama":
        schema = payload["format"]
        messages = payload["messages"]
    else:
        schema = payload["response_format"]["json_schema"]["schema"]
        messages = payload["messages"]
    assert schema["required"] == ["question"] and schema["additionalProperties"] is False
    assert json.loads(messages[1]["content"])["recent_conversation"][0]["question"] == (
        "How much annual leave?"
    )


def test_followup_retrieves_fresh_evidence_and_keeps_the_original_question_in_response():
    calls = []
    text = "Annual leave is now 30 days."
    fresh = SearchHit("b" * 64, "updated.txt", 0.8,
                      DocumentChunk("updated", 0, text, None, 0, len(text)))

    class Retriever:
        def search(self, query, **options):
            calls.append(("retrieve", query, options))
            return [fresh]

    class Provider:
        def rewrite_question(self, question, history):
            calls.append(("rewrite", question, history))
            return "How much annual leave over three years?"

        def generate(self, question, sources):
            calls.append(("generate", question, sources))
            assert sources == [fresh]
            return GeneratedAnswer(supported=True, statements=[
                SupportedStatement(text="The current rate is 30 days per year.", source_ids=[1]),
            ])

    history = [turn(1)]
    agent = RAGAgent(Retriever(), Provider())
    response = agent.ask("And over three years?", history=history, document_id="b" * 64)
    assert [call[0] for call in calls] == ["rewrite", "retrieve", "generate"]
    assert calls[1][1] == calls[2][1] == "How much annual leave over three years?"
    assert calls[1][2]["document_id"] == "b" * 64
    assert response.question == "And over three years?"
    assert response.citations[0].filename == "updated.txt"
    assert "24" not in response.answer
    calls.clear()
    agent.ask("New unrelated question")
    assert [call[0] for call in calls] == ["retrieve", "generate"]
    assert calls[0][1] == "New unrelated question"
