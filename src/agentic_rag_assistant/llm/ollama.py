"""Ollama chat API adapter for sourced, structured answers."""

from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from agentic_rag_assistant.answering import GeneratedAnswer, InvalidGenerationError
from agentic_rag_assistant.llm.common import parse_answer, post_json
from agentic_rag_assistant.llm.tool_support import (
    answer_messages, parse_chat_selection, tool_messages,
)
from agentic_rag_assistant.models import SearchHit, ToolResult
from agentic_rag_assistant.tools import ToolSelection, calculator_definition


class ChatMessage(BaseModel):
    role: Literal["assistant"]
    content: str


class ChatEnvelope(BaseModel):
    model_config = ConfigDict(strict=True)

    done: bool
    done_reason: str | None = None
    message: ChatMessage


class OllamaProvider:
    def __init__(
        self, *, model: str, base_url: str, timeout: float, max_output_tokens: int,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.transport = transport

    def _request(self, conversation: list[dict], *, select_tools: bool = False):
        payload = {
            "model": self.model, "messages": conversation, "stream": False,
            "options": {
                "temperature": 0, "num_ctx": 8_192,
                "num_predict": self.max_output_tokens,
            },
        }
        if select_tools:
            payload["tools"] = [{"type": "function", "function": calculator_definition()}]
        else:
            payload["format"] = GeneratedAnswer.model_json_schema()
        return post_json(
            f"{self.base_url}/api/chat", payload,
            timeout=self.timeout, transport=self.transport,
        )

    def select_tool(self, question: str, sources: list[SearchHit]) -> ToolSelection:
        data = self._request(tool_messages(question, sources), select_tools=True)
        try:
            envelope = ChatEnvelope.model_validate(data)
        except ValidationError as exc:
            raise InvalidGenerationError("Ollama returned an invalid tool envelope.") from exc
        if not envelope.done or envelope.done_reason == "length":
            raise InvalidGenerationError("Ollama did not complete tool selection.")
        return parse_chat_selection(data["message"], ollama=True)

    def generate(
        self, question: str, sources: list[SearchHit], *,
        tool_selection: ToolSelection | None = None, tool_result: ToolResult | None = None,
    ) -> GeneratedAnswer:
        data = self._request(answer_messages(
            question, sources, tool_selection, tool_result, provider="ollama",
        ))
        try:
            envelope = ChatEnvelope.model_validate(data)
        except ValidationError as exc:
            raise InvalidGenerationError("Ollama returned an invalid response envelope.") from exc
        if not envelope.done or envelope.done_reason == "length":
            raise InvalidGenerationError("Ollama did not complete the structured answer.")
        if data["message"].get("tool_calls"):
            raise InvalidGenerationError("Ollama requested a tool during final generation.")
        return parse_answer(envelope.message.content)
