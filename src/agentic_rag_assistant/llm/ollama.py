"""Ollama chat API adapter for sourced, structured answers."""

from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from agentic_rag_assistant.answering import GeneratedAnswer, InvalidGenerationError
from agentic_rag_assistant.llm.common import messages, parse_answer, post_json
from agentic_rag_assistant.models import SearchHit


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

    def generate(self, question: str, sources: list[SearchHit]) -> GeneratedAnswer:
        data = post_json(
            f"{self.base_url}/api/chat",
            {
                "model": self.model,
                "messages": messages(question, sources),
                "format": GeneratedAnswer.model_json_schema(),
                "stream": False,
                "options": {
                    "temperature": 0, "num_ctx": 8_192,
                    "num_predict": self.max_output_tokens,
                },
            },
            timeout=self.timeout, transport=self.transport,
        )
        try:
            envelope = ChatEnvelope.model_validate(data)
        except ValidationError as exc:
            raise InvalidGenerationError("Ollama returned an invalid response envelope.") from exc
        if not envelope.done or envelope.done_reason == "length":
            raise InvalidGenerationError("Ollama did not complete the structured answer.")
        return parse_answer(envelope.message.content)
