"""Remote Hugging Face Inference Providers adapter for structured answers."""

from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from agentic_rag_assistant.answering import (
    GeneratedAnswer, GenerationUnavailableError, InvalidGenerationError,
)
from agentic_rag_assistant.llm.common import messages, parse_answer, post_json
from agentic_rag_assistant.models import SearchHit


class CompletionMessage(BaseModel):
    role: Literal["assistant"]
    content: str | None = None
    refusal: str | None = None


class CompletionChoice(BaseModel):
    finish_reason: str
    message: CompletionMessage


class CompletionEnvelope(BaseModel):
    model_config = ConfigDict(strict=True)

    choices: list[CompletionChoice]


class HuggingFaceProvider:
    def __init__(
        self, *, model: str, token: str, timeout: float, max_output_tokens: int,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self.token = token
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.transport = transport

    def generate(self, question: str, sources: list[SearchHit]) -> GeneratedAnswer:
        if not self.token.strip():
            raise GenerationUnavailableError("Set HF_TOKEN for the Hugging Face provider.")
        data = post_json(
            "https://router.huggingface.co/v1/chat/completions",
            {
                "model": self.model,
                "messages": messages(question, sources),
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "sourced_answer", "strict": True,
                        "schema": GeneratedAnswer.model_json_schema(),
                    },
                },
                "max_tokens": self.max_output_tokens,
                "stream": False,
            },
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.token}"},
            transport=self.transport,
        )
        try:
            envelope = CompletionEnvelope.model_validate(data)
        except ValidationError as exc:
            raise InvalidGenerationError(
                "Hugging Face returned an invalid response envelope."
            ) from exc
        if len(envelope.choices) != 1:
            raise InvalidGenerationError("Hugging Face returned an unexpected number of answers.")
        choice = envelope.choices[0]
        if choice.message.refusal:
            raise InvalidGenerationError("Hugging Face declined to produce a sourced answer.")
        if choice.finish_reason != "stop":
            raise InvalidGenerationError("Hugging Face did not complete the structured answer.")
        if choice.message.content is None:
            raise InvalidGenerationError("Hugging Face returned no structured answer text.")
        return parse_answer(choice.message.content)
