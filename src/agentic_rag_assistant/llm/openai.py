"""OpenAI Responses API adapter for sourced, structured answers."""

import httpx
from pydantic import BaseModel, Field, ValidationError

from agentic_rag_assistant.answering import (
    GeneratedAnswer, GenerationUnavailableError, InvalidGenerationError,
)
from agentic_rag_assistant.llm.common import messages, parse_answer, post_json
from agentic_rag_assistant.models import SearchHit


class ResponseContent(BaseModel):
    type: str
    text: str | None = None


class ResponseOutput(BaseModel):
    type: str
    role: str | None = None
    status: str | None = None
    content: list[ResponseContent] = Field(default_factory=list)


class ResponsesEnvelope(BaseModel):
    status: str
    output: list[ResponseOutput]


class OpenAIProvider:
    def __init__(
        self, *, model: str, api_key: str, timeout: float, max_output_tokens: int,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.transport = transport

    def generate(self, question: str, sources: list[SearchHit]) -> GeneratedAnswer:
        if not self.api_key.strip():
            raise GenerationUnavailableError("Set OPENAI_API_KEY for the OpenAI provider.")
        data = post_json(
            "https://api.openai.com/v1/responses",
            {
                "model": self.model,
                "input": messages(question, sources),
                "text": {"format": {
                    "type": "json_schema", "name": "sourced_answer", "strict": True,
                    "schema": GeneratedAnswer.model_json_schema(),
                }},
                "max_output_tokens": self.max_output_tokens,
                "store": False,
                "stream": False,
            },
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
            transport=self.transport,
        )
        try:
            envelope = ResponsesEnvelope.model_validate(data)
        except ValidationError as exc:
            raise InvalidGenerationError("OpenAI returned an invalid response envelope.") from exc
        if envelope.status != "completed":
            raise InvalidGenerationError("OpenAI did not complete the structured answer.")

        texts = []
        for item in envelope.output:
            if item.type != "message":
                continue
            if item.role != "assistant" or item.status != "completed":
                raise InvalidGenerationError("OpenAI returned an incomplete answer message.")
            for content in item.content:
                if content.type == "refusal":
                    raise InvalidGenerationError("OpenAI declined to produce a sourced answer.")
                if content.type == "output_text" and content.text is not None:
                    texts.append(content.text)
        if not texts:
            raise InvalidGenerationError("OpenAI returned no structured answer text.")
        return parse_answer("".join(texts))
