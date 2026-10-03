"""Remote Hugging Face Inference Providers adapter for structured answers."""

from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from agentic_rag_assistant.answering import (
    GeneratedAnswer, GenerationUnavailableError, InvalidGenerationError,
)
from agentic_rag_assistant.llm.common import parse_answer, post_json
from agentic_rag_assistant.llm.tool_support import (
    answer_messages, parse_chat_selection, tool_messages,
)
from agentic_rag_assistant.memory import (
    StandaloneQuestion, parse_rewritten_question, rewrite_messages,
)
from agentic_rag_assistant.models import ConversationTurn, SearchHit, ToolResult
from agentic_rag_assistant.tools import ToolSelection, calculator_definition


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

    def _request(
        self, conversation: list[dict], *, select_tools: bool = False,
        response_schema: type[BaseModel] = GeneratedAnswer,
    ):
        if not self.token.strip():
            raise GenerationUnavailableError("Set HF_TOKEN for the Hugging Face provider.")
        payload = {
            "model": self.model, "messages": conversation,
            "max_tokens": self.max_output_tokens, "stream": False,
        }
        if select_tools:
            payload.update({
                "tools": [{"type": "function", "function": calculator_definition()}],
                "tool_choice": "auto", "parallel_tool_calls": False,
            })
        else:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "sourced_answer" if response_schema is GeneratedAnswer else "question",
                    "strict": True, "schema": response_schema.model_json_schema(),
                },
            }
        return post_json(
            "https://router.huggingface.co/v1/chat/completions", payload,
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.token}"},
            transport=self.transport,
        )

    def select_tool(self, question: str, sources: list[SearchHit]) -> ToolSelection:
        data = self._request(tool_messages(question, sources), select_tools=True)
        try:
            envelope = CompletionEnvelope.model_validate(data)
        except ValidationError as exc:
            raise InvalidGenerationError("Hugging Face returned an invalid tool envelope.") from exc
        if len(envelope.choices) != 1:
            raise InvalidGenerationError("Hugging Face returned an unexpected number of decisions.")
        choice = envelope.choices[0]
        if choice.finish_reason not in ("stop", "tool_calls"):
            raise InvalidGenerationError("Hugging Face did not complete tool selection.")
        selection = parse_chat_selection(data["choices"][0]["message"])
        if (choice.finish_reason == "tool_calls") != (selection.call is not None):
            raise InvalidGenerationError("Hugging Face returned inconsistent tool selection.")
        return selection

    def generate(
        self, question: str, sources: list[SearchHit], *,
        tool_selection: ToolSelection | None = None, tool_result: ToolResult | None = None,
    ) -> GeneratedAnswer:
        data = self._request(answer_messages(
            question, sources, tool_selection, tool_result, provider="huggingface",
        ))
        return parse_answer(self._extract_text(data))

    def rewrite_question(self, question: str, history: list[ConversationTurn]) -> str:
        data = self._request(
            rewrite_messages(question, history), response_schema=StandaloneQuestion,
        )
        return parse_rewritten_question(self._extract_text(data))

    def _extract_text(self, data) -> str:
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
        if data["choices"][0]["message"].get("tool_calls"):
            raise InvalidGenerationError("Hugging Face requested a tool during final generation.")
        return choice.message.content
