"""OpenAI Responses API adapter for sourced, structured answers."""

import httpx
from pydantic import BaseModel, Field, ValidationError

from agentic_rag_assistant.answering import (
    GeneratedAnswer, GenerationUnavailableError, InvalidGenerationError,
)
from agentic_rag_assistant.llm.common import parse_answer, post_json
from agentic_rag_assistant.llm.tool_support import answer_messages, tool_messages
from agentic_rag_assistant.memory import (
    StandaloneQuestion, parse_rewritten_question, rewrite_messages,
)
from agentic_rag_assistant.models import ConversationTurn, SearchHit, ToolResult
from agentic_rag_assistant.tools import (
    ToolSelection, calculator_definition, parse_calculator_call,
)


class ResponseContent(BaseModel):
    type: str
    text: str | None = None


class ResponseOutput(BaseModel):
    type: str
    role: str | None = None
    status: str | None = None
    content: list[ResponseContent] = Field(default_factory=list)
    name: str | None = None
    call_id: str | None = None
    arguments: str | None = None


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

    def _request(
        self, conversation: list[dict], *, select_tools: bool = False,
        response_schema: type[BaseModel] = GeneratedAnswer,
    ):
        if not self.api_key.strip():
            raise GenerationUnavailableError("Set OPENAI_API_KEY for the OpenAI provider.")
        payload = {
            "model": self.model, "input": conversation,
            "max_output_tokens": self.max_output_tokens, "store": False, "stream": False,
        }
        if select_tools:
            payload.update({
                "tools": [{"type": "function", "strict": True, **calculator_definition()}],
                "tool_choice": "auto", "parallel_tool_calls": False,
                "include": ["reasoning.encrypted_content"],
            })
        else:
            payload["text"] = {"format": {
                "type": "json_schema", "strict": True,
                "name": "sourced_answer" if response_schema is GeneratedAnswer else "question",
                "schema": response_schema.model_json_schema(),
            }}
        return post_json(
            "https://api.openai.com/v1/responses", payload,
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
            transport=self.transport,
        )

    def select_tool(self, question: str, sources: list[SearchHit]) -> ToolSelection:
        data = self._request(tool_messages(question, sources), select_tools=True)
        try:
            envelope = ResponsesEnvelope.model_validate(data)
        except ValidationError as exc:
            raise InvalidGenerationError("OpenAI returned an invalid tool envelope.") from exc
        if envelope.status != "completed":
            raise InvalidGenerationError("OpenAI did not complete tool selection.")
        calls = [item for item in envelope.output if item.type == "function_call"]
        if len(calls) > 1:
            raise InvalidGenerationError("Only one calculator call is allowed per question.")
        for item in envelope.output:
            if item.type not in ("message", "reasoning", "function_call"):
                raise InvalidGenerationError("OpenAI returned an unsupported tool-selection item.")
            if item.type == "message" and (
                item.role != "assistant" or item.status != "completed"
                or any(content.type == "refusal" for content in item.content)
            ):
                raise InvalidGenerationError("OpenAI declined or did not complete tool selection.")
        if not calls:
            if not any(
                content.type == "output_text" and content.text
                for item in envelope.output for content in item.content
            ):
                raise InvalidGenerationError("OpenAI returned no tool-selection decision.")
            return ToolSelection(None, [])
        native = calls[0]
        if native.status not in (None, "completed") or not native.call_id or not native.name:
            raise InvalidGenerationError("OpenAI returned an incomplete function call.")
        if native.arguments is None:
            raise InvalidGenerationError("OpenAI returned no function arguments.")
        call = parse_calculator_call(native.call_id, native.name, native.arguments)
        # With store=false, replay reasoning items (including encrypted content)
        # alongside the function call before submitting its correlated output.
        return ToolSelection(call, data["output"])

    def generate(
        self, question: str, sources: list[SearchHit], *,
        tool_selection: ToolSelection | None = None, tool_result: ToolResult | None = None,
    ) -> GeneratedAnswer:
        data = self._request(answer_messages(
            question, sources, tool_selection, tool_result, provider="openai",
        ))
        return parse_answer(self._extract_text(data))

    def rewrite_question(self, question: str, history: list[ConversationTurn]) -> str:
        data = self._request(
            rewrite_messages(question, history), response_schema=StandaloneQuestion,
        )
        return parse_rewritten_question(self._extract_text(data))

    def _extract_text(self, data) -> str:
        try:
            envelope = ResponsesEnvelope.model_validate(data)
        except ValidationError as exc:
            raise InvalidGenerationError("OpenAI returned an invalid response envelope.") from exc
        if envelope.status != "completed":
            raise InvalidGenerationError("OpenAI did not complete the structured answer.")

        texts = []
        for item in envelope.output:
            if item.type == "function_call":
                raise InvalidGenerationError("OpenAI requested a tool during final generation.")
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
        return "".join(texts)
