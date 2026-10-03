"""Shared structured-output validation and HTTP transport for LLM adapters."""

import ssl
from typing import Any

import httpx
import truststore
from pydantic import ValidationError

from agentic_rag_assistant.answering import (
    SYSTEM_PROMPT,
    GeneratedAnswer,
    GenerationTimeoutError,
    GenerationUnavailableError,
    InvalidGenerationError,
    build_evidence_message,
)
from agentic_rag_assistant.models import SearchHit


def messages(question: str, sources: list[SearchHit]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_evidence_message(question, sources)},
    ]


def post_json(
    url: str, payload: dict[str, Any], *, timeout: float,
    headers: dict[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> Any:
    # Use system trust without changing process-wide SSL behavior.
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    try:
        with httpx.Client(
            timeout=timeout, verify=context, transport=transport,
            follow_redirects=False,
        ) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException as exc:
        raise GenerationTimeoutError("The configured LLM provider timed out.") from exc
    except httpx.HTTPError as exc:
        raise GenerationUnavailableError(
            "The configured LLM provider is unavailable or rejected the request. "
            "Check the model, credentials, and service readiness."
        ) from exc
    except ValueError as exc:
        raise InvalidGenerationError("The LLM provider returned an invalid JSON response.") from exc


def parse_answer(text: str) -> GeneratedAnswer:
    try:
        return GeneratedAnswer.model_validate_json(text)
    except ValidationError as exc:
        raise InvalidGenerationError(
            "The LLM provider returned an invalid structured answer."
        ) from exc
