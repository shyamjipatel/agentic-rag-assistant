"""Select an LLM adapter in one place using application configuration."""

import httpx

from agentic_rag_assistant.answering import (
    GeneratedAnswer, GenerationUnavailableError, LLMProvider,
)
from agentic_rag_assistant.llm.huggingface import HuggingFaceProvider
from agentic_rag_assistant.llm.ollama import OllamaProvider
from agentic_rag_assistant.llm.openai import OpenAIProvider
from agentic_rag_assistant.models import SearchHit
from agentic_rag_assistant.settings import Settings


class UnconfiguredProvider:
    def generate(self, question: str, sources: list[SearchHit]) -> GeneratedAnswer:
        raise GenerationUnavailableError(
            "Set LLM_MODEL for the selected remote provider to generate answers."
        )


def create_llm_provider(
    settings: Settings, *, transport: httpx.BaseTransport | None = None,
) -> LLMProvider:
    if settings.llm_model is None:
        return UnconfiguredProvider()
    if settings.llm_provider == "huggingface":
        token = settings.hf_token.get_secret_value() if settings.hf_token else ""
        return HuggingFaceProvider(
            model=settings.llm_model, token=token,
            timeout=settings.llm_timeout_seconds,
            max_output_tokens=settings.llm_max_output_tokens, transport=transport,
        )
    if settings.llm_provider == "openai":
        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
        return OpenAIProvider(
            model=settings.llm_model, api_key=key,
            timeout=settings.llm_timeout_seconds,
            max_output_tokens=settings.llm_max_output_tokens, transport=transport,
        )
    if settings.llm_provider == "ollama":
        return OllamaProvider(
            model=settings.llm_model, base_url=str(settings.ollama_base_url),
            timeout=settings.llm_timeout_seconds,
            max_output_tokens=settings.llm_max_output_tokens, transport=transport,
        )
    raise GenerationUnavailableError("The configured LLM provider is unsupported.")
