"""Project configuration loaded from environment variables and a local .env."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AnyHttpUrl, Field, SecretStr, StringConstraints
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    database_url: SecretStr = SecretStr(
        "postgresql://rag:rag_local_dev@127.0.0.1:55432/agentic_rag"
    )
    model_cache_dir: Path = Path(".cache/embeddings")
    llm_provider: Literal["openai", "ollama"] | None = None
    llm_model: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] | None = None
    llm_timeout_seconds: float = Field(default=120, gt=0, le=600)
    llm_max_output_tokens: int = Field(default=4_096, ge=128, le=16_384)
    openai_api_key: SecretStr | None = None
    ollama_base_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:11434")


@lru_cache
def get_settings() -> Settings:
    return Settings()
