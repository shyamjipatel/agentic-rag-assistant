"""Project configuration loaded from environment variables and a local .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: SecretStr = SecretStr(
        "postgresql://rag:rag_local_dev@127.0.0.1:55432/agentic_rag"
    )
    model_cache_dir: Path = Path(".cache/embeddings")


@lru_cache
def get_settings() -> Settings:
    return Settings()
