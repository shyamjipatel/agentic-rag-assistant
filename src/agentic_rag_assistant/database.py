"""Explicit initialization of vector storage and conversation tables."""

from importlib.resources import files

import psycopg

from agentic_rag_assistant.embeddings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
from agentic_rag_assistant.settings import get_settings

SCHEMA_VERSION = 1


class StorageUnavailableError(RuntimeError):
    """The vector database is unavailable or has incompatible configuration."""


def connect(database_url: str):
    return psycopg.connect(
        database_url,
        connect_timeout=5,
        options="-c statement_timeout=10000",
        application_name="agentic-rag-assistant",
    )


def check_configuration(connection) -> None:
    row = connection.execute(
        "SELECT schema_version, embedding_model, embedding_dimensions "
        "FROM rag_index_config WHERE singleton = TRUE"
    ).fetchone()
    if row != (SCHEMA_VERSION, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS):
        raise StorageUnavailableError(
            "The database index configuration is incompatible. "
            "Initialize a separate database before changing embedding models."
        )


def initialize_database(database_url: str) -> None:
    schema = files("agentic_rag_assistant").joinpath("schema.sql").read_text()
    with connect(database_url) as connection:
        connection.execute(schema)
        connection.execute(
            "INSERT INTO rag_index_config "
            "(singleton, schema_version, embedding_model, embedding_dimensions) "
            "VALUES (TRUE, %s, %s, %s) ON CONFLICT (singleton) DO NOTHING",
            (SCHEMA_VERSION, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS),
        )
        check_configuration(connection)


if __name__ == "__main__":
    try:
        initialize_database(get_settings().database_url.get_secret_value())
    except (psycopg.Error, StorageUnavailableError):
        raise SystemExit(
            "Database initialization failed. Check DATABASE_URL, database readiness, "
            "and the existing index configuration."
        ) from None
    print("PostgreSQL vector and conversation schemas initialized.")
