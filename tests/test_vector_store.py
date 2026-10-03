import os
from dataclasses import replace

import pytest
from psycopg.conninfo import conninfo_to_dict

from agentic_rag_assistant.database import StorageUnavailableError, connect, initialize_database
from agentic_rag_assistant.embeddings import EMBEDDING_DIMENSIONS
from agentic_rag_assistant.ingestion import ingest_document
from agentic_rag_assistant.vector_store import PostgresVectorStore

pytestmark = pytest.mark.integration


def axis_vector(axis: int):
    values = [0.0] * EMBEDDING_DIMENSIONS
    values[axis] = 1.0
    return values


@pytest.fixture
def store():
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to run PostgreSQL integration tests.")
    if not conninfo_to_dict(url).get("dbname", "").endswith("_test"):
        pytest.fail("Integration tests require a separate database ending in _test.")
    initialize_database(url)
    with connect(url) as connection:
        connection.execute("DELETE FROM documents")
    yield PostgresVectorStore(url)
    with connect(url) as connection:
        connection.execute("DELETE FROM documents")


def test_exact_cosine_ranking_filters_and_source_metadata(store) -> None:
    relevant = ingest_document(filename="relevant.txt", content=b"Relevant source")
    other = ingest_document(filename="other.txt", content=b"Another source")
    store.save_document(relevant, [axis_vector(0)])
    store.save_document(other, [axis_vector(1)])

    results = store.search(axis_vector(0), top_k=5)
    assert [hit.document_id for hit in results] == [relevant.document_id, other.document_id]
    assert [hit.score for hit in results] == pytest.approx([1.0, 0.0])
    assert results[0].chunk == relevant.chunks[0]
    assert len(store.search(axis_vector(0), top_k=1)) == 1
    assert len(store.search(axis_vector(0), top_k=5, min_score=0.5)) == 1
    filtered = store.search(axis_vector(0), top_k=5, document_id=other.document_id)
    assert [hit.document_id for hit in filtered] == [other.document_id]
    assert store.search(axis_vector(0), top_k=5, document_id="f" * 64) == []


def test_reindex_is_idempotent_and_survives_a_new_store_instance(store) -> None:
    document = ingest_document(filename="original.txt", content=b"x" * 1_200)
    vectors = [axis_vector(0), axis_vector(1)]
    store.save_document(document, vectors)
    store.save_document(replace(document, filename="renamed.txt"), vectors)

    reopened = PostgresVectorStore(store.database_url)
    results = reopened.search(axis_vector(0), top_k=20)
    assert len(results) == 2
    assert {hit.filename for hit in results} == {"renamed.txt"}


def test_failed_reindex_rolls_back_document_and_chunks(store) -> None:
    document = ingest_document(filename="original.txt", content=b"x" * 1_200)
    vectors = [axis_vector(0), axis_vector(1)]
    store.save_document(document, vectors)
    invalid = replace(
        document, filename="failed.txt",
        chunks=[replace(chunk, chunk_id="duplicate-id") for chunk in document.chunks],
    )
    with pytest.raises(StorageUnavailableError):
        store.save_document(invalid, vectors)

    results = store.search(axis_vector(0), top_k=20)
    assert len(results) == 2
    assert {hit.filename for hit in results} == {"original.txt"}
    assert {hit.chunk.chunk_id for hit in results} == {chunk.chunk_id for chunk in document.chunks}


def test_incompatible_model_configuration_is_rejected(store) -> None:
    with connect(store.database_url) as connection:
        connection.execute("UPDATE rag_index_config SET embedding_model = 'incompatible'")
    try:
        with pytest.raises(StorageUnavailableError, match="incompatible"):
            store.search(axis_vector(0), top_k=5)
    finally:
        with connect(store.database_url) as connection:
            connection.execute(
                "UPDATE rag_index_config SET embedding_model = %s",
                ("BAAI/bge-small-en-v1.5",),
            )
