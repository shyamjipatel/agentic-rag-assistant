"""Transactional document storage and exact cosine search using pgvector."""

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from agentic_rag_assistant.database import (
    StorageUnavailableError,
    check_configuration,
    connect,
)
from agentic_rag_assistant.embeddings import validate_vector
from agentic_rag_assistant.models import DocumentChunk, IngestedDocument, SearchHit


class PostgresVectorStore:
    def __init__(self, database_url: str):
        self.database_url = database_url

    def save_document(self, document: IngestedDocument, vectors: list[list[float]]) -> None:
        if len(vectors) != len(document.chunks):
            raise ValueError("Every chunk requires exactly one embedding.")
        for vector in vectors:
            validate_vector(vector)
        try:
            with connect(self.database_url) as connection:
                check_configuration(connection)
                register_vector(connection)
                connection.execute(
                    """INSERT INTO documents
                    (document_id, filename, character_count, page_count, skipped_pages)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (document_id) DO UPDATE SET
                        filename = EXCLUDED.filename,
                        character_count = EXCLUDED.character_count,
                        page_count = EXCLUDED.page_count,
                        skipped_pages = EXCLUDED.skipped_pages,
                        indexed_at = CURRENT_TIMESTAMP""",
                    (
                        document.document_id, document.filename, document.character_count,
                        document.page_count, Jsonb(document.skipped_pages),
                    ),
                )
                connection.execute(
                    "DELETE FROM document_chunks WHERE document_id = %s",
                    (document.document_id,),
                )
                with connection.cursor() as cursor:
                    cursor.executemany(
                        """INSERT INTO document_chunks
                        (chunk_id, document_id, chunk_index, text, page_number,
                         start_char, end_char, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                        [
                            (
                                chunk.chunk_id, document.document_id, chunk.chunk_index,
                                chunk.text, chunk.page_number, chunk.start_char, chunk.end_char,
                                Vector(vector),
                            )
                            for chunk, vector in zip(document.chunks, vectors, strict=True)
                        ],
                    )
        except psycopg.Error as exc:
            raise StorageUnavailableError(
                "Vector storage is unavailable. Check database readiness and initialization."
            ) from exc

    def search(
        self, vector: list[float], *, top_k: int,
        document_id: str | None = None, min_score: float | None = None,
    ) -> list[SearchHit]:
        validate_vector(vector)
        try:
            with connect(self.database_url) as connection:
                check_configuration(connection)
                register_vector(connection)
                rows = connection.execute(
                    """SELECT d.document_id, d.filename,
                        1 - (c.embedding <=> %s) AS score,
                        c.chunk_id, c.chunk_index, c.text, c.page_number,
                        c.start_char, c.end_char
                    FROM document_chunks c JOIN documents d USING (document_id)
                    WHERE (%s::text IS NULL OR d.document_id = %s)
                    ORDER BY c.embedding <=> %s, c.chunk_id LIMIT %s""",
                    (Vector(vector), document_id, document_id, Vector(vector), top_k),
                ).fetchall()
            return [
                SearchHit(
                    document_id=row[0], filename=row[1], score=float(row[2]),
                    chunk=DocumentChunk(
                        chunk_id=row[3], chunk_index=row[4], text=row[5],
                        page_number=row[6], start_char=row[7], end_char=row[8],
                    ),
                )
                for row in rows
                if min_score is None or row[2] >= min_score
            ]
        except psycopg.Error as exc:
            raise StorageUnavailableError(
                "Vector search is unavailable. Check database readiness and initialization."
            ) from exc
