CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_index_config (
    singleton boolean PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    schema_version integer NOT NULL,
    embedding_model text NOT NULL,
    embedding_dimensions integer NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    document_id text PRIMARY KEY CHECK (length(document_id) = 64),
    filename text NOT NULL,
    character_count integer NOT NULL CHECK (character_count > 0),
    page_count integer CHECK (page_count > 0),
    skipped_pages jsonb NOT NULL,
    indexed_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id text PRIMARY KEY,
    document_id text NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_index integer NOT NULL CHECK (chunk_index >= 0),
    text text NOT NULL,
    page_number integer CHECK (page_number > 0),
    start_char integer NOT NULL CHECK (start_char >= 0),
    end_char integer NOT NULL CHECK (end_char > start_char),
    embedding vector(384) NOT NULL,
    UNIQUE (document_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id uuid PRIMARY KEY,
    turn_count integer NOT NULL DEFAULT 0 CHECK (turn_count >= 0),
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversation_turns (
    conversation_id uuid NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    turn_number integer NOT NULL CHECK (turn_number > 0),
    response jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (conversation_id, turn_number)
);
