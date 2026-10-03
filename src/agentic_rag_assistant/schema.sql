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

-- Additive upgrades keep existing documents and conversation turns intact.
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema() AND table_name = 'conversations'
            AND column_name = 'title') THEN
        ALTER TABLE conversations ADD COLUMN title text NOT NULL DEFAULT 'New conversation'
            CHECK (length(title) BETWEEN 1 AND 120);
        UPDATE conversations c SET title = left(t.response->>'question', 120)
        FROM conversation_turns t
        WHERE t.conversation_id = c.conversation_id AND t.turn_number = 1
            AND length(t.response->>'question') > 0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema() AND table_name = 'conversations'
            AND column_name = 'updated_at') THEN
        ALTER TABLE conversations ADD COLUMN updated_at timestamptz
            NOT NULL DEFAULT CURRENT_TIMESTAMP;
        UPDATE conversations c SET updated_at = coalesce(
            (SELECT max(t.created_at) FROM conversation_turns t
             WHERE t.conversation_id = c.conversation_id), c.created_at);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS conversations_recent
    ON conversations (updated_at DESC, conversation_id DESC);
