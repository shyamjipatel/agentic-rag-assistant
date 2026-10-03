# agentic-rag-assistant

A portfolio project being built incrementally toward a production-quality RAG
assistant with LangGraph, tool calling, source citations, vector storage, and Docker.

## Milestone 1: FastAPI foundation

The application exposes `GET /health`, returning HTTP 200 with
`{"status": "ok"}`. This is a liveness check: it confirms the API responds, without
checking databases or external services.

## Milestone 2: Document ingestion

`POST /documents/ingest` accepts one multipart field named `file`: a UTF-8 `.txt`
file or a text-based `.pdf`. It returns HTTP 200 with extracted chunks and source
metadata. The operation prepares documents for retrieval; it does not persist
files, create embeddings, or call an LLM.

With the server running, try the included sample:

```sh
curl --fail-with-body -X POST http://127.0.0.1:8000/documents/ingest \
  -F 'file=@examples/knowledge.txt;type=text/plain'
```

For a PDF, replace the file path:

```sh
curl --fail-with-body -X POST http://127.0.0.1:8000/documents/ingest \
  -F 'file=@/absolute/path/guide.pdf;type=application/pdf'
```

You can also upload a file through the endpoint's **Try it out** button at
<http://127.0.0.1:8000/docs>.

The response contains `document_id`, `filename`, `character_count`, `page_count`,
`skipped_pages`, `chunk_count`, and `chunks`. Each chunk contains `chunk_id`,
`chunk_index`, `text`, `page_number`, `start_char`, and `end_char`.

- Documents use a SHA-256 identifier derived from the raw uploaded bytes. Repeated
  uploads of identical bytes produce the same document and chunk identifiers.
  Chunk IDs also include the ingestion version (`v1`) and zero-based chunk index.
- Chunks use 1,000-character windows with 200 characters of overlap. For example,
  a 1,801-character text produces spans `[0, 1000)`, `[800, 1800)`, and
  `[1600, 1801)`. These are character counts, not model-token counts. This simple
  baseline can split words; semantic or token-aware chunking is a later refinement.
- PDF extraction and chunking happen separately for each page. Page numbers are
  one-based. Character offsets refer to the extracted page text, with an exclusive
  end offset. For `.txt`, offsets refer to the decoded file text and page numbers
  are `null`. A leading UTF-8 BOM is removed; other whitespace is preserved.
- Whitespace-only chunks are skipped. PDF pages without extracted text appear in
  `skipped_pages`. PDFs with no extracted text are rejected; OCR is not included.
  PDF layout, tables, and reading order can affect extraction quality
  ([pypdf extraction documentation](https://pypdf.readthedocs.io/en/stable/user/extract-text.html)).
- Limits are 5 MiB per file, 100 PDF pages, and 500,000 extracted characters.
  Unsupported extensions return 415; invalid, encrypted, or empty documents return
  400; exceeded limits return 413; missing file fields return 422. The client MIME
  type is only a hint: parsing validates the actual content for the chosen format.
- The upload filename is source metadata, never a path used to write a file.
  Parsing runs in a worker thread. File limits are checked after multipart parsing;
  deployment request limits and isolated PDF parsing remain future hardening work.

## Milestone 3: Local embeddings and vector retrieval

`POST /documents/index` parses a `.txt` or text-based `.pdf`, embeds each chunk,
and stores the chunks and source metadata in PostgreSQL with pgvector.
`POST /search` retrieves passages by semantic similarity. Responses contain
passages and their sources; answer generation and formatted citations are the
next milestone. `/documents/ingest` remains available as a parsing preview.

### Prepare local storage and the model

Install the updated dependencies using the development instructions below.
Then run these commands from the repository root with Docker Desktop running:

```sh
cp .env.example .env  # First setup only; preserve an existing .env.
docker compose up -d --wait
python -m agentic_rag_assistant.database
python -m agentic_rag_assistant.embeddings
```

The database command initializes the schema explicitly and can be rerun.
The model command downloads the public model on first use, verifies inference,
and caches it under `.cache/embeddings/`. No API key is required. Normal API
requests load the cached model with downloads disabled. `.env` and the model
cache are ignored by Git.

Model preparation uses the operating system's trusted certificates with TLS
verification enabled. It defaults to ordinary HTTPS downloads because the native
Xet transport failed on the development network. Set `HF_HUB_DISABLE_XET=0` before
the command to opt into Xet ([Hugging Face environment variables](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhubdisablexet)).

Compose runs only PostgreSQL for this milestone. It binds to `127.0.0.1:55432`
and keeps data in a named Docker volume. The example credentials are for local
development. If you change the database credentials or port, update
`DATABASE_URL` to match. Changing `.env` does not reset an existing database's
credentials. `docker compose down` stops the database and retains its volume;
`docker compose down -v` deletes the stored documents. API containerization comes
in Phase 9.

### Index documents and search

Start the API as described below, then index the two fictional sample policies:

```sh
curl --fail-with-body http://127.0.0.1:8000/documents/index \
  -F 'file=@examples/leave-policy.txt;type=text/plain'
curl --fail-with-body http://127.0.0.1:8000/documents/index \
  -F 'file=@examples/expense-policy.txt;type=text/plain'

curl --fail-with-body http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "How many vacation days can staff take each year?", "top_k": 2}'
```

The index response reports `document_id`, `filename`, `chunk_count`,
`embedding_model`, and `embedding_dimensions`. Indexing identical file bytes
replaces that document's chunks atomically; it does not add duplicates. If the
same bytes are uploaded under another filename, the latest filename is retained.
A failed replacement leaves the previous stored document intact.

Search returns `query` and `results`. Each result contains `document_id`,
`filename`, `score`, and a `chunk` with its text, identifier, page number, and
character offsets. Optional request fields are `top_k` (1–20, default 5),
`document_id` (restrict to one indexed document), and `min_score` (cosine threshold
between -1 and 1). The query must contain 1–2,000 characters after trimming.
Without a threshold, search returns the nearest passages even when none is
relevant. Scores measure similarity, not answer confidence. An empty index or
unmatched filter returns an empty list.

### Retrieval decisions

- FastEmbed runs `BAAI/bge-small-en-v1.5` locally on CPU using ONNX Runtime.
  This English model produces 384-dimensional vectors and supports inputs up to
  512 tokens. Queries receive the model's retrieval instruction; documents do not
  ([BGE model card](https://huggingface.co/BAAI/bge-small-en-v1.5)).
- The tokenizer checks complete inputs before inference, including the query
  instruction and special tokens. Inputs over 512 tokens return 422 instead of
  being silently truncated. The existing 1,000-character chunks fit ordinary
  English text but can exceed the token limit for unusual text. Shorter passages
  are required in that case; token-aware chunking remains a future refinement.
- PostgreSQL stores document metadata, extracted chunk text, and `vector(384)`
  embeddings together. Raw uploaded files are not stored. The database records
  the expected schema version, model name, and vector dimensions and rejects
  incompatible configuration. A different model requires a separate database
  and re-embedding the documents.
- Search uses exact cosine distance (`<=>`) and reports `1 - distance`. Exact
  search gives a straightforward baseline for the small initial corpus; an
  approximate index can follow when measurements justify it
  ([pgvector documentation](https://github.com/pgvector/pgvector)).
- The HTTP routes delegate to a retrieval service, which coordinates an embedding
  provider and a document store. This keeps HTTP, model inference, and SQL
  independently testable. Parsing, inference, and database work run in worker
  threads. One cached model instance serializes inference in each API process;
  database connections are opened per operation. Connection pooling and ingestion
  jobs remain later scaling work.
- Missing model files or unavailable/uninitialized storage return 503.
  Model loading is lazy, so `/health` and `/documents/ingest` work without model
  preparation or a running database. Authentication and user-specific document
  isolation are not implemented; keep this milestone on a trusted local machine.

## Development environment

Use Python 3.13. The initial environment was verified with Python 3.13.7 on macOS
arm64. Current FastAPI and LangGraph metadata explicitly list Python 3.13 support:
[FastAPI](https://pypi.org/project/fastapi/),
[LangGraph](https://pypi.org/project/langgraph/).
FastEmbed/ONNX Runtime, Psycopg, and pgvector were also installed and verified on
this environment for Milestone 3, so switching to Python 3.12 is not necessary.
Use a current Python 3.13 patch release for deployment.

From the repository root, create an environment if one does not already exist:

```sh
python3.13 -m venv .venv
```

Activate it and install the package, development tools, and pinned dependencies:

```sh
source .venv/bin/activate
python --version
python -m pip install -c requirements.lock -e '.[dev]'
python -m pip check
```

The existing Python 3.13 `.venv` can be used directly. In VS Code, select
`.venv/bin/python` with **Python: Select Interpreter**.

## Run and test

```sh
python -m uvicorn agentic_rag_assistant.main:app --reload
```

Open <http://127.0.0.1:8000/docs>, or check the endpoint from another terminal:

```sh
curl http://127.0.0.1:8000/health
```

Run the unit and HTTP contract tests without model downloads or a database:

```sh
python -m pytest
```

Four PostgreSQL integration tests are skipped unless `TEST_DATABASE_URL` is set.
Use a **separate test database**: the tests delete its document rows before and
after each test and require a database name ending in `_test`. With the default
local development credentials:

```sh
docker compose exec -T postgres createdb -U rag agentic_rag_test  # Once only
TEST_DATABASE_URL=postgresql://rag:rag_local_dev@127.0.0.1:55432/agentic_rag_test \
  python -m pytest
```

The integration tests verify exact ranking, filters, persistence, idempotent
indexing, transactional rollback, and incompatible model configuration. Model
inference is tested independently using a deterministic adapter; the real model
and API were also checked manually with the included sample policies.

## Structure and dependency management

```text
src/agentic_rag_assistant/
    __init__.py
    main.py                 # FastAPI application and health route
    documents.py            # Preview/index routes and upload error mapping
    search.py               # Validated semantic-search HTTP endpoint
    ingestion.py            # Text/PDF parsing and chunk preparation
    chunking.py             # Overlapping text windows with source offsets
    models.py               # Typed document, chunk, and search responses
    embeddings.py           # Local model, input validation, model setup command
    retrieval.py            # Coordinates embedding and storage
    vector_store.py         # Atomic indexing and exact cosine search
    database.py             # Connections, initialization, configuration checks
    schema.sql              # Version 1 PostgreSQL/pgvector schema
    settings.py             # Environment-based configuration
tests/
    conftest.py             # In-memory PDF test payloads
    test_health.py
    test_documents.py
    test_ingestion.py
    test_chunking.py
    test_embeddings.py
    test_retrieval.py
    test_retrieval_api.py
    test_vector_store.py    # Dedicated-database integration tests
examples/
    knowledge.txt           # Shareable upload example
    leave-policy.txt        # Fictional semantic-search sample
    expense-policy.txt      # Fictional semantic-search sample
compose.yaml                # Local PostgreSQL service and persistent volume
.env.example                # Shareable local configuration defaults
pyproject.toml              # Package metadata, direct dependencies, pytest settings
requirements.lock          # Exact runtime and test dependency versions
.python-version            # Python minor version for compatible version managers
```

- The `src/` layout requires installing the package. Tests import the installed
  package instead of relying on the repository root being on Python's import path.
- `pip install -e` makes an editable installation, so source edits take effect
  without reinstalling. Hatchling builds the Python package.
- FastAPI defines the HTTP API; Uvicorn is the ASGI server that serves it. HTTPX2
  supports the current Starlette/FastAPI test client, and pytest runs the test
  ([Starlette test-client documentation](https://starlette.dev/testclient/)). Test
  tools belong in the `dev` extra rather than runtime dependencies.
- `python-multipart` handles file-upload form data; `pypdf` extracts PDF text.
  Plain text parsing and chunking use Python's standard library. FastEmbed runs
  local embeddings, Psycopg connects to PostgreSQL, and pgvector adapts vectors.
  Pydantic Settings loads configuration; Truststore uses system certificates for
  model preparation. LangGraph will be introduced in its own milestone.
- `pyproject.toml` defines allowed dependency ranges. `requirements.lock` pins
  their transitive dependencies for repeatable development installs and is passed
  to pip as a constraints file. It includes test tools and was verified on Python
  3.13/macOS arm64; other platforms must be verified when added. Build tooling is
  resolved separately in pip's isolated build environment.
- The document router handles HTTP concerns and delegates to the ingestion
  service. Parsing, chunking, and document data have separate modules so they can
  be tested without an HTTP request. Standard Python dataclasses define the typed
  response; FastAPI generates its OpenAPI schema.

To deliberately refresh the dependency snapshot, use a clean Python 3.13 virtual
environment, install with `python -m pip install -e '.[dev]'` without constraints,
and generate pins with:

```sh
python -m pip freeze --exclude agentic-rag-assistant > requirements.lock
```

Review the dependency diff and run `python -m pip check` and `python -m pytest`
before committing an update.
