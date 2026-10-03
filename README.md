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

## Development environment

Use Python 3.13. The initial environment was verified with Python 3.13.7 on macOS
arm64. Current FastAPI and LangGraph metadata explicitly list Python 3.13 support:
[FastAPI](https://pypi.org/project/fastapi/),
[LangGraph](https://pypi.org/project/langgraph/).
Recheck compatibility when choosing the vector store and embedding libraries.
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

Run the health, ingestion, chunking, and HTTP contract tests:

```sh
python -m pytest
```

## Structure and dependency management

```text
src/agentic_rag_assistant/
    __init__.py
    main.py                 # FastAPI application and health route
    documents.py            # Upload route and HTTP error mapping
    ingestion.py            # Text/PDF parsing and chunk preparation
    chunking.py             # Overlapping text windows with source offsets
    models.py               # Typed document and chunk data
tests/
    conftest.py             # In-memory PDF test payloads
    test_health.py
    test_documents.py
    test_ingestion.py
    test_chunking.py
examples/
    knowledge.txt           # Shareable upload example
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
  Plain text parsing and chunking use Python's standard library. LangGraph and
  embedding dependencies will be introduced in their own milestones.
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
