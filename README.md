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

## Milestone 4: Answers with source citations

`POST /ask` retrieves evidence, asks the configured LLM for a structured answer,
and resolves source citations against the retrieved chunks. Milestone 5 adds
LangGraph orchestration, and Milestone 6 adds opt-in calculator tool calling.
Conversation memory follows later.

The RAG service depends on an `LLMProvider` protocol, not a concrete provider.
Only the adapter factory selects Ollama, remote Hugging Face, or OpenAI:

```text
POST /ask
    -> RAGAgent -> RetrievalService -> local embeddings + PostgreSQL
                -> LLMProvider
                     -> OllamaProvider       (default: local)
                     -> HuggingFaceProvider  (remote)
                     -> OpenAIProvider       (remote)
                -> validate references and attach stored source metadata
```

### Choose the LLM through configuration

Local inference is the default, both in application settings and `.env.example`.
If the LLM variables are absent or empty, settings select Ollama and `qwen2.5:7b`.
To override the configuration, edit your existing `.env` rather than replacing it.

Default local configuration:

```ini
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

Install Ollama from <https://ollama.com/download>, start its local service, then
download the model named in `LLM_MODEL`:

```sh
ollama pull qwen2.5:7b
```

If you use the standalone CLI, run `ollama serve` in another terminal. The adapter
calls the local `/api/chat` endpoint with a JSON schema, streaming disabled, and
an 8,192-token context. Choose a local model that supports that context size and
structured output. The default [Qwen 2.5 7B model](https://ollama.com/library/qwen2.5:7b)
can be replaced through configuration. Selecting the default does not install
Ollama or download model weights automatically. Ollama cloud models are outside this
adapter's scope ([Ollama Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs)).

For remote Hugging Face inference:

```ini
LLM_PROVIDER=huggingface
LLM_MODEL=Qwen/Qwen3-32B:deepinfra
HF_TOKEN=your-hugging-face-token
```

The adapter uses `https://router.huggingface.co/v1/chat/completions` with strict
JSON-schema output. At [Access Tokens](https://huggingface.co/settings/tokens),
create a fine-grained token with **Make calls to Inference Providers** enabled
([authentication requirements](https://huggingface.co/docs/inference-providers/index#authentication)).
The account also needs inference credits. Only the question and retrieved
passage text are sent. Keep the token in the ignored `.env`, never in
`.env.example` or Git.

A token can authenticate successfully but still receive a 403 from inference
if it lacks this permission. Enable the permission on the existing token, or
create a replacement and update `HF_TOKEN` in `.env`. Restart the API if you
replace the token.

The example pins the DeepInfra backend using the optional `:deepinfra` suffix
([DeepInfra integration](https://huggingface.co/docs/inference-providers/providers/deepinfra)).
Hosted availability can change; verify availability and JSON-schema support for
the selected model/backend pair. You can configure a
different compatible hosted model or backend without changing RAG code
([Inference Providers](https://huggingface.co/docs/inference-providers/index),
[Structured Outputs](https://huggingface.co/docs/inference-providers/guides/structured-output)).
Hosted inference consumes credits and can incur usage charges
([pricing](https://huggingface.co/docs/inference-providers/pricing)).

OpenAI remains available as another remote option:

```ini
LLM_PROVIDER=openai
LLM_MODEL=gpt-6-luna
OPENAI_API_KEY=your-api-key
```

The adapter uses the Responses API with strict JSON-schema output and
`store=false`. The selected model must support this output format. `gpt-6-luna`
is a documented model with Structured Outputs support; access depends on your API
account ([model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna),
[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)).
Only the question and retrieved passage text are sent to OpenAI. Keep your real
key in the ignored `.env`, never in `.env.example` or Git. Calls use paid API
usage; no calls are made during the automated tests.

Set both `LLM_PROVIDER` and `LLM_MODEL` when switching to a remote provider: local
model tags and hosted model identifiers have different formats. If a remote
provider is selected without a model, generation returns an actionable 503 rather
than applying the local model default. Missing credentials also return 503 before
any hosted request is sent. Credentials alone do not select a remote provider.

Restart the API after changing `.env`: settings and service instances are cached
within each process. Switching the LLM provider or model does **not** change the
local embedding model or require reindexing documents. These are separate jobs:
embeddings find passages; the LLM turns those passages into an answer.

Optional settings are `LLM_TIMEOUT_SECONDS` (default 120) and
`LLM_MAX_OUTPUT_TOKENS` (default 4,096). OpenAI reasoning models use the output
budget for reasoning as well as answer tokens; increase it if responses are
incomplete. There are no automatic retries or fallbacks to a different provider.
A local failure remains local even when remote credentials are configured.

### Ask a question

Prepare the database and embeddings, index the sample policies from Milestone 3,
configure a provider, and start the API. Then:

```sh
curl --fail-with-body http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "How many vacation days can staff take each year?", "top_k": 2}'
```

You can also use **Try it out** at <http://127.0.0.1:8000/docs>. An illustrative
response, with the long document/chunk details omitted, is:

```json
{
  "question": "How many vacation days can staff take each year?",
  "answer": "Employees receive 24 days of paid annual leave each year. [1]",
  "answered": true,
  "citations": [
    {"source_id": 1, "filename": "leave-policy.txt", "score": 0.736}
  ],
  "tool_results": []
}
```

Each actual citation also includes `document_id` and the complete `chunk` with
its identifier, text, page number, and character offsets. Source numbers are
local to the answer. Only cited sources appear in `citations`.

The request accepts a trimmed `question` of 1–2,000 characters, `top_k` of 1–5
(default 5), optional `document_id`, and optional `min_score` between -1 and 1.
Milestone 6 adds `use_tools`, a JSON boolean that defaults to `false`.
This smaller evidence limit keeps the initial generation context bounded; the
existing `/search` endpoint still accepts up to 20 results. Without a score
threshold, nearest passages may be unrelated, so the model is instructed to
abstain when the evidence does not answer the question.

If retrieval finds no matching passages, the service returns HTTP 200 with
`answered=false`, a standard insufficient-evidence message, and empty citations
without calling the LLM. The model can also abstain when retrieved text is
unhelpful. Invalid questions or oversized embedding inputs return 422; malformed
model output, incomplete output, or invalid citations return 502; missing provider
configuration or unavailable dependencies return 503; LLM HTTP timeouts return
504. Other endpoints remain usable without configuring an LLM.

### Citation validation and its limits

The model returns short statements with integer source IDs. The backend validates
the structured answer, rejects unknown IDs and contradictory abstentions, adds
inline markers such as `[1]`, and copies citation metadata from the retrieved
records. The model cannot supply a filename or page number in place of that
stored metadata.

These checks establish that a citation refers to evidence supplied to the model.
They do not prove that a passage supports every claim, or that a model cannot
follow malicious instructions in a passage. The prompt treats passages as data
and asks for evidence-based answers; semantic grounding evaluation and broader
prompt-injection defenses remain future quality work.

Tests exercise all three adapters using HTTPX mock transports, including their actual
request/response serialization, timeout and refusal handling, and `/ask` citation
resolution. This does not verify hosted-model access or local-model quality. A
live answer requires a running Ollama model, or credentials and model access for
the selected remote provider. Automated tests do not establish that a particular
token has inference permissions or that a hosted model is currently available.

## Milestone 5: LangGraph orchestration

`POST /ask` now runs through a compiled LangGraph `StateGraph`. The request and
response formats from Milestone 4 are unchanged, including document filters,
validated citations, abstention, and HTTP error handling.

```mermaid
flowchart TD
    Start([Start]) --> Retrieve[Retrieve passages]
    Retrieve --> Evidence{Any passages?}
    Evidence -->|Yes| Generate[Generate structured answer]
    Evidence -->|No| Abstain[Insufficient-evidence response]
    Generate --> Validate[Validate citations and resolve source metadata]
    Validate --> Finish([End])
    Abstain --> Finish
```

The implementation in `agent.py` has three LangGraph concepts:

- **State** carries the question, retrieval filters, passages, structured model
  output, and final response for one request. An output schema limits the graph's
  returned value to the resolved response. Credentials are held by the provider,
  rather than copied into graph state.
- **Nodes** perform retrieval, generation, citation validation, or abstention.
  They reuse the existing retrieval service, provider protocol, and citation
  validator. Provider selection remains in the adapter factory.
- **Edges** define the execution order. A conditional edge after retrieval skips
  generation when there are no passages, so that branch makes no LLM request.

The graph compiles once when the cached agent dependency is created. Each `ask`
invocation starts with fresh state; there is no checkpointer or conversation
history. Tests cover both graph branches, repeated and concurrent invocations,
evidence limits, and failure propagation. Failed nodes propagate their exceptions
to the existing HTTP boundary; there are no graph retries or provider fallbacks.
Without tool mode, each invocation performs one retrieval and at most one
generation request. Milestone 6 adds a separate tool-selection request when enabled.

This is the orchestration foundation for the agent: its route is currently
determined by application code. Phase 6 below introduces model-driven tool selection,
and conversation memory is Phase 7. Introducing them separately keeps each new
behavior understandable and testable
([Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api),
[Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)).

Use the existing `/ask` example above to try the graph. After pulling this change,
install the updated dependencies with
`python -m pip install -c requirements.lock -e '.[dev]'` and restart the API.
The graph change does not require a database migration, model download, or
document reindexing.

## Milestone 6: Calculator tool calling

Enable tool mode on `/ask` to let the model choose whether it needs one calculator
call. Retrieval still runs first, with the same document and similarity filters.
The model receives a native function definition, rather than a request to write
executable Python. The server validates the requested call and performs the arithmetic.

Restart the API after this change, then try:

```sh
curl --fail-with-body http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Using the annual leave policy, calculate the total paid annual leave days for 3 years at the stated yearly rate.", "top_k": 2, "use_tools": true}'
```

With the sample leave policy, the calculator can multiply `24` by `3` to get
`72`. The answer cites the annual rate; the multi-year total is a calculation,
not a promise that the policy permits accumulating that leave balance.

```mermaid
flowchart TD
    Retrieve[Retrieve passages] --> Evidence{Any passages?}
    Evidence -->|No| Abstain[Insufficient evidence]
    Evidence -->|Yes, tools disabled| Generate[Generate structured answer]
    Evidence -->|Yes, tools enabled| Choose[Model chooses a tool]
    Choose -->|No tool needed| Generate
    Choose -->|Calculator requested| Execute[Validate arguments and calculate]
    Execute -->|Native tool result| Generate
    Generate --> Validate[Validate citations and return answer]
```

The optional request field is `use_tools: true`. Omitting it keeps the direct
RAG path. Every answer response now includes an additive `tool_results` list,
empty when no tool was executed. Each executed calculation exposes the actual
server-computed operation, operands, value, and supporting source IDs:

```json
{
  "tool_results": [
    {
      "tool_name": "calculator",
      "operation": "multiply",
      "left": "24",
      "right": "3",
      "value": "72",
      "source_ids": [1]
    }
  ]
}
```

The model chooses the tool; `tools.py` validates and executes it. Its allowlist
contains only `calculator`, with `add`, `subtract`, `multiply`, and `divide`.
Operands are decimal strings limited to 12 integer and 6 fractional digits.
Python `Decimal` uses 28 significant digits; results requiring more precision
are rounded. Arbitrary expressions, code, unknown tools, malformed arguments,
division by zero, and source IDs outside the retrieved evidence are rejected.

Each invocation performs at most one calculation and two LLM requests: tool
selection, then structured answer generation. The final request offers no tools,
and an additional tool call is rejected. Empty retrieval skips both LLM requests.
This bounded round trip introduces tool calling incrementally; multi-step tool
loops and external API tools can build on it later.

All adapters implement the same provider interface. Hugging Face uses native chat
`tool_calls` and a correlated `tool_call_id`; Ollama uses its chat tool messages;
OpenAI uses Responses function-call items and `function_call_output`. Its reasoning
items are replayed with the call output while keeping `store=false`.
See [Hugging Face function calling](https://huggingface.co/docs/inference-providers/guides/function-calling),
[Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling), and
[official OpenAI documentation](https://developers.openai.com/api/docs/guides/function-calling).
The selected model/backend must support native tool calling as well as structured
answers. Automatic provider fallbacks and retries remain disabled.

The backend verifies that calculator source IDs exist and requires a supported
final answer to cite all of them. It does not prove that the model extracted the
right operands, applied the right policy conditions, or described the result
faithfully. The visible tool results make the computation inspectable; semantic
grounding evaluation remains future quality work.

Tests exercise decimal arithmetic, native tool/result round trips for all three
providers, rejected calls, bounded execution, failure mapping, and request
isolation. Conversation memory is still the next milestone.

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
    answers.py              # Ask endpoint and dependency wiring
    agent.py                # LangGraph state, nodes, routing, request isolation
    tools.py                # Calculator schema, validation, decimal execution
    answering.py            # Provider contracts, evidence prompt, citation validation
    llm/
        factory.py          # Selects local or remote LLM adapters from settings
        common.py           # HTTP transport and structured-output validation
        tool_support.py     # Native tool messages and provider-neutral prompts
        openai.py           # OpenAI Responses API adapter
        ollama.py           # Ollama chat API adapter
        huggingface.py      # Remote Hugging Face Inference Providers adapter
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
    test_answering.py
    test_agent.py           # Real graph routing, isolation, and failure boundaries
    test_tools.py           # Decimal arithmetic and invalid calculator inputs
    test_tool_calling.py    # Native tool calling through adapters, graph, and HTTP
    test_llm_providers.py
    test_answers_api.py
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
- HTTPX provides runtime HTTP clients for the LLM adapters and mock transports
  for provider tests. It is a distinct package from the HTTPX2 test-client dependency.
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
