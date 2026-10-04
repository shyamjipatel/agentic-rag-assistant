# Development and verification

## Host setup

Use Python 3.13. The original macOS ARM64 environment was verified with 3.13.7;
the pinned dependencies also build in the Linux ARM64 Docker image. Python 3.12
is not required for this repository's current dependency set.

From the repository root:

```sh
# Only if a virtual environment does not already exist:
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements.lock -e '.[dev]'
python -m pip check

# Only if .env does not already exist:
cp .env.example .env

docker compose up -d --wait postgres
python -m agentic_rag_assistant.database
python -m agentic_rag_assistant.embeddings
python -m uvicorn agentic_rag_assistant.main:app --reload --no-access-log
```

Configure your chosen provider in `.env` before asking questions. Open the browser
workspace at <http://127.0.0.1:8000/> and the API reference at `/docs`. Select
`.venv/bin/python` as the editor interpreter. If Docker's API is already bound to
8000, stop just that API or run the host server with `--port 8001`.

## Unit and HTTP tests

```sh
python -m pytest -q
```

This suite uses real parsers, domain validation, actual LangGraph routing, and
mock HTTP provider transports. PostgreSQL integration tests are skipped without
`TEST_DATABASE_URL`; model downloads and paid inference are never required.

Coverage includes unsupported/invalid uploads, limits, tokenizer behavior,
malformed provider envelopes, timeouts/refusals/truncation, source IDs, calculator
arguments/arithmetic, request isolation, bounded history, safe validation/errors,
request size limits with and without Content-Length, and readiness status mapping.
These are behavior/contract tests; they do not measure a real LLM's answer quality.

## PostgreSQL integration tests

Use a **dedicated database ending in `_test`**. Vector-store tests delete document
rows before/after cases. Conversation tests clean up their own UUIDs; the legacy
migration case uses its own temporary schema. Never point them at development data.

```sh
docker compose up -d --wait postgres
docker compose exec -T postgres createdb -U rag agentic_rag_test  # Once only
TEST_DATABASE_URL=postgresql://rag:rag_local_dev@127.0.0.1:55432/agentic_rag_test \
  python -m pytest -q
```

These cases verify ranking/filters, actual chunk counts, idempotence, persistence,
rollback, schema incompatibility, revision conflicts, pagination, and preservation
of legacy conversations during initialization. Change the test URL if you changed
the local credentials/port. Test guards reject non-`_test` database names.

## Browser workflows and screenshots

Playwright is optional and belongs to `browser-tests`, not runtime dependencies:

```sh
python -m pip install -c requirements.lock -e '.[dev,browser-tests]'
TEST_DATABASE_URL=postgresql://rag:rag_local_dev@127.0.0.1:55432/agentic_rag_test \
  python tests/ui_browser_check.py
```

The script starts a temporary server on 127.0.0.1:8791. It uses the actual FastAPI
routes, parsing/indexing, graph, and PostgreSQL with deterministic embedding/LLM
adapters. It checks TXT/PDF uploads, refresh persistence, session switching during
generation, citations, calculator results, rename/delete, upload/ask failures,
long-history pagination, safe text rendering, and mobile controls.

It launches the installed macOS Chrome by default. Set `UI_CHROME_PATH` to another
Chrome/Chromium executable; CI uses `/usr/bin/google-chrome`. Screenshots go to
`/tmp/agentic-ui-screenshots`; set `UI_SCREENSHOT_DIR` to change that location.
The script cleans up its created records. Do not run it concurrently with the
database integration tests. A deterministic fake in this test harness is not a
production provider or an app fallback mode.

## CI and package verification

[CI](../.github/workflows/ci.yml) runs on pushes to main and pull requests. It uses
Python 3.13 and a pgvector PostgreSQL service, installs constrained dependencies,
runs the full test suite and browser check, and builds a wheel. A separate job
validates Compose, builds the Linux container, checks dependencies, and verifies
non-root execution. Actions are pinned to official release commit hashes and use
read-only repository permissions; checkout does not persist credentials.

To verify the distributable locally:

```sh
python -m pip wheel --no-deps --wheel-dir /tmp/agentic-rag-wheels .
```

The wheel includes `schema.sql` and browser assets. Docker installs the wheel's
package instead of relying on a source mount. The UI has no npm build requirement.

## Full container workflow check

Use an isolated Compose project and a separate `_test` database. The optional
`tests/compose.check.yaml` override adds a deterministic Ollama HTTP fixture on
the private container network; it is never part of normal application startup or
the packaged image. The API still runs its real embedding model, SQL stores,
provider HTTP adapter, graph, citation validation, calculator, and memory.

For example, create a temporary environment file **outside the repository** with
`POSTGRES_DB=agentic_rag_test`, `POSTGRES_PORT=55933`, `API_PORT=8800`, and matching
test credentials. Clear remote credentials in that file. Then:

```sh
docker compose -p agentic-rag-check --env-file /path/to/check.env \
  -f compose.yaml -f tests/compose.check.yaml up --build -d --wait api
TEST_DATABASE_URL=postgresql://rag:rag_test_only@127.0.0.1:55933/agentic_rag_test \
  CONTAINER_CHECK_URL=http://127.0.0.1:8800 python tests/container_check.py
```

Adjust the URL to the actual test credentials. Model preparation requires working
network trust or an imported complete cache as described in the operations guide.
The script checks that the HTTP server uses the supplied test database before any
HTTP mutation. It prints a test-session UUID, which can verify persistence:

```sh
docker compose -p agentic-rag-check --env-file /path/to/check.env \
  -f compose.yaml -f tests/compose.check.yaml restart api
TEST_DATABASE_URL=postgresql://rag:rag_test_only@127.0.0.1:55933/agentic_rag_test \
  CONTAINER_CHECK_URL=http://127.0.0.1:8800 CONTAINER_CHECK_SESSION="<printed-uuid>" \
  python tests/container_check.py
```

The second pass reads both saved turns/citations and the calculator result before
deleting the test session. Documents remain until you clean up this **isolated**
test project's volumes:

```sh
docker compose -p agentic-rag-check --env-file /path/to/check.env \
  -f compose.yaml -f tests/compose.check.yaml --profile app down -v
```

Do not use `down -v` on the real development project.

## Dependency changes

`pyproject.toml` declares direct compatibility ranges. `requirements.lock` constrains
the complete runtime/dev/browser-test dependency set. Use it with `pip -c`; do not
blindly install the entire file into the runtime image. Docker derives runtime
requirements from the project's declared dependencies and installs the app without
test extras.

For a deliberate refresh, use a clean Python 3.13 environment, install all tracked
extras without constraints, and regenerate:

```sh
python -m pip install -e '.[dev,browser-tests]'
python -m pip freeze --exclude agentic-rag-assistant > requirements.lock
python -m pip check
```

Review version changes, run the full suite, and rebuild Docker. Base-image updates
also require verification: the Dockerfile pins its Python image digest, while OS
packages are resolved from the distribution repositories during the build.

## Contribution boundaries

Keep HTTP concerns in routers, orchestration in services/graph nodes, and provider
protocols in adapters. Add tests at the behavior boundary affected by a change.
Do not commit `.env`, tokens, model weights, test databases, or generated local
cache files. Preserve existing documents/conversations in migrations. Keep commits
focused; [the learning notes](learning-notes.md) explain earlier component decisions.
