# agentic-rag-assistant

A portfolio project being built incrementally toward a production-quality RAG
assistant with LangGraph, tool calling, source citations, vector storage, and Docker.

## Milestone 1: FastAPI foundation

The current application exposes `GET /health`, returning HTTP 200 with
`{"status": "ok"}`. This is a liveness check: it confirms the API responds, without
checking databases or external services.

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

Run the HTTP contract test:

```sh
python -m pytest
```

## Structure and dependency management

```text
src/agentic_rag_assistant/
    __init__.py
    main.py                 # FastAPI application and health route
tests/
    test_health.py          # HTTP status, content type, and exact response
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
- `pyproject.toml` defines allowed dependency ranges. `requirements.lock` pins
  their transitive dependencies for repeatable development installs and is passed
  to pip as a constraints file. It includes test tools and was verified on Python
  3.13/macOS arm64; other platforms must be verified when added. Build tooling is
  resolved separately in pip's isolated build environment.
- Keep the first route in `main.py`. Split routes and business logic into modules
  when features introduce that need.

To deliberately refresh the dependency snapshot, use a clean Python 3.13 virtual
environment, install with `python -m pip install -e '.[dev]'` without constraints,
and generate pins with:

```sh
python -m pip freeze --exclude agentic-rag-assistant > requirements.lock
```

Review the dependency diff and run `python -m pip check` and `python -m pytest`
before committing an update.
