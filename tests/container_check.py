"""Check an isolated Docker API with real embeddings/SQL and the Ollama fixture.

Start using compose.yaml + tests/compose.check.yaml in a separate Compose project.
Set CONTAINER_CHECK_URL and use a database whose name ends in _test.
Run again with CONTAINER_CHECK_SESSION to verify persistence after API recreation.
This script refuses to modify a database with a non-test name.
"""

import os
from uuid import UUID

import httpx
from psycopg.conninfo import conninfo_to_dict


def run():
    from agentic_rag_assistant.conversation_store import PostgresConversationStore

    url = os.environ.get("TEST_DATABASE_URL", "")
    if not url or not conninfo_to_dict(url).get("dbname", "").endswith("_test"):
        raise SystemExit("Set TEST_DATABASE_URL to the isolated stack's _test database.")
    base = os.getenv("CONTAINER_CHECK_URL", "http://127.0.0.1:8800")
    with httpx.Client(base_url=base, timeout=30) as client:
        # Prove the API points at the guarded test DB before any HTTP mutation.
        store = PostgresConversationStore(url)
        guard = store.create()
        try:
            assert client.get(f"/conversations/{guard.conversation_id}").status_code == 200, (
                "The HTTP API and TEST_DATABASE_URL must use the same isolated test database."
            )
        finally:
            store.delete(guard.conversation_id)
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {"status": "ready"}
        assert client.get("/").status_code == 200
        for path in ["app.js", "api.js", "render.js", "app.css", "favicon.svg"]:
            assert client.get(f"/static/{path}").status_code == 200
        previous = os.getenv("CONTAINER_CHECK_SESSION")
        if previous:
            identifier = str(UUID(previous))
            loaded = client.get(f"/conversations/{identifier}").json()
            assert loaded["turn_count"] == 2
            citation = loaded["turns"][0]["response"]["citations"][0]
            assert citation["filename"] == "container-policy.txt"
            assert loaded["turns"][1]["response"]["tool_results"][0]["value"] == "72"
            assert client.delete(f"/conversations/{identifier}").status_code == 204
            print("Container persistence verified after API restart; test conversation removed.")
            return
        indexed = client.post("/documents/index", files={"file": (
            "container-policy.txt", b"Employees receive 24 days of annual leave each year.",
            "text/plain",
        )})
        assert indexed.status_code == 200, indexed.text
        document_id = indexed.json()["document_id"]
        found = client.post("/search", json={
            "query": "Annual leave days?", "document_id": document_id,
        })
        assert found.status_code == 200 and found.json()["results"]
        created = client.post("/conversations")
        assert created.status_code == 201
        identifier = created.json()["conversation_id"]
        first = client.post("/ask", json={
            "question": "How many annual leave days?", "conversation_id": identifier,
            "document_id": document_id,
        })
        assert first.status_code == 200, first.text
        assert first.json()["answered"] and "24" in first.json()["answer"]
        assert first.json()["citations"][0]["document_id"] == document_id
        second = client.post("/ask", json={
            "question": "And over three years?", "conversation_id": identifier,
            "document_id": document_id, "use_tools": True,
        })
        assert second.status_code == 200, second.text
        assert second.json()["tool_results"][0]["value"] == "72"
        assert client.get(f"/conversations/{identifier}").json()["turn_count"] == 2
        print(
            "Container checks passed: real embeddings/retrieval, provider HTTP protocol, "
            "graph, citations, calculator, memory."
        )
        print(f"CONTAINER_CHECK_SESSION={identifier}")


if __name__ == "__main__":
    run()
