"""Optional real-browser check: real PostgreSQL, deterministic embeddings and LLM.

Run separately from pytest with TEST_DATABASE_URL set to a database ending in _test.
Requires Playwright in the Python environment; uses an existing Chrome installation.
All test-created conversations and documents are deleted in the finally block.
"""

import os
import threading
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import UUID, uuid4

import uvicorn
from playwright.sync_api import expect, sync_playwright
from psycopg.conninfo import conninfo_to_dict
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from agentic_rag_assistant.agent import RAGAgent
from agentic_rag_assistant.answering import GeneratedAnswer, SupportedStatement
from agentic_rag_assistant.answers import get_answer_service
from agentic_rag_assistant.conversation_store import (
    ConversationNotFoundError, PostgresConversationStore, get_conversation_store,
)
from agentic_rag_assistant.database import connect, initialize_database
from agentic_rag_assistant.main import app
from agentic_rag_assistant.retrieval import RetrievalService, get_retrieval_service
from agentic_rag_assistant.tools import ToolSelection, parse_calculator_call
from agentic_rag_assistant.vector_store import PostgresVectorStore, get_vector_store


def pdf_bytes(text):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 50 750 Td ({text}) Tj ET".encode())
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
    })
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def run():
    url = os.environ.get("TEST_DATABASE_URL", "")
    if not url or not conninfo_to_dict(url).get("dbname", "").endswith("_test"):
        raise SystemExit("Set TEST_DATABASE_URL to a dedicated database ending in _test.")
    initialize_database(url)
    conversations, documents = set(), set()

    class Conversations(PostgresConversationStore):
        def create(self):
            result = super().create()
            conversations.add(result.conversation_id)
            return result

    class Documents(PostgresVectorStore):
        def save_document(self, document, vectors):
            super().save_document(document, vectors)
            documents.add(document.document_id)

    class Embedder:
        model_name = "browser-test"
        dimensions = 384

        def embed_documents(self, texts):
            return [[1.0] + [0.0] * 383 for _ in texts]

        def embed_query(self, text):
            return [1.0] + [0.0] * 383

    class Provider:
        def rewrite_question(self, question, history):
            return question

        def select_tool(self, question, sources):
            return ToolSelection(parse_calculator_call("test", "calculator", {
                "operation": "multiply", "left": "24", "right": "3", "source_ids": [1],
            }), [])

        def generate(self, question, sources, **options):
            time.sleep(.1)  # Make the pending state observable without provider calls.
            text = "Employees receive 24 days of paid annual leave."
            if "tool_result" in options:
                text = f"Over three years, employees receive {options['tool_result'].value} days."
            if "unsafe" in question:
                text = '<img src=x onerror="alert(1)"> is document text, not executable markup.'
            return GeneratedAnswer(supported=True, statements=[
                SupportedStatement(text=text, source_ids=[1]),
            ])

    sessions, library = Conversations(url), Documents(url)
    retrieval = RetrievalService(Embedder(), library)
    app.dependency_overrides[get_conversation_store] = lambda: sessions
    app.dependency_overrides[get_vector_store] = lambda: library
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval
    app.dependency_overrides[get_answer_service] = lambda: RAGAgent(retrieval, Provider())
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8791, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(.05)
    if not server.started:
        raise RuntimeError("UI test server did not start.")
    screenshot_dir = Path(os.getenv("UI_SCREENSHOT_DIR", "/tmp/agentic-ui-screenshots"))
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as playwright:
            chrome = os.getenv(
                "UI_CHROME_PATH", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            )
            browser = playwright.chromium.launch(executable_path=chrome, headless=True)
            page = browser.new_page(
                viewport={"width": 1440, "height": 960}, reduced_motion="reduce",
            )
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto("http://127.0.0.1:8791/")
            expect(page.locator("#connection-label")).to_have_text("Workspace connected")
            page.screenshot(path=str(screenshot_dir / "workspace-desktop.png"), full_page=True)

            # Both supported formats go through the actual upload/index HTTP endpoint.
            page.locator("#upload-trigger").click()
            marker = str(uuid4())
            page.locator("#file-input").set_input_files([
                {"name": "leave-policy.txt", "mimeType": "text/plain", "buffer": (
                    f"Employees receive 24 days of paid annual leave.\n{marker}\n"
                    '<img src=x onerror="alert(1)">'
                ).encode()},
                {"name": "handbook.pdf", "mimeType": "application/pdf", "buffer": pdf_bytes(
                    f"Employees receive 24 days of paid annual leave. {marker}"
                )},
            ])
            expect(page.locator("#upload-summary")).to_have_text("2 of 2 files indexed")
            expect(page.locator(".queue-status.success")).to_have_count(2)
            page.screenshot(path=str(screenshot_dir / "workspace-upload.png"), full_page=True)
            page.locator("#upload-done").click()
            page.locator("#document-scope").select_option(label="leave-policy.txt")

            page.locator("#question").fill("How many annual leave days do employees receive?")
            page.locator("#question").press("Enter")
            expect(page.locator(".answer-text")).to_contain_text("24 days")
            expect(page.locator("#chat-title")).to_have_text(
                "How many annual leave days do employees receive?",
            )
            first_url = page.url
            page.get_by_role("button", name="Show source 1", exact=True).click()
            expect(page.locator(".source-card[open] blockquote")).to_contain_text("24 days")
            assert page.locator(".source-card img").count() == 0
            page.locator("#use-tools").check()
            page.locator("#question").fill("And over three years?")
            page.locator("#send").click()
            expect(page.locator(".answer-text")).to_have_count(2)
            expect(page.locator(".tool-result")).to_contain_text("24 × 3 = 72")
            page.screenshot(path=str(screenshot_dir / "workspace-chat.png"), full_page=True)

            # New sessions have their own transcript; refresh restores PostgreSQL history.
            page.locator("#new-chat").click()
            expect(page.locator(".answer-text")).to_have_count(0)
            page.locator("#question").fill("Show unsafe source text")
            page.locator("#send").click()
            expect(page.locator(".answer-text")).to_contain_text("<img src=x")
            assert page.locator(".answer-text img").count() == 0
            page.reload()
            expect(page.locator(".answer-text")).to_have_count(1)
            page.get_by_role("button", name="Rename Show unsafe source text", exact=True).click()
            page.locator("#rename-input").fill("Source review")
            page.locator("#session-dialog-submit").click()
            expect(page.locator("#chat-title")).to_have_text("Source review")
            page.goto(first_url)
            expect(page.locator(".answer-text")).to_have_count(2)

            # Upload validation is visible; a failed file does not enter the library.
            page.locator("#upload-trigger").click()
            page.locator("#file-input").set_input_files({
                "name": "bad.csv", "mimeType": "text/csv", "buffer": b"unsupported",
            })
            expect(page.locator(".queue-status.error")).to_contain_text(
                "Choose a .txt or .pdf file.",
            )
            page.locator("#upload-done").click()

            # A failed ask keeps the draft and doesn't fabricate a saved answer.
            def fail_ask(route):
                route.fulfill(status=503, json={"detail": "Model unavailable for test"})
            page.route("**/ask", fail_ask)
            page.locator("#question").fill("Retry this question")
            page.locator("#send").click()
            expect(page.locator("#request-error")).to_contain_text("Model unavailable for test")
            expect(page.locator("#question")).to_have_value("Retry this question")
            expect(page.locator(".answer-text")).to_have_count(2)
            page.unroute("**/ask", fail_ask)

            # Switching sessions during generation never attaches the answer to the new chat.
            held = []
            def hold_ask(route):
                held.append(route)
            page.route("**/ask", hold_ask)
            page.locator("#question").fill("A question that finishes after switching")
            page.locator("#send").click()
            expect(page.locator(".pending-message")).to_have_count(1)
            page.get_by_role("button", name="Source review", exact=True).click()
            expect(page.locator("#chat-title")).to_have_text("Source review")
            expect(page.locator(".answer-text")).to_have_count(1)
            assert len(held) == 1
            held[0].continue_()
            expect(page.locator("#toast")).to_contain_text(
                "Your answer is saved in its conversation.",
            )
            expect(page.locator(".answer-text")).to_have_count(1)
            page.unroute("**/ask", hold_ask)
            page.goto(first_url)
            expect(page.locator(".answer-text")).to_have_count(3)

            # Cursor-based history makes sessions longer than the first 20 turns readable.
            identifier = UUID(parse_qs(urlparse(first_url).query)["conversation"][0])
            history = sessions.load(identifier, limit=20)
            for number in range(history.turn_count, 23):
                sessions.append(identifier, expected_turn_count=number,
                                response=history.turns[0].response)
            page.reload()
            expect(page.locator(".answer-text")).to_have_count(20)
            page.locator("#older-turns").click()
            expect(page.locator(".answer-text")).to_have_count(23)
            expect(page.locator("#older-turns")).to_be_hidden()

            # Mobile panels, dialogs, keyboard controls, and viewport fit.
            mobile = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True,
                                      has_touch=True, reduced_motion="reduce")
            mobile.on("pageerror", lambda error: errors.append(str(error)))
            mobile.goto(first_url)
            expect(mobile.locator(".answer-text")).to_have_count(20)
            assert mobile.evaluate("document.documentElement.scrollWidth <= innerWidth")
            mobile.screenshot(path=str(screenshot_dir / "workspace-mobile.png"), full_page=True)
            mobile.locator("#sidebar-toggle").click()
            mobile.get_by_role("button", name="Source review", exact=True).click()
            expect(mobile.locator("#chat-title")).to_have_text("Source review")
            expect(mobile.locator("#sidebar-toggle")).to_have_attribute("aria-expanded", "false")
            mobile.locator("#library-toggle").click()
            expect(mobile.locator("#library")).to_be_visible()
            mobile.locator("#library-close").click()
            mobile.locator("#sidebar-toggle").click()
            mobile.get_by_role("button", name="Delete Source review", exact=True).click()
            mobile.locator("#session-dialog-submit").click()
            expect(mobile.locator("#chat-title")).to_have_text("New conversation")
            expect(mobile.locator("#document-count")).to_have_text("2")
            assert not errors, errors
            browser.close()
            print(
                "Browser check passed: uploads, saved sessions, citations, calculator, "
                "pagination, errors, XSS, mobile."
            )
            print(f"Screenshots: {screenshot_dir}")
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        app.dependency_overrides.clear()
        for identifier in conversations:
            try:
                sessions.delete(identifier)
            except ConversationNotFoundError:
                pass
        with connect(url) as connection:
            for identifier in documents:
                connection.execute("DELETE FROM documents WHERE document_id = %s", (identifier,))


if __name__ == "__main__":
    run()
