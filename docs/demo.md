# Portfolio demo

## A five-minute walkthrough

Start the application using the [README](../README.md). Configure a working local
or remote provider; the container's readiness check intentionally does not probe
or bill that provider. Use only the fictional included sample documents for a
shareable recording. Clear unrelated documents from your recording workspace by
using a separate database/project rather than deleting development data.

1. **Introduce the workspace.** Show the conversation sidebar, document library,
   upload control, and composer. Explain that documents are indexed for retrieval;
   this is not LLM training.
2. **Upload `examples/leave-policy.txt`.** Show indexing progress and the indexed
   document in the library. Optionally upload a text-based PDF to show page citations.
3. **Ask “How many annual leave days do employees receive?”** The fictional policy
   states 24 days. Open the source marker/card and show the exact passage. Explain
   that the server attaches source metadata rather than trusting invented filenames.
4. **Enable Calculator and ask “And over three years at that annual rate?”** Show
   follow-up context, fresh retrieval, and the `24 × 3 = 72` tool result if selected.
   A real model's wording/tool choice can vary; do not promise this is a forced call.
5. **Ask an unsupported question**, such as “What is the pension contribution rate?”
   after indexing only the leave policy. Show the insufficient-evidence path. The
   model's abstention is not a mathematical proof against hallucination.
6. **Create another conversation, rename it, and return to the first.** Refresh the
   browser to demonstrate PostgreSQL persistence. Explain that documents are shared
   while each conversation receives only its own bounded history.
7. **Show `/docs`, `/health`, and `/ready`.** Finish with the graph diagram, test suite,
   Docker startup jobs, and the adapter factory that enables provider switching.

For a client discussion, connect each engineering choice to its purpose:
transactional turns avoid lost updates; request IDs make errors traceable; strict
tool arguments bound execution; cached local embeddings separate retrieval costs
from remote answer generation. Be explicit about authentication, evaluation, and
scaling work needed for that client's actual deployment.

## Screenshots

These images are captured from the application by the optional browser workflow.
They use real HTTP routes, LangGraph, and PostgreSQL with deterministic test model
adapters, not a live-model accuracy evaluation. All example documents are fictional.

### Workspace

![Desktop conversation workspace and shared document library](images/workspace-desktop.png)

### TXT/PDF upload

![Upload queue with successfully indexed TXT and PDF documents](images/workspace-upload.png)

### Cited answer and calculator

![Saved conversation showing source cards and the deterministic calculator result](images/workspace-chat.png)

### Mobile

<img src="images/workspace-mobile.png" alt="Mobile chat with source cards and a calculator result" width="390">

## Reproduce the screenshots

Follow [browser verification](development.md#browser-workflows-and-screenshots).
The test database must end in `_test`; the script removes its own sessions and
documents after completion. To write a new reviewed screenshot set directly:

```sh
UI_SCREENSHOT_DIR=docs/images \
TEST_DATABASE_URL=postgresql://rag:rag_local_dev@127.0.0.1:55432/agentic_rag_test \
  python tests/ui_browser_check.py
```

Do not run the browser check concurrently with the database test suite. Review
screenshots before committing, especially if you changed fixtures or UI text.
For a live provider demo, use the configured production adapters and record its
actual results separately; the test harness does not prove live provider behavior.
