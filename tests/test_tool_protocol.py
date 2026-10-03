"""Provider compatibility cases beyond the graph's tool-call round trips."""

import json

import httpx
import pytest

from agentic_rag_assistant.answering import InvalidGenerationError
from agentic_rag_assistant.llm.openai import OpenAIProvider


@pytest.mark.parametrize("status", [None, "completed", "in_progress", "incomplete"])
def test_openai_function_call_status_is_optional_but_must_be_complete_when_present(status):
    call = {
        "type": "function_call", "id": "fc_demo", "call_id": "call_demo",
        "name": "calculator", "arguments": json.dumps({
            "operation": "multiply", "left": "24", "right": "3", "source_ids": [1],
        }),
    }
    if status is not None:
        call["status"] = status
    envelope = {"status": "completed", "output": [call]}
    adapter = OpenAIProvider(
        model="test-model", api_key="test-only-key", timeout=10, max_output_tokens=512,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=envelope)),
    )
    if status in (None, "completed"):
        selection = adapter.select_tool("Calculate leave over 3 years.", [])
        assert selection.call.call_id == "call_demo"
        assert selection.call.arguments.operation == "multiply"
        assert selection.continuation == [call]
    else:
        with pytest.raises(InvalidGenerationError, match="incomplete"):
            adapter.select_tool("Calculate leave over 3 years.", [])
