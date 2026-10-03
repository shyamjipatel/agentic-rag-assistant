import pytest

from agentic_rag_assistant.answering import InvalidGenerationError
from agentic_rag_assistant.models import DocumentChunk, SearchHit
from agentic_rag_assistant.tools import execute_calculator, parse_calculator_call


def evidence():
    return [SearchHit("a" * 64, "leave.txt", 0.8,
                      DocumentChunk("chunk", 0, "24 days", None, 0, 7))]


@pytest.mark.parametrize(
    "operation,left,right,expected",
    [("add", "0.1", "0.2", "0.3"), ("subtract", "24", "5", "19"),
     ("multiply", "24", "3", "72"), ("divide", "24", "5", "4.8"),
     ("subtract", "2", "5", "-3"), ("multiply", "-0", "3", "0"),
     ("divide", "1", "3", "0.3333333333333333333333333333")],
)
def test_calculator_uses_decimal_arithmetic(operation, left, right, expected):
    call = parse_calculator_call("call-1", "calculator", {
        "operation": operation, "left": left, "right": right, "source_ids": [1, 1],
    })
    result = execute_calculator(call, evidence())
    assert result.value == expected
    assert result.source_ids == [1]


@pytest.mark.parametrize(
    "overrides",
    [{"operation": "eval"}, {"left": "__import__('os')"}, {"left": "NaN"},
     {"left": "Infinity"}, {"left": "1e99"}, {"left": "1234567890123"},
     {"left": "0.1234567"}, {"left": 24}, {"source_ids": []},
     {"source_ids": [0]}, {"source_ids": ["1"]}, {"filename": "invented.pdf"}],
)
def test_invalid_tool_arguments_are_rejected(overrides):
    args = {"operation": "multiply", "left": "24", "right": "3", "source_ids": [1]}
    args.update(overrides)
    with pytest.raises(InvalidGenerationError, match="invalid calculator"):
        parse_calculator_call("call-1", "calculator", args)


@pytest.mark.parametrize("name,arguments", [("shell", {}), ("calculator", "broken JSON")])
def test_unknown_tools_and_malformed_arguments_are_rejected(name, arguments):
    with pytest.raises(InvalidGenerationError):
        parse_calculator_call("call-1", name, arguments)


@pytest.mark.parametrize("right,ids,message", [("0", [1], "zero"), ("3", [2], "evidence")])
def test_execution_rejects_zero_division_and_missing_sources(right, ids, message):
    call = parse_calculator_call("call-1", "calculator", {
        "operation": "divide", "left": "24", "right": right, "source_ids": ids,
    })
    with pytest.raises(InvalidGenerationError, match=message):
        execute_calculator(call, evidence())
