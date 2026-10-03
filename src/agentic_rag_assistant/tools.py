"""Validated calculator calls and deterministic execution over retrieved evidence."""

import json
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from agentic_rag_assistant.answering import MAX_ANSWER_SOURCES, InvalidGenerationError
from agentic_rag_assistant.models import SearchHit, ToolResult

DecimalOperand = Annotated[str, StringConstraints(pattern=r"^-?\d{1,12}(\.\d{1,6})?$")]


class CalculatorArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    operation: Literal["add", "subtract", "multiply", "divide"]
    left: DecimalOperand
    right: DecimalOperand
    source_ids: list[Annotated[int, Field(ge=1, le=MAX_ANSWER_SOURCES)]] = Field(
        min_length=1, max_length=MAX_ANSWER_SOURCES,
    )


class CalculatorCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    call_id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    name: Literal["calculator"]
    arguments: CalculatorArguments


@dataclass(frozen=True)
class ToolSelection:
    call: CalculatorCall | None
    continuation: list[dict[str, Any]]


def calculator_definition() -> dict[str, Any]:
    return {
        "name": "calculator",
        "description": (
            "Perform one arithmetic operation using numbers from the retrieved evidence "
            "and question. Use decimal strings for left/right. Supply the source IDs that "
            "support the calculation. Do not use this tool when evidence is insufficient."
        ),
        "parameters": CalculatorArguments.model_json_schema(),
    }


def parse_calculator_call(call_id: str, name: str, arguments: str | dict) -> CalculatorCall:
    try:
        if isinstance(arguments, str):
            if len(arguments) > 4_096:
                raise ValueError("Oversized arguments")
            arguments = json.loads(arguments)
        return CalculatorCall.model_validate({
            "call_id": call_id, "name": name, "arguments": arguments,
        })
    except (ValueError, ValidationError) as exc:
        raise InvalidGenerationError("The model requested an invalid calculator call.") from exc


def execute_calculator(call: CalculatorCall, sources: list[SearchHit]) -> ToolResult:
    args = call.arguments
    if any(identifier > len(sources) for identifier in args.source_ids):
        raise InvalidGenerationError(
            "The calculator cited a source outside the retrieved evidence."
        )
    left, right = Decimal(args.left), Decimal(args.right)
    with localcontext() as context:
        context.prec = 28
        match args.operation:
            case "add":
                value = left + right
            case "subtract":
                value = left - right
            case "multiply":
                value = left * right
            case "divide":
                if right == 0:
                    raise InvalidGenerationError("The calculator cannot divide by zero.")
                value = left / right
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    if value == 0:
        rendered = "0"
    return ToolResult(
        "calculator", args.operation, args.left, args.right, rendered,
        sorted(set(args.source_ids)),
    )
