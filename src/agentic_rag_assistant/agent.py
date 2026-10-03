"""A bounded LangGraph workflow for answering from retrieved evidence."""

from dataclasses import replace
from typing import Literal, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from agentic_rag_assistant.answering import (
    INSUFFICIENT_EVIDENCE,
    MAX_ANSWER_SOURCES,
    GeneratedAnswer,
    GenerationUnavailableError,
    InvalidGenerationError,
    LLMProvider,
    PassageRetriever,
    resolve_answer,
)
from agentic_rag_assistant.memory import RECENT_TURNS
from agentic_rag_assistant.models import AnswerResponse, ConversationTurn, SearchHit, ToolResult
from agentic_rag_assistant.tools import ToolSelection, execute_calculator


class AgentInput(TypedDict):
    question: str
    top_k: int
    document_id: str | None
    min_score: float | None
    use_tools: NotRequired[bool]
    history: NotRequired[list[ConversationTurn]]


class AgentState(AgentInput):
    sources: NotRequired[list[SearchHit]]
    generated: NotRequired[GeneratedAnswer]
    response: NotRequired[AnswerResponse]
    selection: NotRequired[ToolSelection]
    tool_result: NotRequired[ToolResult]
    standalone_question: NotRequired[str]


class AgentOutput(TypedDict):
    response: AnswerResponse


class RAGAgent:
    """Reuse the graph definition, with fresh state for each invocation."""

    def __init__(self, retriever: PassageRetriever, provider: LLMProvider):
        self.retriever = retriever
        self.provider = provider
        builder = StateGraph(
            AgentState, input_schema=AgentInput, output_schema=AgentOutput,
        )
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("generate", self._generate)
        builder.add_node("validate_citations", self._validate_citations)
        builder.add_node("abstain", self._abstain)
        builder.add_node("choose_tool", self._choose_tool)
        builder.add_node("execute_tool", self._execute_tool)
        builder.add_node("contextualize", self._contextualize)
        builder.add_conditional_edges(
            START, self._route_start, {"contextualize": "contextualize", "retrieve": "retrieve"},
        )
        builder.add_edge("contextualize", "retrieve")
        builder.add_conditional_edges(
            "retrieve", self._route_after_retrieval,
            {"generate": "generate", "abstain": "abstain", "choose_tool": "choose_tool"},
        )
        builder.add_conditional_edges(
            "choose_tool", self._route_after_selection,
            {"execute_tool": "execute_tool", "generate": "generate"},
        )
        builder.add_edge("execute_tool", "generate")
        builder.add_edge("generate", "validate_citations")
        builder.add_edge("validate_citations", END)
        builder.add_edge("abstain", END)
        self.graph = builder.compile(name="rag_answer")

    def _route_start(self, state: AgentInput) -> Literal["contextualize", "retrieve"]:
        return "contextualize" if state.get("history") else "retrieve"

    def _contextualize(self, state: AgentState) -> dict[str, str]:
        return {"standalone_question": self.provider.rewrite_question(
            state["question"], state["history"],
        )}

    def _effective_question(self, state: AgentState) -> str:
        return state.get("standalone_question", state["question"])

    def _retrieve(self, state: AgentState) -> dict[str, list[SearchHit]]:
        sources = self.retriever.search(
            self._effective_question(state), top_k=state["top_k"],
            document_id=state["document_id"], min_score=state["min_score"],
        )
        if len(sources) > state["top_k"]:
            raise ValueError("The retriever exceeded the requested evidence limit.")
        return {"sources": sources}

    def _route_after_retrieval(
        self, state: AgentState,
    ) -> Literal["generate", "abstain", "choose_tool"]:
        if not state["sources"]:
            return "abstain"
        return "choose_tool" if state.get("use_tools", False) else "generate"

    def _choose_tool(self, state: AgentState) -> dict[str, ToolSelection]:
        if not hasattr(self.provider, "select_tool"):
            raise GenerationUnavailableError(
                "The configured provider does not support tool calling."
            )
        return {"selection": self.provider.select_tool(
            self._effective_question(state), state["sources"],
        )}

    def _route_after_selection(self, state: AgentState) -> Literal["execute_tool", "generate"]:
        return "execute_tool" if state["selection"].call is not None else "generate"

    def _execute_tool(self, state: AgentState) -> dict[str, ToolResult]:
        call = state["selection"].call
        if call is None:
            raise InvalidGenerationError("The model returned no calculator call to execute.")
        return {"tool_result": execute_calculator(call, state["sources"])}

    def _generate(self, state: AgentState) -> dict[str, GeneratedAnswer]:
        if "tool_result" in state:
            return {"generated": self.provider.generate(
                self._effective_question(state), state["sources"],
                tool_selection=state["selection"], tool_result=state["tool_result"],
            )}
        return {"generated": self.provider.generate(
            self._effective_question(state), state["sources"],
        )}

    def _validate_citations(self, state: AgentState) -> AgentOutput:
        response = resolve_answer(
            state["question"], state["generated"], state["sources"],
        )
        if "tool_result" in state:
            tool_result = state["tool_result"]
            cited = {citation.source_id for citation in response.citations}
            if response.answered and not set(tool_result.source_ids).issubset(cited):
                raise InvalidGenerationError(
                    "The answer omitted the calculator's supporting sources."
                )
            response = replace(response, tool_results=[tool_result])
        return {"response": response}

    def _abstain(self, state: AgentState) -> AgentOutput:
        return {"response": AnswerResponse(state["question"], INSUFFICIENT_EVIDENCE, False, [])}

    def ask(
        self, question: str, *, top_k: int = MAX_ANSWER_SOURCES,
        document_id: str | None = None, min_score: float | None = None,
        use_tools: bool = False,
        history: list[ConversationTurn] | None = None,
    ) -> AnswerResponse:
        if not 1 <= top_k <= MAX_ANSWER_SOURCES:
            raise ValueError("Answers require between one and five retrieved passages.")
        result = self.graph.invoke({
            "question": question, "top_k": top_k,
            "document_id": document_id, "min_score": min_score,
            "use_tools": use_tools,
            "history": (history or [])[-RECENT_TURNS:],
        })
        return result["response"]
