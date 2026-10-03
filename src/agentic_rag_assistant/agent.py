"""A bounded LangGraph workflow for answering from retrieved evidence."""

from typing import Literal, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from agentic_rag_assistant.answering import (
    INSUFFICIENT_EVIDENCE,
    MAX_ANSWER_SOURCES,
    GeneratedAnswer,
    LLMProvider,
    PassageRetriever,
    resolve_answer,
)
from agentic_rag_assistant.models import AnswerResponse, SearchHit


class AgentInput(TypedDict):
    question: str
    top_k: int
    document_id: str | None
    min_score: float | None


class AgentState(AgentInput):
    sources: NotRequired[list[SearchHit]]
    generated: NotRequired[GeneratedAnswer]
    response: NotRequired[AnswerResponse]


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
        builder.add_edge(START, "retrieve")
        builder.add_conditional_edges(
            "retrieve", self._route_after_retrieval,
            {"generate": "generate", "abstain": "abstain"},
        )
        builder.add_edge("generate", "validate_citations")
        builder.add_edge("validate_citations", END)
        builder.add_edge("abstain", END)
        self.graph = builder.compile(name="rag_answer")

    def _retrieve(self, state: AgentInput) -> dict[str, list[SearchHit]]:
        sources = self.retriever.search(
            state["question"], top_k=state["top_k"],
            document_id=state["document_id"], min_score=state["min_score"],
        )
        if len(sources) > state["top_k"]:
            raise ValueError("The retriever exceeded the requested evidence limit.")
        return {"sources": sources}

    def _route_after_retrieval(self, state: AgentState) -> Literal["generate", "abstain"]:
        return "generate" if state["sources"] else "abstain"

    def _generate(self, state: AgentState) -> dict[str, GeneratedAnswer]:
        return {"generated": self.provider.generate(state["question"], state["sources"])}

    def _validate_citations(self, state: AgentState) -> AgentOutput:
        return {"response": resolve_answer(
            state["question"], state["generated"], state["sources"],
        )}

    def _abstain(self, state: AgentState) -> AgentOutput:
        return {"response": AnswerResponse(state["question"], INSUFFICIENT_EVIDENCE, False, [])}

    def ask(
        self, question: str, *, top_k: int = MAX_ANSWER_SOURCES,
        document_id: str | None = None, min_score: float | None = None,
    ) -> AnswerResponse:
        if not 1 <= top_k <= MAX_ANSWER_SOURCES:
            raise ValueError("Answers require between one and five retrieved passages.")
        result = self.graph.invoke({
            "question": question, "top_k": top_k,
            "document_id": document_id, "min_score": min_score,
        })
        return result["response"]
