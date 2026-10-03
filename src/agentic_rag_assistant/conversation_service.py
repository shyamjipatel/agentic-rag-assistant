"""Load committed history, run an answer, then atomically save its completed turn."""

from typing import Protocol
from uuid import UUID

from agentic_rag_assistant.agent import RAGAgent
from agentic_rag_assistant.memory import RECENT_TURNS
from agentic_rag_assistant.models import AnswerResponse, ConversationHistory


class ConversationStore(Protocol):
    def load(self, identifier: UUID, *, limit: int) -> ConversationHistory: ...

    def append(
        self, identifier: UUID, *, expected_turn_count: int, response: AnswerResponse,
    ) -> AnswerResponse: ...


def answer_with_memory(
    agent: RAGAgent, store: ConversationStore, question: str, *,
    conversation_id: UUID | None, top_k: int, document_id: str | None,
    min_score: float | None, use_tools: bool,
) -> AnswerResponse:
    options = {"top_k": top_k, "document_id": document_id,
               "min_score": min_score, "use_tools": use_tools}
    if conversation_id is None:
        return agent.ask(question, **options)
    context = store.load(conversation_id, limit=RECENT_TURNS)
    response = agent.ask(question, **options, history=context.turns)
    return store.append(
        conversation_id, expected_turn_count=context.turn_count, response=response,
    )
