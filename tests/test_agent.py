"""Exercise real graph routing, request isolation, and failure boundaries."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from agentic_rag_assistant.agent import RAGAgent
from agentic_rag_assistant.answering import (
    INSUFFICIENT_EVIDENCE,
    GeneratedAnswer,
    GenerationTimeoutError,
    SupportedStatement,
)
from agentic_rag_assistant.models import DocumentChunk, SearchHit


def source(name: str) -> SearchHit:
    text = f"Evidence for {name}."
    return SearchHit(
        document_id="a" * 64, filename=f"{name}.txt", score=0.8,
        chunk=DocumentChunk(f"{name}:v1:0", 0, text, None, 0, len(text)),
    )


class Retriever:
    def __init__(self, hits_by_question):
        self.hits_by_question = hits_by_question
        self.calls = []

    def search(self, query, **options):
        self.calls.append(query)
        return self.hits_by_question[query]


class Provider:
    def __init__(self):
        self.calls = []

    def generate(self, question, sources):
        self.calls.append(question)
        return GeneratedAnswer(
            supported=True,
            statements=[SupportedStatement(text=f"Answer for {question}.", source_ids=[1])],
        )


@pytest.mark.parametrize(
    "hits,expected_nodes",
    [([source("policy")], ["retrieve", "generate", "validate_citations"]),
     ([], ["retrieve", "abstain"])],
)
def test_graph_runs_only_the_selected_evidence_branch(hits, expected_nodes):
    provider = Provider()
    agent = RAGAgent(Retriever({"policy": hits}), provider)
    updates = list(agent.graph.stream(
        {"question": "policy", "top_k": 2, "document_id": None, "min_score": None},
        stream_mode="updates",
    ))
    assert [name for update in updates for name in update] == expected_nodes
    response = updates[-1][expected_nodes[-1]]["response"]
    assert response.answered is bool(hits)
    assert provider.calls == (["policy"] if hits else [])


def test_reused_graph_does_not_carry_sources_or_answers_into_the_next_request():
    provider = Provider()
    agent = RAGAgent(Retriever({"policy": [source("policy")], "missing": []}), provider)
    compiled = agent.graph
    first = agent.ask("policy")
    second = agent.ask("missing")
    assert first.citations[0].filename == "policy.txt"
    assert second.question == "missing"
    assert second.answer == INSUFFICIENT_EVIDENCE
    assert second.answered is False
    assert second.citations == []
    assert provider.calls == ["policy"]
    assert agent.graph is compiled


def test_concurrent_requests_keep_their_own_questions_and_sources():
    barrier = Barrier(2)

    class ConcurrentRetriever:
        def search(self, query, **options):
            barrier.wait(timeout=5)
            return [source(query)]

    class ConcurrentProvider:
        def generate(self, question, sources):
            assert sources[0].filename == f"{question}.txt"
            return GeneratedAnswer(
                supported=True,
                statements=[SupportedStatement(text=f"Answer for {question}.", source_ids=[1])],
            )

    agent = RAGAgent(ConcurrentRetriever(), ConcurrentProvider())
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(agent.ask, ["leave", "expense"]))
    for question, result in zip(["leave", "expense"], results, strict=True):
        assert result.question == question
        assert result.answer == f"Answer for {question}. [1]"
        assert result.citations[0].filename == f"{question}.txt"


@pytest.mark.parametrize("top_k", [0, 6])
def test_invalid_evidence_limit_is_rejected_before_running_the_graph(top_k):
    retriever = Retriever({"policy": [source("policy")]})
    provider = Provider()
    with pytest.raises(ValueError, match="between one and five"):
        RAGAgent(retriever, provider).ask("policy", top_k=top_k)
    assert retriever.calls == []
    assert provider.calls == []


def test_retriever_cannot_exceed_the_generation_evidence_limit():
    provider = Provider()
    agent = RAGAgent(Retriever({"policy": [source("leave"), source("expense")]}), provider)
    with pytest.raises(ValueError, match="exceeded"):
        agent.ask("policy", top_k=1)
    assert provider.calls == []


def test_provider_failure_propagates_without_retry_and_next_request_can_succeed():
    class FailsOnceProvider(Provider):
        def generate(self, question, sources):
            if not self.calls:
                self.calls.append(question)
                raise GenerationTimeoutError("Timed out")
            return super().generate(question, sources)

    provider = FailsOnceProvider()
    agent = RAGAgent(Retriever({"policy": [source("policy")]}), provider)
    with pytest.raises(GenerationTimeoutError, match="Timed out"):
        agent.ask("policy")
    assert provider.calls == ["policy"]
    result = agent.ask("policy")
    assert provider.calls == ["policy", "policy"]
    assert result.answered is True
