from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from agentic_rag_assistant.embeddings import (
    EMBEDDING_DIMENSIONS,
    QUERY_INSTRUCTION,
    EmbeddingInputError,
    EmbeddingUnavailableError,
    LocalEmbedder,
    validate_vector,
)


class FakeVector:
    def tolist(self):
        return [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)


class FakeModel:
    def __init__(self):
        self.inputs = []

    def embed(self, texts, batch_size):
        self.inputs.extend(texts)
        return [FakeVector() for _ in texts]


def cached_embedder():
    embedder = LocalEmbedder(Path("unused-test-cache"))
    embedder._model = FakeModel()
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "word": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    embedder._tokenizer = tokenizer
    return embedder


def test_queries_receive_model_instruction_and_passages_do_not() -> None:
    embedder = cached_embedder()
    vectors = embedder.embed_documents(["A passage."])
    query_vector = embedder.embed_query("A question?")

    assert len(vectors) == 1
    assert len(query_vector) == EMBEDDING_DIMENSIONS
    assert embedder._model.inputs == ["A passage.", QUERY_INSTRUCTION + "A question?"]


def test_long_input_is_rejected_before_inference_without_truncation() -> None:
    embedder = cached_embedder()
    with pytest.raises(EmbeddingInputError, match="512-token"):
        embedder.embed_documents(["word " * 513])
    assert embedder._model.inputs == []


@pytest.mark.parametrize(
    "vector",
    [[], [1.0], [0.0] * EMBEDDING_DIMENSIONS,
     [float("nan")] * EMBEDDING_DIMENSIONS, [float("inf")] * EMBEDDING_DIMENSIONS],
)
def test_invalid_vectors_are_rejected(vector) -> None:
    with pytest.raises(EmbeddingUnavailableError):
        validate_vector(vector)


def test_model_load_failure_becomes_an_actionable_service_error(monkeypatch) -> None:
    embedder = LocalEmbedder(Path("unused-test-cache"))

    def fail_load():
        raise OSError("unavailable cache")

    monkeypatch.setattr(embedder, "_load", fail_load)
    with pytest.raises(EmbeddingUnavailableError, match="Prepare the model"):
        embedder.embed_query("A question?")
