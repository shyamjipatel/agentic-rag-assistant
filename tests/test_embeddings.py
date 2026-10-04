import os
import sys
from pathlib import Path
from types import SimpleNamespace

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
    prepare_model,
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


def test_loading_disables_native_telemetry_before_constructing_the_offline_model(monkeypatch):
    monkeypatch.delenv("ORT_DISABLE_TELEMETRY", raising=False)
    prepared = cached_embedder()
    model = prepared._model
    model.model = SimpleNamespace(tokenizer=prepared._tokenizer)

    def make_model(**options):
        assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"
        assert options["local_files_only"] is True
        return model

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=make_model))
    embedder = LocalEmbedder(Path("unused-test-cache"))
    assert len(embedder.embed_query("A question?")) == EMBEDDING_DIMENSIONS


def test_prepared_cache_is_verified_without_any_download(monkeypatch, tmp_path):
    (tmp_path / "model-marker").write_text("populated cache")
    monkeypatch.setattr(LocalEmbedder, "embed_query", lambda self, text: [1.0] * 384)

    def unexpected_download(**kwargs):
        raise AssertionError("A prepared cache must stay offline")

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=unexpected_download))
    prepare_model(tmp_path)


@pytest.mark.parametrize("populated", [False, True])
def test_missing_or_unusable_cache_triggers_explicit_model_preparation(
    monkeypatch, tmp_path, populated,
):
    if populated:
        (tmp_path / "model-marker").write_text("incomplete cache")
    downloads = []

    def verify(self, text):
        if not downloads:
            raise EmbeddingUnavailableError("Missing model")
        return [1.0] * 384

    monkeypatch.setattr(LocalEmbedder, "embed_query", verify)
    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(
        TextEmbedding=lambda **options: downloads.append(options),
    ))
    prepare_model(tmp_path)
    assert len(downloads) == 1
    assert downloads[0]["cache_dir"] == str(tmp_path)
