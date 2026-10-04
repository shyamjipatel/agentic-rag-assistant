"""Local BGE embeddings with explicit input and vector validation."""

import os
from math import isfinite
from pathlib import Path
from threading import Lock
from typing import Protocol

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384
MAX_EMBEDDING_TOKENS = 512
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingInputError(ValueError):
    """An input is too long for the embedding model."""


class EmbeddingUnavailableError(RuntimeError):
    """The local embedding model could not produce usable vectors."""


class EmbeddingProvider(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


def validate_vector(vector: list[float], dimensions: int = EMBEDDING_DIMENSIONS) -> None:
    if len(vector) != dimensions or not all(isfinite(value) for value in vector):
        raise EmbeddingUnavailableError("The embedding model returned an invalid vector.")
    if not any(value != 0 for value in vector):
        raise EmbeddingUnavailableError("The embedding model returned a zero vector.")


class LocalEmbedder:
    model_name = EMBEDDING_MODEL
    dimensions = EMBEDDING_DIMENSIONS

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self._model = None
        self._tokenizer = None
        self._lock = Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        # Disable the native background uploader before ONNX Runtime initializes.
        os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
        from fastembed import TextEmbedding
        from tokenizers import Tokenizer

        model = TextEmbedding(
            model_name=self.model_name,
            cache_dir=str(self.cache_dir),
            threads=2,
            local_files_only=True,
        )
        # Clone the pinned FastEmbed adapter's tokenizer to count untruncated inputs.
        tokenizer = Tokenizer.from_str(model.model.tokenizer.to_str())
        tokenizer.no_truncation()
        tokenizer.no_padding()
        self._tokenizer = tokenizer
        self._model = model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._lock:
            try:
                self._load()
                for encoding in self._tokenizer.encode_batch(texts):
                    if len(encoding.ids) > MAX_EMBEDDING_TOKENS:
                        raise EmbeddingInputError(
                            "A chunk or query exceeds the 512-token embedding limit. "
                            "Use shorter passages or a shorter query."
                        )
                vectors = [
                    vector.tolist()
                    for vector in self._model.embed(texts, batch_size=32)
                ]
                if len(vectors) != len(texts):
                    raise EmbeddingUnavailableError("The model returned an incomplete batch.")
                for vector in vectors:
                    validate_vector(vector)
                return vectors
            except (EmbeddingInputError, EmbeddingUnavailableError):
                raise
            except Exception as exc:
                raise EmbeddingUnavailableError(
                    "Local embeddings are unavailable. Prepare the model with "
                    "python -m agentic_rag_assistant.embeddings."
                ) from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, query: str) -> list[float]:
        return self._embed([QUERY_INSTRUCTION + query])[0]


def prepare_model(cache_dir: Path) -> None:
    """Verify a populated cache offline before attempting an explicit download."""
    if cache_dir.is_dir() and any(cache_dir.iterdir()):
        try:
            LocalEmbedder(cache_dir).embed_query("Verify the local embedding model.")
            return
        except EmbeddingUnavailableError:
            pass
    from fastembed import TextEmbedding

    TextEmbedding(model_name=EMBEDDING_MODEL, cache_dir=str(cache_dir), threads=2)
    LocalEmbedder(cache_dir).embed_query("Verify the local embedding model.")


if __name__ == "__main__":
    # This is an application entry point, not library import-time TLS mutation.
    import truststore

    truststore.inject_into_ssl()
    # Ordinary HTTPS avoids native Xet transport failures on some networks.
    # Users can opt back in by setting HF_HUB_DISABLE_XET=0 explicitly.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    from agentic_rag_assistant.settings import get_settings

    try:
        prepare_model(get_settings().model_cache_dir)
    except Exception:
        raise SystemExit(
            "Model preparation failed. Check download connectivity, TLS trust, "
            "and cache permissions; an existing prepared cache can be imported offline."
        ) from None
    print(f"Local model prepared: {EMBEDDING_MODEL}, {EMBEDDING_DIMENSIONS} dimensions.")
