"""Embedding retrieval for Lacanian RAG queries."""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np
import faiss
from ..core.config import AppConfig
from .openai_client import OpenAIClient, load_openai_config


@dataclass(frozen=True)
class RetrieverConfig:
    """Configuration for vector retrieval."""

    index_path: Path
    metadata_path: Path
    embedding_model: str


class IndexStore:
    """Load and query a FAISS index with metadata."""

    def __init__(self, config: RetrieverConfig):
        self._config = config
        self._index = faiss.read_index(str(config.index_path))
        with open(config.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self._ids = meta["ids"]
        self._metadata = meta["metadata"]

    @property
    def ids(self) -> List[str]:
        """Return the stored chunk ids."""
        return self._ids

    @property
    def metadata(self) -> List[Dict[str, Any]]:
        """Return metadata rows aligned with the index."""
        return self._metadata

    def search(self, vector: np.ndarray, top_k: int):
        """Search the FAISS index for nearest neighbors."""
        return self._index.search(vector, top_k)


class OpenAIEmbedder:
    """Create embeddings using the OpenAI client."""

    def __init__(self, client: OpenAIClient, model: str):
        self._client = client
        self._model = model

    def embed(self, text: str) -> np.ndarray:
        """Embed a query string into a vector suitable for similarity search."""
        embedding = self._client.embed(text, model=self._model)
        return np.array(embedding, dtype="float32").reshape(1, -1)


def _normalize_source_id(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized.lower()


class Retriever:
    """Combine embeddings with an index store to return scored results."""

    def __init__(self, index_store: IndexStore, embedder: OpenAIEmbedder):
        self._index_store = index_store
        self._embedder = embedder

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        source_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search the index for the top_k most similar chunks."""
        logger = logging.getLogger("text_extractor_api")
        debug = logger.isEnabledFor(logging.DEBUG)
        total_start = time.monotonic() if debug else 0.0
        embed_start = time.monotonic() if debug else 0.0
        query_vector = self._embedder.embed(query)
        if debug:
            embed_ms = (time.monotonic() - embed_start) * 1000.0
        normalized_source_id = _normalize_source_id(source_id)
        candidate_k = top_k
        if normalized_source_id:
            # Oversample to keep enough matches after filtering.
            candidate_k = max(top_k, top_k * 5)
        index_size = len(self._index_store.metadata)
        if index_size == 0:
            return []
        candidate_k = min(candidate_k, index_size)
        search_start = time.monotonic() if debug else 0.0
        distances, indices = self._index_store.search(query_vector, candidate_k)
        if debug:
            search_ms = (time.monotonic() - search_start) * 1000.0
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= len(self._index_store.metadata):
                continue
            meta = self._index_store.metadata[idx]
            seminar = meta.get("seminar")
            if normalized_source_id:
                if not seminar or seminar.lower() != normalized_source_id:
                    continue
            result = {
                "score": float(distances[0][i]),
                "id": self._index_store.ids[idx],
                "seminar": meta.get("seminar"),
                "lecon": meta.get("lecon"),
                "pages": meta.get("pages"),
            }
            results.append(result)
            if len(results) >= top_k:
                break
        if debug:
            total_ms = (time.monotonic() - total_start) * 1000.0
            logger.debug(
                "RAG stage=retrieval_detail total_ms=%.2f embed_ms=%.2f faiss_ms=%.2f "
                "top_k=%s candidate_k=%s results=%s source_id=%s",
                total_ms,
                embed_ms,
                search_ms,
                top_k,
                candidate_k,
                len(results),
                source_id or "-",
            )
        return results


class _RetrieverSingleton:
    """Singleton wrapper for the default Retriever instance."""

    _instance: Optional[Retriever] = None

    @classmethod
    def get_instance(cls) -> Retriever:
        """Return the shared Retriever instance."""
        if cls._instance is None:
            cls._instance = cls._build()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear the cached Retriever instance."""
        cls._instance = None

    @staticmethod
    def _build() -> Retriever:
        """Construct a Retriever with default configuration."""
        app_config = AppConfig()
        openai_client = OpenAIClient(load_openai_config())
        config = RetrieverConfig(
            index_path=app_config.vector_index_path,
            metadata_path=app_config.metadata_path,
            embedding_model=openai_client.config.embedding_model,
        )
        index_store = IndexStore(config)
        embedder = OpenAIEmbedder(openai_client, config.embedding_model)
        return Retriever(index_store, embedder)


def get_default_retriever() -> Retriever:
    """Return the shared Retriever instance."""
    return _RetrieverSingleton.get_instance()


def reset_default_retriever() -> None:
    """Reset the shared Retriever instance."""
    _RetrieverSingleton.reset()


def search_similar_chunks(
    query: str,
    top_k: int = 5,
    *,
    source_id: Optional[str] = None,
):
    """Search the index for the top_k most similar chunks."""
    return get_default_retriever().search(
        query,
        top_k=top_k,
        source_id=source_id,
    )
