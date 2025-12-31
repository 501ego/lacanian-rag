"""Build embeddings and FAISS index from chunked texts."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import faiss
from tqdm import tqdm
from ..core.config import AppConfig
from .openai_client import OpenAIClient, load_openai_config

APP_CONFIG = AppConfig()


class ChunkLoader:
    """Load chunked JSON files from disk."""

    def __init__(self, chunks_dir: Path):
        self._chunks_dir = chunks_dir

    def load(self) -> List[Dict[str, Any]]:
        """Load all chunked JSONs into a list of {id, text, metadata} dicts."""
        records = []
        for file in self._chunks_dir.glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                chunk_items = json.load(f)
                for chunk in chunk_items:
                    records.append({
                        "id": f"{file.stem}_{chunk['chunk_id']}",
                        "text": chunk["text"],
                        "metadata": {
                            "seminar": chunk.get("seminar"),
                            "lecon": chunk.get("lecon"),
                            "pages": chunk.get("pages"),
                        },
                    })
        return records


class OpenAIEmbedder:
    """Embed text using the OpenAI client."""

    def __init__(self, client: OpenAIClient, model: str):
        self._client = client
        self._model = model

    def embed(self, text: str) -> List[float]:
        """Embed text with the configured model."""
        return self._client.embed(text, model=self._model)


class FaissIndexBuilder:
    """Build and persist a FAISS index with metadata."""

    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        embedding_dim: int,
        embedder_instance: OpenAIEmbedder,
    ):
        self._index_path = index_path
        self._metadata_path = metadata_path
        self._embedding_dim = embedding_dim
        self._embedder = embedder_instance

    def build(self, chunk_records: List[Dict[str, Any]]):
        """Create a FAISS index from chunk embeddings and persist metadata."""
        if not chunk_records:
            print("No chunk records found. Skipping FAISS index build.")
            return

        index = faiss.IndexFlatL2(self._embedding_dim)
        metadata = []
        ids = []
        embeddings = []

        for chunk in tqdm(chunk_records, desc="Embedding chunks"):
            embedding = self._embedder.embed(chunk["text"])
            embeddings.append(embedding)
            metadata.append(chunk["metadata"])
            ids.append(chunk["id"])

        if not embeddings:
            print("No embeddings generated. Skipping FAISS index build.")
            return

        embeddings_np = np.array(embeddings, dtype="float32")
        if embeddings_np.ndim == 1:
            embeddings_np = embeddings_np.reshape(1, -1)
        if embeddings_np.shape[1] != self._embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self._embedding_dim}, got {embeddings_np.shape[1]}"
            )
        index.add(embeddings_np)

        self._index_path.parent.mkdir(exist_ok=True)
        faiss.write_index(index, str(self._index_path))

        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump({"ids": ids, "metadata": metadata},
                      f, ensure_ascii=False, indent=2)

        print(f"FAISS index saved with {len(chunk_records)} entries.")


class FaissIndexUpdater:
    """Append new embeddings and metadata to an existing FAISS index."""

    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        embedding_dim: int,
        embedder_instance: OpenAIEmbedder,
    ):
        self._index_path = index_path
        self._metadata_path = metadata_path
        self._embedding_dim = embedding_dim
        self._embedder = embedder_instance

    def append(self, chunk_records: List[Dict[str, Any]]) -> int:
        """Append chunk records to the index and metadata store."""
        if not chunk_records:
            return 0
        ids, metadata = self._load_metadata()
        existing_ids = set(ids)
        embeddings = []
        new_ids = []
        new_metadata = []
        for chunk in chunk_records:
            chunk_id = chunk.get("id")
            if not chunk_id or chunk_id in existing_ids:
                continue
            embedding = self._embedder.embed(chunk["text"])
            embeddings.append(embedding)
            new_ids.append(chunk_id)
            new_metadata.append(chunk["metadata"])

        if not embeddings:
            return 0

        embeddings_np = np.array(embeddings, dtype="float32")
        if embeddings_np.ndim == 1:
            embeddings_np = embeddings_np.reshape(1, -1)
        if embeddings_np.shape[1] != self._embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self._embedding_dim}, got {embeddings_np.shape[1]}"
            )

        index = self._load_or_create_index()
        if index.d != self._embedding_dim:
            raise ValueError(
                f"Index dim mismatch: expected {self._embedding_dim}, got {index.d}"
            )
        index.add(embeddings_np)

        ids.extend(new_ids)
        metadata.extend(new_metadata)

        self._index_path.parent.mkdir(exist_ok=True)
        faiss.write_index(index, str(self._index_path))

        self._metadata_path.parent.mkdir(exist_ok=True)
        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump({"ids": ids, "metadata": metadata},
                      f, ensure_ascii=False, indent=2)
        return len(new_ids)

    def _load_or_create_index(self):
        if self._index_path.exists():
            return faiss.read_index(str(self._index_path))
        return faiss.IndexFlatL2(self._embedding_dim)

    def _load_metadata(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        if not self._metadata_path.exists():
            return [], []
        with open(self._metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        ids = payload.get("ids", [])
        metadata = payload.get("metadata", [])
        if not isinstance(ids, list) or not isinstance(metadata, list):
            return [], []
        return ids, metadata


if __name__ == "__main__":
    print("Loading text chunks...")
    openai_client = OpenAIClient(load_openai_config())
    chunk_loader = ChunkLoader(APP_CONFIG.chunks_dir)
    chunk_records = chunk_loader.load()
    print(f"{len(chunk_records)} chunks loaded.")
    embedder = OpenAIEmbedder(
        openai_client,
        openai_client.config.embedding_model,
    )
    index_builder = FaissIndexBuilder(
        index_path=APP_CONFIG.vector_index_path,
        metadata_path=APP_CONFIG.metadata_path,
        embedding_dim=APP_CONFIG.embedding_dim,
        embedder_instance=embedder,
    )
    index_builder.build(chunk_records)
