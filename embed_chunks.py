"""Build embeddings and FAISS index from chunked texts."""

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import faiss
from tqdm import tqdm
from config import AppConfig
from openai_client import OpenAIClient, load_openai_config

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
