"""Index raw text or PDFs into the existing vector store."""

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from ..core.config import AppConfig
from ..infrastructure.embed_chunks import FaissIndexUpdater, OpenAIEmbedder
from ..infrastructure.openai_client import OpenAIClient, load_openai_config
from ..infrastructure.retriever import reset_default_retriever
from ..infrastructure.text_extractor import PDFChunker, TextChunker

APP_CONFIG = AppConfig()


def _normalize_doc_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def _metadata_has_doc(metadata_path, doc_id: str) -> bool:
    if not metadata_path.exists():
        return False
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    ids = payload.get("ids", [])
    if not isinstance(ids, list):
        return False
    prefix = f"{doc_id}_"
    return any(isinstance(item, str) and item.startswith(prefix) for item in ids)


def _seminar_id_from_pdf(pdf_path: Path) -> str:
    return (
        pdf_path.stem.replace(" ", "_")
        .replace("'", "")
        .replace("...", "")
        .replace("__", "_")
    )


def index_text(text: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
    """Chunk and index text, then return metadata about the operation."""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")
    doc_id = _normalize_doc_id(doc_id or f"doc_{uuid4().hex[:12]}")
    if not doc_id:
        raise ValueError("Invalid doc_id.")

    chunks_path = APP_CONFIG.chunks_dir / f"{doc_id}.json"
    if chunks_path.exists() or _metadata_has_doc(APP_CONFIG.metadata_path, doc_id):
        raise ValueError("doc_id already exists.")

    chunker = TextChunker(
        chunk_size=APP_CONFIG.chunk_size,
        overlap=APP_CONFIG.overlap,
    )
    chunks = chunker.chunk_text(text)
    if not chunks:
        raise ValueError("No chunks produced.")

    chunk_payload = []
    chunk_records = []
    for chunk in chunks:
        chunk_payload.append(
            {
                "seminar": doc_id,
                "lecon": None,
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "pages": chunk["pages"],
            }
        )
        chunk_records.append(
            {
                "id": f"{doc_id}_{chunk['chunk_id']}",
                "text": chunk["text"],
                "metadata": {
                    "seminar": doc_id,
                    "lecon": None,
                    "pages": chunk["pages"],
                },
            }
        )

    chunks_path.parent.mkdir(exist_ok=True)
    try:
        chunks_path.write_text(
            json.dumps(chunk_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        openai_client = OpenAIClient(load_openai_config())
        embedder = OpenAIEmbedder(
            openai_client, openai_client.config.embedding_model
        )
        updater = FaissIndexUpdater(
            index_path=APP_CONFIG.vector_index_path,
            metadata_path=APP_CONFIG.metadata_path,
            embedding_dim=APP_CONFIG.embedding_dim,
            embedder_instance=embedder,
        )
        added = updater.append(chunk_records)
    except Exception as exc:
        if chunks_path.exists():
            chunks_path.unlink()
        raise RuntimeError(str(exc)) from exc

    reset_default_retriever()
    return {
        "doc_id": doc_id,
        "chunk_count": len(chunks),
        "added_vectors": added,
        "chunks_path": str(chunks_path),
        "index_path": str(APP_CONFIG.vector_index_path),
        "metadata_path": str(APP_CONFIG.metadata_path),
    }


def index_pdf_bytes(
    file_bytes: bytes,
    filename: str,
    doc_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Store, chunk, and index a PDF upload."""
    if not file_bytes:
        raise ValueError("Empty file.")
    if not filename.lower().endswith(".pdf"):
        raise ValueError("File must be a PDF.")

    if doc_id:
        doc_id = _normalize_doc_id(doc_id)
    else:
        doc_id = _normalize_doc_id(Path(filename).stem)
    if not doc_id:
        doc_id = f"doc_{uuid4().hex[:12]}"

    if _metadata_has_doc(APP_CONFIG.metadata_path, doc_id):
        raise ValueError("doc_id already exists.")

    APP_CONFIG.input_pdf_dir.mkdir(exist_ok=True)
    pdf_path = APP_CONFIG.input_pdf_dir / f"{doc_id}.pdf"
    if pdf_path.exists():
        raise ValueError("doc_id already exists.")

    try:
        pdf_path.write_bytes(file_bytes)
        return _index_pdf_path(pdf_path)
    except ValueError:
        if pdf_path.exists():
            pdf_path.unlink()
        raise
    except Exception as exc:
        if pdf_path.exists():
            pdf_path.unlink()
        raise RuntimeError(str(exc)) from exc


def _index_pdf_path(pdf_path: Path) -> Dict[str, Any]:
    if not pdf_path.exists():
        raise ValueError("PDF file not found.")
    seminar_id = _seminar_id_from_pdf(pdf_path)
    if not seminar_id:
        raise ValueError("Invalid PDF filename.")

    chunks_path = APP_CONFIG.output_dir / f"{seminar_id}.json"
    if chunks_path.exists() or _metadata_has_doc(APP_CONFIG.metadata_path, seminar_id):
        raise ValueError("doc_id already exists.")

    chunker = PDFChunker(
        input_dir=APP_CONFIG.input_pdf_dir,
        output_dir=APP_CONFIG.output_dir,
        chunk_size=APP_CONFIG.chunk_size,
        overlap=APP_CONFIG.overlap,
    )
    chunker.process_pdf_to_json(pdf_path)
    if not chunks_path.exists():
        raise RuntimeError("Chunking failed.")

    chunk_items = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunk_records = []
    for chunk in chunk_items:
        chunk_records.append(
            {
                "id": f"{seminar_id}_{chunk['chunk_id']}",
                "text": chunk["text"],
                "metadata": {
                    "seminar": chunk.get("seminar"),
                    "lecon": chunk.get("lecon"),
                    "pages": chunk.get("pages"),
                },
            }
        )

    try:
        openai_client = OpenAIClient(load_openai_config())
        embedder = OpenAIEmbedder(
            openai_client, openai_client.config.embedding_model
        )
        updater = FaissIndexUpdater(
            index_path=APP_CONFIG.vector_index_path,
            metadata_path=APP_CONFIG.metadata_path,
            embedding_dim=APP_CONFIG.embedding_dim,
            embedder_instance=embedder,
        )
        added = updater.append(chunk_records)
    except Exception as exc:
        if chunks_path.exists():
            chunks_path.unlink()
        raise RuntimeError(str(exc)) from exc

    reset_default_retriever()
    return {
        "doc_id": seminar_id,
        "chunk_count": len(chunk_items),
        "added_vectors": added,
        "chunks_path": str(chunks_path),
        "index_path": str(APP_CONFIG.vector_index_path),
        "metadata_path": str(APP_CONFIG.metadata_path),
    }
