"""FastAPI app exposing RAG query and text indexation endpoints."""

from __future__ import annotations

from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..application.index_text import index_text as run_index_text
from ..application.rag_query import run_rag_query
from .logger import configure_logger, request_logging_middleware

API_DESCRIPTION = (
    "API for Lacanian RAG queries and incremental indexation of raw text. "
    "Swagger UI is available at /docs."
)

TAGS_METADATA = [
    {
        "name": "RAG",
        "description": (
            "Query the existing vector index and return a structured response "
            "with citations and source metadata."
        ),
    },
    {
        "name": "Indexing",
        "description": (
            "Chunk raw text, persist chunks, and append new embeddings to the "
            "existing FAISS index and metadata store."
        ),
    },
]

app = FastAPI(
    title="Text Extractor API",
    description=API_DESCRIPTION,
    version="1.0.0",
    openapi_tags=TAGS_METADATA,
    contact={
        "name": "Text Extractor API",
    },
    license_info={
        "name": "Proprietary",
    },
)

LOGGER = configure_logger()
app.middleware("http")(request_logging_middleware(LOGGER))


class RagQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question.")
    language: Literal["en", "es"] = Field(
        "en", description="Response language."
    )
    top_k: Optional[int] = Field(
        None, ge=1, description="Top-k chunks to retrieve from the index."
    )
    max_sources: Optional[int] = Field(
        None, ge=1, description="Max sources to include in the response."
    )


class SourceMetadataModel(BaseModel):
    seminar_title: str = Field(..., description="Localized seminar title.")
    seminar_id: str = Field(..., description="Seminar identifier.")
    lesson_label: str = Field(..., description="Localized lesson label.")
    lesson_raw: Optional[str] = Field(
        None, description="Raw lesson identifier if available."
    )
    chunk_id: str = Field(..., description="Chunk id within the source.")
    chunk_index: int = Field(..., ge=0, description="Chunk index.")
    pages: List[int] = Field(
        default_factory=list, description="Page numbers for the chunk."
    )


class SourceModel(BaseModel):
    source_index: int = Field(..., ge=1, description="1-based source index.")
    french_quotes: List[str] = Field(
        default_factory=list, description="Exact quotes from context."
    )
    translations: List[str] = Field(
        default_factory=list, description="Translations for each quote."
    )
    translation_critique: str = Field(
        ..., description="Critique and alternatives for the translation."
    )
    context: str = Field(..., description="Contextual summary.")
    lacanian_development: str = Field(
        ..., description="Interpretive development grounded in context."
    )
    source_metadata: SourceMetadataModel = Field(
        ..., description="Metadata for the cited source."
    )


class SourceCatalogEntryModel(BaseModel):
    source_index: int = Field(..., ge=1, description="1-based source index.")
    seminar_title: str = Field(..., description="Localized seminar title.")
    seminar_id: str = Field(..., description="Seminar identifier.")
    lesson_label: str = Field(..., description="Localized lesson label.")
    chunk_id: str = Field(..., description="Chunk id within the source.")
    chunk_index: int = Field(..., ge=0, description="Chunk index.")
    pages: List[int] = Field(
        default_factory=list, description="Page numbers for the chunk."
    )


class RagResultModel(BaseModel):
    language: Literal["en", "es"] = Field(
        ..., description="Language code for the response."
    )
    sources: List[SourceModel] = Field(
        default_factory=list, description="Enriched source responses."
    )
    comparative_trajectory: str = Field(
        ..., description="Cross-source synthesis."
    )
    sources_catalog: List[SourceCatalogEntryModel] = Field(
        default_factory=list,
        description="Compact catalog of the sources used.",
    )


class WarningsModel(BaseModel):
    translation_error: Optional[str] = Field(
        None, description="Translation warning when fallback is used."
    )
    validation_errors: List[str] = Field(
        default_factory=list, description="Schema validation warnings."
    )


class RagQueryResponse(BaseModel):
    result: RagResultModel = Field(..., description="Structured RAG result.")
    warnings: WarningsModel = Field(..., description="Warnings and notes.")


class IndexTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to index.")
    doc_id: Optional[str] = Field(
        None, description="Optional identifier for the document."
    )


class IndexTextResponse(BaseModel):
    doc_id: str = Field(..., description="Final document id used.")
    chunk_count: int = Field(..., description="Total chunks created.")
    added_vectors: int = Field(
        ..., description="Vectors appended to the FAISS index."
    )
    chunks_path: str = Field(
        ..., description="Filesystem path to the stored chunks JSON."
    )
    index_path: str = Field(..., description="Filesystem path to the index.")
    metadata_path: str = Field(
        ..., description="Filesystem path to the metadata JSON."
    )


@app.post(
    "/rag/query",
    response_model=RagQueryResponse,
    summary="Run a RAG query",
    description=(
        "Runs retrieval-augmented generation over the existing FAISS index. "
        "Returns a structured JSON payload with citations and metadata."
    ),
    tags=["RAG"],
    responses={
        200: {
            "description": "RAG response payload.",
            "content": {
                "application/json": {
                    "example": {
                        "result": {
                            "language": "en",
                            "sources": [
                                {
                                    "source_index": 1,
                                    "french_quotes": ["..."],
                                    "translations": ["..."],
                                    "translation_critique": "...",
                                    "context": "...",
                                    "lacanian_development": "...",
                                    "source_metadata": {
                                        "seminar_title": "Seminar XI",
                                        "seminar_id": "Seminar_XI",
                                        "lesson_label": "Lesson 1",
                                        "lesson_raw": "Lecon_1",
                                        "chunk_id": "chunk_001",
                                        "chunk_index": 1,
                                        "pages": [12, 13],
                                    },
                                }
                            ],
                            "comparative_trajectory": "...",
                            "sources_catalog": [
                                {
                                    "source_index": 1,
                                    "seminar_title": "Seminar XI",
                                    "seminar_id": "Seminar_XI",
                                    "lesson_label": "Lesson 1",
                                    "chunk_id": "chunk_001",
                                    "chunk_index": 1,
                                    "pages": [12, 13],
                                }
                            ],
                        },
                        "warnings": {
                            "translation_error": None,
                            "validation_errors": [],
                        },
                    }
                }
            },
        },
        400: {"description": "Invalid request parameters."},
        500: {"description": "Internal server error."},
    },
)
def rag_query(request: RagQueryRequest):
    try:
        payload, warnings = run_rag_query(
            request.question,
            request.language,
            top_k=request.top_k,
            max_sources=request.max_sources,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RagQueryResponse(result=payload, warnings=warnings)


@app.post(
    "/rag/index-text",
    response_model=IndexTextResponse,
    summary="Index raw text",
    description=(
        "Chunks raw text, writes the chunk JSON, and appends new embeddings "
        "to the existing FAISS index and metadata store."
    ),
    tags=["Indexing"],
    responses={
        200: {
            "description": "Indexation result.",
            "content": {
                "application/json": {
                    "example": {
                        "doc_id": "doc_abc123",
                        "chunk_count": 5,
                        "added_vectors": 5,
                        "chunks_path": "text_chunks_json/doc_abc123.json",
                        "index_path": "vector_index/lacan.index",
                        "metadata_path": "vector_index/metadata.json",
                    }
                }
            },
        },
        400: {"description": "Invalid input or empty text."},
        409: {"description": "doc_id already exists."},
        500: {"description": "Indexation error."},
    },
)
def index_text(request: IndexTextRequest):
    try:
        payload = run_index_text(request.text, request.doc_id)
    except ValueError as exc:
        detail = str(exc)
        if "already exists" in detail:
            raise HTTPException(status_code=409, detail=detail) from exc
        raise HTTPException(status_code=400, detail=detail) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IndexTextResponse(**payload)
