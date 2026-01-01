"""FastAPI app exposing RAG query and text indexation endpoints."""

from __future__ import annotations

import json
from typing import List, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..application.index_text import index_pdf_bytes as run_index_pdf
from ..application.rag_query import (
    run_rag_expand,
    run_rag_expand_stream,
    run_rag_query,
    run_rag_query_stream,
)
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LOGGER = configure_logger()
app.middleware("http")(request_logging_middleware(LOGGER))


class RagQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question.")
    language: Literal["en", "es"] = Field(
        "en", description="Response language."
    )
    detail: Literal["concise", "full"] = Field(
        "concise", description="Response detail level."
    )
    stream: bool = Field(
        False, description="Stream response using SSE."
    )
    top_k: Optional[int] = Field(
        None, ge=1, description="Top-k chunks to retrieve from the index."
    )
    max_sources: Optional[int] = Field(
        None, ge=1, description="Max sources to include in the response."
    )
    source_id: Optional[str] = Field(
        None,
        description=(
            "Optional source identifier (seminar/doc id) to restrict retrieval."
        ),
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


class RetrievalCatalogEntryModel(BaseModel):
    rank: int = Field(..., ge=1, description="1-based retrieval rank.")
    score: Optional[float] = Field(None, description="Vector distance score.")
    seminar_id: Optional[str] = Field(None, description="Seminar identifier.")
    seminar_title: str = Field(..., description="Localized seminar title.")
    lesson_label: str = Field(..., description="Localized lesson label.")
    lesson_raw: Optional[str] = Field(
        None, description="Raw lesson identifier if available."
    )
    chunk_id: Optional[str] = Field(None, description="Chunk id within the source.")
    chunk_index: Optional[int] = Field(None, ge=0, description="Chunk index.")
    full_id: Optional[str] = Field(None, description="Full chunk id.")
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
    retrieval_catalog: List[RetrievalCatalogEntryModel] = Field(
        default_factory=list,
        description="Top-k retrieved references with metadata.",
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


class RagExpandRequest(BaseModel):
    language: Optional[Literal["en", "es"]] = Field(
        None, description="Response language override."
    )
    question: Optional[str] = Field(
        None, description="Optional question or focus for the expansion."
    )
    instruction: Optional[str] = Field(
        None, description="Optional task override for the expansion."
    )
    stream: bool = Field(
        False, description="Stream response using SSE."
    )
    retrieval_catalog: Optional[List[RetrievalCatalogEntryModel]] = Field(
        None,
        description="Top-k retrieved references from the initial response.",
    )
    result: Optional[RagResultModel] = Field(
        None, description="Original response payload containing retrieval_catalog."
    )


class RagExpandResultModel(BaseModel):
    language: Literal["en", "es"] = Field(
        ..., description="Language code for the response."
    )
    comparative_trajectory: str = Field(
        ..., description="Deep critical synthesis."
    )


class RagExpandResponse(BaseModel):
    result: RagExpandResultModel = Field(
        ..., description="Deep synthesis response."
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
                            "retrieval_catalog": [
                                {
                                    "rank": 1,
                                    "score": 0.123,
                                    "seminar_id": "Seminar_XI",
                                    "seminar_title": "Seminar XI",
                                    "lesson_label": "Lesson 1",
                                    "lesson_raw": "Lecon_1",
                                    "chunk_id": "chunk_001",
                                    "chunk_index": 1,
                                    "full_id": "Seminar_XI_chunk_001",
                                    "pages": [12, 13],
                                }
                            ],
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
    if request.stream:
        def event_stream():
            try:
                for event, data in run_rag_query_stream(
                    request.question,
                    request.language,
                    detail=request.detail,
                    top_k=request.top_k,
                    max_sources=request.max_sources,
                    source_id=request.source_id,
                ):
                    payload = json.dumps(data, ensure_ascii=False)
                    yield f"event: {event}\ndata: {payload}\n\n"
            except Exception as exc:
                LOGGER.exception("RAG query stream error")
                payload = json.dumps({"detail": str(exc)}, ensure_ascii=False)
                yield f"event: error\ndata: {payload}\n\n"
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    try:
        payload, warnings = run_rag_query(
            request.question,
            request.language,
            detail=request.detail,
            top_k=request.top_k,
            max_sources=request.max_sources,
            source_id=request.source_id,
        )
    except ValueError as exc:
        LOGGER.warning("RAG query failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("RAG query error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RagQueryResponse(result=payload, warnings=warnings)


@app.post(
    "/rag/expand",
    response_model=RagExpandResponse,
    summary="Expand a RAG response",
    description=(
        "Generates a deep critical synthesis using the retrieval_catalog "
        "from a previous RAG response."
    ),
    tags=["RAG"],
    responses={
        200: {"description": "Expanded response payload."},
        400: {"description": "Invalid request parameters."},
        500: {"description": "Internal server error."},
    },
)
def rag_expand(request: RagExpandRequest):
    if request.stream:
        def event_stream():
            try:
                retrieval_catalog = request.retrieval_catalog
                if retrieval_catalog is None and request.result:
                    retrieval_catalog = request.result.retrieval_catalog
                if not retrieval_catalog:
                    raise ValueError("retrieval_catalog is required.")
                language = request.language
                if not language and request.result:
                    language = request.result.language
                if not language:
                    raise ValueError("language is required.")
                for event, data in run_rag_expand_stream(
                    retrieval_catalog,
                    language,
                    question=request.question,
                    instruction=request.instruction,
                ):
                    payload = json.dumps(data, ensure_ascii=False)
                    yield f"event: {event}\ndata: {payload}\n\n"
            except Exception as exc:
                LOGGER.exception("RAG expand stream error")
                payload = json.dumps({"detail": str(exc)}, ensure_ascii=False)
                yield f"event: error\ndata: {payload}\n\n"
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    try:
        retrieval_catalog = request.retrieval_catalog
        if retrieval_catalog is None and request.result:
            retrieval_catalog = request.result.retrieval_catalog
        if not retrieval_catalog:
            raise ValueError("retrieval_catalog is required.")
        language = request.language
        if not language and request.result:
            language = request.result.language
        if not language:
            raise ValueError("language is required.")
        payload = run_rag_expand(
            retrieval_catalog,
            language,
            question=request.question,
            instruction=request.instruction,
        )
    except ValueError as exc:
        LOGGER.warning("RAG expand failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("RAG expand error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RagExpandResponse(result=payload)


@app.post(
    "/rag/index-text",
    response_model=IndexTextResponse,
    summary="Index a PDF",
    description=(
        "Uploads a PDF, chunks it with the PDF chunker, and appends "
        "embeddings to the existing FAISS index and metadata store."
    ),
    tags=["Indexing"],
    responses={
        200: {
            "description": "Indexation result.",
            "content": {
                "application/json": {
                    "example": {
                        "doc_id": "seminar_xi",
                        "chunk_count": 42,
                        "added_vectors": 42,
                        "chunks_path": "text_chunks_json/seminar_xi.json",
                        "index_path": "vector_index/lacan.index",
                        "metadata_path": "vector_index/metadata.json",
                    }
                }
            },
        },
        400: {"description": "Invalid PDF or empty file."},
        409: {"description": "doc_id already exists."},
        500: {"description": "Indexation error."},
    },
)
async def index_text(
    file: UploadFile = File(..., description="PDF file to index."),
    doc_id: Optional[str] = Form(None, description="Optional document id."),
):
    try:
        file_bytes = await file.read()
        payload = run_index_pdf(file_bytes, file.filename or "document.pdf", doc_id)
    except ValueError as exc:
        detail = str(exc)
        if "already exists" in detail:
            raise HTTPException(status_code=409, detail=detail) from exc
        raise HTTPException(status_code=400, detail=detail) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IndexTextResponse(**payload)
