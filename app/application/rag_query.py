"""RAG query use case."""

import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from openai import OpenAIError
from ..core.config import (
    AppConfig,
    JSON_KEYS,
    LANGUAGE_NAMES,
    SOURCE_METADATA_KEYS,
    STRUCTURE_HINT,
    UI_TEXT,
    UNKNOWN_LECON,
)
from ..infrastructure.openai_client import OpenAIClient, load_openai_config
from .rag_services import (
    AuditLogger,
    ChunkRepository,
    LabelResolver,
    PromptBuilder,
    ResponseProcessor,
)
from ..infrastructure.retriever import search_similar_chunks

APP_CONFIG = AppConfig()
openai_client = OpenAIClient(load_openai_config())
LOGGER = logging.getLogger("text_extractor_api")


def _log_stage(name: str, start: float) -> None:
    if not LOGGER.isEnabledFor(logging.DEBUG):
        return
    duration_ms = (time.monotonic() - start) * 1000.0
    LOGGER.debug("RAG stage=%s duration_ms=%.2f", name, duration_ms)


def _prepare_rag_query(
    question: str,
    language_code: str,
    *,
    top_k: Optional[int],
    max_sources: Optional[int],
    source_id: Optional[str],
    detail: Optional[str],
    audit: bool,
) -> Dict[str, Any]:
    total_start = time.monotonic()
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    if language_code not in LANGUAGE_NAMES:
        raise ValueError("Unsupported language code.")

    warnings: Dict[str, Any] = {
        "translation_error": None,
        "validation_errors": [],
    }
    ui_text = UI_TEXT
    language_name = LANGUAGE_NAMES[language_code]
    translation_start = time.monotonic()
    try:
        question_fr = openai_client.translate(
            question, language_name, "French")
    except OpenAIError as exc:
        warnings["translation_error"] = ui_text["translation_error"].format(
            error=exc
        )
        question_fr = question
    finally:
        _log_stage("translation", translation_start)

    detail_level = (detail or "concise").lower().strip()
    if detail_level not in ("concise", "full"):
        detail_level = "concise"

    retrieval_start = time.monotonic()
    chunks = search_similar_chunks(
        question_fr,
        top_k=top_k or APP_CONFIG.top_k,
        source_id=source_id,
    )
    _log_stage("retrieval", retrieval_start)
    if source_id and not chunks:
        raise ValueError(
            f"No chunks found for source_id '{source_id}'. "
            "Check the id or increase top_k."
        )
    retrieval_catalog = _build_retrieval_catalog(
        chunks, language_code, ui_text)
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        key = (chunk.get("id"), chunk.get("seminar"))
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)
    selected_chunks = unique_chunks[: max_sources or APP_CONFIG.max_sources]

    prompt_builder = PromptBuilder(
        chunks_dir=APP_CONFIG.chunks_dir,
        ui_text=ui_text,
        language_names=LANGUAGE_NAMES,
        structure_hint=STRUCTURE_HINT,
        json_keys=JSON_KEYS,
        unknown_lecon=UNKNOWN_LECON,
    )
    prompt_start = time.monotonic()
    prompt, source_entries, context_text = prompt_builder.build(
        question, selected_chunks, language_code, detail_level
    )
    _log_stage("prompt_build", prompt_start)
    audit_logger = AuditLogger(APP_CONFIG.audit_dir) if audit else None
    if audit_logger:
        audit_logger.save_context(
            question,
            question_fr,
            language_code,
            source_entries,
            context_text,
            source_id=source_id,
        )

    if not prompt:
        raise ValueError(ui_text["context_error"])

    return {
        "total_start": total_start,
        "warnings": warnings,
        "prompt": prompt,
        "source_entries": source_entries,
        "context_text": context_text,
        "retrieval_catalog": retrieval_catalog,
        "detail_level": detail_level,
        "audit_logger": audit_logger,
    }


def _parse_chunk_id(raw_id: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    if not raw_id:
        return None, None
    match = re.search(r"chunk_(\d+)$", raw_id)
    if not match:
        return None, None
    chunk_index = int(match.group(1))
    return f"chunk_{match.group(1)}", chunk_index


def _build_retrieval_catalog(
    chunks: List[Dict[str, Any]],
    language_code: str,
    ui_text: Dict[str, str],
) -> List[Dict[str, Any]]:
    repo = ChunkRepository(APP_CONFIG.chunks_dir, ui_text)
    label_resolver = LabelResolver(language_code, UNKNOWN_LECON)
    catalog = []
    for rank, chunk in enumerate(chunks, start=1):
        seminar_id = chunk.get("seminar")
        full_id = chunk.get("id")
        lesson_raw = chunk.get("lecon")
        pages = chunk.get("pages") or []
        if isinstance(pages, list):
            pages = sorted(set(pages))
        chunk_id, chunk_index = _parse_chunk_id(full_id)
        if seminar_id and full_id:
            matching_chunk = repo.get_matching_chunk(seminar_id, full_id)
            if matching_chunk:
                chunk_id = matching_chunk.get("chunk_id") or chunk_id
                if chunk_index is None:
                    chunk_index = matching_chunk.get("chunk_index")
                lesson_raw = matching_chunk.get("lecon") or lesson_raw
                matching_pages = matching_chunk.get("pages")
                if matching_pages:
                    pages = sorted(set(matching_pages))
        catalog.append(
            {
                "rank": rank,
                "score": chunk.get("score"),
                "seminar_id": seminar_id,
                "seminar_title": (
                    label_resolver.get_seminar_title(seminar_id)
                    if seminar_id
                    else ""
                ),
                "lesson_label": label_resolver.get_lesson_label(lesson_raw),
                "lesson_raw": lesson_raw,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "full_id": full_id,
                "pages": pages,
            }
        )
    return catalog


def _build_context_from_catalog(
    retrieval_catalog: List[Dict[str, Any]],
    ui_text: Dict[str, str],
) -> Tuple[str, int]:
    repo = ChunkRepository(APP_CONFIG.chunks_dir, ui_text)
    context_parts = []
    seen = set()
    source_index = 1
    for entry in retrieval_catalog:
        seminar_id = entry.get("seminar_id") or entry.get("seminar")
        raw_id = entry.get("full_id") or entry.get("chunk_id")
        if not seminar_id or not raw_id:
            continue
        key = (seminar_id, raw_id)
        if key in seen:
            continue
        seen.add(key)
        matching_chunk = repo.get_matching_chunk(seminar_id, raw_id)
        if not matching_chunk:
            continue
        context_parts.append(f"[{source_index}] {matching_chunk['text']}")
        source_index += 1
    return "\n\n".join(context_parts), source_index - 1


def _normalize_catalog_entries(
    retrieval_catalog: List[Any],
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for entry in retrieval_catalog:
        if isinstance(entry, dict):
            normalized.append(entry)
            continue
        if hasattr(entry, "model_dump"):
            normalized.append(entry.model_dump())
            continue
        if hasattr(entry, "dict"):
            normalized.append(entry.dict())
            continue
        if hasattr(entry, "__dict__"):
            normalized.append(dict(entry.__dict__))
            continue
    return normalized


def run_rag_query(
    question: str,
    language_code: str,
    *,
    top_k: Optional[int] = None,
    max_sources: Optional[int] = None,
    source_id: Optional[str] = None,
    detail: Optional[str] = None,
    audit: bool = True,
    on_generate: Optional[Callable[[], None]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run a full RAG query and return the JSON payload plus warnings."""
    prepared = _prepare_rag_query(
        question,
        language_code,
        top_k=top_k,
        max_sources=max_sources,
        source_id=source_id,
        detail=detail,
        audit=audit,
    )
    total_start = prepared["total_start"]
    warnings = prepared["warnings"]
    prompt = prepared["prompt"]
    source_entries = prepared["source_entries"]
    context_text = prepared["context_text"]
    retrieval_catalog = prepared["retrieval_catalog"]
    detail_level = prepared["detail_level"]
    audit_logger = prepared["audit_logger"]
    ui_text = UI_TEXT

    if on_generate:
        on_generate()
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant on Lacanian psychoanalysis.",
        },
        {"role": "user", "content": prompt},
    ]
    llm_start = time.monotonic()
    response = openai_client.chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=1200 if detail_level == "concise" else 2600,
        response_format={"type": "json_object"},
    )
    _log_stage("llm", llm_start)
    response_text = response.choices[0].message.content.strip()

    response_processor = ResponseProcessor(
        language_names=LANGUAGE_NAMES,
        json_keys=JSON_KEYS,
        source_metadata_keys=SOURCE_METADATA_KEYS,
        translator=openai_client.translate,
    )
    post_start = time.monotonic()
    parsed_response, parse_error = response_processor.parse(response_text)
    if not parsed_response:
        if audit_logger:
            audit_logger.save_response(response_text, None)
        detail = ui_text["validation_error"]
        if parse_error:
            detail = f"{detail} Parse error: {parse_error}"
        detail = f"{detail} See rag_audit/rag_response_last.txt."
        raise ValueError(detail)

    normalized_response = response_processor.normalize(
        parsed_response, len(source_entries), language_code, context_text
    )
    if not normalized_response:
        if audit_logger:
            audit_logger.save_response(response_text, None)
        raise ValueError(
            f"{ui_text['validation_error']} See rag_audit/rag_response_last.txt."
        )

    response_processor.align_quotes_to_context(
        normalized_response, context_text)
    response_processor.ensure_translations(normalized_response, language_code)
    errors = response_processor.collect_validation_errors(
        normalized_response, len(source_entries), language_code, context_text
    )
    if errors:
        warnings["validation_errors"].extend(errors)
    _log_stage("postprocess", post_start)

    enriched_sources = []
    for idx, source in enumerate(normalized_response[JSON_KEYS["sources"]]):
        meta = source_entries[idx]
        metadata = {
            SOURCE_METADATA_KEYS["seminar_title"]: meta.seminar_title,
            SOURCE_METADATA_KEYS["seminar_id"]: meta.seminar_id,
            SOURCE_METADATA_KEYS["lesson_label"]: meta.lesson_label,
            SOURCE_METADATA_KEYS["lesson_raw"]: meta.lesson_raw,
            SOURCE_METADATA_KEYS["chunk_id"]: meta.chunk_id,
            SOURCE_METADATA_KEYS["chunk_index"]: meta.chunk_index,
            SOURCE_METADATA_KEYS["pages"]: meta.pages,
        }
        enriched = dict(source)
        enriched[JSON_KEYS["source_metadata"]] = metadata
        enriched_sources.append(enriched)

    output_payload = dict(normalized_response)
    output_payload[JSON_KEYS["language"]] = language_code
    output_payload[JSON_KEYS["sources"]] = enriched_sources
    output_payload[JSON_KEYS["sources_catalog"]] = response_processor.build_sources_catalog(
        source_entries
    )
    output_payload["retrieval_catalog"] = retrieval_catalog

    if audit_logger:
        audit_logger.save_response(response_text, output_payload)
    _log_stage("total", total_start)
    return output_payload, warnings


def run_rag_expand(
    retrieval_catalog: List[Dict[str, Any]],
    language_code: str,
    *,
    question: Optional[str] = None,
    instruction: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a deep critical synthesis from a retrieval catalog."""
    total_start = time.monotonic()
    if language_code not in LANGUAGE_NAMES:
        raise ValueError("Unsupported language code.")
    if not retrieval_catalog:
        raise ValueError("retrieval_catalog cannot be empty.")
    normalized_catalog = _normalize_catalog_entries(retrieval_catalog)
    if not normalized_catalog:
        raise ValueError("retrieval_catalog cannot be empty.")

    ui_text = UI_TEXT
    context_start = time.monotonic()
    context_text, total_sources = _build_context_from_catalog(
        normalized_catalog, ui_text)
    _log_stage("expand_context", context_start)
    if not context_text:
        raise ValueError(ui_text["context_error"])

    language_name = LANGUAGE_NAMES[language_code]
    task = instruction or question or "Provide a deep critical synthesis."
    prompt = f"""
You are a helpful assistant specialized in Jacques Lacan's work. Use only the context below to answer the task. Do not make up information.

Context:
{context_text}

Task:
{task}

Instructions:
- Respond only in {language_name} for all string values
- Use every source from [1] to [{total_sources}] in order; do not skip any source
- Cite claims with numbered references like [1], [2], etc. inside the text
- Write "comparative_trajectory" as a long critical synthesis (not a summary): 2-3 paragraphs, at least 8 sentences total and at least 900 characters; use a Lacanian voice; highlight tensions, shifts, or stakes across sources; explain what changes in the conceptual position; avoid adding facts not in the context and mark speculation explicitly
- If "language" is "es", avoid English words in string values (except [n] citations)
- Output JSON only with exactly these keys:
  {{
    "language": "{language_code}",
    "comparative_trajectory": "..."
  }}

Begin your answer below:
"""
    llm_start = time.monotonic()
    response = openai_client.chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant on Lacanian psychoanalysis.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=1800,
        response_format={"type": "json_object"},
    )
    _log_stage("expand_llm", llm_start)
    response_text = response.choices[0].message.content.strip()

    response_processor = ResponseProcessor(
        language_names=LANGUAGE_NAMES,
        json_keys=JSON_KEYS,
        source_metadata_keys=SOURCE_METADATA_KEYS,
        translator=openai_client.translate,
    )
    parsed_response, parse_error = response_processor.parse(response_text)
    if not parsed_response:
        detail = ui_text["validation_error"]
        if parse_error:
            detail = f"{detail} Parse error: {parse_error}"
        raise ValueError(detail)

    trajectory = parsed_response.get(JSON_KEYS["comparative_trajectory"])
    if not isinstance(trajectory, str) or not trajectory.strip():
        raise ValueError(ui_text["validation_error"])

    output_payload = {
        JSON_KEYS["language"]: language_code,
        JSON_KEYS["comparative_trajectory"]: trajectory.strip(),
    }
    _log_stage("expand_total", total_start)
    return output_payload


def run_rag_query_stream(
    question: str,
    language_code: str,
    *,
    top_k: Optional[int] = None,
    max_sources: Optional[int] = None,
    source_id: Optional[str] = None,
    detail: Optional[str] = None,
    audit: bool = True,
    on_generate: Optional[Callable[[], None]] = None,
):
    """Stream a RAG query as events: meta, delta, done."""
    prepared = _prepare_rag_query(
        question,
        language_code,
        top_k=top_k,
        max_sources=max_sources,
        source_id=source_id,
        detail=detail,
        audit=audit,
    )
    total_start = prepared["total_start"]
    warnings = prepared["warnings"]
    prompt = prepared["prompt"]
    source_entries = prepared["source_entries"]
    context_text = prepared["context_text"]
    retrieval_catalog = prepared["retrieval_catalog"]
    detail_level = prepared["detail_level"]
    audit_logger = prepared["audit_logger"]

    yield "meta", {
        "language": language_code,
        "detail": detail_level,
        "retrieval_catalog": retrieval_catalog,
    }

    if on_generate:
        on_generate()
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant on Lacanian psychoanalysis.",
        },
        {"role": "user", "content": prompt},
    ]
    llm_start = time.monotonic()
    response_text_parts: List[str] = []
    for delta in openai_client.chat_completion_stream(
        messages=messages,
        temperature=0.7,
        max_tokens=1200 if detail_level == "concise" else 2600,
        response_format={"type": "json_object"},
    ):
        response_text_parts.append(delta)
        yield "delta", {"text": delta}
    _log_stage("llm", llm_start)
    response_text = "".join(response_text_parts).strip()

    response_processor = ResponseProcessor(
        language_names=LANGUAGE_NAMES,
        json_keys=JSON_KEYS,
        source_metadata_keys=SOURCE_METADATA_KEYS,
        translator=openai_client.translate,
    )
    post_start = time.monotonic()
    parsed_response, parse_error = response_processor.parse(response_text)
    if not parsed_response:
        if audit_logger:
            audit_logger.save_response(response_text, None)
        detail_msg = UI_TEXT["validation_error"]
        if parse_error:
            detail_msg = f"{detail_msg} Parse error: {parse_error}"
        raise ValueError(detail_msg)

    normalized_response = response_processor.normalize(
        parsed_response, len(source_entries), language_code, context_text
    )
    if not normalized_response:
        if audit_logger:
            audit_logger.save_response(response_text, None)
        raise ValueError(UI_TEXT["validation_error"])

    response_processor.align_quotes_to_context(
        normalized_response, context_text)
    response_processor.ensure_translations(normalized_response, language_code)
    errors = response_processor.collect_validation_errors(
        normalized_response, len(source_entries), language_code, context_text
    )
    if errors:
        warnings["validation_errors"].extend(errors)
    _log_stage("postprocess", post_start)

    enriched_sources = []
    for idx, source in enumerate(normalized_response[JSON_KEYS["sources"]]):
        meta = source_entries[idx]
        metadata = {
            SOURCE_METADATA_KEYS["seminar_title"]: meta.seminar_title,
            SOURCE_METADATA_KEYS["seminar_id"]: meta.seminar_id,
            SOURCE_METADATA_KEYS["lesson_label"]: meta.lesson_label,
            SOURCE_METADATA_KEYS["lesson_raw"]: meta.lesson_raw,
            SOURCE_METADATA_KEYS["chunk_id"]: meta.chunk_id,
            SOURCE_METADATA_KEYS["chunk_index"]: meta.chunk_index,
            SOURCE_METADATA_KEYS["pages"]: meta.pages,
        }
        enriched = dict(source)
        enriched[JSON_KEYS["source_metadata"]] = metadata
        enriched_sources.append(enriched)

    output_payload = dict(normalized_response)
    output_payload[JSON_KEYS["language"]] = language_code
    output_payload[JSON_KEYS["sources"]] = enriched_sources
    output_payload[JSON_KEYS["sources_catalog"]] = response_processor.build_sources_catalog(
        source_entries
    )
    output_payload["retrieval_catalog"] = retrieval_catalog

    if audit_logger:
        audit_logger.save_response(response_text, output_payload)
    _log_stage("total", total_start)
    yield "done", {"result": output_payload, "warnings": warnings}


def run_rag_expand_stream(
    retrieval_catalog: List[Dict[str, Any]],
    language_code: str,
    *,
    question: Optional[str] = None,
    instruction: Optional[str] = None,
):
    """Stream a deep critical synthesis as events."""
    total_start = time.monotonic()
    if language_code not in LANGUAGE_NAMES:
        raise ValueError("Unsupported language code.")
    if not retrieval_catalog:
        raise ValueError("retrieval_catalog cannot be empty.")
    normalized_catalog = _normalize_catalog_entries(retrieval_catalog)
    if not normalized_catalog:
        raise ValueError("retrieval_catalog cannot be empty.")

    ui_text = UI_TEXT
    context_start = time.monotonic()
    context_text, total_sources = _build_context_from_catalog(
        normalized_catalog, ui_text)
    _log_stage("expand_context", context_start)
    if not context_text:
        raise ValueError(ui_text["context_error"])

    language_name = LANGUAGE_NAMES[language_code]
    task = instruction or question or "Provide a deep critical synthesis."
    prompt = f"""
You are a helpful assistant specialized in Jacques Lacan's work. Use only the context below to answer the task. Do not make up information.

Context:
{context_text}

Task:
{task}

Instructions:
- Respond only in {language_name} for all string values
- Use every source from [1] to [{total_sources}] in order; do not skip any source
- Cite claims with numbered references like [1], [2], etc. inside the text
- Write "comparative_trajectory" as a long critical synthesis (not a summary): 2-3 paragraphs, at least 8 sentences total and at least 900 characters; use a Lacanian voice; highlight tensions, shifts, or stakes across sources; explain what changes in the conceptual position; avoid adding facts not in the context and mark speculation explicitly
- If "language" is "es", avoid English words in string values (except [n] citations)
- Output JSON only with exactly these keys:
  {{
    "language": "{language_code}",
    "comparative_trajectory": "..."
  }}

Begin your answer below:
"""
    yield "meta", {"language": language_code, "sources": total_sources}

    llm_start = time.monotonic()
    response_text_parts: List[str] = []
    for delta in openai_client.chat_completion_stream(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant on Lacanian psychoanalysis.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=2000,
        response_format={"type": "json_object"},
    ):
        response_text_parts.append(delta)
        yield "delta", {"text": delta}
    _log_stage("expand_llm", llm_start)
    response_text = "".join(response_text_parts).strip()

    response_processor = ResponseProcessor(
        language_names=LANGUAGE_NAMES,
        json_keys=JSON_KEYS,
        source_metadata_keys=SOURCE_METADATA_KEYS,
        translator=openai_client.translate,
    )
    parsed_response, parse_error = response_processor.parse(response_text)
    if not parsed_response:
        detail_msg = ui_text["validation_error"]
        if parse_error:
            detail_msg = f"{detail_msg} Parse error: {parse_error}"
        raise ValueError(detail_msg)

    trajectory = parsed_response.get(JSON_KEYS["comparative_trajectory"])
    if not isinstance(trajectory, str) or not trajectory.strip():
        raise ValueError(ui_text["validation_error"])

    output_payload = {
        JSON_KEYS["language"]: language_code,
        JSON_KEYS["comparative_trajectory"]: trajectory.strip(),
    }
    _log_stage("expand_total", total_start)
    yield "done", {"result": output_payload}
