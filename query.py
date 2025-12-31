"""Query entry point for Lacanian RAG answers."""

import json
import sys
from openai import OpenAIError
from config import (
    AppConfig,
    JSON_KEYS,
    LANGUAGE_NAMES,
    SOURCE_METADATA_KEYS,
    STRUCTURE_HINT,
    UI_TEXT,
    UNKNOWN_LECON,
)
from openai_client import OpenAIClient, load_openai_config
from rag_services import AuditLogger, PromptBuilder, ResponseProcessor
from retriever import search_similar_chunks

APP_CONFIG = AppConfig()
openai_client = OpenAIClient(load_openai_config())


def choose_language() -> str:
    """Prompt until a supported language code is selected."""
    while True:
        choice = input("Choose language (en/es): ").strip().lower()
        if choice in LANGUAGE_NAMES:
            return choice
        print("Please choose 'en' or 'es'.")


def query_lacan(question: str, language_code: str) -> None:
    """Run a full RAG query and print JSON output."""
    ui_text = UI_TEXT
    print(ui_text["searching"])
    language_name = LANGUAGE_NAMES[language_code]
    try:
        question_fr = openai_client.translate(
            question, language_name, "French")
    except OpenAIError as exc:
        print(ui_text["translation_error"].format(error=exc))
        question_fr = question

    chunks = search_similar_chunks(question_fr, top_k=APP_CONFIG.top_k)
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        key = (chunk.get("id"), chunk.get("seminar"))
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)
    selected_chunks = unique_chunks[:APP_CONFIG.max_sources]

    prompt_builder = PromptBuilder(
        chunks_dir=APP_CONFIG.chunks_dir,
        ui_text=ui_text,
        language_names=LANGUAGE_NAMES,
        structure_hint=STRUCTURE_HINT,
        json_keys=JSON_KEYS,
        unknown_lecon=UNKNOWN_LECON,
    )
    prompt, source_entries, context_text = prompt_builder.build(
        question, selected_chunks, language_code
    )
    audit_logger = AuditLogger(APP_CONFIG.audit_dir)
    audit_logger.save_context(
        question, question_fr, language_code, source_entries, context_text
    )

    if not prompt:
        print(f"\n{ui_text['context_error']}")
        return

    print(f"\n{ui_text['generating']}\n")
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant on Lacanian psychoanalysis.",
        },
        {"role": "user", "content": prompt},
    ]
    response = openai_client.chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=2600,
        response_format={"type": "json_object"},
    )
    response_text = response.choices[0].message.content.strip()

    response_processor = ResponseProcessor(
        language_names=LANGUAGE_NAMES,
        json_keys=JSON_KEYS,
        source_metadata_keys=SOURCE_METADATA_KEYS,
        translator=openai_client.translate,
    )
    parsed_response = response_processor.parse(response_text)
    if not parsed_response:
        print(f"\n{ui_text['validation_error']}", file=sys.stderr)
        print(response_text, file=sys.stderr)
        audit_logger.save_response(response_text, None)
        return

    normalized_response = response_processor.normalize(
        parsed_response, len(source_entries), language_code, context_text
    )
    if not normalized_response:
        print(f"\n{ui_text['validation_error']}", file=sys.stderr)
        print(response_text, file=sys.stderr)
        audit_logger.save_response(response_text, None)
        return

    response_processor.align_quotes_to_context(
        normalized_response, context_text)
    response_processor.ensure_translations(normalized_response, language_code)
    errors = response_processor.collect_validation_errors(
        normalized_response, len(source_entries), language_code, context_text
    )
    if errors:
        print(f"\n{ui_text['validation_error']}", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)

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

    audit_logger.save_response(response_text, output_payload)
    print(json.dumps(output_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    selected_language = choose_language()
    question_text = input(f"{UI_TEXT['ask_question']}: ")
    query_lacan(question_text, selected_language)
