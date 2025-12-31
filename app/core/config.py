"""Central configuration for the project."""

from dataclasses import dataclass
from pathlib import Path

LANGUAGE_NAMES = {"en": "English", "es": "Spanish"}
UNKNOWN_LECON = {"en": "Unknown lesson", "es": "Lección desconocida"}
UI_TEXT = {
    "ask_question": "Ask a question about Lacan",
    "searching": "Searching relevant fragments...",
    "generating": "Generating response with GPT...",
    "context_error": "[ERROR] Could not build the context. Check chunks or metadata.",
    "file_error": "[!] Error opening file {path}: {error}",
    "missing_chunk": "[!] No matching chunk found in {seminar} for id {chunk_id}",
    "translation_error": "[!] Translation failed, using the original question: {error}",
    "validation_error": "[ERROR] The model output did not follow the JSON schema.",
}
JSON_KEYS = {
    "language": "language",
    "sources": "sources",
    "source_index": "source_index",
    "french_quotes": "french_quotes",
    "translations": "translations",
    "translation_critique": "translation_critique",
    "context": "context",
    "lacanian_development": "lacanian_development",
    "comparative_trajectory": "comparative_trajectory",
    "source_metadata": "source_metadata",
    "sources_catalog": "sources_catalog",
}
SOURCE_METADATA_KEYS = {
    "seminar_title": "seminar_title",
    "seminar_id": "seminar_id",
    "lesson_label": "lesson_label",
    "lesson_raw": "lesson_raw",
    "chunk_id": "chunk_id",
    "chunk_index": "chunk_index",
    "pages": "pages",
}
STRUCTURE_HINT = (
    "Output must be valid JSON only, with no extra text.\n"
    "Schema:\n"
    "{\n"
    '  "language": "en|es",\n'
    '  "sources": [\n'
    "    {\n"
    '      "source_index": 1,\n'
    '      "french_quotes": ["..."],\n'
    '      "translations": ["..."],\n'
    '      "translation_critique": "...",\n'
    '      "context": "...",\n'
    '      "lacanian_development": "..."\n'
    "    }\n"
    "  ],\n"
    '  "comparative_trajectory": "..." \n'
    "}\n"
    "Use the exact keys shown above.\n"
    "Use \\n\\n inside strings to separate paragraphs when needed."
)


@dataclass(frozen=True)
class AppConfig:
    """Project-wide configuration values."""

    top_k: int = 5
    max_sources: int = 4
    chunks_dir: Path = Path("text_chunks_json")
    audit_dir: Path = Path("rag_audit")
    vector_index_path: Path = Path("vector_index/lacan.index")
    metadata_path: Path = Path("vector_index/metadata.json")
    input_pdf_dir: Path = Path("data_pdfs")
    output_dir: Path = Path("text_chunks_json")
    chunk_size: int = 1100
    overlap: int = 200
    embedding_dim: int = 3072
