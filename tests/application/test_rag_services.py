import json
from pathlib import Path
import types

from app.application import rag_services
from app.core.config import JSON_KEYS, LANGUAGE_NAMES, UI_TEXT, UNKNOWN_LECON


def _write_chunks(path: Path):
    payload = [
        {
            "seminar": "s1",
            "lecon": "Leçon 1",
            "chunk_id": "chunk_001",
            "chunk_index": 1,
            "text": "Bonjour le monde.",
            "pages": [1],
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_chunk_repository_get_matching_chunk(tmp_path):
    chunk_path = tmp_path / "s1.json"
    _write_chunks(chunk_path)
    repo = rag_services.ChunkRepository(tmp_path, UI_TEXT)
    assert repo.get_matching_chunk("s1", "chunk_001")["chunk_id"] == "chunk_001"
    assert repo.get_matching_chunk("s1", "s1_chunk_001")["chunk_id"] == "chunk_001"
    assert repo.get_matching_chunk("s1", "s1_1")["chunk_id"] == "chunk_001"


def test_chunk_repository_raw_id_fallback(tmp_path):
    payload = [
        {
            "seminar": "s1",
            "lecon": "Leçon 1",
            "chunk_id": "random",
            "chunk_index": 7,
            "text": "Bonjour.",
            "pages": [1],
        }
    ]
    (tmp_path / "s1.json").write_text(json.dumps(payload), encoding="utf-8")
    repo = rag_services.ChunkRepository(tmp_path, UI_TEXT)
    assert repo.get_matching_chunk("s1", "random")["chunk_id"] == "random"
    assert repo.get_matching_chunk("s1", "id_7")["chunk_index"] == 7


def test_chunk_repository_handles_missing_file(tmp_path, capsys):
    repo = rag_services.ChunkRepository(tmp_path, UI_TEXT)
    assert repo.get_matching_chunk("missing", "chunk_001") is None
    captured = capsys.readouterr()
    assert "Error" in captured.out


def test_label_resolver():
    resolver = rag_services.LabelResolver("en", UNKNOWN_LECON)
    assert resolver.get_seminar_title("S1_Ecrits_techniques")
    assert resolver.get_lesson_label(None).startswith("Unknown")


def test_prompt_builder_build(tmp_path):
    chunk_path = tmp_path / "s1.json"
    _write_chunks(chunk_path)
    builder = rag_services.PromptBuilder(
        chunks_dir=tmp_path,
        ui_text=UI_TEXT,
        language_names=LANGUAGE_NAMES,
        structure_hint="HINT",
        json_keys=JSON_KEYS,
        unknown_lecon=UNKNOWN_LECON,
    )
    prompt, sources, context_text = builder.build(
        "Question?",
        [{"seminar": "s1", "id": "s1_chunk_001"}],
        "en",
        detail_level="full",
    )
    assert "Question" in prompt
    assert "Bonjour" in context_text
    assert sources[0].seminar_id == "s1"


def test_prompt_builder_empty_chunks(tmp_path):
    builder = rag_services.PromptBuilder(
        chunks_dir=tmp_path,
        ui_text=UI_TEXT,
        language_names=LANGUAGE_NAMES,
        structure_hint="HINT",
        json_keys=JSON_KEYS,
        unknown_lecon=UNKNOWN_LECON,
    )
    prompt, sources, context_text = builder.build("Question?", [], "en")
    assert prompt is None
    assert sources == []
    assert context_text == ""


def test_prompt_builder_missing_seminar_and_chunk(tmp_path, capsys):
    chunk_path = tmp_path / "s1.json"
    _write_chunks(chunk_path)
    builder = rag_services.PromptBuilder(
        chunks_dir=tmp_path,
        ui_text=UI_TEXT,
        language_names=LANGUAGE_NAMES,
        structure_hint="HINT",
        json_keys=JSON_KEYS,
        unknown_lecon=UNKNOWN_LECON,
    )
    prompt, sources, context_text = builder.build(
        "Question?",
        [{"id": "s1_chunk_001"}, {"seminar": "missing", "id": "missing_chunk"}],
        "en",
    )
    assert prompt is None
    assert sources == []
    assert context_text == ""
    captured = capsys.readouterr()
    assert "missing" in captured.out


def test_prompt_builder_concise_detail(tmp_path):
    chunk_path = tmp_path / "s1.json"
    _write_chunks(chunk_path)
    builder = rag_services.PromptBuilder(
        chunks_dir=tmp_path,
        ui_text=UI_TEXT,
        language_names=LANGUAGE_NAMES,
        structure_hint="HINT",
        json_keys=JSON_KEYS,
        unknown_lecon=UNKNOWN_LECON,
    )
    prompt, sources, context_text = builder.build(
        "Question?",
        [{"seminar": "s1", "id": "s1_chunk_001"}],
        "en",
        detail_level="weird",
    )
    assert "Keep translation_critique" in prompt
    assert sources
    assert context_text


def test_response_processor_parse_and_normalize():
    processor = rag_services.ResponseProcessor(
        language_names=LANGUAGE_NAMES,
        json_keys=JSON_KEYS,
        source_metadata_keys={"chunk_id": "chunk_id"},
        translator=lambda text, *_args: f"t:{text}",
    )
    response_text = """```json
    {"language":"en","sources":[{"source_index":1,"french_quotes":["Bonjour"],"translations":["Hello"],"translation_critique":"ok","context":"ctx","lacanian_development":"dev"}],"comparative_trajectory":"traj"}
    ```"""
    parsed, error = processor.parse(response_text)
    assert error is None
    context_text = "[1] Bonjour"
    normalized = processor.normalize(parsed, 1, "en", context_text)
    assert normalized[JSON_KEYS["sources"]][0][JSON_KEYS["french_quotes"]]


def test_response_processor_parse_repairs_json():
    processor = rag_services.ResponseProcessor(
        language_names=LANGUAGE_NAMES,
        json_keys=JSON_KEYS,
        source_metadata_keys={"chunk_id": "chunk_id"},
        translator=lambda text, *_args: text,
    )
    response_text = """```json
    {“language”:“en”,“sources”:[],“comparative_trajectory”:“x”,}
    ```"""
    parsed, error = processor.parse(response_text)
    assert error is None
    assert parsed[JSON_KEYS["language"]] == "en"


def test_response_processor_normalize_defaults():
    processor = rag_services.ResponseProcessor(
        language_names=LANGUAGE_NAMES,
        json_keys=JSON_KEYS,
        source_metadata_keys={"chunk_id": "chunk_id"},
        translator=lambda text, *_args: text,
    )
    response_json = {
        JSON_KEYS["sources"]: "bad",
        JSON_KEYS["comparative_trajectory"]: "traj",
    }
    context_text = "[1] Bonjour. Salut."
    normalized = processor.normalize(response_json, 1, "en", context_text)
    source = normalized[JSON_KEYS["sources"]][0]
    assert source[JSON_KEYS["french_quotes"]]
    assert len(source[JSON_KEYS["translations"]]) == len(source[JSON_KEYS["french_quotes"]])


def test_response_processor_normalize_non_dict():
    processor = rag_services.ResponseProcessor(
        language_names=LANGUAGE_NAMES,
        json_keys=JSON_KEYS,
        source_metadata_keys={"chunk_id": "chunk_id"},
        translator=lambda text, *_args: text,
    )
    assert processor.normalize("bad", 1, "en", "") is None


def test_response_processor_ensure_translations():
    processor = rag_services.ResponseProcessor(
        language_names=LANGUAGE_NAMES,
        json_keys=JSON_KEYS,
        source_metadata_keys={"chunk_id": "chunk_id"},
        translator=lambda text, *_args: f"t:{text}",
    )
    response_json = {
        JSON_KEYS["sources"]: [
            {
                JSON_KEYS["french_quotes"]: ["Bonjour"],
                JSON_KEYS["translations"]: [""],
            }
        ]
    }
    processor.ensure_translations(response_json, "en")
    assert response_json[JSON_KEYS["sources"]][0][JSON_KEYS["translations"]] == ["t:Bonjour"]


def test_response_processor_alignment_and_validation():
    processor = rag_services.ResponseProcessor(
        language_names=LANGUAGE_NAMES,
        json_keys=JSON_KEYS,
        source_metadata_keys={"chunk_id": "chunk_id"},
        translator=lambda text, *_args: f"t:{text}",
    )
    response_json = {
        JSON_KEYS["language"]: "en",
        JSON_KEYS["sources"]: [
            {
                JSON_KEYS["source_index"]: 1,
                JSON_KEYS["french_quotes"]: ["Bon\"jour"],
                JSON_KEYS["translations"]: ["Hello"],
                JSON_KEYS["translation_critique"]: "crit",
                JSON_KEYS["context"]: "ctx",
                JSON_KEYS["lacanian_development"]: "dev",
            }
        ],
        JSON_KEYS["comparative_trajectory"]: "traj",
    }
    context_text = "[1] Bon\"jour"
    processor.align_quotes_to_context(response_json, context_text)
    processor.ensure_translations(response_json, "en")
    errors = processor.collect_validation_errors(response_json, 1, "en", context_text)
    assert errors == []


def test_response_processor_collect_validation_errors():
    processor = rag_services.ResponseProcessor(
        language_names=LANGUAGE_NAMES,
        json_keys=JSON_KEYS,
        source_metadata_keys={"chunk_id": "chunk_id"},
        translator=lambda text, *_args: text,
    )
    assert processor.collect_validation_errors("bad", 1, "en") == [
        "Response is not a JSON object."
    ]

    errors = processor.collect_validation_errors(
        {JSON_KEYS["language"]: "xx", JSON_KEYS["sources"]: "bad"}, 1, "en"
    )
    assert '"sources" must be an array.' in errors

    response_json = {
        JSON_KEYS["language"]: "en",
        JSON_KEYS["sources"]: [
            {
                JSON_KEYS["source_index"]: 1,
                JSON_KEYS["french_quotes"]: ["Bonjour"],
                JSON_KEYS["translations"]: ["Hello"],
                JSON_KEYS["translation_critique"]: "crit",
                JSON_KEYS["context"]: "ctx",
                JSON_KEYS["lacanian_development"]: "dev",
            },
            "bad",
        ],
        JSON_KEYS["comparative_trajectory"]: "traj",
    }
    errors = processor.collect_validation_errors(response_json, 2, "en")
    assert "Source 2 is not an object." in errors

    response_json = {
        JSON_KEYS["language"]: "es",
        JSON_KEYS["sources"]: [
            {
                JSON_KEYS["source_index"]: 2,
                JSON_KEYS["french_quotes"]: [""],
                JSON_KEYS["translations"]: [" "],
                JSON_KEYS["translation_critique"]: "the thing",
                JSON_KEYS["context"]: "",
                JSON_KEYS["lacanian_development"]: "",
            },
        ],
        JSON_KEYS["comparative_trajectory"]: "the path",
    }
    errors = processor.collect_validation_errors(response_json, 1, "es")
    assert "Source 1 has incorrect source_index." in errors
    assert "Source 1 has empty quote strings." in errors
    assert "Source 1 has empty translation strings." in errors
    assert "comparative_trajectory contains English." in errors

    response_json = {
        JSON_KEYS["language"]: "en",
        JSON_KEYS["sources"]: [
            {
                JSON_KEYS["source_index"]: 1,
                JSON_KEYS["french_quotes"]: ["Not in context"],
                JSON_KEYS["translations"]: ["T"],
                JSON_KEYS["translation_critique"]: "crit",
                JSON_KEYS["context"]: "ctx",
                JSON_KEYS["lacanian_development"]: "dev",
            },
            {
                JSON_KEYS["source_index"]: 2,
                JSON_KEYS["french_quotes"]: ["Bonjour"],
                JSON_KEYS["translations"]: ["Hello"],
                JSON_KEYS["translation_critique"]: "crit",
                JSON_KEYS["context"]: "ctx",
                JSON_KEYS["lacanian_development"]: "dev",
            },
        ],
        JSON_KEYS["comparative_trajectory"]: "traj",
    }
    errors = processor.collect_validation_errors(
        response_json, 2, "en", context_text="[1] Bonjour"
    )
    assert "includes a quote not in context" in " ".join(errors)


def test_response_processor_quote_validation_with_context():
    processor = rag_services.ResponseProcessor(
        language_names=LANGUAGE_NAMES,
        json_keys=JSON_KEYS,
        source_metadata_keys={"chunk_id": "chunk_id"},
        translator=lambda text, *_args: text,
    )
    response_json = {
        JSON_KEYS["language"]: "en",
        JSON_KEYS["sources"]: [
            {
                JSON_KEYS["source_index"]: 1,
                JSON_KEYS["french_quotes"]: ["Not in context"],
                JSON_KEYS["translations"]: ["T"],
                JSON_KEYS["translation_critique"]: "crit",
                JSON_KEYS["context"]: "ctx",
                JSON_KEYS["lacanian_development"]: "dev",
            },
            {
                JSON_KEYS["source_index"]: 2,
                JSON_KEYS["french_quotes"]: ["Bonjour"],
                JSON_KEYS["translations"]: ["Hello"],
                JSON_KEYS["translation_critique"]: "crit",
                JSON_KEYS["context"]: "ctx",
                JSON_KEYS["lacanian_development"]: "dev",
            },
        ],
        JSON_KEYS["comparative_trajectory"]: "traj",
    }
    errors = processor.collect_validation_errors(
        response_json, 2, "en", context_text="[1] Bonjour"
    )
    assert "includes a quote not in context" in " ".join(errors)


def test_response_processor_helpers():
    processor = rag_services.ResponseProcessor(
        language_names=LANGUAGE_NAMES,
        json_keys=JSON_KEYS,
        source_metadata_keys={"chunk_id": "chunk_id"},
        translator=lambda text, *_args: text,
    )
    assert processor._contains_english_tokens("the path") is True
    assert processor._contains_english_tokens(123) is False
    assert processor._normalize_char_for_alignment("’") == "'"
    assert processor._normalize_char_for_alignment("“") == '"'
    assert processor._normalize_char_for_alignment("–") == "-"
    assert processor._normalize_text_no_ws(123) == ("", [])
    assert processor._snap_quote_to_context("", "segment") == ""
    assert processor._snap_quote_to_context("abc", "def") == "abc"
    assert processor._extract_default_quotes("") == []
    assert processor._extract_default_quotes("One. Two.") == ["One.", "Two."]


def test_snap_quote_to_context_normalizes():
    processor = rag_services.ResponseProcessor(
        language_names=LANGUAGE_NAMES,
        json_keys=JSON_KEYS,
        source_metadata_keys={"chunk_id": "chunk_id"},
        translator=lambda text, *_args: text,
    )
    segment = "L’etre est la."
    quote = "L'etre est la."
    assert processor._snap_quote_to_context(quote, segment) == segment


def test_audit_logger_writes_files(tmp_path):
    logger = rag_services.AuditLogger(tmp_path)
    entry = rag_services.SourceEntry(
        source_index=1,
        seminar_id="s1",
        seminar_title="Seminar",
        lesson_label="Lesson",
        lesson_raw=None,
        chunk_id="chunk_001",
        chunk_index=1,
        full_id="s1_chunk_001",
        pages=[1],
    )
    logger.save_context("q", "qf", "en", [entry], "ctx", source_id="s1")
    logger.save_response("raw", {"ok": True})
    assert (tmp_path / "rag_context_last.json").exists()
    assert (tmp_path / "rag_response_last.json").exists()


def test_audit_logger_save_response_none(tmp_path):
    logger = rag_services.AuditLogger(tmp_path)
    logger.save_response("raw", None)
    assert (tmp_path / "rag_response_last.txt").exists()
