import types

import pytest

from app.application import rag_query
from app.application.rag_services import AuditLogger, SourceEntry
from app.core.config import AppConfig, JSON_KEYS, UI_TEXT


class DummyClient:
    def __init__(self):
        self.translate_calls = []
        self.chat_calls = []
        self.stream_parts = []

    def translate(self, text, *_args, **_kwargs):
        self.translate_calls.append(text)
        return f"fr:{text}"

    def chat_completion(self, *args, **kwargs):
        self.chat_calls.append((args, kwargs))
        content = kwargs.pop("_content", None) or "{}"
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
        )

    def chat_completion_stream(self, *args, **kwargs):
        _ = args, kwargs
        for part in self.stream_parts:
            yield part


def test_parse_chunk_id():
    assert rag_query._parse_chunk_id("abc") == (None, None)
    assert rag_query._parse_chunk_id("chunk_001") == ("chunk_001", 1)


def test_normalize_catalog_entries():
    class Obj:
        def __init__(self):
            self.a = 1

    class DictLike:
        def dict(self):
            return {"b": 2}

    class DumpLike:
        def model_dump(self):
            return {"c": 3}

    normalized = rag_query._normalize_catalog_entries([Obj(), DictLike(), DumpLike(), {"d": 4}])
    assert {"a": 1} in normalized
    assert {"b": 2} in normalized
    assert {"c": 3} in normalized
    assert {"d": 4} in normalized


def test_build_context_from_catalog(tmp_path, monkeypatch):
    chunk_payload = [
        {
            "seminar": "s1",
            "lecon": "Leçon 1",
            "chunk_id": "chunk_001",
            "chunk_index": 1,
            "text": "Bonjour.",
            "pages": [1],
        }
    ]
    (tmp_path / "s1.json").write_text(
        __import__("json").dumps(chunk_payload), encoding="utf-8"
    )
    monkeypatch.setattr(rag_query, "APP_CONFIG", rag_query.AppConfig(chunks_dir=tmp_path))
    catalog = [
        {"seminar_id": "s1", "full_id": "s1_chunk_001"},
        {"seminar_id": "s1", "full_id": "s1_chunk_001"},
    ]
    context, count = rag_query._build_context_from_catalog(catalog, UI_TEXT)
    assert "Bonjour" in context
    assert count == 1


def test_prepare_rag_query_handles_translation_error(monkeypatch):
    dummy = DummyClient()

    class DummyError(Exception):
        pass

    def explode(*_args, **_kwargs):
        raise DummyError("boom")

    monkeypatch.setattr(rag_query, "openai_client", dummy)
    monkeypatch.setattr(rag_query, "OpenAIError", DummyError)
    monkeypatch.setattr(rag_query, "search_similar_chunks", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(rag_query, "_build_retrieval_catalog", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(rag_query.PromptBuilder, "build", lambda *_args, **_kwargs: ("prompt", [], "ctx"))
    dummy.translate = explode

    result = rag_query._prepare_rag_query(
        "Question",
        "en",
        top_k=None,
        max_sources=None,
        source_id=None,
        detail="unknown",
        audit=False,
    )
    assert result["warnings"]["translation_error"]
    assert result["detail_level"] == "concise"


def test_prepare_rag_query_invalid_inputs():
    with pytest.raises(ValueError):
        rag_query._prepare_rag_query(
            "",
            "en",
            top_k=None,
            max_sources=None,
            source_id=None,
            detail=None,
            audit=False,
        )
    with pytest.raises(ValueError):
        rag_query._prepare_rag_query(
            "Question",
            "xx",
            top_k=None,
            max_sources=None,
            source_id=None,
            detail=None,
            audit=False,
        )


def test_prepare_rag_query_source_id_missing(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(rag_query, "openai_client", dummy)
    monkeypatch.setattr(rag_query, "search_similar_chunks", lambda *_args, **_kwargs: [])
    with pytest.raises(ValueError):
        rag_query._prepare_rag_query(
            "Question",
            "en",
            top_k=None,
            max_sources=None,
            source_id="s1",
            detail=None,
            audit=False,
        )


def test_prepare_rag_query_prompt_missing(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(rag_query, "openai_client", dummy)
    monkeypatch.setattr(
        rag_query, "search_similar_chunks", lambda *_args, **_kwargs: [{"id": "s1_chunk_001", "seminar": "s1"}]
    )
    monkeypatch.setattr(rag_query, "_build_retrieval_catalog", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(rag_query.PromptBuilder, "build", lambda *_args, **_kwargs: (None, [], ""))
    with pytest.raises(ValueError):
        rag_query._prepare_rag_query(
            "Question",
            "en",
            top_k=None,
            max_sources=None,
            source_id=None,
            detail=None,
            audit=False,
        )


def test_prepare_rag_query_audit_and_dedupe(tmp_path, monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(rag_query, "openai_client", dummy)
    app_config = AppConfig(
        chunks_dir=tmp_path,
        audit_dir=tmp_path / "audit",
    )
    monkeypatch.setattr(rag_query, "APP_CONFIG", app_config)
    chunk_payload = [
        {
            "seminar": "s1",
            "lecon": "Leçon 1",
            "chunk_id": "chunk_001",
            "chunk_index": 1,
            "text": "Bonjour.",
            "pages": [1],
        }
    ]
    (tmp_path / "s1.json").write_text(
        __import__("json").dumps(chunk_payload), encoding="utf-8"
    )
    monkeypatch.setattr(
        rag_query,
        "search_similar_chunks",
        lambda *_args, **_kwargs: [
            {"id": "s1_chunk_001", "seminar": "s1", "lecon": "Leçon 1", "pages": [1]},
            {"id": "s1_chunk_001", "seminar": "s1", "lecon": "Leçon 1", "pages": [1]},
        ],
    )
    result = rag_query._prepare_rag_query(
        "Question",
        "en",
        top_k=None,
        max_sources=None,
        source_id=None,
        detail=None,
        audit=True,
    )
    assert result["prompt"]
    assert (app_config.audit_dir / "rag_context_last.json").exists()


def test_build_retrieval_catalog(tmp_path, monkeypatch):
    app_config = AppConfig(chunks_dir=tmp_path)
    monkeypatch.setattr(rag_query, "APP_CONFIG", app_config)
    chunk_payload = [
        {
            "seminar": "s1",
            "lecon": "Leçon 1",
            "chunk_id": "chunk_001",
            "chunk_index": 1,
            "text": "Bonjour.",
            "pages": [1, 2],
        }
    ]
    (tmp_path / "s1.json").write_text(
        __import__("json").dumps(chunk_payload), encoding="utf-8"
    )
    catalog = rag_query._build_retrieval_catalog(
        [{"id": "s1_chunk_001", "seminar": "s1", "lecon": "Leçon 1", "pages": [2, 2, 1]}],
        "en",
        UI_TEXT,
    )
    assert catalog[0]["chunk_id"] == "chunk_001"
    assert catalog[0]["pages"] == [1, 2]


def test_build_context_from_catalog_skips_missing(tmp_path, monkeypatch):
    app_config = AppConfig(chunks_dir=tmp_path)
    monkeypatch.setattr(rag_query, "APP_CONFIG", app_config)
    (tmp_path / "s1.json").write_text(
        __import__("json").dumps(
            [
                {
                    "seminar": "s1",
                    "lecon": "Leçon 1",
                    "chunk_id": "chunk_001",
                    "chunk_index": 1,
                    "text": "Bonjour.",
                    "pages": [1],
                }
            ]
        ),
        encoding="utf-8",
    )
    catalog = [
        {"seminar_id": None, "full_id": "s1_chunk_001"},
        {"seminar_id": "s1", "full_id": None},
        {"seminar_id": "s1", "full_id": "s1_chunk_001"},
    ]
    context, count = rag_query._build_context_from_catalog(catalog, UI_TEXT)
    assert count == 1
    assert "Bonjour" in context


def test_run_rag_query_success(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(rag_query, "openai_client", dummy)

    source_entries = [
        SourceEntry(
            source_index=1,
            seminar_id="s1",
            seminar_title="Seminar",
            lesson_label="Lesson",
            lesson_raw="Leçon 1",
            chunk_id="chunk_001",
            chunk_index=1,
            full_id="s1_chunk_001",
            pages=[1],
        )
    ]
    prepared = {
        "total_start": 0.0,
        "warnings": {"translation_error": None, "validation_errors": []},
        "prompt": "prompt",
        "source_entries": source_entries,
        "context_text": "[1] Bonjour",
        "retrieval_catalog": [],
        "detail_level": "concise",
        "audit_logger": None,
    }
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
            }
        ],
        JSON_KEYS["comparative_trajectory"]: "traj",
    }
    dummy.chat_completion = lambda *args, **kwargs: types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(message=types.SimpleNamespace(content=__import__("json").dumps(response_json)))
        ]
    )
    monkeypatch.setattr(rag_query, "_prepare_rag_query", lambda *_args, **_kwargs: prepared)

    payload, warnings = rag_query.run_rag_query("Q", "en", audit=False)
    assert payload[JSON_KEYS["language"]] == "en"
    assert warnings["translation_error"] is None


def test_run_rag_query_parse_error(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(rag_query, "openai_client", dummy)
    monkeypatch.setattr(rag_query, "_prepare_rag_query", lambda *_args, **_kwargs: {
        "total_start": 0.0,
        "warnings": {"translation_error": None, "validation_errors": []},
        "prompt": "prompt",
        "source_entries": [],
        "context_text": "",
        "retrieval_catalog": [],
        "detail_level": "concise",
        "audit_logger": None,
    })
    dummy.chat_completion = lambda *args, **kwargs: types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="not json"))]
    )
    with pytest.raises(ValueError):
        rag_query.run_rag_query("Q", "en", audit=False)


def test_run_rag_query_parse_error_with_audit(tmp_path, monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(rag_query, "openai_client", dummy)
    audit_logger = AuditLogger(tmp_path)
    prepared = {
        "total_start": 0.0,
        "warnings": {"translation_error": None, "validation_errors": []},
        "prompt": "prompt",
        "source_entries": [],
        "context_text": "",
        "retrieval_catalog": [],
        "detail_level": "concise",
        "audit_logger": audit_logger,
    }
    monkeypatch.setattr(rag_query, "_prepare_rag_query", lambda *_args, **_kwargs: prepared)
    dummy.chat_completion = lambda *args, **kwargs: types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="not json"))]
    )
    with pytest.raises(ValueError):
        rag_query.run_rag_query("Q", "en", audit=False)
    assert (tmp_path / "rag_response_last.txt").exists()


def test_run_rag_query_normalize_error(tmp_path, monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(rag_query, "openai_client", dummy)
    audit_logger = AuditLogger(tmp_path)
    prepared = {
        "total_start": 0.0,
        "warnings": {"translation_error": None, "validation_errors": []},
        "prompt": "prompt",
        "source_entries": [],
        "context_text": "",
        "retrieval_catalog": [],
        "detail_level": "concise",
        "audit_logger": audit_logger,
    }
    monkeypatch.setattr(rag_query, "_prepare_rag_query", lambda *_args, **_kwargs: prepared)
    response_json = {JSON_KEYS["language"]: "en", JSON_KEYS["sources"]: []}
    dummy.chat_completion = lambda *args, **kwargs: types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(message=types.SimpleNamespace(content=__import__("json").dumps(response_json)))
        ]
    )
    monkeypatch.setattr(rag_query.ResponseProcessor, "normalize", lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError):
        rag_query.run_rag_query("Q", "en", audit=False)
    assert (tmp_path / "rag_response_last.txt").exists()


def test_run_rag_query_collects_warnings(tmp_path, monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(rag_query, "openai_client", dummy)
    audit_logger = AuditLogger(tmp_path)
    source_entries = [
        SourceEntry(
            source_index=1,
            seminar_id="s1",
            seminar_title="Seminar",
            lesson_label="Lesson",
            lesson_raw="Leçon 1",
            chunk_id="chunk_001",
            chunk_index=1,
            full_id="s1_chunk_001",
            pages=[1],
        )
    ]
    prepared = {
        "total_start": 0.0,
        "warnings": {"translation_error": None, "validation_errors": []},
        "prompt": "prompt",
        "source_entries": source_entries,
        "context_text": "[1] Bonjour",
        "retrieval_catalog": [],
        "detail_level": "concise",
        "audit_logger": audit_logger,
    }
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
            }
        ],
        JSON_KEYS["comparative_trajectory"]: "traj",
    }
    dummy.chat_completion = lambda *args, **kwargs: types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(message=types.SimpleNamespace(content=__import__("json").dumps(response_json)))
        ]
    )
    monkeypatch.setattr(rag_query, "_prepare_rag_query", lambda *_args, **_kwargs: prepared)

    called = {"count": 0}

    def on_generate():
        called["count"] += 1

    monkeypatch.setattr(
        rag_query.ResponseProcessor,
        "collect_validation_errors",
        lambda *_args, **_kwargs: ["warn"],
    )
    payload, warnings = rag_query.run_rag_query("Q", "en", audit=False, on_generate=on_generate)
    assert payload[JSON_KEYS["language"]] == "en"
    assert warnings["validation_errors"] == ["warn"]
    assert called["count"] == 1
    assert (tmp_path / "rag_response_last.json").exists()


def test_run_rag_expand_success(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(rag_query, "openai_client", dummy)
    monkeypatch.setattr(rag_query, "_build_context_from_catalog", lambda *_args, **_kwargs: ("[1] ctx", 1))
    response_json = {
        JSON_KEYS["language"]: "en",
        JSON_KEYS["comparative_trajectory"]: "traj",
    }
    dummy.chat_completion = lambda *args, **kwargs: types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(message=types.SimpleNamespace(content=__import__("json").dumps(response_json)))
        ]
    )
    output = rag_query.run_rag_expand([{"seminar_id": "s1", "full_id": "s1_chunk_001"}], "en")
    assert output[JSON_KEYS["comparative_trajectory"]] == "traj"


def test_run_rag_expand_errors(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(rag_query, "openai_client", dummy)
    with pytest.raises(ValueError):
        rag_query.run_rag_expand([], "en")
    with pytest.raises(ValueError):
        rag_query.run_rag_expand([{"seminar_id": "s1"}], "xx")

    monkeypatch.setattr(rag_query, "_build_context_from_catalog", lambda *_args, **_kwargs: ("", 0))
    with pytest.raises(ValueError):
        rag_query.run_rag_expand([{"seminar_id": "s1", "full_id": "s1_chunk_001"}], "en")


def test_run_rag_expand_parse_error(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(rag_query, "openai_client", dummy)
    monkeypatch.setattr(rag_query, "_build_context_from_catalog", lambda *_args, **_kwargs: ("[1] ctx", 1))
    dummy.chat_completion = lambda *args, **kwargs: types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="bad json"))]
    )
    with pytest.raises(ValueError):
        rag_query.run_rag_expand([{"seminar_id": "s1", "full_id": "s1_chunk_001"}], "en")


def test_run_rag_expand_invalid_trajectory(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(rag_query, "openai_client", dummy)
    monkeypatch.setattr(rag_query, "_build_context_from_catalog", lambda *_args, **_kwargs: ("[1] ctx", 1))
    response_json = {JSON_KEYS["language"]: "en", JSON_KEYS["comparative_trajectory"]: " "}
    dummy.chat_completion = lambda *args, **kwargs: types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(message=types.SimpleNamespace(content=__import__("json").dumps(response_json)))
        ]
    )
    with pytest.raises(ValueError):
        rag_query.run_rag_expand([{"seminar_id": "s1", "full_id": "s1_chunk_001"}], "en")


def test_run_rag_query_stream(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(rag_query, "openai_client", dummy)
    source_entries = [
        SourceEntry(
            source_index=1,
            seminar_id="s1",
            seminar_title="Seminar",
            lesson_label="Lesson",
            lesson_raw="Leçon 1",
            chunk_id="chunk_001",
            chunk_index=1,
            full_id="s1_chunk_001",
            pages=[1],
        )
    ]
    monkeypatch.setattr(rag_query, "_prepare_rag_query", lambda *_args, **_kwargs: {
        "total_start": 0.0,
        "warnings": {"translation_error": None, "validation_errors": []},
        "prompt": "prompt",
        "source_entries": source_entries,
        "context_text": "[1] Bonjour",
        "retrieval_catalog": [],
        "detail_level": "concise",
        "audit_logger": None,
    })
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
            }
        ],
        JSON_KEYS["comparative_trajectory"]: "traj",
    }
    dummy.stream_parts = list(__import__("json").dumps(response_json))
    events = list(rag_query.run_rag_query_stream("Q", "en", audit=False))
    assert events[0][0] == "meta"
    assert events[-1][0] == "done"


def test_run_rag_query_stream_parse_error(tmp_path, monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(rag_query, "openai_client", dummy)
    audit_logger = AuditLogger(tmp_path)
    monkeypatch.setattr(rag_query, "_prepare_rag_query", lambda *_args, **_kwargs: {
        "total_start": 0.0,
        "warnings": {"translation_error": None, "validation_errors": []},
        "prompt": "prompt",
        "source_entries": [],
        "context_text": "",
        "retrieval_catalog": [],
        "detail_level": "concise",
        "audit_logger": audit_logger,
    })
    dummy.stream_parts = list("not json")
    with pytest.raises(ValueError):
        list(rag_query.run_rag_query_stream("Q", "en", audit=False))
    assert (tmp_path / "rag_response_last.txt").exists()


def test_run_rag_query_stream_normalize_error(tmp_path, monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(rag_query, "openai_client", dummy)
    audit_logger = AuditLogger(tmp_path)
    monkeypatch.setattr(rag_query, "_prepare_rag_query", lambda *_args, **_kwargs: {
        "total_start": 0.0,
        "warnings": {"translation_error": None, "validation_errors": []},
        "prompt": "prompt",
        "source_entries": [],
        "context_text": "",
        "retrieval_catalog": [],
        "detail_level": "concise",
        "audit_logger": audit_logger,
    })
    response_json = {JSON_KEYS["language"]: "en", JSON_KEYS["sources"]: []}
    dummy.stream_parts = list(__import__("json").dumps(response_json))
    monkeypatch.setattr(rag_query.ResponseProcessor, "normalize", lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError):
        list(rag_query.run_rag_query_stream("Q", "en", audit=False))
    assert (tmp_path / "rag_response_last.txt").exists()


def test_run_rag_query_stream_collects_warnings(tmp_path, monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(rag_query, "openai_client", dummy)
    audit_logger = AuditLogger(tmp_path)
    source_entries = [
        SourceEntry(
            source_index=1,
            seminar_id="s1",
            seminar_title="Seminar",
            lesson_label="Lesson",
            lesson_raw="Leçon 1",
            chunk_id="chunk_001",
            chunk_index=1,
            full_id="s1_chunk_001",
            pages=[1],
        )
    ]
    monkeypatch.setattr(rag_query, "_prepare_rag_query", lambda *_args, **_kwargs: {
        "total_start": 0.0,
        "warnings": {"translation_error": None, "validation_errors": []},
        "prompt": "prompt",
        "source_entries": source_entries,
        "context_text": "[1] Bonjour",
        "retrieval_catalog": [],
        "detail_level": "concise",
        "audit_logger": audit_logger,
    })
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
            }
        ],
        JSON_KEYS["comparative_trajectory"]: "traj",
    }
    dummy.stream_parts = list(__import__("json").dumps(response_json))
    called = {"count": 0}

    def on_generate():
        called["count"] += 1

    monkeypatch.setattr(
        rag_query.ResponseProcessor,
        "collect_validation_errors",
        lambda *_args, **_kwargs: ["warn"],
    )
    events = list(rag_query.run_rag_query_stream("Q", "en", audit=False, on_generate=on_generate))
    assert events[-1][1]["warnings"]["validation_errors"] == ["warn"]
    assert called["count"] == 1
    assert (tmp_path / "rag_response_last.json").exists()


def test_run_rag_expand_stream(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(rag_query, "openai_client", dummy)
    monkeypatch.setattr(rag_query, "_build_context_from_catalog", lambda *_args, **_kwargs: ("[1] ctx", 1))
    response_json = {
        JSON_KEYS["language"]: "en",
        JSON_KEYS["comparative_trajectory"]: "traj",
    }
    dummy.stream_parts = list(__import__("json").dumps(response_json))
    events = list(
        rag_query.run_rag_expand_stream(
            [{"seminar_id": "s1", "full_id": "s1_chunk_001"}],
            "en",
        )
    )
    assert events[0][0] == "meta"
    assert events[-1][0] == "done"


def test_run_rag_expand_stream_errors(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(rag_query, "openai_client", dummy)
    with pytest.raises(ValueError):
        list(rag_query.run_rag_expand_stream([], "en"))
    with pytest.raises(ValueError):
        list(rag_query.run_rag_expand_stream([{"seminar_id": "s1"}], "xx"))

    monkeypatch.setattr(rag_query, "_build_context_from_catalog", lambda *_args, **_kwargs: ("", 0))
    with pytest.raises(ValueError):
        list(rag_query.run_rag_expand_stream([{"seminar_id": "s1", "full_id": "s1_chunk_001"}], "en"))


def test_run_rag_expand_stream_parse_error(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(rag_query, "openai_client", dummy)
    monkeypatch.setattr(rag_query, "_build_context_from_catalog", lambda *_args, **_kwargs: ("[1] ctx", 1))
    dummy.stream_parts = list("bad json")
    with pytest.raises(ValueError):
        list(rag_query.run_rag_expand_stream([{"seminar_id": "s1", "full_id": "s1_chunk_001"}], "en"))


def test_run_rag_expand_stream_invalid_trajectory(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(rag_query, "openai_client", dummy)
    monkeypatch.setattr(rag_query, "_build_context_from_catalog", lambda *_args, **_kwargs: ("[1] ctx", 1))
    response_json = {JSON_KEYS["language"]: "en", JSON_KEYS["comparative_trajectory"]: " "}
    dummy.stream_parts = list(__import__("json").dumps(response_json))
    with pytest.raises(ValueError):
        list(rag_query.run_rag_expand_stream([{"seminar_id": "s1", "full_id": "s1_chunk_001"}], "en"))
