import json
from pathlib import Path

import pytest

from app.application import index_text as index_module
from app.core.config import AppConfig


class DummyUpdater:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.calls = []

    def append(self, records):
        self.calls.append(records)
        return len(records)


class DummyEmbedder:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class DummyClient:
    def __init__(self, *args, **kwargs):
        self.config = type("Cfg", (), {"embedding_model": "embed"})()


def test_normalize_doc_id():
    assert index_module._normalize_doc_id(" Hello World ") == "Hello_World"
    assert index_module._normalize_doc_id("!!!") == ""


def test_metadata_has_doc(tmp_path):
    meta_path = tmp_path / "meta.json"
    assert index_module._metadata_has_doc(meta_path, "doc") is False
    meta_path.write_text(json.dumps({"ids": ["doc_abc"]}), encoding="utf-8")
    assert index_module._metadata_has_doc(meta_path, "doc") is True


def test_metadata_has_doc_invalid_payload(tmp_path):
    meta_path = tmp_path / "meta.json"
    meta_path.write_text("{bad json", encoding="utf-8")
    assert index_module._metadata_has_doc(meta_path, "doc") is False

    meta_path.write_text(json.dumps({"ids": "not-a-list"}), encoding="utf-8")
    assert index_module._metadata_has_doc(meta_path, "doc") is False


def test_seminar_id_from_pdf(tmp_path):
    path = tmp_path / "My Seminar.pdf"
    assert index_module._seminar_id_from_pdf(path) == "My_Seminar"


def test_index_text_success(tmp_path, monkeypatch):
    app_config = AppConfig(
        chunks_dir=tmp_path / "chunks",
        audit_dir=tmp_path / "audit",
        vector_index_path=tmp_path / "index" / "lacan.index",
        metadata_path=tmp_path / "index" / "metadata.json",
        input_pdf_dir=tmp_path / "pdfs",
        output_dir=tmp_path / "out",
    )
    monkeypatch.setattr(index_module, "APP_CONFIG", app_config)
    monkeypatch.setattr(index_module, "OpenAIClient", DummyClient)
    monkeypatch.setattr(index_module, "load_openai_config", lambda: object())
    monkeypatch.setattr(index_module, "OpenAIEmbedder", DummyEmbedder)
    monkeypatch.setattr(index_module, "FaissIndexUpdater", DummyUpdater)

    payload = index_module.index_text("hello world", doc_id="doc_1")
    assert payload["doc_id"] == "doc_1"
    assert payload["chunk_count"] >= 1
    assert Path(payload["chunks_path"]).exists()


def test_index_text_rejects_empty():
    with pytest.raises(ValueError):
        index_module.index_text("  ")


def test_index_text_invalid_doc_id():
    with pytest.raises(ValueError):
        index_module.index_text("hello", doc_id="!!!")


def test_index_text_no_chunks(tmp_path, monkeypatch):
    app_config = AppConfig(
        chunks_dir=tmp_path / "chunks",
        audit_dir=tmp_path / "audit",
        vector_index_path=tmp_path / "index" / "lacan.index",
        metadata_path=tmp_path / "index" / "metadata.json",
        input_pdf_dir=tmp_path / "pdfs",
        output_dir=tmp_path / "out",
    )
    monkeypatch.setattr(index_module, "APP_CONFIG", app_config)

    class EmptyChunker:
        def __init__(self, *args, **kwargs):
            pass

        def chunk_text(self, _text):
            return []

    monkeypatch.setattr(index_module, "TextChunker", EmptyChunker)
    with pytest.raises(ValueError):
        index_module.index_text("hello", doc_id="doc_empty")


def test_index_text_duplicate_doc_id(tmp_path, monkeypatch):
    app_config = AppConfig(
        chunks_dir=tmp_path / "chunks",
        audit_dir=tmp_path / "audit",
        vector_index_path=tmp_path / "index" / "lacan.index",
        metadata_path=tmp_path / "index" / "metadata.json",
        input_pdf_dir=tmp_path / "pdfs",
        output_dir=tmp_path / "out",
    )
    monkeypatch.setattr(index_module, "APP_CONFIG", app_config)
    app_config.chunks_dir.mkdir(parents=True)
    (app_config.chunks_dir / "doc_1.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        index_module.index_text("hello", doc_id="doc_1")


def test_index_text_cleanup_on_failure(tmp_path, monkeypatch):
    class FailingUpdater(DummyUpdater):
        def append(self, records):
            raise RuntimeError("boom")

    app_config = AppConfig(
        chunks_dir=tmp_path / "chunks",
        audit_dir=tmp_path / "audit",
        vector_index_path=tmp_path / "index" / "lacan.index",
        metadata_path=tmp_path / "index" / "metadata.json",
        input_pdf_dir=tmp_path / "pdfs",
        output_dir=tmp_path / "out",
    )
    monkeypatch.setattr(index_module, "APP_CONFIG", app_config)
    monkeypatch.setattr(index_module, "OpenAIClient", DummyClient)
    monkeypatch.setattr(index_module, "load_openai_config", lambda: object())
    monkeypatch.setattr(index_module, "OpenAIEmbedder", DummyEmbedder)
    monkeypatch.setattr(index_module, "FaissIndexUpdater", FailingUpdater)

    with pytest.raises(RuntimeError):
        index_module.index_text("hello", doc_id="doc_2")
    assert not (app_config.chunks_dir / "doc_2.json").exists()


def test_index_pdf_bytes_validates(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        index_module.index_pdf_bytes(b"", "file.pdf")
    with pytest.raises(ValueError):
        index_module.index_pdf_bytes(b"%PDF", "file.txt")

    monkeypatch.setattr(index_module, "_metadata_has_doc", lambda *_args: True)
    with pytest.raises(ValueError):
        index_module.index_pdf_bytes(b"%PDF", "file.pdf")


def test_index_pdf_bytes_cleanup_on_error(tmp_path, monkeypatch):
    app_config = AppConfig(
        chunks_dir=tmp_path / "chunks",
        audit_dir=tmp_path / "audit",
        vector_index_path=tmp_path / "index" / "lacan.index",
        metadata_path=tmp_path / "index" / "metadata.json",
        input_pdf_dir=tmp_path / "pdfs",
        output_dir=tmp_path / "out",
    )
    monkeypatch.setattr(index_module, "APP_CONFIG", app_config)
    app_config.input_pdf_dir.mkdir(parents=True)

    def boom(_path):
        raise ValueError("fail")

    monkeypatch.setattr(index_module, "_index_pdf_path", boom)
    with pytest.raises(ValueError):
        index_module.index_pdf_bytes(b"%PDF", "doc.pdf")
    assert not any(app_config.input_pdf_dir.glob("*.pdf"))


def test_index_pdf_bytes_doc_id_normalization(tmp_path, monkeypatch):
    app_config = AppConfig(
        chunks_dir=tmp_path / "chunks",
        audit_dir=tmp_path / "audit",
        vector_index_path=tmp_path / "index" / "lacan.index",
        metadata_path=tmp_path / "index" / "metadata.json",
        input_pdf_dir=tmp_path / "pdfs",
        output_dir=tmp_path / "out",
    )
    monkeypatch.setattr(index_module, "APP_CONFIG", app_config)
    app_config.input_pdf_dir.mkdir(parents=True)

    def fake_index(pdf_path):
        return {"doc_id": pdf_path.stem}

    monkeypatch.setattr(index_module, "_index_pdf_path", fake_index)
    payload = index_module.index_pdf_bytes(b"%PDF", "doc.pdf", doc_id="!!!")
    assert payload["doc_id"].startswith("doc_")


def test_index_pdf_bytes_existing_path(tmp_path, monkeypatch):
    app_config = AppConfig(
        chunks_dir=tmp_path / "chunks",
        audit_dir=tmp_path / "audit",
        vector_index_path=tmp_path / "index" / "lacan.index",
        metadata_path=tmp_path / "index" / "metadata.json",
        input_pdf_dir=tmp_path / "pdfs",
        output_dir=tmp_path / "out",
    )
    monkeypatch.setattr(index_module, "APP_CONFIG", app_config)
    app_config.input_pdf_dir.mkdir(parents=True)
    existing = app_config.input_pdf_dir / "doc_1.pdf"
    existing.write_bytes(b"%PDF")

    with pytest.raises(ValueError):
        index_module.index_pdf_bytes(b"%PDF", "doc.pdf", doc_id="doc_1")


def test_index_pdf_bytes_wraps_exception(tmp_path, monkeypatch):
    app_config = AppConfig(
        chunks_dir=tmp_path / "chunks",
        audit_dir=tmp_path / "audit",
        vector_index_path=tmp_path / "index" / "lacan.index",
        metadata_path=tmp_path / "index" / "metadata.json",
        input_pdf_dir=tmp_path / "pdfs",
        output_dir=tmp_path / "out",
    )
    monkeypatch.setattr(index_module, "APP_CONFIG", app_config)
    app_config.input_pdf_dir.mkdir(parents=True)

    def boom(_path):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(index_module, "_index_pdf_path", boom)
    with pytest.raises(RuntimeError):
        index_module.index_pdf_bytes(b"%PDF", "doc.pdf", doc_id="doc_2")


def test_index_pdf_path_success(tmp_path, monkeypatch):
    app_config = AppConfig(
        chunks_dir=tmp_path / "chunks",
        audit_dir=tmp_path / "audit",
        vector_index_path=tmp_path / "index" / "lacan.index",
        metadata_path=tmp_path / "index" / "metadata.json",
        input_pdf_dir=tmp_path / "pdfs",
        output_dir=tmp_path / "out",
    )
    monkeypatch.setattr(index_module, "APP_CONFIG", app_config)
    monkeypatch.setattr(index_module, "OpenAIClient", DummyClient)
    monkeypatch.setattr(index_module, "load_openai_config", lambda: object())
    monkeypatch.setattr(index_module, "OpenAIEmbedder", DummyEmbedder)
    monkeypatch.setattr(index_module, "FaissIndexUpdater", DummyUpdater)
    monkeypatch.setattr(index_module, "reset_default_retriever", lambda: None)

    pdf_path = app_config.input_pdf_dir / "Seminar.pdf"
    app_config.input_pdf_dir.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF")

    def fake_process(_self, _path):
        app_config.output_dir.mkdir(parents=True, exist_ok=True)
        output = app_config.output_dir / "Seminar.json"
        output.write_text(
            json.dumps(
                [
                    {
                        "seminar": "Seminar",
                        "lecon": None,
                        "chunk_id": "chunk_000",
                        "chunk_index": 0,
                        "text": "hi",
                        "pages": [1],
                    }
                ]
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(index_module.PDFChunker, "process_pdf_to_json", fake_process)
    payload = index_module._index_pdf_path(pdf_path)
    assert payload["doc_id"] == "Seminar"
    assert payload["chunk_count"] == 1


def test_index_pdf_path_missing_file(tmp_path, monkeypatch):
    app_config = AppConfig(
        chunks_dir=tmp_path / "chunks",
        audit_dir=tmp_path / "audit",
        vector_index_path=tmp_path / "index" / "lacan.index",
        metadata_path=tmp_path / "index" / "metadata.json",
        input_pdf_dir=tmp_path / "pdfs",
        output_dir=tmp_path / "out",
    )
    monkeypatch.setattr(index_module, "APP_CONFIG", app_config)
    with pytest.raises(ValueError):
        index_module._index_pdf_path(tmp_path / "missing.pdf")


def test_index_pdf_path_invalid_filename(tmp_path, monkeypatch):
    app_config = AppConfig(
        chunks_dir=tmp_path / "chunks",
        audit_dir=tmp_path / "audit",
        vector_index_path=tmp_path / "index" / "lacan.index",
        metadata_path=tmp_path / "index" / "metadata.json",
        input_pdf_dir=tmp_path / "pdfs",
        output_dir=tmp_path / "out",
    )
    monkeypatch.setattr(index_module, "APP_CONFIG", app_config)
    app_config.input_pdf_dir.mkdir(parents=True)
    pdf_path = app_config.input_pdf_dir / "Seminar.pdf"
    pdf_path.write_bytes(b"%PDF")
    monkeypatch.setattr(index_module, "_seminar_id_from_pdf", lambda _path: "")
    with pytest.raises(ValueError):
        index_module._index_pdf_path(pdf_path)


def test_index_pdf_path_existing_chunks(tmp_path, monkeypatch):
    app_config = AppConfig(
        chunks_dir=tmp_path / "chunks",
        audit_dir=tmp_path / "audit",
        vector_index_path=tmp_path / "index" / "lacan.index",
        metadata_path=tmp_path / "index" / "metadata.json",
        input_pdf_dir=tmp_path / "pdfs",
        output_dir=tmp_path / "out",
    )
    monkeypatch.setattr(index_module, "APP_CONFIG", app_config)
    app_config.input_pdf_dir.mkdir(parents=True)
    app_config.output_dir.mkdir(parents=True)
    pdf_path = app_config.input_pdf_dir / "Seminar.pdf"
    pdf_path.write_bytes(b"%PDF")
    (app_config.output_dir / "Seminar.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        index_module._index_pdf_path(pdf_path)


def test_index_pdf_path_chunking_failed(tmp_path, monkeypatch):
    app_config = AppConfig(
        chunks_dir=tmp_path / "chunks",
        audit_dir=tmp_path / "audit",
        vector_index_path=tmp_path / "index" / "lacan.index",
        metadata_path=tmp_path / "index" / "metadata.json",
        input_pdf_dir=tmp_path / "pdfs",
        output_dir=tmp_path / "out",
    )
    monkeypatch.setattr(index_module, "APP_CONFIG", app_config)
    app_config.input_pdf_dir.mkdir(parents=True)
    pdf_path = app_config.input_pdf_dir / "Seminar.pdf"
    pdf_path.write_bytes(b"%PDF")

    def no_write(_self, _path):
        return None

    monkeypatch.setattr(index_module.PDFChunker, "process_pdf_to_json", no_write)
    with pytest.raises(RuntimeError):
        index_module._index_pdf_path(pdf_path)


def test_index_pdf_path_wraps_exception(tmp_path, monkeypatch):
    class FailingUpdater(DummyUpdater):
        def append(self, records):
            raise RuntimeError("boom")

    app_config = AppConfig(
        chunks_dir=tmp_path / "chunks",
        audit_dir=tmp_path / "audit",
        vector_index_path=tmp_path / "index" / "lacan.index",
        metadata_path=tmp_path / "index" / "metadata.json",
        input_pdf_dir=tmp_path / "pdfs",
        output_dir=tmp_path / "out",
    )
    monkeypatch.setattr(index_module, "APP_CONFIG", app_config)
    monkeypatch.setattr(index_module, "OpenAIClient", DummyClient)
    monkeypatch.setattr(index_module, "load_openai_config", lambda: object())
    monkeypatch.setattr(index_module, "OpenAIEmbedder", DummyEmbedder)
    monkeypatch.setattr(index_module, "FaissIndexUpdater", FailingUpdater)
    monkeypatch.setattr(index_module, "reset_default_retriever", lambda: None)

    app_config.input_pdf_dir.mkdir(parents=True)
    app_config.output_dir.mkdir(parents=True)
    pdf_path = app_config.input_pdf_dir / "Seminar.pdf"
    pdf_path.write_bytes(b"%PDF")
    chunk_path = app_config.output_dir / "Seminar.json"

    def fake_process(_self, _path):
        chunk_path.write_text(
            json.dumps(
                [
                    {
                        "seminar": "Seminar",
                        "lecon": None,
                        "chunk_id": "chunk_000",
                        "chunk_index": 0,
                        "text": "hi",
                        "pages": [1],
                    }
                ]
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(index_module.PDFChunker, "process_pdf_to_json", fake_process)
    with pytest.raises(RuntimeError):
        index_module._index_pdf_path(pdf_path)
    assert not chunk_path.exists()
