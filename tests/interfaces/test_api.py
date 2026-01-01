import types

import pytest

from app.interfaces import api as api_module


class DummyUpload:
    def __init__(self, content=b"%PDF", filename="doc.pdf"):
        self._content = content
        self.filename = filename

    async def read(self):
        return self._content


def test_rag_query_success(monkeypatch):
    payload = {
        "language": "en",
        "sources": [
            {
                "source_index": 1,
                "french_quotes": ["Bonjour"],
                "translations": ["Hello"],
                "translation_critique": "crit",
                "context": "ctx",
                "lacanian_development": "dev",
                "source_metadata": {
                    "seminar_title": "Seminar",
                    "seminar_id": "s1",
                    "lesson_label": "Lesson",
                    "lesson_raw": None,
                    "chunk_id": "chunk_001",
                    "chunk_index": 1,
                    "pages": [1],
                },
            }
        ],
        "comparative_trajectory": "traj",
        "retrieval_catalog": [
            {
                "rank": 1,
                "score": 0.1,
                "seminar_id": "s1",
                "seminar_title": "Seminar",
                "lesson_label": "Lesson",
                "lesson_raw": None,
                "chunk_id": "chunk_001",
                "chunk_index": 1,
                "full_id": "s1_chunk_001",
                "pages": [1],
            }
        ],
        "sources_catalog": [
            {
                "source_index": 1,
                "seminar_title": "Seminar",
                "seminar_id": "s1",
                "lesson_label": "Lesson",
                "chunk_id": "chunk_001",
                "chunk_index": 1,
                "pages": [1],
            }
        ],
    }
    monkeypatch.setattr(
        api_module,
        "run_rag_query",
        lambda *_args, **_kwargs: (payload, {"translation_error": None, "validation_errors": []}),
    )
    request = types.SimpleNamespace(
        question="Q",
        language="en",
        detail="concise",
        stream=False,
        top_k=None,
        max_sources=None,
        source_id=None,
    )
    response = api_module.rag_query(request)
    result = response.result
    language = result["language"] if isinstance(result, dict) else result.language
    assert language == "en"


def test_rag_query_stream(monkeypatch):
    class DummyStream:
        def __init__(self, content, **_kwargs):
            self.content = list(content)

    monkeypatch.setattr(api_module, "StreamingResponse", DummyStream)
    monkeypatch.setattr(api_module, "run_rag_query_stream", lambda *_args, **_kwargs: [("meta", {"ok": True})])
    request = types.SimpleNamespace(
        question="Q",
        language="en",
        detail="concise",
        stream=True,
        top_k=None,
        max_sources=None,
        source_id=None,
    )
    response = api_module.rag_query(request)
    assert response.content


def test_rag_query_stream_error(monkeypatch):
    class DummyStream:
        def __init__(self, content, **_kwargs):
            self.content = list(content)

    monkeypatch.setattr(api_module, "StreamingResponse", DummyStream)
    def boom(*_args, **_kwargs):
        raise RuntimeError("stream boom")

    monkeypatch.setattr(api_module, "run_rag_query_stream", boom)
    request = types.SimpleNamespace(
        question="Q",
        language="en",
        detail="concise",
        stream=True,
        top_k=None,
        max_sources=None,
        source_id=None,
    )
    response = api_module.rag_query(request)
    assert any("event: error" in chunk for chunk in response.content)


def test_rag_query_error(monkeypatch):
    def boom(*_args, **_kwargs):
        raise ValueError("bad")

    monkeypatch.setattr(api_module, "run_rag_query", boom)
    request = types.SimpleNamespace(
        question="Q",
        language="en",
        detail="concise",
        stream=False,
        top_k=None,
        max_sources=None,
        source_id=None,
    )
    with pytest.raises(Exception) as exc:
        api_module.rag_query(request)
    assert "bad" in str(exc.value)


def test_rag_query_server_error(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("server error")

    monkeypatch.setattr(api_module, "run_rag_query", boom)
    request = types.SimpleNamespace(
        question="Q",
        language="en",
        detail="concise",
        stream=False,
        top_k=None,
        max_sources=None,
        source_id=None,
    )
    with pytest.raises(Exception) as exc:
        api_module.rag_query(request)
    assert "server error" in str(exc.value)


def test_rag_expand_success(monkeypatch):
    monkeypatch.setattr(api_module, "run_rag_expand", lambda *_args, **_kwargs: {"language": "en", "comparative_trajectory": "traj"})
    request = types.SimpleNamespace(
        stream=False,
        retrieval_catalog=[{"rank": 1}],
        result=None,
        language="en",
        question=None,
        instruction=None,
    )
    response = api_module.rag_expand(request)
    result = response.result
    language = result["language"] if isinstance(result, dict) else result.language
    assert language == "en"


def test_rag_expand_stream(monkeypatch):
    class DummyStream:
        def __init__(self, content, **_kwargs):
            self.content = list(content)

    monkeypatch.setattr(api_module, "StreamingResponse", DummyStream)
    monkeypatch.setattr(
        api_module,
        "run_rag_expand_stream",
        lambda *_args, **_kwargs: [("meta", {"ok": True})],
    )
    request = types.SimpleNamespace(
        stream=True,
        retrieval_catalog=[{"rank": 1}],
        result=None,
        language="en",
        question=None,
        instruction=None,
    )
    response = api_module.rag_expand(request)
    assert response.content


def test_rag_expand_stream_error(monkeypatch):
    class DummyStream:
        def __init__(self, content, **_kwargs):
            self.content = list(content)

    monkeypatch.setattr(api_module, "StreamingResponse", DummyStream)
    def boom(*_args, **_kwargs):
        raise RuntimeError("stream boom")

    monkeypatch.setattr(api_module, "run_rag_expand_stream", boom)
    request = types.SimpleNamespace(
        stream=True,
        retrieval_catalog=[{"rank": 1}],
        result=None,
        language="en",
        question=None,
        instruction=None,
    )
    response = api_module.rag_expand(request)
    assert any("event: error" in chunk for chunk in response.content)


def test_rag_expand_with_result_fallback(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "run_rag_expand",
        lambda *_args, **_kwargs: {"language": "en", "comparative_trajectory": "traj"},
    )
    result = types.SimpleNamespace(retrieval_catalog=[{"rank": 1}], language="en")
    request = types.SimpleNamespace(
        stream=False,
        retrieval_catalog=None,
        result=result,
        language=None,
        question=None,
        instruction=None,
    )
    response = api_module.rag_expand(request)
    result_payload = response.result
    language = result_payload["language"] if isinstance(result_payload, dict) else result_payload.language
    assert language == "en"


def test_rag_expand_missing_catalog(monkeypatch):
    request = types.SimpleNamespace(
        stream=False,
        retrieval_catalog=None,
        result=None,
        language="en",
        question=None,
        instruction=None,
    )
    with pytest.raises(Exception):
        api_module.rag_expand(request)


def test_rag_expand_server_error(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("server error")

    monkeypatch.setattr(api_module, "run_rag_expand", boom)
    request = types.SimpleNamespace(
        stream=False,
        retrieval_catalog=[{"rank": 1}],
        result=None,
        language="en",
        question=None,
        instruction=None,
    )
    with pytest.raises(Exception) as exc:
        api_module.rag_expand(request)
    assert "server error" in str(exc.value)


def test_index_text_success(monkeypatch):
    monkeypatch.setattr(api_module, "run_index_pdf", lambda *_args, **_kwargs: {
        "doc_id": "doc",
        "chunk_count": 1,
        "added_vectors": 1,
        "chunks_path": "x",
        "index_path": "y",
        "metadata_path": "z",
    })

    response = api_module.index_text
    upload = DummyUpload()
    result = __import__("asyncio").run(response(upload, None))
    assert result.doc_id == "doc"


def test_index_text_conflict(monkeypatch):
    def conflict(*_args, **_kwargs):
        raise ValueError("already exists")

    monkeypatch.setattr(api_module, "run_index_pdf", conflict)
    upload = DummyUpload()
    with pytest.raises(Exception):
        __import__("asyncio").run(api_module.index_text(upload, None))


def test_index_text_server_error(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(api_module, "run_index_pdf", boom)
    upload = DummyUpload()
    with pytest.raises(Exception):
        __import__("asyncio").run(api_module.index_text(upload, None))
