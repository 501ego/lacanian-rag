import json
import types

import numpy as np

from app.infrastructure import retriever


class DummyIndexStore:
    def __init__(self, ids, metadata, distances=None, indices=None):
        self._ids = ids
        self._metadata = metadata
        self._distances = distances or [[0.1 for _ in ids]]
        self._indices = indices or [list(range(len(ids)))]

    @property
    def ids(self):
        return self._ids

    @property
    def metadata(self):
        return self._metadata

    def search(self, _vector, _top_k):
        return self._distances, self._indices


class DummyEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return [0.0, 0.1]


def test_normalize_source_id():
    assert retriever._normalize_source_id(None) is None
    assert retriever._normalize_source_id(" ") is None
    assert retriever._normalize_source_id(" Foo ") == "foo"


def test_index_store_reads_metadata(tmp_path, monkeypatch):
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(
        json.dumps({"ids": ["a"], "metadata": [{"seminar": "s1"}]}),
        encoding="utf-8",
    )

    class DummyIndex:
        def search(self, vector, top_k):
            return [[0.1]], [[0]]

    monkeypatch.setattr(retriever.faiss, "read_index", lambda _path: DummyIndex())
    config = retriever.RetrieverConfig(
        index_path=tmp_path / "index.bin",
        metadata_path=meta_path,
        embedding_model="embed",
    )
    store = retriever.IndexStore(config)
    assert store.ids == ["a"]
    assert store.metadata == [{"seminar": "s1"}]
    distances, indices = store.search(np.array([[0.0]], dtype="float32"), 1)
    assert distances == [[0.1]]
    assert indices == [[0]]


def test_openai_embedder_returns_array(monkeypatch):
    class DummyClient:
        def embed(self, _text, model=None):
            _ = model
            return [0.1, 0.2]

    embedder = retriever.OpenAIEmbedder(DummyClient(), "model")
    vector = embedder.embed("q")
    assert getattr(vector, "shape", None) == (1, 2)


def test_retriever_search_filters_and_limits():
    index_store = DummyIndexStore(
        ids=["a", "b", "c"],
        metadata=[
            {"seminar": "s1", "lecon": "l1", "pages": [1]},
            {"seminar": "s2", "lecon": "l2", "pages": [2]},
            {"seminar": "s1", "lecon": "l3", "pages": [3]},
        ],
        distances=[[0.1, 0.2, 0.3]],
        indices=[[0, 1, 2]],
    )
    retr = retriever.Retriever(index_store, DummyEmbedder())
    results = retr.search("q", top_k=1, source_id="s1")
    assert len(results) == 1
    assert results[0]["seminar"] == "s1"


def test_retriever_search_debug_and_skip(monkeypatch, caplog):
    index_store = DummyIndexStore(
        ids=["a"],
        metadata=[{"seminar": "s1"}],
        distances=[[0.1]],
        indices=[[5]],
    )
    retr = retriever.Retriever(index_store, DummyEmbedder())
    caplog.set_level("DEBUG")
    results = retr.search("q", top_k=1)
    assert results == []


def test_retriever_search_handles_empty_index():
    index_store = DummyIndexStore(ids=[], metadata=[], distances=[[]], indices=[[]])
    retr = retriever.Retriever(index_store, DummyEmbedder())
    assert retr.search("q", top_k=3) == []


def test_singleton_helpers(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(retriever._RetrieverSingleton, "_instance", None)
    monkeypatch.setattr(retriever._RetrieverSingleton, "_build", lambda: sentinel)
    assert retriever.get_default_retriever() is sentinel
    retriever.reset_default_retriever()
    assert retriever._RetrieverSingleton._instance is None


def test_singleton_build(monkeypatch):
    class DummyClient:
        def __init__(self, _cfg):
            self.config = type("Cfg", (), {"embedding_model": "embed"})()

    class DummyIndexStore:
        def __init__(self, config):
            self.config = config

        @property
        def metadata(self):
            return []

    class DummyEmbedder:
        def __init__(self, client, model):
            self.client = client
            self.model = model

    monkeypatch.setattr(retriever, "OpenAIClient", DummyClient)
    monkeypatch.setattr(retriever, "load_openai_config", lambda: object())
    monkeypatch.setattr(retriever, "IndexStore", DummyIndexStore)
    monkeypatch.setattr(retriever, "OpenAIEmbedder", DummyEmbedder)
    instance = retriever._RetrieverSingleton._build()
    assert isinstance(instance, retriever.Retriever)


def test_search_similar_chunks_delegates(monkeypatch):
    dummy = types.SimpleNamespace(search=lambda *args, **kwargs: ["ok"])
    monkeypatch.setattr(retriever, "get_default_retriever", lambda: dummy)
    assert retriever.search_similar_chunks("q") == ["ok"]
