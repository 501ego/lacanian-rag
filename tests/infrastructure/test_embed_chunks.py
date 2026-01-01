import json
import types

import pytest

from app.infrastructure import embed_chunks


class DummyEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return [0.1, 0.2]


def test_chunk_loader_loads_records(tmp_path):
    payload = [
        {"chunk_id": "chunk_000", "text": "hello", "seminar": "s1", "lecon": None, "pages": [1]},
    ]
    file_path = tmp_path / "s1.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")
    loader = embed_chunks.ChunkLoader(tmp_path)
    records = loader.load()
    assert records[0]["id"] == "s1_chunk_000"


def test_faiss_index_builder_skips_empty(capsys, tmp_path):
    builder = embed_chunks.FaissIndexBuilder(
        index_path=tmp_path / "index.bin",
        metadata_path=tmp_path / "meta.json",
        embedding_dim=2,
        embedder_instance=DummyEmbedder(),
    )
    builder.build([])
    captured = capsys.readouterr()
    assert "Skipping" in captured.out


def test_faiss_index_builder_builds(tmp_path, monkeypatch):
    embedder = DummyEmbedder()
    builder = embed_chunks.FaissIndexBuilder(
        index_path=tmp_path / "index.bin",
        metadata_path=tmp_path / "meta.json",
        embedding_dim=2,
        embedder_instance=embedder,
    )
    records = [
        {"id": "s1_chunk_000", "text": "hi", "metadata": {"seminar": "s1"}},
    ]
    builder.build(records)
    assert (tmp_path / "meta.json").exists()


def test_openai_embedder_calls_client():
    class DummyClient:
        def __init__(self):
            self.calls = []

        def embed(self, text, model=None):
            self.calls.append((text, model))
            return [0.3, 0.4]

    client = DummyClient()
    embedder = embed_chunks.OpenAIEmbedder(client, "model")
    assert embedder.embed("hi") == [0.3, 0.4]
    assert client.calls == [("hi", "model")]


def test_faiss_index_builder_single_dim(tmp_path):
    class ScalarEmbedder:
        def embed(self, _text):
            return 0.1

    builder = embed_chunks.FaissIndexBuilder(
        index_path=tmp_path / "index.bin",
        metadata_path=tmp_path / "meta.json",
        embedding_dim=1,
        embedder_instance=ScalarEmbedder(),
    )
    records = [{"id": "s1_chunk_000", "text": "hi", "metadata": {"seminar": "s1"}}]
    builder.build(records)
    assert (tmp_path / "meta.json").exists()


def test_faiss_index_builder_dim_mismatch(tmp_path):
    embedder = DummyEmbedder()
    builder = embed_chunks.FaissIndexBuilder(
        index_path=tmp_path / "index.bin",
        metadata_path=tmp_path / "meta.json",
        embedding_dim=3,
        embedder_instance=embedder,
    )
    records = [{"id": "s1_chunk_000", "text": "hi", "metadata": {"seminar": "s1"}}]
    with pytest.raises(ValueError):
        builder.build(records)


def test_faiss_index_updater_append(tmp_path, monkeypatch):
    embedder = DummyEmbedder()
    updater = embed_chunks.FaissIndexUpdater(
        index_path=tmp_path / "index.bin",
        metadata_path=tmp_path / "meta.json",
        embedding_dim=2,
        embedder_instance=embedder,
    )
    records = [
        {"id": "s1_chunk_000", "text": "hi", "metadata": {"seminar": "s1"}},
        {"id": "s1_chunk_001", "text": "ho", "metadata": {"seminar": "s1"}},
    ]
    added = updater.append(records)
    assert added == 2

    # Duplicate id should be skipped.
    added_again = updater.append(records[:1])
    assert added_again == 0


def test_faiss_index_updater_empty_records(tmp_path):
    embedder = DummyEmbedder()
    updater = embed_chunks.FaissIndexUpdater(
        index_path=tmp_path / "index.bin",
        metadata_path=tmp_path / "meta.json",
        embedding_dim=2,
        embedder_instance=embedder,
    )
    assert updater.append([]) == 0


def test_faiss_index_updater_handles_bad_metadata(tmp_path):
    meta_path = tmp_path / "meta.json"
    meta_path.write_text("{bad json", encoding="utf-8")
    updater = embed_chunks.FaissIndexUpdater(
        index_path=tmp_path / "index.bin",
        metadata_path=meta_path,
        embedding_dim=2,
        embedder_instance=DummyEmbedder(),
    )
    with pytest.raises(json.JSONDecodeError):
        updater._load_metadata()


def test_faiss_index_updater_metadata_wrong_types(tmp_path):
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps({"ids": [], "metadata": "bad"}), encoding="utf-8")
    updater = embed_chunks.FaissIndexUpdater(
        index_path=tmp_path / "index.bin",
        metadata_path=meta_path,
        embedding_dim=2,
        embedder_instance=DummyEmbedder(),
    )
    assert updater._load_metadata() == ([], [])


def test_faiss_index_updater_index_dim_mismatch(tmp_path, monkeypatch):
    embedder = DummyEmbedder()
    updater = embed_chunks.FaissIndexUpdater(
        index_path=tmp_path / "index.bin",
        metadata_path=tmp_path / "meta.json",
        embedding_dim=2,
        embedder_instance=embedder,
    )

    class DummyIndex:
        d = 3

        def add(self, _vectors):
            return None

    monkeypatch.setattr(updater, "_load_or_create_index", lambda: DummyIndex())
    records = [{"id": "s1_chunk_000", "text": "hi", "metadata": {"seminar": "s1"}}]
    with pytest.raises(ValueError):
        updater.append(records)


def test_faiss_index_updater_load_or_create_index(tmp_path, monkeypatch):
    embedder = DummyEmbedder()
    updater = embed_chunks.FaissIndexUpdater(
        index_path=tmp_path / "index.bin",
        metadata_path=tmp_path / "meta.json",
        embedding_dim=2,
        embedder_instance=embedder,
    )
    updater._index_path.write_bytes(b"index")

    class DummyIndex:
        d = 2

    monkeypatch.setattr(embed_chunks.faiss, "read_index", lambda _path: DummyIndex())
    index = updater._load_or_create_index()
    assert isinstance(index, DummyIndex)


def test_faiss_index_updater_dim_mismatch(tmp_path):
    embedder = DummyEmbedder()
    updater = embed_chunks.FaissIndexUpdater(
        index_path=tmp_path / "index.bin",
        metadata_path=tmp_path / "meta.json",
        embedding_dim=3,
        embedder_instance=embedder,
    )
    records = [
        {"id": "s1_chunk_000", "text": "hi", "metadata": {"seminar": "s1"}},
    ]
    try:
        updater.append(records)
    except ValueError as exc:
        assert "Embedding dim mismatch" in str(exc)
