import os
import types

import pytest

from app.infrastructure import openai_client


class DummyChat:
    def __init__(self):
        self.completions = self
        self.calls = []
        self.stream_events = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return self.stream_events
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]
        )


class DummyEmbeddings:
    def __init__(self, embedding):
        self.embedding = embedding
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=self.embedding)])


class DummyOpenAI:
    def __init__(self, embedding=None):
        self.chat = DummyChat()
        self.embeddings = DummyEmbeddings(embedding or [0.1, 0.2])


def test_load_openai_config_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        openai_client.load_openai_config()


def test_load_openai_config_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = openai_client.load_openai_config()
    assert config.gpt_model
    assert config.translation_model
    assert config.embedding_model


def test_prefers_max_completion_tokens():
    assert openai_client.OpenAIClient._prefers_max_completion_tokens("gpt-5") is True
    assert openai_client.OpenAIClient._prefers_max_completion_tokens("gpt-4o") is False


def test_supports_temperature():
    assert openai_client.OpenAIClient._supports_temperature("gpt-5") is False
    assert openai_client.OpenAIClient._supports_temperature("gpt-4o") is True


def test_chat_completion_adjusts_kwargs(monkeypatch):
    config = openai_client.OpenAIConfig(
        api_key="key",
        gpt_model="gpt-5-mini",
        translation_model="gpt-4o-mini",
        embedding_model="text-embedding-3-large",
    )
    client = openai_client.OpenAIClient(config)
    dummy = DummyOpenAI()
    client._client = dummy

    client.chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=123,
        temperature=0.5,
    )
    call = dummy.chat.calls[-1]
    assert "max_tokens" not in call
    assert call.get("max_completion_tokens") == 123
    assert "temperature" not in call


def test_config_property():
    config = openai_client.OpenAIConfig(
        api_key="key",
        gpt_model="gpt-4o",
        translation_model="gpt-4o-mini",
        embedding_model="text-embedding-3-large",
    )
    client = openai_client.OpenAIClient(config)
    assert client.config is config


def test_chat_completion_stream_yields_content():
    config = openai_client.OpenAIConfig(
        api_key="key",
        gpt_model="gpt-4o",
        translation_model="gpt-4o-mini",
        embedding_model="text-embedding-3-large",
    )
    client = openai_client.OpenAIClient(config)
    dummy = DummyOpenAI()
    dummy.chat.stream_events = [
        types.SimpleNamespace(choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content="a"))]),
        types.SimpleNamespace(choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=None))]),
        types.SimpleNamespace(choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content="b"))]),
    ]
    client._client = dummy

    chunks = list(
        client.chat_completion_stream(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.1,
        )
    )
    assert chunks == ["a", "b"]


def test_chat_completion_stream_adjusts_kwargs():
    config = openai_client.OpenAIConfig(
        api_key="key",
        gpt_model="gpt-5-mini",
        translation_model="gpt-4o-mini",
        embedding_model="text-embedding-3-large",
    )
    client = openai_client.OpenAIClient(config)
    dummy = DummyOpenAI()
    dummy.chat.stream_events = [
        types.SimpleNamespace(choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content="a"))]),
    ]
    client._client = dummy
    list(
        client.chat_completion_stream(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
            temperature=0.5,
        )
    )
    call = dummy.chat.calls[-1]
    assert call.get("max_completion_tokens") == 10
    assert "temperature" not in call


def test_translate_and_embed(monkeypatch):
    config = openai_client.OpenAIConfig(
        api_key="key",
        gpt_model="gpt-4o",
        translation_model="gpt-4o-mini",
        embedding_model="text-embedding-3-large",
    )
    client = openai_client.OpenAIClient(config)
    dummy = DummyOpenAI(embedding=[1.0, 2.0, 3.0])
    client._client = dummy

    response = client.translate("hola", "Spanish", "English")
    assert response == "ok"

    embedding = client.embed("text")
    assert embedding == [1.0, 2.0, 3.0]
