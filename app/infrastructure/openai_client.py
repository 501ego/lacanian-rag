"""OpenAI client wrapper and configuration helpers."""

from dataclasses import dataclass
import os
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass(frozen=True)
class OpenAIConfig:
    """Configuration for OpenAI client usage."""

    api_key: str
    gpt_model: str
    translation_model: str
    embedding_model: str


def load_openai_config() -> OpenAIConfig:
    """Load OpenAI configuration from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")
    return OpenAIConfig(
        api_key=api_key,
        gpt_model=os.getenv("OPENAI_GPT_MODEL", "gpt-4o"),
        translation_model=os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
    )


class OpenAIClient:
    """Thin wrapper around the OpenAI SDK for app-specific needs."""

    def __init__(self, config: OpenAIConfig):
        self._config = config
        self._client = OpenAI(api_key=config.api_key)

    @property
    def config(self) -> OpenAIConfig:
        """Return the active OpenAI configuration."""
        return self._config

    def chat_completion(
        self,
        messages: Iterable[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        """Create a chat completion with the configured default model."""
        return self._client.chat.completions.create(
            model=model or self._config.gpt_model,
            messages=list(messages),
            **kwargs,
        )

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        *,
        model: Optional[str] = None,
    ) -> str:
        """Translate text using the configured translation model."""
        response = self.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a translation engine. "
                        f"Translate from {source_language} to {target_language}. "
                        "Output only the translated text without quotes or extra text."
                    ),
                },
                {"role": "user", "content": text},
            ],
            model=model or self._config.translation_model,
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    def embed(self, text: str, *, model: Optional[str] = None) -> List[float]:
        """Embed text into a vector using the configured embedding model."""
        response = self._client.embeddings.create(
            model=model or self._config.embedding_model,
            input=text,
        )
        return response.data[0].embedding
