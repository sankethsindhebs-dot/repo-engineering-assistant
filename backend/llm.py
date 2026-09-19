"""Minimal provider boundary for probes; Ollama is a candidate adapter only."""

import asyncio
from dataclasses import dataclass
import math
import re
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from backend.config import Settings

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class ProviderError(RuntimeError):
    """Safe diagnostic: never contains raw server responses or request text."""


@dataclass(frozen=True)
class EmbeddingIdentity:
    provider: str
    model: str
    digest: str


class ModelProvider(Protocol):
    async def generate(self, prompt: str, response_model: type[ResponseT]) -> ResponseT: ...
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
    async def embedding_identity(self) -> EmbeddingIdentity: ...
    async def aclose(self) -> None: ...


class _Message(BaseModel):
    model_config = ConfigDict(strict=True)
    role: str
    content: str


class _ChatResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    model: str
    message: _Message
    done: bool


class _EmbeddingResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    model: str
    embeddings: list[list[float]]


class OllamaProvider:
    def __init__(self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.client = httpx.AsyncClient(
            base_url=settings.model_base_url,
            timeout=settings.model_timeout_seconds,
            trust_env=False,
            follow_redirects=False,
            transport=transport,
        )

    async def _request(self, path: str, payload: dict | None = None) -> dict:
        try:
            async with asyncio.timeout(self.settings.model_timeout_seconds):
                response = (await self.client.get(path) if payload is None
                            else await self.client.post(path, json=payload))
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"Model endpoint returned HTTP {exc.response.status_code}") from None
        except (httpx.RequestError, TimeoutError) as exc:
            raise ProviderError(f"Model request failed ({type(exc).__name__})") from None
        except ValueError:
            raise ProviderError("Model endpoint returned invalid JSON") from None
        if not isinstance(body, dict) or "error" in body:
            raise ProviderError("Model endpoint returned an invalid response envelope")
        return body

    @staticmethod
    def _check_model(actual: str, expected: str | None) -> None:
        if actual not in {expected, f"{expected}:latest"}:
            raise ProviderError("Response model does not match the configured model")

    async def generate(self, prompt: str, response_model: type[ResponseT]) -> ResponseT:
        if not self.settings.model_generation_model:
            raise ProviderError("Generation model is not configured")
        body = await self._request("/api/chat", {
            "model": self.settings.model_generation_model,
            "messages": [{"role": "user", "content": prompt}],
            "format": response_model.model_json_schema(),
            "stream": False,
            "options": {"temperature": 0, "num_predict": 256},
        })
        try:
            response = _ChatResponse.model_validate(body)
            self._check_model(response.model, self.settings.model_generation_model)
            if not response.done or response.message.role != "assistant":
                raise ProviderError("Model returned an incomplete/non-assistant response")
            return response_model.model_validate_json(response.message.content, strict=True)
        except ValidationError:
            raise ProviderError("Generation response failed schema validation") from None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.settings.model_embedding_model:
            raise ProviderError("Embedding model is not configured")
        if not texts or any(not text.strip() for text in texts):
            raise ProviderError("Embedding input must contain nonempty text")
        body = await self._request("/api/embed", {
            "model": self.settings.model_embedding_model,
            "input": texts,
            "truncate": False,
        })
        try:
            response = _EmbeddingResponse.model_validate(body)
        except ValidationError:
            raise ProviderError("Embedding response failed schema validation") from None
        self._check_model(response.model, self.settings.model_embedding_model)
        vectors = response.embeddings
        if len(vectors) != len(texts) or not vectors or not vectors[0]:
            raise ProviderError("Embedding response has the wrong vector count or empty vectors")
        dimension = len(vectors[0])
        if any(
            len(vector) != dimension
            or not all(math.isfinite(value) for value in vector)
            or not any(value != 0 for value in vector)
            for vector in vectors
        ):
            raise ProviderError("Embedding vectors must be finite, nonzero and equal-dimensional")
        return vectors

    async def embedding_identity(self) -> EmbeddingIdentity:
        name = self.settings.model_embedding_model
        if not name:
            raise ProviderError("Embedding model is not configured")
        normalized = name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"
        body = await self._request("/api/tags")
        models = body.get("models")
        if not isinstance(models, list):
            raise ProviderError("Model metadata response has invalid schema")
        matches = [item for item in models if isinstance(item, dict)
                   and item.get("name") == normalized]
        if len(matches) != 1 or not isinstance(matches[0].get("digest"), str):
            raise ProviderError("Embedding model identity is unavailable")
        digest = matches[0]["digest"].removeprefix("sha256:")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ProviderError("Embedding model digest has invalid schema")
        return EmbeddingIdentity("ollama", normalized, digest.lower())

    async def aclose(self) -> None:
        await self.client.aclose()


def create_provider(settings: Settings) -> ModelProvider:
    if settings.model_provider == "ollama":
        return OllamaProvider(settings)
    raise ProviderError("Model provider is disabled; no model feasibility has been established")
