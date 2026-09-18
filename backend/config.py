"""Phase 0 settings. Provider selection remains an explicit local experiment."""

from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEO4J_VERSION = "5.26.30"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_username: Literal["neo4j"] = "neo4j"
    neo4j_database: str = Field(default="neo4j", min_length=1)
    neo4j_password: SecretStr | None = None
    neo4j_timeout_seconds: float = Field(default=5, gt=0, le=60)

    model_provider: Literal["disabled", "ollama"] = "disabled"
    model_base_url: str = "http://127.0.0.1:11434"
    model_generation_model: str | None = None
    model_embedding_model: str | None = None
    model_timeout_seconds: float = Field(default=120, gt=0, le=600)

    @field_validator("neo4j_uri")
    @classmethod
    def validate_neo4j_uri(cls, value: str) -> str:
        parts = urlsplit(value)
        if (
            parts.scheme not in {"bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc"}
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.path not in {"", "/"}
            or parts.query
            or parts.fragment
        ):
            raise ValueError("Use a Neo4j/Bolt URI without credentials, path, query or fragment")
        _ = parts.port  # Also reject invalid port numbers.
        return value.rstrip("/")

    @field_validator("model_base_url")
    @classmethod
    def validate_local_provider_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if (
            parts.scheme not in {"http", "https"}
            or parts.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parts.username is not None
            or parts.password is not None
            or parts.path not in {"", "/"}
            or parts.query
            or parts.fragment
        ):
            raise ValueError("Phase 0 supports a loopback model endpoint without credentials")
        _ = parts.port
        return value.rstrip("/")

    @field_validator("neo4j_password", mode="before")
    @classmethod
    def empty_password_is_unconfigured(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("neo4j_password")
    @classmethod
    def validate_password(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None:
            password = value.get_secret_value()
            if len(password) < 8 or password.lower().startswith(("changeme", "replace-", "<")):
                raise ValueError("Set a local password of at least eight characters; do not use a placeholder")
        return value

    @model_validator(mode="after")
    def validate_provider_selection(self) -> "Settings":
        if self.model_provider == "ollama":
            for name in (self.model_generation_model, self.model_embedding_model):
                if not name or not name.strip() or name.endswith(":cloud"):
                    raise ValueError("Explicit local generation and embedding model names are required")
        return self
