"""Centralized, Pydantic-validated application settings."""

import os
from urllib.parse import quote, urlunsplit

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Load and validate environment variables from `.env`.

    - `extra="ignore"` lets `.env` keep legacy keys (`POSTGRES_URI`, `REDIS_URL`,
      old Groq keys, etc.) without failing startup.
    - Required fields raise a clear validation error at import time if missing.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- JINA AI (embeddings + reranker) ---
    JINA_API_KEY: str | None = None

    # --- OPENAI & OPENROUTER LLM ---
    OPENAI_API_KEY: str | None = None
    JUDGE_OPENAI_API_KEY: str | None = None
    OPEN_ROUTER_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None

    # --- PORTKEY LLM GATEWAY ---
    PORTKEY_API_KEY: str | None = None
    PORTKEY_API_KEY_1: str | None = None
    PORTKEY_API_KEY_2: str | None = None
    PORTKEY_API_KEY_3: str | None = None
    PORTKEY_PRIMARY_SLUG: str = "rag"
    PORTKEY_FALLBACK_SLUG: str = "brag"
    LLM_MODEL: str = Field(
        default="openai/gpt-oss-120b",
        validation_alias=AliasChoices("LLM_MODEL", "PORTKEY_MODEL", "MODEL_NAME"),
    )
    # Portkey saved config is referenced by its system-generated `pc-...` ID.
    # Required when block_inline_config is enabled on the workspace.
    PORTKEY_PRIMARY_CONFIG_ID: str | None = None

    # --- QDRANT VECTOR DB ---
    QDRANT_URL: str | None = Field(default=None, validation_alias=AliasChoices("QDRANT_URL", "QDRANT_CLUSTER_ENDPOINT"))
    QDRANT_API_KEY: str | None = Field(default=None, validation_alias=AliasChoices("QDRANT_API_KEY", "QDRANT_SECURITY"))
    QDRANT_SECURITY: str | None = Field(
        default=None, validation_alias=AliasChoices("QDRANT_SECURITY", "QDRANT_API_KEY")
    )
    QDRANT_COLLECTION: str = "enterprise_rag_v1"

    # --- PINECONE VECTOR DB ---
    PINECONE_API_KEY: str | None = Field(default=None, validation_alias=AliasChoices("PINECONE_API_KEY"))
    PINECONE_HOST: str | None = Field(default=None, validation_alias=AliasChoices("PINECONE_HOST"))
    PINECONE_INDEX_NAME: str = Field(
        default="legal-enterprise-knowledge-base",
        validation_alias=AliasChoices("PINECONE_INDEX_NAME", "PINECONE_COLLECTION"),
    )
    PINECONE_COLLECTION: str = "legal-enterprise-knowledge-base"

    # --- NEON SERVERLESS POSTGRES (LangGraph checkpointer) ---
    NEON_DB_URL: str | None = None

    # --- UPSTASH REDIS (rate limiting) ---
    UPSTASH_REDIS_REST_URL: str | None = None
    UPSTASH_REDIS_REST_TOKEN: str | None = None

    # --- API SAFETY ---
    API_KEY: str | None = Field(default=None, alias="RAG_API_KEY")
    RATE_LIMIT_PER_MINUTE: int = 120
    STRICT_STARTUP: bool = False

    # --- OBSERVABILITY ---
    LOGFIRE_TOKEN: str | None = None
    LOGFIRE_WRITE_TOKEN: str | None = None
    LOGFIRE_BASE_URL: str | None = "https://logfire-us.pydantic.dev"  # Default US Logfire API endpoint
    LANGSMITH_TRACING: str = "true"
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str = "rag_scale_test"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    @field_validator(
        "JINA_API_KEY",
        "OPENAI_API_KEY",
        "JUDGE_OPENAI_API_KEY",
        "PORTKEY_API_KEY",
        "PORTKEY_PRIMARY_SLUG",
        "PORTKEY_FALLBACK_SLUG",
        "PORTKEY_PRIMARY_CONFIG_ID",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_HOST",
        "NEON_DB_URL",
        "UPSTASH_REDIS_REST_URL",
        "UPSTASH_REDIS_REST_TOKEN",
        "API_KEY",
        "LOGFIRE_TOKEN",
        "LOGFIRE_WRITE_TOKEN",
        "LOGFIRE_BASE_URL",
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
        mode="before",
    )
    @classmethod
    def _strip_whitespace(cls, v: str | None) -> str | None:
        """Strip leading and trailing whitespace from string environment variables.

        Prevents invalid HTTP header exceptions (e.g. h11/httpcore LocalProtocolError)
        caused by trailing spaces in API keys or Portkey slugs (such as 'rag   ').
        """
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("PINECONE_API_KEY", mode="before")
    @classmethod
    def _empty_pinecone_key_as_none(cls, v):
        """Treat empty PINECONE_API_KEY as unset so local Pinecone doesn't receive a blank header."""
        if v == "" or v is None:
            return None
        if isinstance(v, str):
            return v.strip()
        return v

    @property
    def judge_api_key(self) -> str | None:
        """Dedicated judge key, falling back to the main OpenAI key."""
        return self.JUDGE_OPENAI_API_KEY or self.OPENAI_API_KEY

    @property
    def postgres_uri(self) -> str:
        """LangGraph Postgres checkpointer URI (Neon).

        Serverless Postgres closes idle connections, so append TCP keepalive
        options to keep the connection pool healthy between requests.
        """
        if not self.NEON_DB_URL:
            return ""
        base = self.NEON_DB_URL.rstrip("/")
        keepalive = "keepalives=1&keepalives_idle=30&keepalives_interval=10&keepalives_count=5"
        if "?" in base:
            return f"{base}&{keepalive}"
        return f"{base}?{keepalive}"

    @property
    def redis_url(self) -> str:
        """TLS Redis URL derived from Upstash REST credentials.

        Upstash exposes the same host for REST and TLS Redis. The REST token is
        used as the Redis password under the default username. The result is
        passed to `limits` for rate limiting and to the health checker.
        """
        if not self.UPSTASH_REDIS_REST_URL or not self.UPSTASH_REDIS_REST_TOKEN:
            return ""
        host = self.UPSTASH_REDIS_REST_URL.replace("https://", "").replace("http://", "").rstrip("/")
        token = quote(self.UPSTASH_REDIS_REST_TOKEN, safe="")
        netloc = f"default:{token}@{host}"
        return urlunsplit(("rediss", netloc, "/0", "ssl_cert_reqs=required", ""))


# Singleton used across the app.
settings = Settings()


def apply_langchain_env():
    """Write LangSmith/LangChain settings to os.environ for automatic tracing.

    Tracing is only activated when both LANGSMITH_TRACING and LANGSMITH_API_KEY
    are set — enabling tracing without a key causes LangChain to emit 401 noise
    on every LangGraph step.
    """
    if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", settings.LANGSMITH_TRACING)
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.LANGSMITH_API_KEY)
    if settings.LANGSMITH_PROJECT:
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.LANGSMITH_PROJECT)
    if settings.LANGSMITH_ENDPOINT:
        os.environ.setdefault("LANGCHAIN_ENDPOINT", settings.LANGSMITH_ENDPOINT)


apply_langchain_env()
