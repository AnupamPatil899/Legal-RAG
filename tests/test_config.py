"""Tests for app.config Settings validators and header formatting."""

from app.config import Settings


def test_settings_strips_whitespace():
    """Verify that Settings strips whitespace from configuration values."""
    settings = Settings(
        JINA_API_KEY=" jina-key ",
        OPENAI_API_KEY=" openai-key ",
        PORTKEY_API_KEY=" portkey-key ",
        PORTKEY_PRIMARY_SLUG=" rag   ",
        PORTKEY_PRIMARY_CONFIG_ID=" pc-12345   ",
        QDRANT_URL="https://qdrant.example.com ",
        NEON_DB_URL=" postgresql://user:pass@host/db ",
        UPSTASH_REDIS_REST_URL=" https://redis.example.com ",
        UPSTASH_REDIS_REST_TOKEN=" token ",
    )
    assert settings.PORTKEY_PRIMARY_SLUG == "rag"
    assert settings.PORTKEY_PRIMARY_CONFIG_ID == "pc-12345"
    assert settings.PORTKEY_API_KEY == "portkey-key"
    assert settings.OPENAI_API_KEY == "openai-key"
    assert settings.JINA_API_KEY == "jina-key"
