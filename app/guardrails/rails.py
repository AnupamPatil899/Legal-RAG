import logfire
from langchain_openai import ChatOpenAI
from nemoguardrails import LLMRails, RailsConfig

from app.config import settings
from app.guardrails.colang_rules import (
    COLANG_CONTENT,
    JAILBREAK_PATTERNS,
    OFF_TOPIC_PATTERNS,
    RAIL_INDICATORS,
    REFUSE_JAILBREAK_MSG,
    REFUSE_OFF_TOPIC_MSG,
    YAML_CONTENT,
)

_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses dynamic model from settings.LLM_MODEL (e.g. openai/gpt-oss-120b) via Portkey.
    """
    global _rails

    slug = (getattr(settings, "PORTKEY_PRIMARY_SLUG", "") or "").strip()
    api_key = (getattr(settings, "PORTKEY_API_KEY", "") or "").strip()
    model_name = (getattr(settings, "LLM_MODEL", "openai/gpt-oss-120b") or "").strip()
    full_model = model_name if model_name.startswith("@") else f"@{slug}/{model_name}"

    guard_llm = ChatOpenAI(
        api_key=api_key or "dummy",
        base_url="https://api.portkey.ai/v1",
        model=full_model,
        default_headers={
            "x-portkey-api-key": api_key,
            "x-portkey-virtual-key": slug,
        },
    )

    config = RailsConfig.from_content(colang_content=COLANG_CONTENT, yaml_content=YAML_CONTENT)
    _rails = LLMRails(config, llm=guard_llm)
    logfire.info(f"🛡️ NeMo Guardrails initialised ({model_name}).")


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate and deterministic safety filters.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    msg_lower = (message or "").lower().strip()

    # 1. Fast deterministic check for adversarial / jailbreak attempts
    for pattern in JAILBREAK_PATTERNS:
        if pattern in msg_lower:
            logfire.info(f"🛡️ Guardrails deterministic fired on '{pattern}' | query='{message[:80]}'")
            return True, REFUSE_JAILBREAK_MSG

    # 2. Fast deterministic check for off-topic non-contract requests
    for pattern in OFF_TOPIC_PATTERNS:
        if pattern in msg_lower:
            logfire.info(f"🛡️ Guardrails off-topic fired on '{pattern}' | query='{message[:80]}'")
            return True, REFUSE_OFF_TOPIC_MSG

    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        try:
            result = _rails.generate(messages=[{"role": "user", "content": message}])
            content = result.get("content", "") if isinstance(result, dict) else str(result)

            fired = any(indicator in content for indicator in RAIL_INDICATORS)
            if fired:
                logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
                return True, content

        except Exception as e:
            logfire.warning(f"NeMo rails evaluation warning: {e}")

        logfire.info("✅ Guardrails passed.")
        return False, None
