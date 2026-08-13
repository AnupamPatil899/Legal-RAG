import json
import re

import logfire

from app.agents.state import AgentState
from app.config import settings
from app.gateway import get_langchain_llm

# Portkey-backed LLM: fallback + cache + retry — same .invoke() interface as ChatOpenAI
llm = get_langchain_llm(feature="planner")


def _extract_json(text: str) -> dict | None:
    """Safely extracts JSON dict from markdown blocks or raw text."""
    try:
        # Try direct parse
        return json.loads(text.strip())
    except Exception:
        pass

    # Try extracting inside code block ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # Try finding any outermost JSON object { ... }
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    return None


def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation
    and extracts structured entities (company, document type, clause keywords)
    for high-precision metadata and hybrid search.
    """
    # Get the conversation history (excluding the latest message)
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
    You are an intelligent Assistant Planner for an SEC Legal Contract RAG system.
    Analyze the conversation history and the latest user message.

    CONVERSATION HISTORY:
    {history}

    LATEST MESSAGE:
    "{user_message}"

    Task:
    1. If the latest message is a greeting (hi, hello, thanks) or a conversational question that can be answered using ONLY the conversation history above (e.g., "what is my name"), respond with 'CONVERSATIONAL'.
    2. If the user asks a domain-specific question (e.g., regarding legal contracts, clauses, parties, liabilities, standard terms, or enterprise SEC filings):
       Output a JSON object with:
       - "intent": "TECHNICAL"
       - "search_query": "Refined dense search query preserving exact contract names, legal terms, acronyms (ISO, LLI, EOQ, 8-K, 10-K), and parties"
       - "company": "Target company / party name if mentioned (e.g. 'Inmode', 'Invasix', 'Flextronics', 'Exact Sciences', 'Upjohn', 'Bioeq'), or empty string '' if none"
       - "doc_type": "Agreement / filing type if mentioned (e.g. 'Turn-Key Manufacturing Agreement', '8-K', 'License Agreement', 'Supply Agreement'), or empty string ''"
       - "keywords": ["List", "of", "domain", "keywords", "acronyms"]

    Output ONLY 'CONVERSATIONAL' or the JSON object.
    """

    with logfire.span("🧠 Planner Decision"):
        decision = ""
        from langchain_openai import ChatOpenAI

        from app.gateway.client import PORTKEY_GATEWAY_URL, _make_headers, get_all_portkey_keys, get_all_portkey_slugs

        keys = get_all_portkey_keys()
        slugs = get_all_portkey_slugs()
        primary_model = (settings.LLM_MODEL or "openai/gpt-oss-120b").strip()
        candidate_models = [primary_model]
        for alt in ("openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"):
            if alt not in candidate_models:
                candidate_models.append(alt)

        last_planner_err = None
        for api_key in keys:
            if decision:
                break
            for slug in slugs:
                if decision:
                    break
                for m in candidate_models:
                    try:
                        full_m = m if m.startswith("@") else f"@{slug}/{m}"
                        p_llm = ChatOpenAI(
                            api_key=api_key,
                            base_url=PORTKEY_GATEWAY_URL,
                            model=full_m,
                            default_headers=_make_headers("planner", api_key=api_key),
                            max_retries=1,
                        )
                        decision = p_llm.invoke(prompt).content.strip()
                        break
                    except Exception as e:
                        last_planner_err = e
                        err_str = str(e).lower()
                        if "429" in err_str or "rate limit" in err_str:
                            logfire.warning(
                                f"⚠️ Planner model '{m}' on key ...{api_key[-4:]} rate-limited. Trying fallback..."
                            )
                            continue
                        else:
                            logfire.warning(f"⚠️ Planner model '{m}' failed: {e}. Trying fallback...")
                            continue

        if not decision and last_planner_err:
            raise last_planner_err

        logfire.info(f"Planner raw output: {decision}")

    if decision == "CONVERSATIONAL" or "CONVERSATIONAL" in decision.upper() and "{" not in decision:
        return {
            "current_query": "CONVERSATIONAL",
            "extracted_entities": {},
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"],
        }

    parsed = _extract_json(decision)
    if parsed and isinstance(parsed, dict):
        search_query = parsed.get("search_query") or user_message
        company = parsed.get("company") or ""
        doc_type = parsed.get("doc_type") or ""
        keywords = parsed.get("keywords") or []
        entities = {
            "company": company,
            "doc_type": doc_type,
            "keywords": keywords,
        }
        plan_desc = [
            "Intent: Technical",
            f"Search Term: {search_query}",
        ]
        if company:
            plan_desc.append(f"Target Entity: {company}")
        if doc_type:
            plan_desc.append(f"Doc Type: {doc_type}")

        return {
            "current_query": search_query,
            "extracted_entities": entities,
            "status": f"Technical research needed. Searching for: {search_query}",
            "plan": plan_desc,
        }

    # Fallback to plain string if JSON parse fails
    clean_query = decision.replace("```json", "").replace("```", "").strip()
    return {
        "current_query": clean_query or user_message,
        "extracted_entities": {},
        "status": f"Technical research needed. Searching for: {clean_query or user_message}",
        "plan": ["Intent: Technical", f"Search Term: {clean_query or user_message}"],
    }
