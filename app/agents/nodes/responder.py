import logfire
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from app.agents.state import AgentState
from app.config import settings
from app.gateway import extract_cache_status


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Portkey client (not LangChain) so we can read the
    x-portkey-cache-status response header and surface Cache: Hit in the UI.
    """
    query = state["current_query"]

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt = f"""
        You are a Senior Legal Contract Specialist.
        Answer the user's question accurately and directly using ONLY the CONTRACT CONTEXT provided below.

        GUIDELINES:
        1. Base your answer strictly on the relevant clauses, numbers, standards (e.g. ISO standards), party names, and terms in the CONTRACT CONTEXT.
        2. Identify the contract and parties when stating obligations or terms.
        3. Be clear, direct, and concise. Do not include unnecessary disclaimers if the answer is present in the context.
        4. If the context does not contain enough information to answer the question, state that clearly.

        CONTRACT CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    with logfire.span("✍️ LLM Synthesis"):
        try:
            response = _generate_response(prompt)
            content = response.choices[0].message.content
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                logfire.info("⚡ Gateway Cache Hit — response served from Portkey cache.")
                plan_update = state["plan"] + ["Cache: Hit ⚡"]
                status = "Cache hit — instant response."
            else:
                logfire.info("✅ Response synthesised via LLM.")
                plan_update = state["plan"]
                status = "Response generated."

            return {
                "final_answer": content,
                "status": status,
                "plan": plan_update,
                "messages": [{"role": "assistant", "content": content}],
            }

        except Exception as e:
            logfire.error(f"LLM Generation failed after retries: {e}")
            raise e


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    reraise=True,
    before_sleep=before_sleep_log(logfire, "warning"),
)
def _generate_response(prompt: str):
    """Call the LLM gateway with multi-key, multi-slug, and multi-model fallbacks."""
    from openai import OpenAI

    from app.gateway.client import PORTKEY_GATEWAY_URL, _make_headers, get_all_portkey_keys, get_all_portkey_slugs

    keys = get_all_portkey_keys()
    slugs = get_all_portkey_slugs()

    primary_model = (settings.LLM_MODEL or "openai/gpt-oss-120b").strip()
    candidate_models = [primary_model]
    for alt in ("openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"):
        if alt not in candidate_models:
            candidate_models.append(alt)

    last_err = None
    for api_key in keys:
        for slug in slugs:
            headers = _make_headers("responder", api_key=api_key)
            client = OpenAI(api_key=api_key, base_url=PORTKEY_GATEWAY_URL, default_headers=headers)
            for model_name in candidate_models:
                try:
                    full_model = model_name if model_name.startswith("@") else f"@{slug}/{model_name}"
                    return client.chat.completions.create(
                        model=full_model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                except Exception as e:
                    last_err = e
                    err_str = str(e).lower()
                    if "429" in err_str or "rate limit" in err_str or "quota" in err_str:
                        logfire.warning(
                            f"⚠️ Model '{model_name}' on key ...{api_key[-4:]} rate-limited. Trying fallback..."
                        )
                        continue
                    else:
                        logfire.warning(f"⚠️ Portkey '{full_model}' failed: {e}. Trying next...")
                        continue

    raise last_err or RuntimeError("All Portkey keys, slugs and models failed.")
