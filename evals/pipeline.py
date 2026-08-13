"""
Phase 1 — Live Pipeline.
Calls the running FastAPI /query endpoint for each golden sample.
Captures: actual_response (truncated to 300 chars), actual_contexts (from sources),
and actual_tools_called (detected from thought_process).
"""

import copy
import json
import os
import time
import uuid

import logfire
import requests
from dotenv import load_dotenv

BACKEND_BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
API_URL = f"{BACKEND_BASE_URL}/query"
STATUS_URL_TEMPLATE = f"{BACKEND_BASE_URL}/query/status/{{job_id}}"
RESPONSE_TRUNCATE = 2000
BATCH_SIZE = 3  # 4 questions per batch (~8k-10k tokens, well under Groq 12K TPM limit)
BATCH_COOLDOWN = 120  # 65s cooldown to let the 1-minute sliding TPM window reset to 0
DELAY_BETWEEN_CALLS = 2  # seconds between calls within a batch
REQUEST_TIMEOUT = 180  # seconds
POLL_INTERVAL = 3  # seconds between job status polls
MAX_POLL_ATTEMPTS = 60  # ~3 minutes max wait per sample


def _get_headers() -> dict:
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)
    load_dotenv(override=True)
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("RAG_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def detect_tool(thought_process: list) -> str:
    """
    Maps the thought_process list from /query response to a tool name.
    Planner sets:  'Technical' / 'Vector Search' / 'Graph RAG' → retrieve_documents
                   'Conversational' / 'Memory'                 → direct_answer
                   'Guardrails Fired' / 'Blocked'              → guardrails
    """
    if not thought_process:
        return "unknown"
    joined = " ".join(thought_process).lower()
    if "guardrails" in joined or "blocked" in joined:
        return "guardrails"
    if any(
        k in joined for k in ["technical", "vector search", "search term", "context retrieved", "graph rag", "retriev"]
    ):
        return "retrieve_documents"
    if "conversational" in joined or "memory" in joined:
        return "direct_answer"
    return "retrieve_documents"


def _poll_for_result(job_id: str) -> dict:
    """Poll /query/status until the Celery job completes or times out."""
    url = STATUS_URL_TEMPLATE.format(job_id=job_id)
    headers = _get_headers()
    for attempt in range(MAX_POLL_ATTEMPTS):
        with logfire.span("🔄 Eval polling job", job_id=job_id, attempt=attempt + 1):
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

        status = data.get("status", "UNKNOWN")
        if status == "SUCCESS":
            return data.get("result", {})
        if status == "FAILURE":
            error = data.get("error", "unknown failure")
            raise RuntimeError(f"RAG job failed: {error}")
        time.sleep(POLL_INTERVAL)

    raise RuntimeError(f"Polling timed out for job {job_id}")


def _fetch_query_result(question: str, thread_id: str, max_retries: int = 5) -> dict:
    """Submit a query and return the final result (handling 429 rate limits, sync block + async jobs)."""
    for attempt in range(max_retries):
        resp = requests.post(
            API_URL,
            json={"q": question, "thread_id": thread_id},
            headers=_get_headers(),
            timeout=REQUEST_TIMEOUT,
        )

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 15))
            logfire.warning(
                f"⏳ Rate limited (429) on /query. Waiting {retry_after}s before retry (attempt {attempt + 1}/{max_retries})..."
            )
            time.sleep(retry_after)
            continue

        resp.raise_for_status()
        data = resp.json()

        # Guardrails can block synchronously without creating a job.
        if data.get("status") == "Blocked by guardrails." or "answer" in data:
            return data

        job_id = data.get("job_id")
        if not job_id:
            raise RuntimeError(f"Unexpected /query response: {data}")

        return _poll_for_result(job_id)

    raise RuntimeError("Exceeded maximum retries on /query due to 429 rate limits.")


def run_pipeline(
    golden_dataset: dict,
    progress_callback=None,
    checkpoint_path: str = None,
    force_refresh: bool = False,
) -> dict:
    """
    Enriches each rag_sample in golden_dataset with live API results.
    - If a sample already has a non-empty actual_response, it is SKIPPED and reused.
    - Immediately saves progress to disk after each new response so no progress is ever lost.
    - Groups newly-made queries in batches of BATCH_SIZE with a BATCH_COOLDOWN to stay within Groq 12K TPM limits.
    """
    dataset = copy.deepcopy(golden_dataset)
    samples = dataset["rag_samples"]
    n = len(samples)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")

    new_queries_executed = 0

    with logfire.span("🚀 Eval Phase 1 — Live Pipeline", total_samples=n):
        for i, sample in enumerate(samples):
            question = sample["question"]
            existing_response = sample.get("actual_response")
            sid = sample.get("id", i + 1)

            # Check if this sample was already processed in a previous run
            if (
                not force_refresh
                and existing_response
                and existing_response.strip()
                and len(existing_response.strip()) > 30
                and not existing_response.startswith("⚠️")
            ):
                logfire.info(f"⏩ Sample {sid} already has a response; skipping (cached).")
                if progress_callback:
                    progress_callback(i + 1, n, f"Sample {sid}: Cached", "cached")
                continue

            if progress_callback:
                progress_callback(i, n, question, "calling")

            with logfire.span(
                f"📤 Live Query {i + 1}/{n}",
                question=question[:80],
                domain=sample.get("domain", ""),
            ):
                try:
                    unique_thread = f"eval_{uuid.uuid4().hex[:8]}_{i}"
                    data = _fetch_query_result(question, thread_id=unique_thread)

                    raw_answer = data.get("answer") or ""
                    thought_process = data.get("thought_process") or []
                    sources = data.get("sources") or []

                    sample["actual_response"] = raw_answer[:RESPONSE_TRUNCATE]
                    sample["actual_contexts"] = sources[:5]
                    sample["actual_tools_called"] = [detect_tool(thought_process)]

                    logfire.info(
                        "✅ Response captured",
                        tool=sample["actual_tools_called"][0],
                        response_chars=len(raw_answer),
                        context_chunks=len(sources),
                    )

                except requests.exceptions.ConnectionError:
                    logfire.error("❌ Cannot reach FastAPI — is the app running on :8000?")
                    sample["actual_response"] = ""
                    sample["actual_contexts"] = sample.get("relevant_contexts", [])
                    sample["actual_tools_called"] = ["unknown"]

                except Exception as e:
                    logfire.error(f"❌ Query failed: {e}")
                    sample["actual_response"] = ""
                    sample["actual_contexts"] = sample.get("relevant_contexts", [])
                    sample["actual_tools_called"] = ["unknown"]

            # Save incrementally after EVERY new question
            save_results(dataset, checkpoint_path)
            new_queries_executed += 1

            if progress_callback:
                progress_callback(i, n, question, "done", sample["actual_response"])

            if i < n - 1:
                # If we completed a batch of newly executed queries, cooldown to reset TPM
                if new_queries_executed > 0 and new_queries_executed % BATCH_SIZE == 0:
                    batch_num = new_queries_executed // BATCH_SIZE
                    for remaining in range(BATCH_COOLDOWN, 0, -5):
                        if progress_callback:
                            msg = f"⏳ Batch {batch_num} complete ({new_queries_executed} queries done). Cooldown ({remaining}s) to reset Groq 12K TPM limit..."
                            progress_callback(i, n, msg, "cooldown", sample["actual_response"])
                        time.sleep(5)
                else:
                    time.sleep(DELAY_BETWEEN_CALLS)

    return dataset


def save_results(dataset: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)


def load_golden_dataset() -> dict:
    golden_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(golden_path) as f:
        return json.load(f)
