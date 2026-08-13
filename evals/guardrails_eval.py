"""
Guardrails binary evaluation.
Sends each test input to the live /query API and checks if the guardrail fired.
Classifies each result as TP / TN / FP / FN and computes precision + recall.
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


def _get_headers() -> dict:
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)
    load_dotenv(override=True)
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("RAG_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _is_blocked(response_json: dict) -> bool:
    tp = response_json.get("thought_process") or []
    status = response_json.get("status", "")
    return "Blocked by guardrails." in status or any("guardrails fired" in step.lower() for step in tp)


GUARDRAILS_BATCH_SIZE = 3
GUARDRAILS_BATCH_COOLDOWN = 120
DELAY_BETWEEN_CALLS = 2


def run_guardrails_eval(
    guardrails_samples: list,
    progress_callback=None,
    checkpoint_path: str = None,
    full_dataset: dict = None,
    force_refresh: bool = False,
) -> list:
    """
    Runs each guardrails test case against the live API.
    - If a test sample already has actual_blocked and result and force_refresh=False, it is SKIPPED and reused.
    - Immediately saves progress to disk after each new test.
    - Processes newly-run tests in batches of GUARDRAILS_BATCH_SIZE with a GUARDRAILS_BATCH_COOLDOWN between batches.
    """
    samples = copy.deepcopy(guardrails_samples)
    n = len(samples)
    new_tests_executed = 0

    if checkpoint_path is None:
        checkpoint_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")

    with logfire.span("🛡️ Eval — Guardrails Tests", total=n):
        for i, sample in enumerate(samples):
            if (
                not force_refresh
                and sample.get("actual_blocked") is not None
                and sample.get("result") in ["TP", "TN", "FP", "FN"]
            ):
                logfire.info(f"⏩ [Guardrail {sample['id']}] Reusing saved result: {sample['result']}")
                if progress_callback:
                    progress_callback(i, n, sample["input"], "cached")
                continue

            if progress_callback:
                progress_callback(i, n, sample["input"], "calling")

            with logfire.span(
                f"🛡️ Test {sample['id']}",
                input_text=sample["input"][:80],
                expected_blocked=sample["expected_blocked"],
            ):
                blocked = False
                for attempt in range(5):
                    try:
                        resp = requests.post(
                            API_URL,
                            json={"q": sample["input"], "thread_id": f"guardrail_{uuid.uuid4().hex[:8]}_{i}"},
                            headers=_get_headers(),
                            timeout=30,
                        )
                        if resp.status_code == 429:
                            time.sleep(15)
                            continue
                        resp.raise_for_status()
                        blocked = _is_blocked(resp.json())
                        break
                    except requests.exceptions.ConnectionError:
                        logfire.error("❌ Cannot reach FastAPI — is the app running on :8000?")
                        break
                    except Exception as e:
                        logfire.error(f"❌ Guardrails test error: {e}")
                        break

                expected = sample["expected_blocked"]
                sample["actual_blocked"] = blocked

                if expected and blocked:
                    sample["result"] = "TP"
                elif expected and not blocked:
                    sample["result"] = "FN"
                elif not expected and not blocked:
                    sample["result"] = "TN"
                else:
                    sample["result"] = "FP"

                new_tests_executed += 1

                # Incremental persistence to disk
                if checkpoint_path and os.path.exists(checkpoint_path):
                    try:
                        with open(checkpoint_path, "r") as f:
                            disk_data = json.load(f)
                        disk_data["guardrails_samples"] = samples
                        with open(checkpoint_path, "w") as f:
                            json.dump(disk_data, f, indent=2)
                    except Exception as e:
                        logfire.warning(f"Could not save guardrails checkpoint: {e}")

                logfire.info(
                    f"🛡️ {sample['result']}",
                    expected_blocked=expected,
                    actual_blocked=blocked,
                    input_preview=sample["input"][:60],
                )

            if i < n - 1:
                # If a batch of newly executed guardrails tests has finished, pause 120s to reset TPM
                if new_tests_executed > 0 and new_tests_executed % GUARDRAILS_BATCH_SIZE == 0:
                    batch_num = new_tests_executed // GUARDRAILS_BATCH_SIZE
                    for remaining in range(GUARDRAILS_BATCH_COOLDOWN, 0, -5):
                        if progress_callback:
                            msg = f"⏳ Guardrails batch {batch_num} complete ({new_tests_executed} tests done). Cooldown ({remaining}s) to reset TPM limit..."
                            progress_callback(i, n, msg, "cooldown")
                        time.sleep(5)
                else:
                    time.sleep(DELAY_BETWEEN_CALLS)

    return samples


def compute_guardrails_metrics(results: list) -> dict:
    tp = sum(1 for r in results if r["result"] == "TP")
    tn = sum(1 for r in results if r["result"] == "TN")
    fp = sum(1 for r in results if r["result"] == "FP")
    fn = sum(1 for r in results if r["result"] == "FN")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / len(results) if results else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "accuracy": round(accuracy, 3),
        "total": len(results),
        "correct": tp + tn,
    }
