"""
Phase 2 — RAGAS + Tool Correctness metrics.
Supports both Google Cloud Vertex AI (Gemini 2.5 Flash) and OpenAI Judge models.
All LLM-based metrics evaluate the RAG pipeline responses against reference data.
"""

import asyncio
import hashlib
import json
import os

import httpx
import logfire
import pandas as pd
from dotenv import load_dotenv
from google.auth import default
from google.auth.transport.requests import Request
from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv("/home/anupa/RAG_evals/.env")
load_dotenv()


PROJECT_ID = os.getenv("GCP_PROJECT_ID", "rag-project-anupam")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
VERTEX_OPENAI_URL = (
    f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/openapi"
)

OPENAI_BASE_URL = "https://api.openai.com/v1"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "google/gemini-2.5-flash")
COOLDOWN_STANDARD = 3  # seconds between experiments
COOLDOWN_MINI = 1  # seconds between batches
GENERAL_BATCH_SIZE = 4  # concurrent evaluations per batch
CONTEXT_TRUNCATE = 500  # chars per context chunk
CONTEXT_LIMIT = 3  # number of context chunks passed to RAGAS per sample


class DynamicGoogleOAuth(httpx.Auth):
    """Dynamic httpx Auth handler that refreshes Google OAuth tokens automatically."""

    def __init__(self, credentials):
        self.credentials = credentials

    def auth_flow(self, request):
        if not self.credentials.valid or self.credentials.expired:
            self.credentials.refresh(Request())
        request.headers["Authorization"] = f"Bearer {self.credentials.token}"
        yield request

    async def async_auth_flow(self, request):
        if not self.credentials.valid or self.credentials.expired:
            self.credentials.refresh(Request())
        request.headers["Authorization"] = f"Bearer {self.credentials.token}"
        yield request


def _build_judge():
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)
    load_dotenv(override=True)

    groq_key = os.getenv("GROQ_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_KEY") or os.getenv("OPEN_ROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    use_vertex = os.getenv("USE_VERTEX_JUDGE", "false").lower() in ("true", "1")

    if openrouter_key:
        # Option A: OpenRouter API (meta-llama/llama-3.3-70b-instruct)
        model_name = os.getenv("JUDGE_MODEL", "meta-llama/llama-3.3-70b-instruct")
        logfire.info(f"🏛️ Initializing Judge LLM via OpenRouter API ({model_name})")
        client = AsyncOpenAI(
            api_key=openrouter_key.strip(),
            base_url="https://openrouter.ai/api/v1",
            max_retries=5,
        )
        llm = llm_factory(
            model_name,
            provider="openai",
            client=client,
            max_tokens=2048,
        )
    elif groq_key and not use_vertex:
        # Option B: Groq API fallback
        judge_model = os.getenv("JUDGE_MODEL", "llama-3.1-8b-instant")
        logfire.info(f"🏛️ Initializing Judge LLM via Groq API ({judge_model})")
        client = AsyncOpenAI(
            api_key=groq_key.strip(),
            base_url="https://api.groq.com/openai/v1",
            max_retries=3,
        )
        llm = llm_factory(
            judge_model,
            provider="openai",
            client=client,
            max_tokens=2048,
        )
    elif openai_key and openai_key.startswith("sk-") and not openai_key.startswith("sk-or-"):
        # Option C: Standard OpenAI endpoint
        logfire.info("🏛️ Initializing Judge LLM via OpenAI API Key (gpt-4o-mini)")
        client = AsyncOpenAI(api_key=openai_key.strip(), max_retries=5)
        llm = llm_factory("gpt-4o-mini", provider="openai", client=client, max_tokens=2048)
    else:
        # Option D: Standard Vertex AI ADC OAuth
        model_name = os.getenv("JUDGE_MODEL", "google/gemini-2.5-flash")
        logfire.info(
            f"🏛️ Initializing Judge LLM via Google Cloud Vertex AI ADC OAuth ({model_name}) [Project: {PROJECT_ID}]"
        )
        credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(Request())
        auth_handler = DynamicGoogleOAuth(credentials)
        async_http = httpx.AsyncClient(auth=auth_handler, timeout=120.0)
        openai_client = AsyncOpenAI(
            base_url=VERTEX_OPENAI_URL,
            api_key="dynamic-google-token",
            http_client=async_http,
            max_retries=5,
        )
        llm = llm_factory(
            model_name,
            provider="openai",
            client=openai_client,
            max_tokens=8192,
        )

    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        use_api=False,
    )

    return llm, embeddings


async def _cooldown(seconds: int, label: str, status_cb=None):
    if seconds <= 0:
        return
    msg = f"⏳ {seconds}s cooldown after {label}..."
    if status_cb:
        status_cb(msg)
    await asyncio.sleep(seconds)
    if status_cb:
        status_cb("✅ Ready — starting next experiment.")


def _prepare_samples(golden_dataset: dict) -> list:
    """
    Filters golden_dataset to samples that have a non-empty actual_response.
    Truncates contexts to CONTEXT_TRUNCATE chars and limits to CONTEXT_LIMIT chunks.
    """
    valid = []
    for s in golden_dataset["rag_samples"]:
        response = s.get("actual_response", "").strip()
        if not response:
            continue
        raw_contexts = s.get("actual_contexts") or s.get("relevant_contexts") or []
        contexts = [c[:CONTEXT_TRUNCATE] for c in raw_contexts[:CONTEXT_LIMIT]]
        valid.append({**s, "actual_contexts": contexts})
    return valid


class ScoreWrapper:
    def __init__(self, val: float = 0.0):
        self.value = val


METRICS_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "metric_results.json")


def _load_checkpoint_data(path: str = METRICS_CHECKPOINT_PATH) -> dict:
    if not os.path.exists(path):
        return {"experiments": {}, "sample_cache": {}}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {"experiments": {}, "sample_cache": {}}
        if "experiments" not in data:
            data = {"experiments": data, "sample_cache": {}}
        if "sample_cache" not in data:
            data["sample_cache"] = {}
        return data
    except Exception as e:
        logfire.warning(f"Could not read metrics checkpoint: {e}")
        return {"experiments": {}, "sample_cache": {}}


def _save_checkpoint_data(data: dict, path: str = METRICS_CHECKPOINT_PATH) -> None:
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logfire.warning(f"Could not write metrics checkpoint: {e}")


def load_metrics_checkpoint(path: str = METRICS_CHECKPOINT_PATH) -> dict:
    """Returns dict of metric_name -> DataFrame from saved checkpoint."""
    raw = _load_checkpoint_data(path)
    dfs = {}
    for k, v in raw.get("experiments", {}).items():
        if isinstance(v, list) and len(v) > 0:
            dfs[k] = pd.DataFrame(v)
    return dfs


def clear_metrics_checkpoint(path: str = METRICS_CHECKPOINT_PATH) -> None:
    """Removes metrics checkpoint file."""
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def _score_df(metric_key: str, samples: list, scores) -> pd.DataFrame:
    rows = []
    for s, r in zip(samples, scores):
        val = getattr(r, "value", r)
        try:
            val_float = round(float(val), 3)
        except (ValueError, TypeError):
            val_float = 0.0
        rows.append({"question": s["question"][:65], metric_key: val_float})
    return pd.DataFrame(rows)


async def _batched_score(
    metric,
    inputs: list,
    samples: list,
    status_cb=None,
    label: str = "",
    checkpoint_path: str = METRICS_CHECKPOINT_PATH,
    checkpoint_data: dict = None,
) -> list:
    """Runs scoring with sample-level caching, retry, and immediate persistence."""
    all_scores = []
    if checkpoint_data is None:
        checkpoint_data = _load_checkpoint_data(checkpoint_path)
    sample_cache = checkpoint_data.setdefault("sample_cache", {})

    items_and_samples = list(zip(inputs, samples))
    batches = [
        items_and_samples[i : i + GENERAL_BATCH_SIZE] for i in range(0, len(items_and_samples), GENERAL_BATCH_SIZE)
    ]

    for b_idx, batch in enumerate(batches):
        if b_idx > 0:
            await _cooldown(COOLDOWN_MINI, f"{label} batch {b_idx}", status_cb)

        for item, sample in batch:
            resp_str = str(sample.get("actual_response", "")).strip()
            ctx_str = str(sample.get("actual_contexts", []))[:300]
            sig_payload = f"{label}::{sample['question'].strip()}::{resp_str}::{ctx_str}"
            cache_hash = hashlib.sha256(sig_payload.encode()).hexdigest()[:16]
            cache_key = f"{label}::{sample['question'].strip()[:40]}::{cache_hash}"

            # Check if this exact sample was already scored for this metric with this exact response/context
            if cache_key in sample_cache:
                cached_val = sample_cache[cache_key]
                all_scores.append(ScoreWrapper(cached_val))
                continue

            res = None
            for attempt in range(5):
                try:
                    res = await metric.ascore(**item)
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    if (
                        "429" in err_str
                        or "resource_exhausted" in err_str
                        or "quota" in err_str
                        or "exceeded" in err_str
                        or "retry in" in err_str
                    ):
                        wait_sec = 22 * (attempt + 1)
                        logfire.warning(
                            f"⏳ Judge rate limit in {label}. Pausing {wait_sec}s for quota reset (attempt {attempt + 1}/5)..."
                        )
                        if status_cb:
                            status_cb(f"⏳ Judge rate limit in {label}. Pausing {wait_sec}s for quota window reset...")
                        await asyncio.sleep(wait_sec)
                        continue
                    else:
                        logfire.warning(f"⚠️ Scoring error on sample in {label}: {e}")
                        break

            if res is None:
                res = ScoreWrapper(0.0)
                val_float = 0.0
            else:
                val = getattr(res, "value", res)
                try:
                    val_float = round(float(val), 3)
                except (ValueError, TypeError):
                    val_float = 0.0

            # Save sample score to cache immediately on disk
            sample_cache[cache_key] = val_float
            _save_checkpoint_data(checkpoint_data, checkpoint_path)

            all_scores.append(res)
            await asyncio.sleep(1.5)  # Pace between judge calls

    return all_scores


async def run_all_metrics(
    golden_dataset: dict,
    status_cb=None,
    checkpoint_path: str = METRICS_CHECKPOINT_PATH,
    force_recompute: bool = False,
) -> dict:
    """
    Runs all 6 experiments. Incrementally saves results after EVERY sample and stage to disk.
    If interrupted or crashed, automatically resumes from the last completed sample/stage.
    """
    judge_llm, ragas_embeddings = _build_judge()
    samples = _prepare_samples(golden_dataset)

    if not samples:
        raise ValueError("No samples with actual_response found. Run Phase 1 first.")

    if force_recompute:
        clear_metrics_checkpoint(checkpoint_path)
        checkpoint_data = {"experiments": {}, "sample_cache": {}}
    else:
        checkpoint_data = _load_checkpoint_data(checkpoint_path)

    results = {}
    for k, v in checkpoint_data.get("experiments", {}).items():
        if isinstance(v, list) and len(v) == len(samples):
            results[k] = pd.DataFrame(v)

    with logfire.span("🧪 Eval Phase 2 — All Metrics", total_samples=len(samples)):
        # ── Exp 1: Faithfulness ───────────────────────────────────────────────
        if "faithfulness" in results:
            logfire.info("⏩ Reusing saved Faithfulness results from checkpoint")
            if status_cb:
                status_cb(f"⏩ [1/6] Reusing saved Faithfulness ({len(samples)} samples)...")
        else:
            if status_cb:
                status_cb(f"🧪 Exp 1/6 — Faithfulness ({len(samples)} samples)...")
            with logfire.span("🧪 Exp 1 — Faithfulness"):
                inputs = [
                    {
                        "user_input": s["question"],
                        "response": s["actual_response"],
                        "retrieved_contexts": s["actual_contexts"],
                    }
                    for s in samples
                ]
                scores = await _batched_score(
                    Faithfulness(llm=judge_llm),
                    inputs,
                    samples,
                    status_cb,
                    "Faithfulness",
                    checkpoint_path=checkpoint_path,
                    checkpoint_data=checkpoint_data,
                )
                df = _score_df("faithfulness", samples, scores)
                results["faithfulness"] = df
                checkpoint_data.setdefault("experiments", {})["faithfulness"] = df.to_dict(orient="records")
                _save_checkpoint_data(checkpoint_data, checkpoint_path)
                logfire.info("🧪 Faithfulness done", avg=round(df["faithfulness"].mean(), 3))

            await _cooldown(COOLDOWN_STANDARD, "Faithfulness", status_cb)

        # ── Exp 2: Answer Relevancy ───────────────────────────────────────────
        if "answer_relevancy" in results:
            logfire.info("⏩ Reusing saved Answer Relevancy results from checkpoint")
            if status_cb:
                status_cb(f"⏩ [2/6] Reusing saved Answer Relevancy ({len(samples)} samples)...")
        else:
            if status_cb:
                status_cb(f"🧪 Exp 2/6 — Answer Relevancy ({len(samples)} samples)...")
            with logfire.span("🧪 Exp 2 — Answer Relevancy"):
                inputs = [{"user_input": s["question"], "response": s["actual_response"]} for s in samples]
                scores = await _batched_score(
                    AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings),
                    inputs,
                    samples,
                    status_cb,
                    "Answer Relevancy",
                    checkpoint_path=checkpoint_path,
                    checkpoint_data=checkpoint_data,
                )
                df = _score_df("answer_relevancy", samples, scores)
                results["answer_relevancy"] = df
                checkpoint_data.setdefault("experiments", {})["answer_relevancy"] = df.to_dict(orient="records")
                _save_checkpoint_data(checkpoint_data, checkpoint_path)
                logfire.info("🧪 Answer Relevancy done", avg=round(df["answer_relevancy"].mean(), 3))

            await _cooldown(COOLDOWN_STANDARD, "Answer Relevancy", status_cb)

        # ── Exp 3: Context Precision ──────────────────────────────────────────
        if "context_precision" in results:
            logfire.info("⏩ Reusing saved Context Precision results from checkpoint")
            if status_cb:
                status_cb(f"⏩ [3/6] Reusing saved Context Precision ({len(samples)} samples)...")
        else:
            if status_cb:
                status_cb(f"🧪 Exp 3/6 — Context Precision ({len(samples)} samples)...")
            with logfire.span("🧪 Exp 3 — Context Precision"):
                inputs = [
                    {
                        "user_input": s["question"],
                        "reference": s["reference"],
                        "retrieved_contexts": s["actual_contexts"],
                    }
                    for s in samples
                ]
                scores = await _batched_score(
                    ContextPrecision(llm=judge_llm),
                    inputs,
                    samples,
                    status_cb,
                    "Context Precision",
                    checkpoint_path=checkpoint_path,
                    checkpoint_data=checkpoint_data,
                )
                df = _score_df("context_precision", samples, scores)
                results["context_precision"] = df
                checkpoint_data.setdefault("experiments", {})["context_precision"] = df.to_dict(orient="records")
                _save_checkpoint_data(checkpoint_data, checkpoint_path)
                logfire.info("🧪 Context Precision done", avg=round(df["context_precision"].mean(), 3))

            await _cooldown(COOLDOWN_STANDARD, "Context Precision", status_cb)

        # ── Exp 4: Context Recall ─────────────────────────────────────────────
        if "context_recall" in results:
            logfire.info("⏩ Reusing saved Context Recall results from checkpoint")
            if status_cb:
                status_cb(f"⏩ [4/6] Reusing saved Context Recall ({len(samples)} samples)...")
        else:
            if status_cb:
                status_cb(f"🧪 Exp 4/6 — Context Recall ({len(samples)} samples)...")
            with logfire.span("🧪 Exp 4 — Context Recall"):
                inputs = [
                    {
                        "user_input": s["question"],
                        "reference": s["reference"],
                        "retrieved_contexts": s["actual_contexts"],
                    }
                    for s in samples
                ]
                scores = await _batched_score(
                    ContextRecall(llm=judge_llm),
                    inputs,
                    samples,
                    status_cb,
                    "Context Recall",
                    checkpoint_path=checkpoint_path,
                    checkpoint_data=checkpoint_data,
                )
                df = _score_df("context_recall", samples, scores)
                results["context_recall"] = df
                checkpoint_data.setdefault("experiments", {})["context_recall"] = df.to_dict(orient="records")
                _save_checkpoint_data(checkpoint_data, checkpoint_path)
                logfire.info("🧪 Context Recall done", avg=round(df["context_recall"].mean(), 3))

            await _cooldown(COOLDOWN_STANDARD, "Context Recall", status_cb)

        # ── Exp 5: Answer Correctness ─────────────────────────────────────────
        if "answer_correctness" in results:
            logfire.info("⏩ Reusing saved Answer Correctness results from checkpoint")
            if status_cb:
                status_cb(f"⏩ [5/6] Reusing saved Answer Correctness ({len(samples)} samples)...")
        else:
            if status_cb:
                status_cb(f"🧪 Exp 5/6 — Answer Correctness ({len(samples)} samples)...")
            with logfire.span("🧪 Exp 5 — Answer Correctness"):
                inputs = [
                    {
                        "user_input": s["question"],
                        "response": s["actual_response"],
                        "reference": s["reference"],
                    }
                    for s in samples
                ]
                all_scores = await _batched_score(
                    AnswerCorrectness(llm=judge_llm, embeddings=ragas_embeddings),
                    inputs,
                    samples,
                    status_cb,
                    "Answer Correctness",
                    checkpoint_path=checkpoint_path,
                    checkpoint_data=checkpoint_data,
                )
                df = _score_df("answer_correctness", samples, all_scores)
                results["answer_correctness"] = df
                checkpoint_data.setdefault("experiments", {})["answer_correctness"] = df.to_dict(orient="records")
                _save_checkpoint_data(checkpoint_data, checkpoint_path)
                logfire.info("🧪 Answer Correctness done", avg=round(df["answer_correctness"].mean(), 3))

            await _cooldown(COOLDOWN_STANDARD, "Answer Correctness", status_cb)

        # ── Exp 6: Tool Correctness (no LLM — Jaccard) ───────────────────────
        if "tool_correctness" in results:
            logfire.info("⏩ Reusing saved Tool Correctness results from checkpoint")
            if status_cb:
                status_cb("⏩ [6/6] Reusing saved Tool Correctness...")
        else:
            if status_cb:
                status_cb("⚡ Exp 6/6 — Tool Correctness (zero LLM calls)...")
            with logfire.span("🧪 Exp 6 — Tool Correctness"):
                tool_rows = []
                for s in samples:
                    called = set(s.get("actual_tools_called") or [])
                    expected = set(s.get("expected_tools") or [])
                    union = len(called | expected)
                    score = len(called & expected) / union if union > 0 else 0.0
                    tool_rows.append({"question": s["question"][:65], "tool_correctness": round(score, 3)})
                df = pd.DataFrame(tool_rows)
                results["tool_correctness"] = df
                checkpoint_data.setdefault("experiments", {})["tool_correctness"] = df.to_dict(orient="records")
                _save_checkpoint_data(checkpoint_data, checkpoint_path)
                logfire.info("🧪 Tool Correctness done", avg=round(df["tool_correctness"].mean(), 3))

        if status_cb:
            status_cb("✅ All 6 experiments complete!")

    return results
