# Evaluation Pipeline — Changelog & Architecture

This document captures all changes made to the RAG evaluation pipeline, retrieval system, guardrails, and LLM routing across the optimization sessions.

---

## Changes Overview

### 1. Qdrant Vector Retrieval — Entity Filter Fix

**Problem:** Qdrant returned `400 Bad Request: Invalid json path: 'Document Name'` whenever `Document Name` was included in query filters. This caused silent fallback to unfiltered dense search across all ~19,000 vectors with no company-specific filtering.

**Root Cause:** Qdrant `FieldCondition(key=...)` does not support keys with spaces (e.g., `'Document Name'`, `'Parties-Answer'`). Only simple alphanumeric payload fields are valid.

**Fix (in `app/services/retrieval/vectordb_service.py`):**
- Restricted Qdrant filter keys strictly to valid payload fields: `source` and `Parties`
- Built `_build_distinctive_tokens()` to extract company-specific entity names from query text, stripping corporate stopwords (`Inc`, `Ltd`, `Corp`, `Agreement`, etc.)
- Added company synonym mappings (e.g., `Inmode → Invasix, InmodeLtd`)
- Removed generic contractor/partner names (`Flextronics`, `Xencor`) from synonym lists — these are counterparties shared across multiple contracts and cause cross-company noise

**Result:** 12/12 Golden Samples now retrieve their exact target contract document.

---

### 2. Context Window Expansion

**Changes (in `app/agents/nodes/retriever.py`):**
- Increased Qdrant candidate retrieval limit from default → `limit=40`
- Increased Jina AI reranker output from `top_n=5` → `top_n=8` chunks

**Why:** Ensures full legal clause definitions and operational provisions reach the LLM, rather than truncated fragments.

---

### 3. Guardrails — Deterministic Off-Topic Filtering

**Problem:** Off-topic queries like *"Tell me a funny joke"* were not being blocked by NeMo Guardrails and were leaking through to the LLM.

**Fix (in `app/guardrails/colang_rules.py` & `app/guardrails/rails.py`):**
- Added fast deterministic `OFF_TOPIC_PATTERNS` list (`joke`, `poem`, `weather`, `recipe`, etc.)
- Pattern check runs before NeMo colang evaluation for instant blocking

**Result:** Guardrails accuracy improved from 3/6 → **6/6 (Precision 1.00, Recall 1.00)**.

---

### 4. Multi-Portkey Key & Model Rotation

**Problem:** Single Portkey API key would hit Groq 429 rate limits after a few queries, causing pipeline failures during evaluation runs.

**Changes:**
- **`app/config.py`:** Added `PORTKEY_API_KEY_1`, `PORTKEY_API_KEY_2`, `PORTKEY_API_KEY_3`, `OPEN_ROUTER_KEY`, `OPENROUTER_API_KEY` to `Settings`
- **`app/gateway/client.py`:** Added `get_all_portkey_keys()` and `get_all_portkey_slugs()` helpers; updated `_make_headers()` to accept dynamic `api_key`
- **`app/agents/nodes/planner.py`:** Wrapped planner LLM invocation in multi-key, multi-slug, multi-model fallback loop
- **`app/agents/nodes/responder.py`:** Same fallback rotation applied to response generation

**Model Fallback Chain:**
```
openai/gpt-oss-120b → openai/gpt-oss-20b → llama-3.3-70b-versatile → llama-3.1-8b-instant
```

**Key Rotation:** Cycles through all 4 Portkey accounts × 2 slugs (`rag`, `brag`) × 4 models on 429 errors.

---

### 5. Judge LLM — OpenRouter Integration

**Problem:** The RAGAS evaluation judge was failing due to exhausted Vertex AI / OpenAI credits.

**Fix (in `evals/metrics.py`):**
- Updated `_build_judge()` to prioritize `OPEN_ROUTER_KEY` with `meta-llama/llama-3.3-70b-instruct` (max_tokens=2048) via `https://openrouter.ai/api/v1`
- Automatic fallback to Groq `llama-3.1-8b-instant` if OpenRouter is unreachable

---

### 6. Evaluation Cache — SHA-256 Hashing

**Problem:** Previous cache keys used `(label, question)` tuples, meaning stale results from old pipeline runs would persist even after retrieval improvements.

**Fix (in `evals/metrics.py`):**
- Replaced static cache keys with SHA-256 hashes of `(label, question, response, retrieved_contexts)`
- Any change to the RAG pipeline output automatically invalidates the cache

---

### 7. Streamlit Eval Pipeline — Force Refresh

**Problem:** Clicking "Run Live Pipeline (Step 2)" in Streamlit reused cached stale responses and guardrail results instead of querying the live backend.

**Fix (in `evals/app.py` & `evals/guardrails_eval.py`):**
- Added `force_refresh=True` parameter to `run_pipeline()` and `run_guardrails_eval()`
- Step 2 now always queries the live FastAPI backend, ensuring fresh responses

---

### 8. Headless Evaluation Script

**New file: `scripts/run_complete_eval_and_summary.py`**

Runs all 6 RAGAS metrics + guardrails evaluation headlessly (no Streamlit needed) and prints a final summary table. Useful for CI/CD or quick terminal-based benchmarking.

---

## Final Benchmark Results

| Metric | Score | Status |
|:---|:---:|:---:|
| **Tool Correctness** | 1.00 | ✅ Good |
| **🛡️ Guardrails Accuracy** | 6/6 (100%) | ✅ Good |
| **Context Recall** | 0.80 | ✅ Good |
| **Context Precision** | 0.70 | ⚠️ Fair |
| **Faithfulness** | 0.69 | ⚠️ Fair |
| **Answer Relevancy** | 0.51 | ❌ Poor |
| **Answer Correctness** | 0.33 | ❌ Poor |

### Known Limitations & Improvement Paths

- **Answer Correctness (0.33):** The golden `reference` answers are short (1-3 sentences) while the RAG produces detailed multi-paragraph responses. The F1 calculation penalizes extra correct statements as false positives. **Fix:** Expand reference answers to match RAG detail level.
- **Answer Relevancy (0.51):** Hedging language ("The excerpts do not contain...") generates reverse questions that diverge from the original query. **Fix:** Tighten LLM system prompts to eliminate disclaimers.

---

## Files Modified

| File | Change Summary |
|:---|:---|
| `app/config.py` | Added multi-Portkey keys, OpenRouter keys |
| `app/gateway/client.py` | Added `get_all_portkey_keys()`, `get_all_portkey_slugs()`, dynamic `api_key` in headers |
| `app/agents/nodes/planner.py` | Multi-key/slug/model fallback loop on 429 |
| `app/agents/nodes/responder.py` | Same fallback rotation for response generation |
| `app/services/retrieval/vectordb_service.py` | Fixed Qdrant filter keys, entity extraction, company synonyms |
| `app/agents/nodes/retriever.py` | Increased candidate limit (40) and reranker top_n (8) |
| `app/guardrails/colang_rules.py` | Added off-topic pattern definitions |
| `app/guardrails/rails.py` | Added deterministic pattern-match filter |
| `evals/metrics.py` | OpenRouter judge, SHA-256 cache hashing |
| `evals/app.py` | Added `force_refresh=True` to Step 2 |
| `evals/guardrails_eval.py` | Added `force_refresh` parameter support |
| `scripts/run_complete_eval_and_summary.py` | **[NEW]** Headless evaluation runner |
