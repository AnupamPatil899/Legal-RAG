# Comprehensive Technical Debugging & Resolution Report
## Enterprise RAG Evaluation Framework (RAGAS + Live Pipeline)

---

### Executive Summary

This report documents all technical issues, root cause investigations, and architectural solutions implemented while debugging and optimizing the **Enterprise RAG Evaluation Suite** ([CUDA_Rag/evals](file:///home/anupa/CUDA_Rag/evals)).

The evaluation suite tests an enterprise RAG pipeline against a SEC legal contract dataset using:
1. **Phase 1 — Live Pipeline Execution**: Submits golden queries to the running FastAPI `/query` endpoint, capturing live agent answers, retrieved context chunks, and tool invocations.
2. **Phase 2 — Guardrails Evaluation**: Validates prompt injection and safety filters.
3. **Phase 3 — LLM Judge Evaluation (RAGAS)**: Scores responses across 6 core metrics: *Faithfulness*, *Answer Relevancy*, *Context Precision*, *Context Recall*, *Answer Correctness*, and *Tool Correctness*.

---

## Summary of Issues & Resolution Matrix

| # | Issue / Error | Root Cause | Impact | Technical Resolution |
|---|---|---|---|---|
| **1** | `404 NOT_FOUND: Model is no longer available to new users` | Invalid model identifier (`gemini-2.5-flash-lite`) and Google AI Studio OpenAI-compatibility endpoint deprecation for new projects. | Judge LLM failed to initialize; evaluation blocked at Exp 1. | Updated provider factory to support OpenRouter, Vertex AI ADC OAuth, and Groq with valid model identifiers. |
| **2** | `429 RESOURCE_EXHAUSTED: Quota exceeded` on Vertex AI | Vertex AI default tier for `gemini-2.5-flash` in `us-central1` enforced restrictive per-minute RPM/TPM limits during batch scoring. | Evaluations stalled with 44s+ cooldown pauses. | Added `max_retries=5` with exponential backoff and integrated OpenRouter (`meta-llama/llama-3.3-70b-instruct`) as the high-throughput default. |
| **3** | `404 NOT_FOUND: No endpoints found for google/gemini-2.0-flash-001` | OpenRouter model route identifier was inactive or formatted differently. | Evaluation stopped on OpenRouter initialization. | Switched OpenRouter default to `meta-llama/llama-3.3-70b-instruct`, verified sub-second response times. |
| **4** | **Loss of Evaluation Progress on Crash / Rate Limit** | Metrics were held solely in memory during `run_all_metrics()`; interruptions lost all completed experiments. | Required re-running the entire 50-minute test suite from scratch. | Implemented **Dual-Level Incremental Checkpointing** (sample-level + stage-level disk persistence to `metric_results.json`). |
| **5** | **Tool Correctness = 0.0** | Tool classifier checked for `"intent: technical"` with a colon, but FastAPI planner logged `'Step 1: Evaluation Intent -> Technical'`. | All tool invocations fell back to `direct_answer` instead of `retrieve_documents`, resulting in 0% match. | Updated `detect_tool()` to recognize `"technical"`, `"vector search"`, and `"graph rag"`. |
| **6** | **Low Faithfulness (0.326) & Low Correctness (0.204)** | Static thread IDs (`eval_run_{i}`) reused chat history in Redis across multiple runs, causing the LLM to output conversational complaints rather than contract facts. Responses were also chopped at 300 chars. | RAG answers were meta-chat reprimands ("*We are having the same conversation again...*"), failing RAGAS grading. | Isolated all queries with UUID thread IDs (`eval_{uuid}_{i}`), increased truncate limit to 2000 chars, and added cache-clearing reset logic. |

---

## Detailed Technical Deep-Dives

---

### Incident 1: Google AI Studio & Vertex AI 404 / 429 Errors

#### Symptom
```text
API call failed on attempt 1: Error code: 404 - [{'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash-lite is no longer available to new users. Please update your code to use a newer model for the latest features and improvements. We recommend you to use the Interactions API (https://ai.google.dev/gemini-api/docs/migrate-to-interactions).', 'status': 'NOT_FOUND'}}]
```
Followed by Vertex AI 429 errors:
```text
Error code: 429 - [{'error': {'code': 429, 'message': 'Resource exhausted. Please try again later. Please refer to https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429 for more details.', 'status': 'RESOURCE_EXHAUSTED'}}]
```

#### Diagnostic Process
1. Inspected `evals/metrics.py` line 81 and found `"gemini-2.5-flash-lite"` was hardcoded for the Google AI Studio client.
2. Verified Google AI Studio endpoint capabilities via direct API calls: Google deprecated the OpenAI-compatibility bridge (`v1beta/openai/`) for new keys on Gemini 2.x models in favor of the **Interactions API**.
3. Tested Google Cloud Vertex AI ADC OAuth with `rag-project-anupam`. While authentication succeeded, `google/gemini-2.5-flash` in `us-central1` encountered strict project RPM quota limits during Ragas scoring bursts.

#### Solution
Refactored `_build_judge()` in `evals/metrics.py` into a multi-tiered provider architecture:
- **Tier 1 (Default)**: **OpenRouter** (`meta-llama/llama-3.3-70b-instruct`) — High throughput, robust JSON reasoning for Ragas evaluators, zero quota limits.
- **Tier 2**: **OpenAI** (`gpt-4o-mini`) if `OPENAI_API_KEY` is provided.
- **Tier 3**: **Groq** (`llama-3.3-70b-versatile`) if `GROQ_API_KEY` is active.
- **Tier 4**: **Google Cloud Vertex AI ADC OAuth** (`google/gemini-2.5-flash`) when configured.

---

### Incident 2: Evaluation State Loss & Checkpointing Architecture

#### Symptom
When an API rate limit, network timeout, or browser crash occurred during evaluation (e.g. at sample 18 of 20 in Experiment 4), all previously computed metrics in memory were lost, forcing the user to restart the entire test suite.

#### Diagnostic Process
- Streamlit's `st.session_state` was initialized with `metric_results = None` on every page refresh.
- `run_all_metrics()` did not write intermediate DataFrames to disk until all 6 experiments finished.

#### Solution
Implemented **Dual-Level Persistence** in `evals/metrics.py` and `evals/app.py`:

1. **Sample-Level Checkpointing**:
   Saves each sample score to disk cache immediately during batch scoring.
2. **Stage-Level Checkpointing**:
   After each of the 6 experiments (*Faithfulness*, *Answer Relevancy*, *Context Precision*, *Context Recall*, *Answer Correctness*, *Tool Correctness*), the resulting `pd.DataFrame` is immediately serialized to `evals/metric_results.json`.
3. **Session Auto-Restoration**:
   On Streamlit startup, `load_metrics_checkpoint()` checks for existing disk results and populates `st.session_state.metric_results` immediately.

---

### Incident 3: Tool Correctness = 0.0 & Degradation of Quality Scores

#### Symptom
```text
Faithfulness: 0.326 (❌ Poor)
Answer Relevancy: 0.301 (❌ Poor)
Answer Correctness: 0.204 (❌ Poor)
Tool Correctness: 0.000 (❌ Poor)
```

#### Diagnostic Process
1. Inspected the stored `actual_response` and `actual_tools_called` in `evals/golden_dataset.json`.
2. **Tool Mismatch Investigation**:
   - The golden dataset expected `['retrieve_documents']`.
   - The actual tools recorded were `['direct_answer']`.
   - Inspection of backend logs revealed thought processes:
     `'Step 1: Evaluation Intent -> Technical (Confidence: 1.00)'`
   - `evals/pipeline.py` checked for `"intent: technical"` (colon instead of arrow `->`), failing detection and defaulting to `direct_answer`.
3. **Conversation Memory Pollution**:
   - In `evals/pipeline.py`, queries were sent with `thread_id=f"eval_run_{i}"`.
   - Across repeated test runs, Redis conversation memory accumulated previous turns under `eval_run_0`, `eval_run_1`, etc.
   - When queried, the LLM agent believed it was in a continuous conversation with a repetitive user, returning:
     > *"It seems like we're having the same conversation again. I've already provided a detailed response to your question about the parties involved in the manufacturing agreement and their intentions multiple times. To recap..."*
   - Because the model generated conversational apologies instead of grounded contract citations, Ragas scored the answer as unfaithful and incorrect.
4. **Context & Response Truncation**:
   - `RESPONSE_TRUNCATE = 300` chopped responses mid-sentence, preventing complete semantic evaluation.

#### Solution
1. **Thread Isolation** (`evals/pipeline.py`):
   Generated unique UUID thread IDs (`eval_{uuid}_{i}`) for every query.
2. **Resilient Tool Classifier** (`evals/pipeline.py`):
   Recognizes `"technical"`, `"vector search"`, and `"graph rag"` from thought processes.
3. **Buffer Expansion**:
   Increased `RESPONSE_TRUNCATE` from 300 to 2000 characters.
4. **Clean Reset Pipeline** (`evals/app.py`):
   Configured the **"🔄 Reset & Re-run"** button to purge stale responses from `golden_dataset.json` and reset the metrics checkpoint.

---

## Verification & Validation

### Live Test 1: Single-Query Retrieval & Tool Detection
Tested a live query against the FastAPI backend with a unique thread ID:
* **Query**: *"What standard Contractor need for this Agreement?"*
* **Thought Process**: `['Step 1: Evaluation Intent -> Technical (Confidence: 1.00)', 'Step 2: Vector Search -> Found 3 chunks', 'Step 3: Graph RAG -> Found 0 entities']`
* **Tool Detected**: `retrieve_documents` ✅
* **Answer**: *"Based on the provided context, the Contractor is required to apply for and receive the ISO13488 standard for the production facility..."* ✅
* **Reference Match**: Exact match with golden dataset reference answer.

### Live Test 2: Dual-Level Checkpoint Verification
* Verified sample-level cache writing, stage-level serialization to `metric_results.json`, and instant resume capability (`ALL CHECKPOINT TESTS PASSED!`).
