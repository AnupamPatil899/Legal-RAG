# ClauseAI — Agentic Legal Document Intelligence Engine

A production-grade, enterprise-level **legal contract RAG system** built with **LangGraph**, **Portkey LLM Gateway**, **Qdrant Vector Database**, and **Jina AI Embeddings/Reranker**. ClauseAI retrieves and reasons over SEC contract filings, distinguishes between technical "True Data" and random "Noisy Data" using semantic re-ranking, history-aware planning, and NeMo Guardrails for input/output safety.

## 🚀 Live Demo & API Endpoints

- **Frontend Streamlit UI**: [https://rag-ui-976087180091.asia-south1.run.app/](https://rag-ui-976087180091.asia-south1.run.app/)
- **Backend FastAPI Service**: [https://rag-api-976087180091.asia-south1.run.app/](https://rag-api-976087180091.asia-south1.run.app/)


---

## Key Features

- **Agentic Intelligence**: LangGraph for cyclic reasoning, multi-step planning, and conversation memory.
- **Guardrails**: NeMo Guardrails + deterministic off-topic pattern filtering blocks jailbreak, injection, and irrelevant inputs before any retrieval.
- **LLM Gateway**: Portkey routes all LLM calls with automatic multi-key, multi-model fallback across 4 API accounts and a cascading model chain (`gpt-oss-120b → gpt-oss-20b → llama-3.3-70b → llama-3.1-8b`).
- **Enterprise Search**: **Qdrant Vector Database** with entity-aware filtering (company name extraction + synonym mapping) + Jina AI Reranker API for semantic reranking.
- **Jina AI Embeddings**: `jina-embeddings-v3` (1024-dim) via Jina API, with local `mxbai-embed-large-v1` fallback.
- **Local Document Parsing**: PDF, HTML, TXT, DOCX, PPTX parsed entirely on-device — no external OCR service.
- **Observability**: Full trace nesting with **Pydantic Logfire** and **LangSmith** across every agent node.
- **Metrics**: Prometheus `/metrics` endpoint with custom RAG and guardrails counters.
- **Synchronous `/query`**: The LangGraph pipeline runs directly inside the `/query` endpoint and returns the final answer.
- **API Security & Rate Limiting**: Bearer-token authentication via GCP Secret Manager and Redis-backed (or in-memory) rate limiting.
- **Evaluation Suite**: RAGAS-powered eval pipeline (6 metrics + guardrails accuracy) with Streamlit demo app, headless CLI runner, and OpenRouter 70B judge.

---

## Agent Intelligence Flow

```mermaid
graph TD
    User((User)) --> UI[Streamlit Chat UI]
    UI -->|Bearer Auth| API[FastAPI /query]

    subgraph Security Layer
        API --> Guard{NeMo Guardrails<br/>+ Off-Topic Patterns}
    end

    Guard -->|"🛡️ Blocked"| BlockResp[Guardrail Response]
    BlockResp --> UI

    Guard -->|"✅ Pass"| Planner

    subgraph LangGraph Agent
        Planner{Planner Node<br/>Intent Classification}
        Planner -->|"Conversational /<br/>Clarification"| Responder
        Planner -->|"Technical /<br/>Document Query"| Retriever

        subgraph Retrieval Pipeline
            Retriever[Retriever Node] -->|Entity-Filtered<br/>Vector Search| Qdrant[(Qdrant Cloud<br/>19K+ Vectors)]
            Qdrant -->|Top 40 Candidates| Reranker[Jina AI Reranker v3<br/>Semantic Reranking]
            Reranker -->|Top 8 Chunks| Responder
        end

        Responder[Responder Node<br/>LLM Answer Generation]
    end

    subgraph LLM Gateway
        Responder -.->|Multi-Key Rotation| Portkey[Portkey Gateway<br/>4 Keys × 2 Slugs × 4 Models]
        Planner -.->|Rate Limit Fallback| Portkey
    end

    Responder --> UI
    Responder -.->|Conversation History| Memory[(LangGraph<br/>Postgres Checkpointer)]

    subgraph Evaluation Suite
        Judge[OpenRouter 70B Judge] -.-> Metrics[RAGAS Metrics<br/>6 Experiments]
        Metrics -.-> Dashboard[Streamlit Eval Dashboard<br/>3-Tab UI]
    end

    style Guard fill:#ff6b6b,stroke:#c0392b,color:#fff
    style Qdrant fill:#4ecdc4,stroke:#1a535c,color:#fff
    style Portkey fill:#a29bfe,stroke:#6c5ce7,color:#fff
    style Memory fill:#ffeaa7,stroke:#fdcb6e,color:#333
    style Reranker fill:#74b9ff,stroke:#0984e3,color:#fff
```

---

## Evaluation Results

The pipeline is evaluated using **RAGAS** metrics with a **Meta Llama 3.3 70B** judge model via OpenRouter, across 12 golden benchmark samples from 5 SEC contract filings.

| Metric | Score | Status | What It Measures |
|:---|:---:|:---:|:---|
| **Tool Correctness** | 1.00 |  Good | Did the pipeline call the correct tool? |
| **🛡️ Guardrails Accuracy** | 6/6 |  Good | Off-topic/jailbreak blocking (Precision 1.00, Recall 1.00) |
| **Context Recall** | 0.80 |  Good | Does retrieved context cover all reference information? |
| **Context Precision** | 0.70 |  Fair | Are relevant chunks ranked at the top of results? |
| **Faithfulness** | 0.69 |  Fair | Are answer claims supported by retrieved context? |


---

## Project Structure

```text
├── app/
│   ├── agents/
│   │   └── nodes/          # Planner, Retriever, Responder LangGraph nodes
│   ├── gateway/            # Portkey LLM gateway — multi-key rotation + fallback routing
│   ├── guardrails/         # NeMo Guardrails + deterministic off-topic pattern filtering
│   ├── ingestion/
│   │   ├── chunking/       # Paragraph-based text splitter (1500 char max)
│   │   └── loaders/        # Local parsers — PDF (pypdf), HTML, TXT, DOCX, PPTX
│   ├── services/
│   │   └── retrieval/      # Jina AI embeddings + Qdrant entity-filtered search + Jina reranking
│   ├── config.py           # Centralized environment variable management (multi-key support)
│   └── main.py             # FastAPI entrypoint — guardrails gate + /query endpoint
├── evals/                  # RAGAS evaluation suite + Streamlit 3-tab demo
│   ├── app.py              # Streamlit eval dashboard (3 steps: pipeline → guardrails → metrics)
│   ├── metrics.py          # RAGAS metric runner (OpenRouter 70B judge + SHA-256 cache)
│   ├── guardrails_eval.py  # Guardrails binary classification evaluator
│   ├── golden_dataset.json # 12 RAG samples + 6 guardrails test cases
│   └── metric_results.json # Checkpointed evaluation scores
├── ui/                     # Streamlit chat interface with reasoning step transparency
├── processed_data/         # Auto-generated — parsed & chunked JSON output per document
├── DOCS/                   # Architectural, postmortem, eval changelog and operational guides
├── DATA/                   # Sample datasets (True vs Noisy documentation)
├── Dockerfile              # Multi-stage container definition
└── requirements.txt        # Pinned dependencies
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangChain + LangGraph |
| LLMs | Groq (gpt-oss-120b / llama-3.3-70b) via **Portkey** gateway (4-key rotation) |
| Guardrails | NeMo Guardrails + Deterministic Pattern Filtering |
| Vector DB | **Qdrant Cloud (Managed SaaS)** with entity-aware filtering / Pinecone (Fallback) |
| Reranking | Jina AI Reranker API (`jina-reranker-v3`) — top 40 → top 8 |
| Embeddings | Jina AI `jina-embeddings-v3` (1024-dim) + local mxbai fallback |
| Document Parsing | pypdf + pdfplumber (local, no OCR service) |
| Persistence | Neon Serverless Postgres (LangGraph checkpointer) |
| Rate Limiting | Upstash Redis |
| Observability | Pydantic Logfire + LangSmith |
| Evaluation | RAGAS (6 metrics) + OpenRouter Meta Llama 3.3 70B Judge |

---

## Getting Started

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file with the following keys:

```env
# OpenAI LLM
OPENAI_API_KEY = "your-openai-api-key"

# LLM Gateway — Multi-Key Portkey Rotation
PORTKEY_API_KEY = "your-portkey-api-key"
PORTKEY_API_KEY_1 = "your-portkey-key-2"
PORTKEY_API_KEY_2 = "your-portkey-key-3"
PORTKEY_API_KEY_3 = "your-portkey-key-4"
PORTKEY_PRIMARY_CONFIG_ID = "pc-xxxxxxxxxxxxxxxx"

# OpenRouter (Evaluation Judge)
OPEN_ROUTER_KEY = "sk-or-v1-..."

# Jina AI Embeddings + Reranker API
JINA_API_KEY = "your-jina-api-key"

# Vector DB — Qdrant Cloud (Managed SaaS)
QDRANT_URL = "https://your-cluster-id.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "your-qdrant-cloud-api-key"
QDRANT_COLLECTION = "enterprise_rag_v1"

# Vector DB — Pinecone Setup (Fallback)
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_INDEX_NAME = "legal-enterprise-knowledge-base"

# Production persistence (Neon) & cache (Upstash Redis)
NEON_DB_URL = "postgresql://user:password@host.neon.tech/enterprise_rag?sslmode=require"
UPSTASH_REDIS_REST_URL = "https://your-db.upstash.io"
UPSTASH_REDIS_REST_TOKEN = "your-upstash-token"

# API safety
RAG_API_KEY = "your-rag-api-bearer-key"       # set to require bearer auth
RATE_LIMIT_PER_MINUTE = 20

# Observability
LOGFIRE_TOKEN = "..."
LANGSMITH_API_KEY = "..."
LANGSMITH_PROJECT = "enterprise_rag"
LANGSMITH_TRACING = true
LANGSMITH_ENDPOINT = https://api.smith.langchain.com

# Backend (for Streamlit UI)
BACKEND_URL = "http://localhost:8000"
```

---

### 3. Vector DB Setup, Data Ingestion & Cloud Migration

#### A. Initialize Qdrant Collection
Create the `enterprise_rag_v1` collection with 1024-dimensional vectors matching `jina-embeddings-v3`:

```bash
python scripts/create_qdrant_collection.py
```

*(Optionally for Pinecone fallback: run `python scripts/create_pinecone_index.py`)*

#### B. Ingest Data (Direct from Source)
Parses all documents in `DATA/` (PDF, HTML, TXT, DOCX, PPTX), generates embeddings via Jina AI, saves chunk metadata to `processed_data/`, and bulk-indexes vectors into Qdrant:

```bash
python -m app.ingestion.processor DATA --wipe
```

> Pass `--wipe` to re-create the vector collection and clean existing points before indexing.

#### C. Zero-Cost Vector Migration to Qdrant Cloud
If vectors were previously stored in a local or GCP Docker Qdrant container, transfer all points, 1024-dim dense vectors, and metadata directly to Qdrant Cloud **without calling embedding APIs or spending credits**:

```bash
DEST_QDRANT_URL="https://your-cluster-id.gcp.cloud.qdrant.io:6333" DEST_QDRANT_API_KEY="your-key" python scripts/migrate_to_qdrant_cloud.py
```

> 📖 **Architecture & Deep Dive Documentation**:
> - [Qdrant Failure Postmortem & Cloud Migration Architecture](DOCS/QDRANT_FAILURE_POSTMORTEM_AND_MIGRATION.md)
> - [Qdrant Optimization & Troubleshooting Guide](DOCS/QDRANT_OPTIMIZATION.md)
> - [Evaluation Pipeline Changelog](DOCS/EVAL_PIPELINE_CHANGELOG.md)

---

## Deployment & GCP Cloud Run Setup

### 1. Automated Deployment via GitHub Actions (`cd.yml`)

The repository includes GitHub Actions workflows (`.github/workflows/ci.yml` and `.github/workflows/cd.yml`) that automatically test, build, and deploy container images to **GCP Cloud Run** (`rag-api` and `rag-ui`) on push to `main`.

### 2. API Authentication & GCP Secret Manager

Secure your API endpoints by storing `RAG_API_KEY` and database credentials in **GCP Secret Manager**:

```bash
# 1. Create secret in Secret Manager
gcloud secrets create RAG_API_KEY --replication-policy="automatic"
echo -n "YOUR_SECRET_BEARER_KEY" | gcloud secrets versions add RAG_API_KEY --data-file=-

# 2. Grant Cloud Run service account access to read secrets
PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format="value(projectNumber)")
gcloud secrets add-iam-policy-binding RAG_API_KEY \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# 3. Mount secrets as environment variables on Cloud Run
gcloud run services update rag-api \
  --update-secrets=RAG_API_KEY=RAG_API_KEY:latest \
  --region asia-south1

gcloud run services update rag-ui \
  --update-secrets=RAG_API_KEY=RAG_API_KEY:latest \
  --region asia-south1
```

---

### 4. Launching Locally

You can verify all external connections before starting the server:
```bash
python -m app.services.health.connection_checker
```

Run backend and frontend:
```bash
# Terminal 1 — FastAPI backend
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Streamlit UI
streamlit run ui/dashboard.py
```

### 5. Query the API

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RAG_API_KEY" \
  -d '{"q": "What are the parties in the Inmode manufacturing agreement?", "thread_id": "user-1"}'

# Response: {"question": "...", "answer": "...", "thought_process": [...], "status": "...", "sources": [...]}
```

### 6. Run the eval suite

```bash
# Headless CLI runner — prints final summary table (no Streamlit needed)
python scripts/run_complete_eval_and_summary.py

# Or use the Streamlit 3-tab eval dashboard
streamlit run evals/app.py

# Or the legacy headless runner
python -m evals.run_evals
```

### 7. Run tests locally

```bash
# Lint + format checks
ruff check app tests evals
ruff format --check app tests evals

# Unit tests
LOGFIRE_IGNORE_NO_CONFIG=1 pytest tests/
```

---

*Built for High-Scale Enterprise Document Intelligence.*

