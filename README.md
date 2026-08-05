# Enterprise Agentic RAG (Scalable Pipeline)

A production-grade, enterprise-level RAG system built with **LangGraph**, **Portkey LLM Gateway**, **OpenAI**, **Qdrant Vector Database**, and **Jina AI Embeddings/Reranker**. The system distinguishes between technical "True Data" and random "Noisy Data" using semantic re-ranking, history-aware planning, and NeMo Guardrails for input/output safety.

## 🚀 Live Demo & API Endpoints

- **Frontend Streamlit UI**: [https://rag-ui-976087180091.asia-south1.run.app/](https://rag-ui-976087180091.asia-south1.run.app/)
- **Backend FastAPI Service**: [https://rag-api-976087180091.asia-south1.run.app/](https://rag-api-976087180091.asia-south1.run.app/)


---

## Key Features

- **Agentic Intelligence**: LangGraph for cyclic reasoning, multi-step planning, and conversation memory.
- **Guardrails**: NeMo Guardrails gate blocks off-topic, jailbreak, and injection inputs before any retrieval.
- **LLM Gateway**: Portkey routes all LLM calls with automatic fallback between Gemini, Kimi, and Llama via your configured Portkey virtual providers.
- **Enterprise Search**: **Qdrant Vector Database** for high-performance vector search + Jina AI Reranker API for semantic reranking (with Pinecone fallback).
- **Jina AI Embeddings**: `jina-embeddings-v3` (1024-dim) via Jina API, with local `mxbai-embed-large-v1` fallback.
- **Local Document Parsing**: PDF, HTML, TXT, DOCX, PPTX parsed entirely on-device — no external OCR service.
- **Observability**: Full trace nesting with **Pydantic Logfire** and **LangSmith** across every agent node.
- **Metrics**: Prometheus `/metrics` endpoint with custom RAG and guardrails counters.
- **Synchronous `/query`**: The LangGraph pipeline runs directly inside the `/query` endpoint and returns the final answer.
- **API Security & Rate Limiting**: Bearer-token authentication via GCP Secret Manager and Redis-backed (or in-memory) rate limiting.
- **Evaluation Suite**: RAGAS-powered eval pipeline (6 metrics) with a dedicated Streamlit demo app and a headless `evals/run_evals.py` script.

---

## Agent Intelligence Flow

```mermaid
graph TD
    User((User)) --> UI[Streamlit UI]
    UI --> API[FastAPI /query]
    API --> Guard{NeMo Guardrails}
    Guard -->|Blocked| UI
    Guard -->|Pass| Planner{Planner Node}
    Planner -->|Conversational| Responder[Responder Node]
    Planner -->|Technical| Retriever[Retriever Node]
    Retriever --> Qdrant[Qdrant Vector DB]
    Qdrant --> Reranker[Jina AI Reranker API]
    Reranker --> Responder
    Responder --> UI
    Responder -.-> Memory[(LangGraph Postgres Checkpointer)]
```

---

## Project Structure

```text
├── app/
│   ├── agents/
│   │   └── nodes/       # Planner, Retriever, Responder LangGraph nodes
│   ├── gateway/         # Portkey LLM gateway — primary + fallback routing
│   ├── guardrails/      # NeMo Guardrails input/output filtering
│   ├── ingestion/
│   │   ├── chunking/    # Paragraph-based text splitter (1500 char max)
│   │   └── loaders/     # Local parsers — PDF (pypdf), HTML, TXT, DOCX, PPTX
│   ├── services/
│   │   └── retrieval/   # Jina AI embeddings + Qdrant / Pinecone search + Jina AI reranking
│   ├── config.py        # Centralized environment variable management
│   └── main.py          # FastAPI entrypoint — guardrails gate + /query endpoint
├── evals/               # RAGAS evaluation suite + Streamlit 3-tab demo
├── scripts/             # Collection initialization scripts for Qdrant and Pinecone
├── ui/                  # Streamlit chat interface with reasoning step transparency
├── processed_data/      # Auto-generated — parsed & chunked JSON output per document
├── DOCS/                # Architectural and operational guides
├── DATA/                # Sample datasets (True vs Noisy documentation)
├── Dockerfile           # Multi-stage container definition
└── requirements.txt     # Pinned dependencies
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangChain + LangGraph |
| LLMs | Kimi + Llama/Gemini fallback via **Portkey** gateway |
| Guardrails | NeMo Guardrails |
| Vector DB | **Qdrant** (Primary) / Pinecone (Fallback) |
| Reranking | Jina AI Reranker API (`jina-reranker-v3`) |
| Embeddings | Jina AI `jina-embeddings-v3` (1024-dim) + local mxbai fallback |
| Document Parsing | pypdf + pdfplumber (local, no OCR service) |
| Persistence | Neon Serverless Postgres (LangGraph checkpointer) |
| Rate Limiting | Upstash Redis |
| Observability | Pydantic Logfire + LangSmith |
| Evaluation | RAGAS + custom Tool Correctness (Jaccard) |

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

# LLM Gateway
PORTKEY_API_KEY = "your-portkey-api-key"
PORTKEY_PRIMARY_CONFIG_ID = "pc-xxxxxxxxxxxxxxxx"

# Jina AI Embeddings + Reranker API
JINA_API_KEY = "your-jina-api-key"

# Vector DB — Qdrant Setup (Primary)
QDRANT_URL = "https://your-qdrant-cluster.cloud.qdrant.io:443"
QDRANT_API_KEY = "your-qdrant-api-key"
QDRANT_COLLECTION = "enterprise_rag"

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

# Evals
JUDGE_OPENAI_API_KEY = "..."

# Backend (for Streamlit UI)
BACKEND_URL = "http://localhost:8000"
```

---

### 3. Vector DB Setup & Data Ingestion

#### A. Initialize Qdrant Collection
Create the `enterprise_rag` collection with 1024-dimensional vectors matching `jina-embeddings-v3`:

```bash
python scripts/create_qdrant_collection.py
```

*(Optionally for Pinecone fallback: run `python scripts/create_pinecone_index.py`)*

#### B. Ingest Data
Parses all documents in `DATA/` (PDF, HTML, TXT, DOCX, PPTX), generates embeddings via Jina AI, saves chunk metadata to `processed_data/`, and indexes vectors into Qdrant:

```bash
python -m app.ingestion.processor DATA --wipe
```

> Pass `--wipe` to re-create the vector collection and clean existing points before indexing.

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

```powershell
curl -X POST "http://localhost:8000/query" `
  -H "Content-Type: application/json" `
  -d '{"q": "How do I start Redis for a Contracts work queue?", "thread_id": "user-1"}'

# Response: {"question": "...", "answer": "...", "thought_process": [...], "status": "...", "sources": [...]}
```

### 6. Run the eval suite

```powershell
# Headless CLI runner (requires backend on :8000)
python -m evals.run_evals

# Or use the Streamlit demo
streamlit run evals/app.py
```

### 7. Run tests locally

```powershell
# Lint + format checks
ruff check app tests evals
ruff format --check app tests evals

# Unit tests
$env:LOGFIRE_IGNORE_NO_CONFIG=1
pytest tests/
```

---

*Built for High-Scale Enterprise Document Intelligence.*
