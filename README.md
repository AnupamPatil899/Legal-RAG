# Enterprise Agentic RAG (Scalable Pipeline)

A production-grade, enterprise-level RAG system built with **LangGraph**, **Portkey LLM Gateway**, **OpenAI**, and **Jina AI Embeddings/Reranker**. The system distinguishes between technical "True Data" and random "Noisy Data" using semantic re-ranking, history-aware planning, and NeMo Guardrails for input/output safety.

## Key Features

- **Agentic Intelligence**: LangGraph for cyclic reasoning, multi-step planning, and conversation memory.
- **Guardrails**: NeMo Guardrails gate blocks off-topic, jailbreak, and injection inputs before any retrieval.
- **LLM Gateway**: Portkey routes all LLM calls with automatic fallback between Gemini, Kimi and llama via your configured Portkey virtual providers.
- **Enterprise Search**: Pinecone Cloud for high-performance vector search + Jina AI Reranker API for semantic reranking.
- **Jina AI Embeddings**: `jina-embeddings-v3` (1024-dim) via Jina API, with local `mxbai-embed-large-v1` fallback.
- **Local Document Parsing**: PDF, HTML, TXT, DOCX, PPTX parsed entirely on-device — no external OCR service.
- **Observability**: Full trace nesting with **Pydantic Logfire** and **LangSmith** across every agent node.
- **Metrics**: Prometheus `/metrics` endpoint with custom RAG and guardrails counters.
- **Synchronous `/query`**: The LangGraph pipeline runs directly inside the `/query` endpoint and returns the final answer.
- **API Key & Rate Limiting**: Optional bearer-token auth and Redis-backed (or in-memory) rate limiting.
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
    Retriever --> Reranker[Jina AI Reranker API]
    Reranker --> Responder
    Responder --> UI
    Responder -.-> Memory[(LangGraph MemorySaver)]
```

---

## Project Structure

```text
├── app/
│   ├── agents/
│   │   └── nodes/       # Planner, Retriever, Responder LangGraph nodes
│   ├── gateway/         # Portkey LLM gateway — primary + fallback Groq routing
│   ├── guardrails/      # NeMo Guardrails input/output filtering
│   ├── ingestion/
│   │   ├── chunking/    # Paragraph-based text splitter (1500 char max)
│   │   └── loaders/     # Local parsers — PDF (pypdf), HTML, TXT, DOCX, PPTX
│   ├── services/
│   │   └── retrieval/   # Jina AI embeddings + Pinecone Hybrid search + Jina AI reranking
│   ├── config.py        # Centralized environment variable management
│   └── main.py          # FastAPI entrypoint — guardrails gate + /query endpoint
├── evals/               # RAGAS evaluation suite + Streamlit 3-tab demo
├── ui/                  # Streamlit chat interface with reasoning step transparency
├── processed_data/      # Auto-generated — parsed & chunked JSON output per document
├── DOCS/                # Architectural and operational guides
├── DATA/                # Sample datasets (True vs Noisy documentation)
├── Dockerfile           # Container definition (retained for reference)
└── requirements.txt     # Pinned dependencies
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangChain + LangGraph |
| LLMs | Kimi + Llama/Gemini fallback via **Portkey** gateway |
| Guardrails | NeMo Guardrails |
| Vector DB | Pinecone Cloud |
| Reranking | Jina AI Reranker API (`jina-reranker-v3`) |
| Embeddings | Jina AI `jina-embeddings-v3` (1024-dim) + local mxbai fallback |
| Document Parsing | pypdf + pdfplumber (local, no OCR service) |
| Observability | Pydantic Logfire + LangSmith |
| Evaluation | RAGAS + custom Tool Correctness (Jaccard) |

---

## Getting Started

### 1. Install dependencies

```powershell
python -m venv tenvv
.\tenvv\Scripts\activate
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

# Vector DB — Pinecone Setup
# For Pinecone Cloud: set PINECONE_API_KEY (leave PINECONE_HOST empty)
PINECONE_API_KEY = "your-pinecone-api-key"
# For Local Development via Docker: uncomment PINECONE_HOST
# PINECONE_HOST = "http://localhost:5081"
PINECONE_INDEX_NAME = "legal-enterprise-knowledge-base"

# Production persistence (Neon) & cache (Upstash Redis)
NEON_DB_URL = "postgresql://user:password@host.neon.tech/enterprise_rag?sslmode=require"
UPSTASH_REDIS_REST_URL = "https://your-db.upstash.io"
UPSTASH_REDIS_REST_TOKEN = "your-upstash-token"

# API safety
RAG_API_KEY = ""                       # set in production to require bearer auth
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

### 3. Initialize Pinecone & Data Ingestion

#### A. Create the Pinecone Index
For local development using Docker Compose:
```bash
# Start local Pinecone emulator container
docker compose up -d pinecone-local

# Create local or cloud Pinecone index (1024-dim hybrid dotproduct metric)
python scripts/create_pinecone_index.py
```

#### B. Ingest Data
Parses all documents in `DATA/`, chunks them, saves metadata to `processed_data/`, and indexes vectors into Pinecone.

```powershell
python -m app.ingestion.processor DATA --wipe
```

> Pass `--wipe` to drop and recreate the Pinecone index. Omit it to append to an existing index.

---

## Deployment & CI/CD (GitHub Actions)

### How Pinecone Works in GitHub Actions & Cloud Run

1. **GitHub Actions CI (`ci.yml`)**:
   - Automated unit tests run against mocks (`MagicMock`). No live Pinecone cluster or local Docker container is needed for CI test runs.

2. **Hosting Local Pinecone Container on GCP (Cloud Run)**:
   You can run the free `pinecone-local` Docker container image directly on GCP Cloud Run:
   ```bash
   gcloud run deploy pinecone-local \
     --image=ghcr.io/pinecone-io/pinecone-local:latest \
     --port=5081 \
     --region=us-central1 \
     --memory=2Gi \
     --allow-unauthenticated
   ```
   *Take note of the service URL generated by GCP (e.g. `https://pinecone-local-xyz123-uc.a.run.app`).*

3. **Create Index & Ingest Data into GCP Pinecone Container**:
   - **Create Index**:
     ```bash
     PINECONE_HOST="https://pinecone-local-xyz123-uc.a.run.app" python scripts/create_pinecone_index.py
     ```
   - **Ingest Data**:
     ```bash
     PINECONE_HOST="https://pinecone-local-xyz123-uc.a.run.app" python -m app.ingestion.processor DATA --wipe
     ```

4. **Linking `rag-api` & `rag-ui` to GCP Pinecone Container**:
   - Update `rag-api` on GCP Cloud Run to use your container host:
     ```bash
     gcloud run services update rag-api \
       --region=us-central1 \
       --set-env-vars="PINECONE_HOST=https://pinecone-local-xyz123-uc.a.run.app,PINECONE_API_KEY=pclocal"
     ```
   - For automated deployments via GitHub Actions (`cd.yml`), add `PINECONE_HOST` in GitHub Repository Secrets and pass `--set-env-vars=PINECONE_HOST=${{ secrets.PINECONE_HOST }}` in `cd.yml`.


---

### 4. Launch the app


The `/query` endpoint runs the LangGraph pipeline synchronously. You only need the FastAPI server and (optionally) the Streamlit UI. Redis and Postgres are managed by Upstash and Neon; no local persistence services are required.

> **Tip:** You can verify all external connections before starting the server:
> ```bash
> python -m app.services.health.connection_checker
> ```

```powershell
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
