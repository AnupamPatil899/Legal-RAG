import logfire
import requests
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from app.config import settings

_JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
_JINA_RERANK_MODEL = "jina-reranker-v3"

_ranker = None


class _JinaReranker:
    """Thin wrapper around the Jina Reranker API."""

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[str]:
        """Score and reorder documents against the query via the Jina API."""
        response = requests.post(
            _JINA_RERANK_URL,
            headers={
                "Authorization": f"Bearer {settings.JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": _JINA_RERANK_MODEL,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": True,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()

        results = payload.get("results", [])
        # Results are already sorted by relevance_score descending
        reranked_docs = []
        for res in results[:top_n]:
            doc = res.get("document")
            if isinstance(doc, dict):
                doc_text = doc.get("text", "")
            elif isinstance(doc, str):
                doc_text = doc
            else:
                index = res.get("index")
                if index is not None and 0 <= index < len(documents):
                    doc_text = documents[index]
                else:
                    doc_text = str(doc or "")
            if doc_text:
                reranked_docs.append(doc_text)

        return reranked_docs


def _get_ranker() -> _JinaReranker:
    """Returns the Jina Reranker wrapper (lazy singleton)."""
    global _ranker
    if _ranker is None:
        logfire.info("🧠 Initializing Jina Reranker v3 via API...")
        _ranker = _JinaReranker()
    return _ranker


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    reraise=True,
    before_sleep=before_sleep_log(logfire, "warning"),
)
def _rerank(query: str, documents: list[str], top_n: int) -> list[str]:
    """Core Jina API reranking with retry on transient failures."""
    ranker = _get_ranker()
    return ranker.rerank(query, documents, top_n)


_local_ranker = None


def _get_local_ranker():
    global _local_ranker
    if _local_ranker is None:
        try:
            from sentence_transformers import CrossEncoder

            _local_ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"Warning: Could not initialize local CrossEncoder: {e}")
            _local_ranker = False
    return _local_ranker


def _rerank_local(query: str, documents: list[str], top_n: int) -> list[str]:
    ranker = _get_local_ranker()
    if not ranker:
        return documents[:top_n]
    try:
        pairs = [[query, d] for d in documents]
        scores = ranker.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_n]]
    except Exception as e:
        print(f"Local CrossEncoder rerank failed: {e}")
        return documents[:top_n]


_jina_disabled = False


def rerank_documents(query: str, documents: list[str], top_n: int = 5) -> list[str]:
    """
    Refines retrieval results by re-scoring documents against the query semantically.
    Uses Jina Reranker API when active with retry; falls back to input document order on failure.
    If Jina API key is not configured, uses local CrossEncoder.
    """
    global _jina_disabled
    if not documents:
        return []

    if settings.JINA_API_KEY and not _jina_disabled:
        try:
            return _rerank(query, documents, top_n)
        except Exception as e:
            logfire.warning(f"Jina Reranker API unavailable ({e}). Falling back to input document order.")
            _jina_disabled = True
            return documents[:top_n]

    return _rerank_local(query, documents, top_n)
