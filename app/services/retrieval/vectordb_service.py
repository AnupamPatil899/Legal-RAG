# import logfire
import os
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from tenacity import retry, stop_after_attempt, wait_exponential

# Patch Pinecone's check_realistic_host to support custom/docker hostnames (e.g. 'pinecone-local') without dots
try:
    import pinecone.pinecone as _pinecone_mod

    _pinecone_mod.check_realistic_host = lambda host: None
except (ImportError, AttributeError):
    pass

try:
    import pinecone.pinecone_asyncio as _pinecone_async_mod

    _pinecone_async_mod.check_realistic_host = lambda host: None
except (ImportError, AttributeError):
    pass


from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.config import settings
from app.services.retrieval.embedding import embed_query

load_dotenv()


def get_qdrant_client():
    url = getattr(settings, "QDRANT_URL", None) or os.getenv("QDRANT_URL")
    if not url:
        return None
    if url.startswith("https://") and ":" not in url[8:]:
        url = f"{url.rstrip('/')}:443"
    api_key = (
        getattr(settings, "QDRANT_SECURITY", None)
        or getattr(settings, "QDRANT_API_KEY", None)
        or os.getenv("QDRANT_SECURITY")
        or os.getenv("QDRANT_API_KEY")
    )
    return QdrantClient(
        url=url,
        api_key=api_key,
        prefer_grpc=False,
        check_compatibility=False,
        timeout=15,
    )


def get_pinecone_index(pc_client: Pinecone, index_name: str = None):
    """
    Returns a Pinecone Index instance.
    For local Pinecone container (where PINECONE_HOST is configured), dynamically resolves
    the local index host (port 5082) so requests work seamlessly.
    """
    if pc_client is None:
        return None
    if index_name is None:
        index_name = getattr(settings, "PINECONE_INDEX_NAME", "legal-enterprise-knowledge-base")
    host = getattr(settings, "PINECONE_HOST", None) or os.getenv("PINECONE_HOST")
    if host:
        parsed = urlparse(host if host.startswith("http") else f"http://{host}")
        scheme = parsed.scheme or "http"
        hostname = parsed.hostname or "localhost"
        try:
            r = requests.get(f"{host.rstrip('/')}/indexes", timeout=2)
            if r.status_code == 200:
                data = r.json()
                for idx in data.get("indexes", []):
                    if idx.get("name") == index_name:
                        target_host = idx.get("host", "")
                        if scheme == "https":
                            target_host = host.rstrip("/")
                        elif ":" in target_host:
                            port = target_host.split(":")[-1]
                            target_host = f"{scheme}://{hostname}:{port}"
                        elif not target_host.startswith("http"):
                            target_host = f"{scheme}://{target_host}"
                        return pc_client.Index(name=index_name, host=target_host)
        except Exception:
            pass
        index_host = host.rstrip("/") if scheme == "https" else f"{scheme}://{hostname}:5082"
        return pc_client.Index(name=index_name, host=index_host)
    return pc_client.Index(index_name)


# Initialize Pinecone Client & BM25 Encoder
_pinecone_api_key = getattr(settings, "PINECONE_API_KEY", None) or os.getenv("PINECONE_API_KEY") or "pclocal"
_pinecone_host = getattr(settings, "PINECONE_HOST", None) or os.getenv("PINECONE_HOST")

_pc_kwargs = {"api_key": _pinecone_api_key}
if _pinecone_host:
    _pc_kwargs["host"] = _pinecone_host

client = Pinecone(**_pc_kwargs) if (_pinecone_api_key or _pinecone_host) else None
bm25 = BM25Encoder.default()

MASTER_INDEX_NAME = getattr(settings, "PINECONE_INDEX_NAME", "legal-enterprise-knowledge-base")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    reraise=True,
    #    before_sleep=before_sleep_log(logfire, "warning"),
)
def _search_enterprise_knowledge(query: str, folder_name: str = None, limit: int = 8):
    """Internal hybrid/dense search with retry logic and optional metadata filtering."""
    qdrant = get_qdrant_client()
    if qdrant:
        collection_name = getattr(settings, "QDRANT_COLLECTION", "enterprise_rag")
        dense_vector = embed_query(query)

        qdrant_filter = None
        if folder_name:
            qdrant_filter = Filter(must=[FieldCondition(key="folder_name", match=MatchValue(value=folder_name))])

        res = qdrant.query_points(
            collection_name=collection_name,
            query=dense_vector,
            query_filter=qdrant_filter,
            limit=limit,
        )

        results = []
        for point in res.points:
            payload = point.payload or {}
            results.append(
                {
                    "content": payload.get("text", ""),
                    "source": payload.get("source", "Unknown"),
                    "folder": payload.get("folder_name", "Unknown"),
                    "score": point.score,
                }
            )
        return results

    if client is None:
        raise RuntimeError("Neither QDRANT_URL nor PINECONE_API_KEY/PINECONE_HOST is set")

    dense_vector = embed_query(query)
    sparse_vector = bm25.encode_queries(query)

    # 1. Connect to the Single Master Index
    index = get_pinecone_index(client, MASTER_INDEX_NAME)

    # 2. Build the Metadata Filter
    search_kwargs = {"vector": dense_vector, "sparse_vector": sparse_vector, "top_k": limit, "include_metadata": True}

    # If a folder_name is provided, attach the Pinecone equality filter
    if folder_name:
        search_kwargs["filter"] = {"folder_name": {"$eq": folder_name}}

    # 3. Execute Query
    response = index.query(**search_kwargs)

    results = []
    for match in response.matches:
        results.append(
            {
                "content": match.metadata.get("text", ""),
                "source": match.metadata.get("source", "Unknown"),
                "folder": match.metadata.get("folder_name", "Unknown"),
                "score": match.score,
            }
        )

    return results


def search_enterprise_knowledge(query: str, folder_name: str = None, limit: int = 8):
    """
    Performs a high-precision search in the enterprise knowledge base.
    Optionally filters by `folder_name` metadata.
    """
    try:
        return _search_enterprise_knowledge(query, folder_name=folder_name, limit=limit)
    except Exception as e:
        # logfire.error(f"❌ Vector Search Failed after retries: {e}")
        print(f"❌ Vector Search Failed after retries: {e}")
        return []
