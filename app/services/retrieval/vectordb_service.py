# import logfire
import os

from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.services.retrieval.embedding import embed_query

load_dotenv()


# Initialize Pinecone Client & BM25 Encoder
client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
bm25 = BM25Encoder.default()

MASTER_INDEX_NAME = getattr(settings, "PINECONE_INDEX_NAME", "legal-enterprise-knowledge-base")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    reraise=True,
    #    before_sleep=before_sleep_log(logfire, "warning"),
)
def _search_enterprise_knowledge(query: str, folder_name: str = None, limit: int = 8):
    """Internal hybrid search with retry logic and optional metadata filtering."""
    dense_vector = embed_query(query)
    sparse_vector = bm25.encode_queries(query)

    # 1. Connect to the Single Master Index
    index = client.Index(MASTER_INDEX_NAME)

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
    Performs a high-precision HYBRID search in the enterprise knowledge base.
    Optionally filters by `folder_name` metadata.
    """
    try:
        return _search_enterprise_knowledge(query, folder_name=folder_name, limit=limit)
    except Exception as e:
        # logfire.error(f"❌ Pinecone Search Failed after retries: {e}")
        print(f"❌ Pinecone Search Failed after retries: {e}")
        return []
