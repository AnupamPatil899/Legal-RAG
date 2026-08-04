import os
import sys

import requests
from pinecone import Pinecone, ServerlessSpec

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.services.retrieval.embedding import get_embedding_dim
from app.services.retrieval.vectordb_service import get_pinecone_index

# 1. Resolve Pinecone Connection Parameters
api_key = getattr(settings, "PINECONE_API_KEY", None) or os.getenv("PINECONE_API_KEY") or "pclocal"
host = getattr(settings, "PINECONE_HOST", None) or os.getenv("PINECONE_HOST", "http://localhost:5081")

pc_kwargs = {"api_key": api_key}
if host:
    pc_kwargs["host"] = host

# Connect to Pinecone (Local or Cloud)
pc = Pinecone(**pc_kwargs)

INDEX_NAME = getattr(settings, "PINECONE_INDEX_NAME", "legal-enterprise-knowledge-base")
DIMENSION = get_embedding_dim()  # 1024 for jina-embeddings-v3 / mxbai-embed-large-v1
METRIC = "dotproduct"  # Required for Hybrid Search (Dense + Sparse)


# 2. Check and Create Index
try:
    if host:
        r = requests.get(f"{host.rstrip('/')}/indexes", timeout=2)
        existing_indexes = [idx.get("name") for idx in r.json().get("indexes", [])] if r.status_code == 200 else []
    else:
        existing_indexes = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        create_kwargs = {
            "name": INDEX_NAME,
            "dimension": DIMENSION,
            "metric": METRIC,
        }
        if not host or "localhost" not in host:
            create_kwargs["spec"] = ServerlessSpec(cloud="aws", region="us-east-1")
        pc.create_index(**create_kwargs)
        print(f"✅ Successfully created index '{INDEX_NAME}' (dimension={DIMENSION}, metric='{METRIC}')")
    else:
        print(f"ℹ️ Index '{INDEX_NAME}' already exists")

    index = get_pinecone_index(pc, INDEX_NAME)
    stats = index.describe_index_stats() if index else {}
    print(f"\nIndex '{INDEX_NAME}' stats: {stats}")
except Exception as e:
    print(f"❌ Could not connect to Pinecone at '{host}': {e}")
    print("\n💡 Tip: If running locally, make sure your Pinecone container is running:")
    print("   docker compose up -d pinecone-local\n")
    sys.exit(1)
