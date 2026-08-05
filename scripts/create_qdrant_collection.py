import os
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.services.retrieval.embedding import get_embedding_dim

QDRANT_URL = settings.QDRANT_URL or os.getenv("QDRANT_URL")
if not QDRANT_URL:
    print("❌ Error: QDRANT_URL environment variable is not configured.")
    sys.exit(1)

if QDRANT_URL.startswith("https://") and ":" not in QDRANT_URL[8:]:
    QDRANT_URL = f"{QDRANT_URL.rstrip('/')}:443"

QDRANT_API_KEY = (
    getattr(settings, "QDRANT_SECURITY", None)
    or getattr(settings, "QDRANT_API_KEY", None)
    or os.getenv("QDRANT_SECURITY")
    or os.getenv("QDRANT_API_KEY")
)
COLLECTION_NAME = settings.QDRANT_COLLECTION or os.getenv("QDRANT_COLLECTION", "enterprise_rag")
DIMENSION = get_embedding_dim()  # 1024 for jina-embeddings-v3 / mxbai-embed-large-v1

print(f"Connecting to Qdrant at '{QDRANT_URL}'...")
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False,
    check_compatibility=False,
    timeout=30,
)

try:
    if not client.collection_exists(COLLECTION_NAME):
        print(f"Creating collection '{COLLECTION_NAME}' (dimension={DIMENSION}, distance=DOT/COSINE)...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=DIMENSION, distance=Distance.DOT),
        )
        print(f"✅ Successfully created Qdrant collection '{COLLECTION_NAME}'")
    else:
        print(f"ℹ️ Qdrant collection '{COLLECTION_NAME}' already exists")

    info = client.get_collection(COLLECTION_NAME)
    print(f"\nCollection '{COLLECTION_NAME}' status: {info.status}, points_count: {info.points_count}")
except Exception as e:
    print(f"❌ Failed to connect or create collection on Qdrant at '{QDRANT_URL}': {e}")
    sys.exit(1)
