import os
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, HnswConfigDiff, OptimizersConfigDiff, VectorParams

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
DIMENSION = get_embedding_dim()

TARGET_COLLECTION = os.getenv("TARGET_COLLECTION", "enterprise_rag_v2")

print(f"Connecting to Qdrant at '{QDRANT_URL}'...")
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False,
    check_compatibility=False,
    timeout=90,
)

if not client.collection_exists(COLLECTION_NAME):
    print(f"❌ Source collection '{COLLECTION_NAME}' does not exist.")
    sys.exit(1)

info = client.get_collection(COLLECTION_NAME)
print(f"Source Collection '{COLLECTION_NAME}': status={info.status}, points={info.points_count}, segments={info.segments_count}")

if info.points_count == 0:
    print("No points found to migrate.")
    sys.exit(0)

print(f"\n📥 Scrolling all {info.points_count} points from '{COLLECTION_NAME}'...")
all_points = []
next_offset = None

while True:
    records, next_offset = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=2500,
        offset=next_offset,
        with_payload=True,
        with_vectors=True,
    )
    all_points.extend(records)
    print(f"  Fetched {len(all_points)} / {info.points_count} points...")
    if next_offset is None or not records:
        break

print(f"✅ Successfully fetched {len(all_points)} points.")

# Create the clean target collection enterprise_rag_v2
print(f"\n✨ Creating clean target collection '{TARGET_COLLECTION}'...")
if client.collection_exists(TARGET_COLLECTION):
    try:
        client.delete_collection(TARGET_COLLECTION)
    except Exception:
        pass

client.create_collection(
    collection_name=TARGET_COLLECTION,
    vectors_config=VectorParams(size=DIMENSION, distance=Distance.DOT),
    hnsw_config=HnswConfigDiff(max_indexing_threads=2),
    optimizers_config=OptimizersConfigDiff(
        indexing_threshold=2000,
        max_optimization_threads=2,
        default_segment_number=2,
    ),
)

# Bulk upsert in large chunks of 4,500 points to keep segment count low (~4 segments total)
BATCH_SIZE = 4500
print(f"\n🚀 Bulk upserting {len(all_points)} points into '{TARGET_COLLECTION}' in batches of {BATCH_SIZE}...")
from qdrant_client.models import PointStruct

for i in range(0, len(all_points), BATCH_SIZE):
    batch = all_points[i : i + BATCH_SIZE]
    points_to_upsert = [
        PointStruct(id=r.id, vector=r.vector, payload=r.payload)
        for r in batch
    ]
    client.upsert(collection_name=TARGET_COLLECTION, points=points_to_upsert)
    print(f"  Batch {i // BATCH_SIZE + 1}: Upserted {len(points_to_upsert)} points...")

final_info = client.get_collection(TARGET_COLLECTION)
print(f"\n✅ Clean Migration Complete! Target '{TARGET_COLLECTION}' state: status={final_info.status}, points={final_info.points_count}, indexed={final_info.indexed_vectors_count}, segments={final_info.segments_count}")

# Optionally try deleting old fragmented collection
try:
    print(f"\n🧹 Attempting cleanup of old fragmented collection '{COLLECTION_NAME}'...")
    client.delete_collection(COLLECTION_NAME)
    print(f"✅ Deleted old collection '{COLLECTION_NAME}'.")
except Exception as e:
    print(f"ℹ️ Old collection cleanup skipped (will self-clean on next Qdrant restart): {e}")

