import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# ==========================================
# 1. Configuration Settings
# ==========================================

# Source (Your GCP Qdrant instance - currently configured in .env)
SOURCE_URL = os.getenv("SOURCE_QDRANT_URL") or os.getenv("QDRANT_URL")
SOURCE_API_KEY = os.getenv("SOURCE_QDRANT_API_KEY") or os.getenv("QDRANT_SECURITY") or os.getenv("QDRANT_API_KEY")
SOURCE_COLLECTION = os.getenv("SOURCE_COLLECTION") or os.getenv("QDRANT_COLLECTION") or "enterprise_rag_v1"

# Destination (Qdrant Cloud)
DEST_URL = (
    os.getenv("DEST_QDRANT_URL")
    or os.getenv("QDRANT_CLOUD_URL")
    or os.getenv("QDRANT_CLUSTER_ENDPOINT")
)
DEST_API_KEY = (
    os.getenv("DEST_QDRANT_API_KEY")
    or os.getenv("QDRANT_CLOUD_API_KEY")
)
DEST_COLLECTION = os.getenv("DEST_COLLECTION") or SOURCE_COLLECTION

# Vector Dimension (1024 for jina-embeddings-v3 / mxbai-embed-large-v1)
DIMENSION = int(os.getenv("EMBEDDING_DIM", "1024"))
BATCH_SIZE = 500  # Safe batch size for cloud REST uploads


def main():
    global DEST_URL, DEST_API_KEY, SOURCE_URL, SOURCE_API_KEY

    print("=" * 65)
    print("🚀 GCP QDRANT ➔ QDRANT CLOUD ZERO-COST MIGRATION TOOL")
    print("=" * 65)

    if not SOURCE_URL:
        SOURCE_URL = input("Enter Source Qdrant URL (GCP/Docker): ").strip()
    if not SOURCE_API_KEY:
        SOURCE_API_KEY = os.getenv("QDRANT_SECURITY")

    # If Cloud credentials are not in environment, prompt interactively
    if not DEST_URL:
        print("\n🔑 Please enter your Qdrant Cloud Cluster details:")
        DEST_URL = input("  Qdrant Cloud URL (e.g. https://xxx.gcp.cloud.qdrant.io:6333): ").strip()
    if not DEST_API_KEY:
        DEST_API_KEY = input("  Qdrant Cloud API Key: ").strip()

    if not DEST_URL or not DEST_API_KEY:
        print("❌ Error: Destination Qdrant Cloud URL and API Key are required.")
        sys.exit(1)

    # Normalize URLs
    if SOURCE_URL.startswith("https://") and ":" not in SOURCE_URL[8:]:
        SOURCE_URL = f"{SOURCE_URL.rstrip('/')}:443"
    if DEST_URL.startswith("https://") and ":" not in DEST_URL[8:]:
        DEST_URL = f"{DEST_URL.rstrip('/')}:6333"

    print(f"\n📡 [1/5] Connecting to Source (GCP Qdrant): {SOURCE_URL}...")
    try:
        source_client = QdrantClient(
            url=SOURCE_URL,
            api_key=SOURCE_API_KEY,
            prefer_grpc=False,
            check_compatibility=False,
            timeout=120,
        )
        
        # Check source collections
        collections = [c.name for c in source_client.get_collections().collections]
        print(f"   Available Source Collections: {collections}")
        
        target_src = SOURCE_COLLECTION
        if target_src not in collections:
            if len(collections) == 1:
                target_src = collections[0]
                print(f"   Using only available source collection: '{target_src}'")
            else:
                print(f"❌ Source collection '{target_src}' not found in {collections}.")
                sys.exit(1)

        src_info = source_client.get_collection(target_src)
        total_points = src_info.points_count
        print(f"✅ Connected to Source! Found {total_points:,} points in '{target_src}' (Status: {src_info.status}).")
    except Exception as e:
        print(f"❌ Failed to connect to Source Qdrant at '{SOURCE_URL}': {e}")
        print("\n💡 TIP: If your source Docker is running on another port or host, pass it via:")
        print("   SOURCE_QDRANT_URL=\"http://<IP>:6333\" python scripts/migrate_to_qdrant_cloud.py")
        sys.exit(1)

    if total_points == 0:
        print("⚠️ No points to migrate in source collection.")
        sys.exit(0)

    print(f"\n☁️ [2/5] Connecting to Qdrant Cloud: {DEST_URL}...")
    try:
        dest_client = QdrantClient(
            url=DEST_URL,
            api_key=DEST_API_KEY,
            prefer_grpc=False,
            check_compatibility=False,
            timeout=120,
        )
        dest_collections = [c.name for c in dest_client.get_collections().collections]
        print(f"✅ Connected to Qdrant Cloud! Existing cloud collections: {dest_collections}")
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant Cloud at '{DEST_URL}': {e}")
        sys.exit(1)

    print(f"\n🏗️ [3/5] Setting up target collection '{DEST_COLLECTION}' on Qdrant Cloud...")
    if dest_client.collection_exists(DEST_COLLECTION):
        print(f"   Collection '{DEST_COLLECTION}' already exists on Qdrant Cloud.")
    else:
        print(f"   Creating collection '{DEST_COLLECTION}' (dim={DIMENSION}, distance=DOT)...")
        dest_client.create_collection(
            collection_name=DEST_COLLECTION,
            vectors_config=VectorParams(size=DIMENSION, distance=Distance.DOT),
        )
        print(f"✅ Created collection '{DEST_COLLECTION}' on Qdrant Cloud.")

    print(f"\n📦 [4/5] Transferring {total_points:,} points (Vectors + Payloads) with ZERO API costs...")
    
    next_offset = None
    transferred = 0
    batch_num = 0

    while True:
        # Scroll points from source
        records, next_offset = source_client.scroll(
            collection_name=target_src,
            limit=BATCH_SIZE,
            offset=next_offset,
            with_payload=True,
            with_vectors=True,
        )

        if not records:
            break

        # Convert to PointStruct format
        points_to_upsert = [
            PointStruct(id=r.id, vector=r.vector, payload=r.payload)
            for r in records
        ]

        # Upsert directly into Qdrant Cloud
        dest_client.upsert(
            collection_name=DEST_COLLECTION,
            points=points_to_upsert,
        )

        transferred += len(points_to_upsert)
        batch_num += 1
        pct = (transferred / total_points) * 100
        print(f"   Batch {batch_num:03d}: Uploaded {transferred:,} / {total_points:,} points ({pct:.1f}%)...")

        if next_offset is None:
            break

    print(f"\n✅ All {transferred:,} points transferred to Qdrant Cloud!")

    print("\n🔍 [5/5] Verifying Qdrant Cloud Indexing & Health Status...")
    for attempt in range(15):
        dest_info = dest_client.get_collection(DEST_COLLECTION)
        print(f"   Status: {dest_info.status} | Points: {dest_info.points_count:,} | Indexed Vectors: {dest_info.indexed_vectors_count:,}")
        if dest_info.status.name.lower() == "green" and dest_info.indexed_vectors_count == dest_info.points_count:
            break
        time.sleep(2)

    final_info = dest_client.get_collection(DEST_COLLECTION)
    print("\n" + "=" * 65)
    print(f"🎉 MIGRATION SUCCESSFUL!")
    print(f"   Target Collection: {DEST_COLLECTION}")
    print(f"   Final Status:      {final_info.status}")
    print(f"   Total Points:      {final_info.points_count:,}")
    print(f"   Indexed Vectors:   {final_info.indexed_vectors_count:,}")
    print(f"   Embedding Credits: 0 (Zero API credits used)")
    print("=" * 65)
    print("\n👉 NEXT STEP: Update your .env to point to Qdrant Cloud:")
    print(f"   QDRANT_URL={DEST_URL}")
    print(f"   QDRANT_SECURITY={DEST_API_KEY}")
    print(f"   QDRANT_COLLECTION={DEST_COLLECTION}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
