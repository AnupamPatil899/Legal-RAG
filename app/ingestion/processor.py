import ast
import json
import os
import sys
import time
import uuid
import logfire
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from pinecone import Pinecone, ServerlessSpec

from app.config import settings

logfire.configure()
from app.ingestion.chunking.splitter import chunk_text
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.text import parse_text
from app.services.retrieval.embedding import embed_texts, get_embedding_dim

from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointStruct,
    TextIndexParams,
    TokenizerType,
    VectorParams,
)

from app.services.retrieval.vectordb_service import get_pinecone_index, get_qdrant_client

PROCESSED_DATA_DIR = "processed_data"

# Initialize Qdrant Client (if configured)
qdrant_client = get_qdrant_client()
QDRANT_COLLECTION = (
    getattr(settings, "QDRANT_COLLECTION", None) or os.getenv("QDRANT_COLLECTION") or "enterprise_rag_v1"
)

# Initialize Pinecone Client
_pinecone_api_key = getattr(settings, "PINECONE_API_KEY", None) or os.getenv("PINECONE_API_KEY") or "pclocal"
_pinecone_host = getattr(settings, "PINECONE_HOST", None) or os.getenv("PINECONE_HOST")

_pc_kwargs = {"api_key": _pinecone_api_key}
if _pinecone_host:
    _pc_kwargs["host"] = _pinecone_host

pc = Pinecone(**_pc_kwargs) if (_pinecone_api_key or _pinecone_host) else None

_bm25_encoder = None


def get_bm25_encoder():
    global _bm25_encoder
    if _bm25_encoder is None:
        try:
            from pinecone_text.sparse import BM25Encoder

            _bm25_encoder = BM25Encoder.default()
        except Exception:
            _bm25_encoder = None
    return _bm25_encoder


# Define single master index name
MASTER_INDEX_NAME = getattr(settings, "PINECONE_INDEX_NAME", "legal-enterprise-knowledge-base")

# Global point batch queue to keep segment count low
BATCH_QDRANT_POINTS = []
BATCH_SIZE = 100
UPSERT_SUB_BATCH = 100  # Max points per single Qdrant upsert HTTP call
TOTAL_UPSERTED = 0

# Load Metadata globally to avoid reading the file repeatedly
METADATA_PATHS = [
    "/home/anupa/CUDA_Rag/DATA/master_clauses_updated_final.csv",
    "DATA/master_clauses_updated_final.csv",
    "DATA/metadata.csv",
    "/home/anupa/CUDA_Rag/DATA/metadata.csv",
]

METADATA_MAP = {}
for p in METADATA_PATHS:
    if os.path.exists(p):
        try:
            df_meta = pd.read_csv(p, encoding="Windows-1252").fillna("")
            if "Filename" in df_meta.columns:
                METADATA_MAP = df_meta.set_index("Filename").to_dict(orient="index")
                print(f"✅ Loaded metadata from {p} ({len(METADATA_MAP)} files).")
                break
        except Exception as e:
            print(f"Warning: Failed to load metadata from {p}: {e}")

if not METADATA_MAP:
    logfire.warning("metadata.csv not found. Operating without extended metadata.")
    print("metadata.csv not found. Operating without extended metadata.")


def _clean_meta_value(val) -> str:
    """Helper to clean stringified python lists from CSV columns into readable text."""
    if not val:
        return ""
    if isinstance(val, (list, tuple)):
        return ", ".join(str(x) for x in val if str(x).strip())
    val_str = str(val).strip()
    if val_str.startswith("[") and val_str.endswith("]"):
        try:
            parsed = ast.literal_eval(val_str)
            if isinstance(parsed, (list, tuple)):
                return ", ".join(str(x) for x in parsed if str(x).strip())
        except Exception:
            return val_str.strip("[]'\" ")
    return val_str


def _ensure_payload_indices(client, collection_name: str):
    """Creates text and keyword payload indices in Qdrant for high-speed filtering."""
    if not client:
        return
    text_fields = ["Parties", "Parties-Answer", "source"]
    for f in text_fields:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=f,
                field_schema=TextIndexParams(
                    type=PayloadSchemaType.TEXT,
                    tokenizer=TokenizerType.WORD,
                    lowercase=True,
                ),
            )
        except Exception:
            pass

    keyword_fields = ["folder_name"]
    for f in keyword_fields:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=f,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass


# Ensure the collection/index exists at startup
dim = get_embedding_dim()

if qdrant_client:
    try:
        if "--wipe" in sys.argv:
            if qdrant_client.collection_exists(QDRANT_COLLECTION):
                print(f"🗑️ --wipe passed: Deleting collection '{QDRANT_COLLECTION}'...")
                qdrant_client.delete_collection(QDRANT_COLLECTION)

            # Wait for Qdrant to fully remove the storage folder on disk
            for attempt in range(15):
                if not qdrant_client.collection_exists(QDRANT_COLLECTION):
                    break
                time.sleep(2)
            else:
                print("⚠️ Warning: Collection still reported as existing after 30s wait.")

            # Retry create_collection in case the disk folder lingers
            created = False
            for attempt in range(10):
                try:
                    qdrant_client.create_collection(
                        collection_name=QDRANT_COLLECTION,
                        vectors_config=VectorParams(size=dim, distance=Distance.DOT),
                        hnsw_config=HnswConfigDiff(max_indexing_threads=2),
                        optimizers_config=OptimizersConfigDiff(
                            indexing_threshold=2000,
                            max_optimization_threads=2,
                            default_segment_number=2,
                        ),
                    )
                    created = True
                    print(f"✅ Created new Qdrant collection: {QDRANT_COLLECTION}")
                    _ensure_payload_indices(qdrant_client, QDRANT_COLLECTION)
                    break
                except Exception as ce:
                    if "already exists" in str(ce) and attempt < 9:
                        print(f"   Qdrant storage still cleaning up, retrying in 3s... (attempt {attempt + 1}/10)")
                        time.sleep(3)
                    else:
                        raise
            if not created:
                print("❌ Failed to create collection after retries.")
        else:
            # No --wipe: just ensure collection exists and indices are present
            if not qdrant_client.collection_exists(QDRANT_COLLECTION):
                qdrant_client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=VectorParams(size=dim, distance=Distance.DOT),
                    hnsw_config=HnswConfigDiff(max_indexing_threads=2),
                    optimizers_config=OptimizersConfigDiff(
                        indexing_threshold=2000,
                        max_optimization_threads=2,
                        default_segment_number=2,
                    ),
                )
                logfire.info(f"Created new Qdrant collection: {QDRANT_COLLECTION}")
                print(f"Created new Qdrant collection: {QDRANT_COLLECTION}")
            _ensure_payload_indices(qdrant_client, QDRANT_COLLECTION)
    except Exception as e:
        print(f"Note: Qdrant collection check/creation skipped: {e}")

master_index = None
if pc and not qdrant_client:
    try:
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if MASTER_INDEX_NAME not in existing_indexes:
            create_kwargs = {
                "name": MASTER_INDEX_NAME,
                "dimension": dim,
                "metric": "dotproduct",  # Required for Hybrid Search
            }
            if not _pinecone_host:
                create_kwargs["spec"] = ServerlessSpec(cloud="aws", region="us-east-1")
            pc.create_index(**create_kwargs)
            logfire.info(f"Created new master Pinecone index: {MASTER_INDEX_NAME}")
            print(f"Created new master Pinecone index: {MASTER_INDEX_NAME}")
        master_index = get_pinecone_index(pc, MASTER_INDEX_NAME)
    except Exception as e:
        print(f"Note: Pinecone index check/creation skipped: {e}")


def flush_qdrant_points():
    global BATCH_QDRANT_POINTS, TOTAL_UPSERTED
    if qdrant_client and BATCH_QDRANT_POINTS:
        try:
            if not qdrant_client.collection_exists(QDRANT_COLLECTION):
                qdrant_client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=VectorParams(size=dim, distance=Distance.DOT),
                    hnsw_config=HnswConfigDiff(max_indexing_threads=2),
                    optimizers_config=OptimizersConfigDiff(
                        indexing_threshold=2000,
                        max_optimization_threads=2,
                        default_segment_number=2,
                    ),
                )
                print(f"✨ Auto-created Qdrant collection '{QDRANT_COLLECTION}' before bulk upsert.")
                _ensure_payload_indices(qdrant_client, QDRANT_COLLECTION)
        except Exception as e:
            print(f"Note: Qdrant collection auto-creation check: {e}")

        # Upsert in sub-batches to avoid 413 Request Entity Too Large from Qdrant REST API
        total = len(BATCH_QDRANT_POINTS)
        for i in range(0, total, UPSERT_SUB_BATCH):
            sub_batch = BATCH_QDRANT_POINTS[i : i + UPSERT_SUB_BATCH]
            qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=sub_batch)
        TOTAL_UPSERTED += total
        logfire.info(f"Bulk indexed {total} points to Qdrant. (Total in collection: {TOTAL_UPSERTED})")
        print(f"🚀 Indexed {total} points to Qdrant Cloud. (Total stored so far: {TOTAL_UPSERTED})")
        BATCH_QDRANT_POINTS = []


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def process_file(file_path: str, filename: str, source_type: str):
    global BATCH_QDRANT_POINTS
    with logfire.span("Processing File", file=filename, source=source_type):
        try:
            # 1. Extract text
            ext = filename.lower().rsplit(".", 1)[-1]
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)
            elif ext == "txt":
                full_text = parse_text(file_path)
            elif ext in ("docx", "pptx"):
                from app.ingestion.loaders.office import parse_office

                full_text = parse_office(file_path)
            else:
                logfire.warning(f"Skipping unsupported file type: {filename}")
                print(f"Skipping unsupported file type: {filename}")
                return

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from {filename} — skipping.")
                print(f"No text extracted from {filename} — skipping.")
                return

            # 2. Chunk text
            raw_chunks = chunk_text(full_text)
            if not raw_chunks:
                return

            # 3. Grab & Format Metadata for Contextual Prepending
            file_meta = METADATA_MAP.get(filename, {})
            if "folder_name" not in file_meta:
                file_meta["folder_name"] = source_type

            doc_name = _clean_meta_value(
                file_meta.get("Document Name") or file_meta.get("Document Name-Answer") or filename
            )
            parties = _clean_meta_value(file_meta.get("Parties") or file_meta.get("Parties-Answer") or "")
            gov_law = _clean_meta_value(file_meta.get("Governing Law") or file_meta.get("Governing Law-Answer") or "")

            # Build contextual header
            header_parts = []
            if doc_name:
                header_parts.append(f"DOCUMENT: {doc_name}")
            if parties:
                header_parts.append(f"PARTIES: {parties}")
            if gov_law:
                header_parts.append(f"GOVERNING LAW: {gov_law}")
            header_parts.append(f"SOURCE: {filename}")
            header = f"[{' | '.join(header_parts)}]"

            # Prepend header to each chunk so the vector representation captures contract identity
            contextual_chunks = [f"{header}\n\n{rc}" for rc in raw_chunks]

            save_processed_locally(  # noqa: F841
                {"filename": filename, "source_type": source_type, "header": header, "chunks": contextual_chunks},
                source_type,
                filename,
            )

            # 4. Vectorize & Queue for Bulk Upsert
            with logfire.span("Vectorizing & Indexing"):
                dense_embeddings = embed_texts(contextual_chunks)

                if qdrant_client:
                    for ctx_chunk, raw_chunk, dense in zip(contextual_chunks, raw_chunks, dense_embeddings):
                        payload = dict(file_meta)
                        payload["text"] = ctx_chunk
                        payload["raw_text"] = raw_chunk
                        payload["header"] = header
                        payload["source"] = filename
                        payload["source_type"] = source_type

                        BATCH_QDRANT_POINTS.append(
                            PointStruct(
                                id=str(uuid.uuid4()),
                                vector=dense,
                                payload=payload,
                            )
                        )
                        if len(BATCH_QDRANT_POINTS) >= BATCH_SIZE:
                            flush_qdrant_points()

                elif master_index:
                    encoder = get_bm25_encoder()
                    sparse_embeddings = (
                        encoder.encode_documents(contextual_chunks) if encoder else [{}] * len(contextual_chunks)
                    )
                    points = []
                    for ctx_chunk, raw_chunk, dense, sparse in zip(
                        contextual_chunks, raw_chunks, dense_embeddings, sparse_embeddings
                    ):
                        payload = dict(file_meta)
                        payload["text"] = ctx_chunk
                        payload["raw_text"] = raw_chunk
                        payload["header"] = header
                        payload["source"] = filename
                        payload["source_type"] = source_type

                        if (
                            sparse
                            and isinstance(sparse, dict)
                            and "indices" in sparse
                            and len(sparse["indices"]) > 2048
                        ):
                            pairs = sorted(zip(sparse["indices"], sparse["values"]), key=lambda x: x[1], reverse=True)[
                                :2048
                            ]
                            sparse = {"indices": [p[0] for p in pairs], "values": [p[1] for p in pairs]}

                        points.append(
                            {"id": str(uuid.uuid4()), "values": dense, "sparse_values": sparse, "metadata": payload}
                        )

                    master_index.upsert(vectors=points)
                    logfire.info(f"Indexed {len(points)} points to '{MASTER_INDEX_NAME}'.")
                    print(f"Indexed {len(points)} points to '{MASTER_INDEX_NAME}'.")
                else:
                    print("⚠️ No active Vector DB client configured for indexing.")

        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")
            print(f"Failed to process {filename}: {e}")


def process_directory(dir_path: str, source_type: str):
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        total_files = len(files)
        print(f"📁 Found {total_files} files in {dir_path}. Starting ingestion...")
        for idx, filename in enumerate(files, 1):
            print(f"[{idx}/{total_files}] Processing {filename}...")
            process_file(os.path.join(dir_path, filename), filename, source_type)


def run_universal_ingestion(base_dir: str, explicit_source_type: str = None):
    with logfire.span("Universal Ingestion Started", base_directory=base_dir):
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not subdirs:
            source_type = explicit_source_type or "general"
            process_directory(base_dir, source_type)
        else:
            for subdir in subdirs:
                process_directory(os.path.join(base_dir, subdir), subdir)

        # Flush any remaining queued points
        flush_qdrant_points()


if __name__ == "__main__":
    clean_args = [a for a in sys.argv if a != "--wipe"]
    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        sys.exit(1)

    run_universal_ingestion(target_dir, explicit_source_type=explicit_type)
    logfire.info("Ingestion job completed.")
    print(f"🎉 Ingestion job completed! Total points stored in Qdrant: {TOTAL_UPSERTED}")
