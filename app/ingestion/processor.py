import json
import os
import sys
import uuid
import logfire
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
# load_dotenv(override=True)


from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

from app.config import settings

# # Initialize Logfire
# logfire_token = getattr(settings, "LOGFIRE_WRITE_TOKEN", None) or settings.LOGFIRE_TOKEN
# logfire_base_url = settings.LOGFIRE_BASE_URL

# if logfire_token:
#     os.environ["LOGFIRE_TOKEN"] = logfire_token
#     logfire_kwargs = {"token": logfire_token}
#     if logfire_base_url:
#         logfire_kwargs["advanced"] = logfire.AdvancedOptions(base_url=logfire_base_url)
#     logfire.configure(**logfire_kwargs)
# else:
#     logfire.configure(send_to_logfire=False)
logfire.configure()
from app.ingestion.chunking.splitter import chunk_text
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.text import parse_text
from app.services.retrieval.embedding import embed_texts, get_embedding_dim


from app.services.retrieval.vectordb_service import get_pinecone_index

PROCESSED_DATA_DIR = "processed_data"

# Initialize Pinecone Client
_pinecone_api_key = getattr(settings, "PINECONE_API_KEY", None) or os.getenv("PINECONE_API_KEY") or "pclocal"
_pinecone_host = getattr(settings, "PINECONE_HOST", None) or os.getenv("PINECONE_HOST")

_pc_kwargs = {"api_key": _pinecone_api_key}
if _pinecone_host:
    _pc_kwargs["host"] = _pinecone_host

pc = Pinecone(**_pc_kwargs)

# Initialize BM25 Encoder for Hybrid Search (Sparse Vectors)
bm25 = BM25Encoder.default()


# Define your single master index name
MASTER_INDEX_NAME = getattr(settings, "PINECONE_INDEX_NAME", "legal-enterprise-knowledge-base")


# Load Metadata globally to avoid reading the file repeatedly
METADATA_CSV_PATH = "/home/anupa/CUDA_Rag/DATA/master_clauses_updated_final.csv"
if os.path.exists(METADATA_CSV_PATH):
    df_meta = pd.read_csv(METADATA_CSV_PATH, encoding="Windows-1252").fillna("")
    METADATA_MAP = df_meta.set_index("Filename").to_dict(orient="index")
else:
    logfire.warning("metadata.csv not found. Operating without extended metadata. logfire")
    print("metadata.csv not found. Operating without extended metadata.")
    METADATA_MAP = {}

# Ensure the master index exists ONCE at startup
try:
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if MASTER_INDEX_NAME not in existing_indexes:
        dim = get_embedding_dim()
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
except Exception as e:
    print(f"Note: Pinecone index check/creation skipped: {e}")

# Connect to the single master index
master_index = get_pinecone_index(pc, MASTER_INDEX_NAME)


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def process_file(file_path: str, filename: str, source_type: str):
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
            chunks = chunk_text(full_text)
            if not chunks:
                return

            save_processed_locally(  # noqa: F841
                {"filename": filename, "source_type": source_type, "chunks": chunks}, source_type, filename
            )

            # 3. Grab Metadata (folder_name is already inside file_meta)
            file_meta = METADATA_MAP.get(filename, {})
            # Ensure folder_name exists so we can filter on it later
            if "folder_name" not in file_meta:
                file_meta["folder_name"] = source_type

            # 4. Vectorize & Upsert to Master Index
            with logfire.span("Vectorizing & Indexing"):
                # if 2 > 1:
                dense_embeddings = embed_texts(chunks)
                sparse_embeddings = bm25.encode_documents(chunks)

                points = []
                for chunk, dense, sparse in zip(chunks, dense_embeddings, sparse_embeddings):
                    payload = dict(file_meta)
                    payload["text"] = chunk
                    payload["source"] = filename
                    payload["source_type"] = source_type

                    # Pinecone limits sparse vector size to max 2048 elements
                    if sparse and isinstance(sparse, dict) and "indices" in sparse and len(sparse["indices"]) > 2048:
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

        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")
            print(f"Failed to process {filename}: {e}")


def process_directory(dir_path: str, source_type: str):
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        # if 2 > 1:
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        for filename in files:
            process_file(os.path.join(dir_path, filename), filename, source_type)


def run_universal_ingestion(base_dir: str, explicit_source_type: str = None):
    with logfire.span("Universal Ingestion Started", base_directory=base_dir):
        # if 2 > 1:
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not subdirs:
            source_type = explicit_source_type or "general"
            process_directory(base_dir, source_type)
        else:
            for subdir in subdirs:
                process_directory(os.path.join(base_dir, subdir), subdir)


if __name__ == "__main__":
    clean_args = [a for a in sys.argv if a != "--wipe"]
    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        sys.exit(1)

    run_universal_ingestion(target_dir, explicit_source_type=explicit_type)
    logfire.info("Ingestion job completed.")
    print("Ingestion job completed.")
