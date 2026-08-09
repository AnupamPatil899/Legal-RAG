# Qdrant Vector Database Optimization & Troubleshooting Guide

This guide documents the architecture, performance tuning, errors encountered, root causes, and resolution strategies for the **Qdrant Vector Database** integrated into the Enterprise Agentic RAG pipeline.

---

## 1. System Specifications & Canonical Collection

- **Active Collection Name**: `enterprise_rag`
- **Vector Dimension**: `1024` (matching `jina-embeddings-v3` / `mxbai-embed-large-v1`)
- **Distance Metric**: `Dot` (Cosine similarity normalized)
- **Current Production State**:
  - **Status**: `GREEN`
  - **Points Count**: `19,050`
  - **Indexed Vectors Count**: `19,050` (100% HNSW indexed)
  - **Active Segments**: `2`

---

## 2. Issues Encountered, Root Causes & Solutions

### Issue 1: Collection Status `grey` and `indexed_vectors_count: 0`

#### Symptom:
After ingesting 19,050 vectors into Qdrant, queries were working via fallback linear scanning, but the collection status remained `grey` and `indexed_vectors_count` was `0`.

#### Root Cause:
1. **Severe Segment Fragmentation**: Ingestion uploaded data file-by-file in small batches, creating **74 to 90 tiny segments** on disk (~250 vectors per segment).
2. **Indexing Threshold Mismatch**: Qdrant's default `indexing_threshold` is set to `10,000` vectors per segment. Because no individual small segment reached 10,000 vectors, the background optimizer never triggered HNSW graph construction.

#### Solution:
Updated collection configuration via `client.update_collection()` in [scripts/create_qdrant_collection.py](file:///home/anupa/CUDA_Rag/scripts/create_qdrant_collection.py) to lower `indexing_threshold` to `2000` and set `default_segment_number=2`.

---

### Issue 2: Collection Status `red` with `Too many open files (os error 24)`

#### Symptom:
Qdrant logged the following runtime error:
```json
{
  "status": "red",
  "optimizer_status": {
    "error": "Service runtime error: IO Error: failed to rename file from ... to ...: Too many open files (os error 24)"
  }
}
```

#### Root Cause:
1. **Linux Container File Descriptor Limits (`ulimit -n 1024`)**:
   Cloud Run containers have a default soft limit of 1,024 open file descriptors.
2. **File Descriptor Explosion**:
   Each Qdrant segment opens ~15 file handles (binary vector files, HNSW graphs, payload DB, WAL logs).
   Having 92 unmerged segments opened `92 × 15 = ~1,380` file handles simultaneously.
3. **Parallel Optimization Overload**:
   When `max_optimization_threads` was set to `2`, Qdrant attempted parallel segment merges, exceeding the container's 1,024 file descriptor limit. The OS rejected file operations with Linux Error 24 (`Too many open files`).

#### Solution:
1. **Single-Threaded Sequential Optimization**:
   Configured `max_optimization_threads=1` and `max_indexing_threads=1` in [scripts/create_qdrant_collection.py](file:///home/anupa/CUDA_Rag/scripts/create_qdrant_collection.py), keeping open file handles under ~200 (well within the 1,024 limit).
2. **Bulk Ingestion Batching**:
   Updated [app/ingestion/processor.py](file:///home/anupa/CUDA_Rag/app/ingestion/processor.py#L60-L180) to queue points in memory across files and flush in **large bulk batches of 4,500 points**.
   - **Before**: 90 files → 90 separate segments.
   - **After**: 90 files → **2 segments**.

---

### Issue 3: Collection Name Clarification (`enterprise_rag` vs `enterprise_rag_test`)

#### Question:
*Why were `enterprise_rag` and `enterprise_rag_test` both mentioned during troubleshooting?*

#### Explanation:
- **`enterprise_rag`**: The **canonical, permanent collection name** configured across [.env](file:///home/anupa/CUDA_Rag/.env), [app/config.py](file:///home/anupa/CUDA_Rag/app/config.py#L45), [app/main.py](file:///home/anupa/CUDA_Rag/app/main.py), and [app/ingestion/processor.py](file:///home/anupa/CUDA_Rag/app/ingestion/processor.py).
- **`enterprise_rag_test`**: A temporary migration target collection created during file-descriptor limit debugging to verify zero-loss data scrolling. All points have been safely consolidated into `enterprise_rag`.

---

### Issue 4: Dashboard shows 0 Points Temporarily During Ingestion

#### Symptom:
When running `python -m app.ingestion.processor DATA --wipe`, opening the Qdrant Dashboard (`/dashboard#/collections`) initially shows **0 Points** or `No Points present`.

#### Root Cause:
1. **`--wipe` Lifecycle**: At script startup, `--wipe` immediately deletes the old collection and re-creates a clean, empty `enterprise_rag` collection.
2. **In-Memory Embedding Overhead**: `processor.py` parses all documents and calls `embed_texts()` via Jina AI API in memory before pushing vectors to Qdrant.
3. **Delayed Batch Flush**: Points are held in memory until `BATCH_QDRANT_POINTS` hits 4,500 points (or when ingestion finishes), at which point `flush_qdrant_points()` executes bulk upsert. During embedding computation (~1-2 minutes), Qdrant accurately reports 0 points until the first batch is flushed.

---

## 3. Operations & Emergency Runbook

### Quick Diagnostic One-Liner
Check current collection status, point count, indexed vector count, and segment count:
```bash
python -c "from app.services.retrieval.vectordb_service import get_qdrant_client; client = get_qdrant_client(); info = client.get_collection('enterprise_rag'); print(f'Status: {info.status} | Points: {info.points_count} | Indexed: {info.indexed_vectors_count} | Segments: {info.segments_count}')"
```

### Apply Collection Optimizer Configuration
Update `indexing_threshold`, `max_optimization_threads`, and `default_segment_number` without re-ingesting data:
```bash
python scripts/create_qdrant_collection.py
```

### Defragment & Migrate Segments
If segments proliferate (>50 segments) due to ad-hoc single-point inserts, scroll all points into large 4,500-point batches:
```bash
python scripts/defrag_qdrant.py
```

### Clean Data Re-Ingestion
Parse, embed, and bulk-upsert all documents from `DATA/` into a clean collection:
```bash
python -m app.ingestion.processor DATA --wipe
```

---

## 4. Architectural Summary & Best Practices

1. **Maintain Batch Upserts**: Keep `BATCH_SIZE = 4500` in `processor.py` to keep segment counts between **2 and 5 segments**, staying safely within container file descriptor limits (`ulimit -n 1024`).
2. **Use Recommended Optimizer Settings**:
   ```python
   optimizer_config = OptimizersConfigDiff(
       indexing_threshold=2000,
       max_optimization_threads=2,
       default_segment_number=2,
   )
   ```

