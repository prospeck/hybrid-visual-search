# Four-minute demo

## 0:00-0:35 - Problem and two signals

“YOLO answers what named objects and counts are present. CLIP answers what a scene means. This
system combines both so users can search concepts and enforce exact object constraints.”

## 0:35-1:20 - Semantic search

Search `people riding bicycles in a city`. Show ranked images, cosine score, object metadata,
and optional boxes. Explain shared image/text embedding space and exact FAISS inner product.

## 1:20-2:00 - Structured filter

Add `person >= 2` and `bicycle >= 1` with AND mode. Contrast with OR. Change maximum count to
show deterministic filtering after semantic candidate retrieval.

## 2:00-2:40 - Ingestion and API

Upload an image. Show content hashing, duplicate skip, YOLO detections, CLIP embedding, SQLite
metadata, and FAISS persistence. Open <http://localhost:8020/docs> and show typed endpoints.

## 2:40-3:25 - Engineering evidence

Open `reports/BENCHMARK.md`: 6.239 images/s indexing, 56.676 ms p50, 70.754 ms p95. State the
exact CPU/model/corpus boundary and that cold load and GPU performance are not claimed.

## 3:25-4:00 - Scale discussion

Explain why Flat is correct for 30 images and how HNSW/IVF-PQ, object storage, PostgreSQL,
queued GPU batches, index versioning, access controls, and recall@k evaluation extend it.

