# Case study: from YOLO demo to hybrid retrieval platform

The supplied baseline successfully ran YOLO and filtered JSON metadata in Streamlit, but it had
hardcoded CUDA, machine-specific absolute paths, no semantic retrieval, no API, no persistence
contract, no tests, and ambiguous maximum-count filtering.

The rebuilt system treats object detection, semantic encoding, vector retrieval, metadata, and
presentation as independent components. A natural-language query searches real CLIP image
embeddings through FAISS, while YOLO counts enforce exact minimum/maximum constraints. SQLite
links vector IDs to hashes, bounding boxes, model versions, and managed files. FastAPI and a
Streamlit client expose the same typed contract.

On a bounded 30-image COCO sample, the verified CPU path indexed 6.239 images/second and served
20 warm end-to-end searches at p50 56.676 ms and p95 70.754 ms. The most important limitation is
scale: exact FAISS retrieval on 30 images demonstrates integration and measurement mechanics,
not million-image performance. The next experiment would compare Flat, HNSW, and IVF-PQ on a
larger labeled corpus using recall@k and concurrent latency.

