# Verified CPU benchmark

The measured run used 30 COCO validation images imported from the user-supplied archive,
YOLO11n, OpenCLIP ViT-B/32 with LAION2B weights, and `faiss.IndexFlatIP`. PyTorch was CPU-only;
the host RTX 3050 was detected but did not execute inference.

| Metric | Result |
|---|---:|
| Warm indexing throughput | 6.239 images/second |
| Warm indexing time | 160.279 ms/image |
| Search samples | 20 |
| Search p50 | 56.676 ms |
| Search p95 | 70.754 ms |
| Search mean | 56.708 ms |

Search timing includes CLIP text encoding, FAISS retrieval, SQLite metadata reads, object
filtering, and score fusion. First-run model download/load is intentionally excluded and must
be reported separately. This small-corpus result is a reproducible local baseline, not a claim
about million-image scale or GPU performance.

