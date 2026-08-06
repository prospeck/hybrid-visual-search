from __future__ import annotations

import hashlib
import shutil
import statistics
import time
from pathlib import Path

from PIL import Image

from visual_search.config import Settings
from visual_search.engines import (
    Detector,
    Encoder,
    HashDetector,
    HashEncoder,
    OpenCLIPEncoder,
    YOLODetector,
)
from visual_search.filters import object_filter_score
from visual_search.models import (
    BenchmarkReport,
    IndexResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from visual_search.store import MetadataStore
from visual_search.vector_index import VectorIndex


def build_detector(settings: Settings) -> Detector:
    if settings.detector_backend == "hash":
        return HashDetector()
    return YOLODetector(settings.detector_model, settings.detector_confidence, settings.device)


def build_encoder(settings: Settings) -> Encoder:
    if settings.encoder_backend == "hash":
        return HashEncoder(settings.embedding_dimension)
    return OpenCLIPEncoder(settings.clip_model, settings.clip_pretrained, settings.device)


class VisualSearchService:
    def __init__(
        self,
        settings: Settings,
        detector: Detector | None = None,
        encoder: Encoder | None = None,
    ) -> None:
        self.settings = settings
        settings.ensure_directories()
        self.detector = detector or build_detector(settings)
        self.encoder = encoder or build_encoder(settings)
        self.store = MetadataStore(settings.database_path)
        self.index = VectorIndex(
            settings.index_path, settings.index_map_path, self.encoder.dimension
        )

    def index_paths(self, paths: list[Path], *, copy_images: bool = True) -> IndexResponse:
        if len(paths) > self.settings.max_batch_images:
            raise ValueError("Batch exceeds configured maximum")
        started = time.perf_counter()
        indexed = skipped = failed = 0
        for source in paths:
            try:
                if source.suffix.lower() not in self.settings.allowed_extensions:
                    failed += 1
                    continue
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                if self.store.contains_hash(digest):
                    skipped += 1
                    continue
                destination = self.settings.data_dir / f"{digest[:12]}-{source.name}"
                if copy_images:
                    shutil.copy2(source, destination)
                else:
                    destination = source.resolve()
                with Image.open(destination) as image:
                    width, height = image.size
                    image.verify()
                detections = self.detector.detect(destination)
                vector = self.encoder.encode_image(destination)
                image_id = self.store.add_image(
                    filename=source.name,
                    path=destination,
                    sha256=digest,
                    width=width,
                    height=height,
                    detector_model=self.detector.name,
                    embedding_model=self.encoder.name,
                    detections=detections,
                )
                self.index.add(image_id, vector)
                indexed += 1
            except Exception as error:
                failed += 1
                self.store.event("index_error", {"path": str(source), "error": str(error)})
        self.index.save()
        duration_ms = (time.perf_counter() - started) * 1000
        response = IndexResponse(
            indexed=indexed,
            skipped_duplicates=skipped,
            failed=failed,
            duration_ms=round(duration_ms, 3),
            images_per_second=round(indexed / max(duration_ms / 1000, 1e-9), 3),
            detector_device=self.detector.device,
            vector_backend=self.index.backend,
        )
        self.store.event("batch_index", response.model_dump())
        return response

    def search(self, request: SearchRequest) -> SearchResponse:
        started = time.perf_counter()
        if not request.query and not request.objects:
            raise ValueError("Provide a semantic query, object filters, or both")
        if request.query:
            candidate_count = min(max(request.top_k * 10, 100), max(len(self.index), 1))
            ranked = self.index.search(self.encoder.encode_text(request.query), candidate_count)
        else:
            ranked = [(record.image_id, 0.0) for record in self.store.list_images()]
        hits: list[SearchHit] = []
        for image_id, semantic_score in ranked:
            record = self.store.get(image_id)
            if record is None:
                continue
            passed, object_score = object_filter_score(
                record.object_counts, request.objects, request.object_mode
            )
            if not passed:
                continue
            semantic_normalized = max(0.0, min(1.0, (semantic_score + 1) / 2))
            weight = request.semantic_weight if request.query else 0.0
            score = weight * semantic_normalized + (1 - weight) * object_score
            hits.append(
                SearchHit(
                    image_id=record.image_id,
                    filename=record.filename,
                    image_url=f"/v1/images/{record.image_id}/content",
                    score=round(score, 6),
                    semantic_score=round(semantic_score, 6),
                    object_score=round(object_score, 6),
                    object_counts=record.object_counts,
                    detections=record.detections,
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        elapsed = (time.perf_counter() - started) * 1000
        response = SearchResponse(
            query=request.query,
            hits=hits[: request.top_k],
            total_candidates=len(ranked),
            latency_ms=round(elapsed, 3),
            vector_backend=self.index.backend,
        )
        self.store.event(
            "search",
            {
                "query": request.query,
                "filters": [item.model_dump() for item in request.objects],
                "latency_ms": response.latency_ms,
                "hits": len(response.hits),
            },
        )
        return response

    def benchmark(self, queries: list[str], indexing: IndexResponse) -> BenchmarkReport:
        latencies = [
            self.search(SearchRequest(query=query, top_k=10)).latency_ms for query in queries
        ]
        ordered = sorted(latencies)
        p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
        return BenchmarkReport(
            indexed_images=self.store.count(),
            device=self.encoder.device,
            detector_model=self.detector.name,
            embedding_model=self.encoder.name,
            vector_backend=self.index.backend,
            indexing_images_per_second=indexing.images_per_second,
            indexing_ms_per_image=round(indexing.duration_ms / max(indexing.indexed, 1), 3),
            search_samples=len(latencies),
            search_p50_ms=round(statistics.median(latencies), 3),
            search_p95_ms=round(ordered[p95_index], 3),
            search_mean_ms=round(statistics.mean(latencies), 3),
            notes=[
                "Cold model load is excluded; search timings include text encoding and filtering."
            ],
        )
