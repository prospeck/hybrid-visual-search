from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    label: str = Field(min_length=1, max_length=100)
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox


class ImageRecord(BaseModel):
    image_id: int
    filename: str
    path: Path
    sha256: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    detector_model: str
    embedding_model: str
    detections: list[Detection] = Field(default_factory=list)

    @property
    def object_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for detection in self.detections:
            counts[detection.label] = counts.get(detection.label, 0) + 1
        return counts


class ObjectConstraint(BaseModel):
    label: str = Field(min_length=1, max_length=100)
    min_count: int = Field(default=1, ge=0, le=10000)
    max_count: int | None = Field(default=None, ge=0, le=10000)

    @field_validator("max_count")
    @classmethod
    def maximum_must_be_valid(cls, value: int | None, info: ValidationInfo) -> int | None:
        minimum = info.data.get("min_count", 1)
        if value is not None and value < minimum:
            raise ValueError("max_count must be greater than or equal to min_count")
        return value


class SearchRequest(BaseModel):
    query: str = Field(default="", max_length=500)
    objects: list[ObjectConstraint] = Field(default_factory=list, max_length=20)
    object_mode: Literal["all", "any"] = "all"
    top_k: int = Field(default=12, ge=1, le=100)
    semantic_weight: float = Field(default=0.75, ge=0, le=1)

    @field_validator("query")
    @classmethod
    def query_or_filter_required(cls, value: str) -> str:
        return value.strip()


class SearchHit(BaseModel):
    image_id: int
    filename: str
    image_url: str
    score: float
    semantic_score: float
    object_score: float
    object_counts: dict[str, int]
    detections: list[Detection]


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    total_candidates: int
    latency_ms: float
    vector_backend: str


class IndexResponse(BaseModel):
    indexed: int
    skipped_duplicates: int
    failed: int
    duration_ms: float
    images_per_second: float
    detector_device: str
    vector_backend: str


class HealthResponse(BaseModel):
    status: str
    indexed_images: int
    detector_backend: str
    encoder_backend: str
    vector_backend: str
    device: str


class BenchmarkReport(BaseModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    indexed_images: int
    device: str
    detector_model: str
    embedding_model: str
    vector_backend: str
    indexing_images_per_second: float
    indexing_ms_per_image: float
    search_samples: int
    search_p50_ms: float
    search_p95_ms: float
    search_mean_ms: float
    notes: list[str] = Field(default_factory=list)
