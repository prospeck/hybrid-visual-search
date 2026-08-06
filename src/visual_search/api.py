from __future__ import annotations

import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated, cast

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageDraw

from visual_search.config import Settings
from visual_search.models import HealthResponse, IndexResponse, SearchRequest, SearchResponse
from visual_search.service import VisualSearchService
from visual_search.store import MetadataStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    settings.ensure_directories()
    app.state.settings = settings
    app.state.search_service = None
    yield


app = FastAPI(
    title="Hybrid Visual Search API",
    version="1.0.0",
    description="YOLO object filtering fused with CLIP/FAISS semantic retrieval.",
    lifespan=lifespan,
)


def settings() -> Settings:
    return cast(Settings, app.state.settings)


def service() -> VisualSearchService:
    current = app.state.search_service
    if current is None:
        current = VisualSearchService(settings())
        app.state.search_service = current
    return cast(VisualSearchService, current)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    configuration = settings()
    store = MetadataStore(configuration.database_path)
    current = app.state.search_service
    return HealthResponse(
        status="ok",
        indexed_images=store.count(),
        detector_backend=configuration.detector_backend,
        encoder_backend=configuration.encoder_backend,
        vector_backend=current.index.backend if current else "not_loaded",
        device=current.encoder.device if current else configuration.device,
    )


@app.post("/v1/index/upload", response_model=IndexResponse)
def upload_image(file: Annotated[UploadFile, File()]) -> IndexResponse:
    configuration = settings()
    original_name = Path(file.filename or "upload.jpg").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in configuration.allowed_extensions:
        raise HTTPException(status_code=415, detail="Unsupported image extension")
    with tempfile.TemporaryDirectory(prefix="visual-search-upload-") as temporary:
        path = Path(temporary) / original_name
        with path.open("wb") as destination:
            shutil.copyfileobj(file.file, destination)
        if path.stat().st_size > configuration.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image exceeds configured size limit")
        return service().index_paths([path])


@app.post("/v1/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    try:
        return service().search(request)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/v1/images/{image_id}/content")
def image_content(image_id: int, annotated: bool = False) -> Response:
    record = MetadataStore(settings().database_path).get(image_id)
    if record is None or not record.path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    if not annotated:
        return FileResponse(record.path)
    with Image.open(record.path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for detection in record.detections:
        box = detection.bbox
        draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline="#00ff88", width=3)
        draw.text((box.x1 + 3, box.y1 + 3), detection.label, fill="#00ff88")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return Response(content=buffer.getvalue(), media_type="image/jpeg")


@app.get("/v1/images")
def images(limit: int = Query(default=100, ge=1, le=1000)) -> list[dict[str, object]]:
    return [
        {
            "image_id": item.image_id,
            "filename": item.filename,
            "image_url": f"/v1/images/{item.image_id}/content",
            "object_counts": item.object_counts,
        }
        for item in MetadataStore(settings().database_path).list_images(limit)
    ]


@app.get("/v1/classes")
def classes() -> list[str]:
    return MetadataStore(settings().database_path).classes()
