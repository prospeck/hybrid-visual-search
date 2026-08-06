from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VISUAL_SEARCH_", extra="ignore")

    data_dir: Path = Path("runtime/images")
    database_path: Path = Path("runtime/metadata.db")
    index_path: Path = Path("runtime/vectors.index")
    index_map_path: Path = Path("runtime/vectors.json")
    detector_backend: Literal["yolo", "hash"] = "yolo"
    detector_model: str = "yolo11n.pt"
    detector_confidence: float = Field(default=0.30, ge=0, le=1)
    encoder_backend: Literal["openclip", "hash"] = "openclip"
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    embedding_dimension: int = Field(default=512, ge=8, le=4096)
    max_upload_mb: int = Field(default=20, ge=1, le=200)
    max_batch_images: int = Field(default=5000, ge=1, le=100000)
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
