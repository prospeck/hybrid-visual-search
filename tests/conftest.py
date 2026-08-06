from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from visual_search.config import Settings
from visual_search.engines import HashDetector, HashEncoder
from visual_search.service import VisualSearchService


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "images",
        database_path=tmp_path / "metadata.db",
        index_path=tmp_path / "vectors.index",
        index_map_path=tmp_path / "vectors.json",
        detector_backend="hash",
        encoder_backend="hash",
        embedding_dimension=64,
    )


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "source"
    directory.mkdir()
    for name, color in [
        ("red_car.jpg", "red"),
        ("person_bicycle.jpg", "blue"),
        ("dog_beach.jpg", "yellow"),
        ("food_table.jpg", "green"),
    ]:
        Image.new("RGB", (80, 60), color).save(directory / name)
    return directory


@pytest.fixture
def service(settings: Settings) -> VisualSearchService:
    return VisualSearchService(settings, HashDetector(), HashEncoder(64))
