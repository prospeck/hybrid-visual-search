from __future__ import annotations

import zipfile
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from visual_search.api import app
from visual_search.cli import benchmark, import_zip, index_directory, parser, search


def test_zip_import_is_flat_and_limited(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    Image.new("RGB", (10, 10), "red").save(source)
    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.write(source, "nested/source.jpg")
        output.writestr("nested/ignore.txt", "no")
    destination = tmp_path / "output"
    assert import_zip(archive, destination, 1) == 0
    assert (destination / "source.jpg").exists()


def test_api_health_upload_search_and_content(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISUAL_SEARCH_DATA_DIR", str(tmp_path / "images"))
    monkeypatch.setenv("VISUAL_SEARCH_DATABASE_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setenv("VISUAL_SEARCH_INDEX_PATH", str(tmp_path / "vectors.index"))
    monkeypatch.setenv("VISUAL_SEARCH_INDEX_MAP_PATH", str(tmp_path / "vectors.json"))
    monkeypatch.setenv("VISUAL_SEARCH_DETECTOR_BACKEND", "hash")
    monkeypatch.setenv("VISUAL_SEARCH_ENCODER_BACKEND", "hash")
    monkeypatch.setenv("VISUAL_SEARCH_EMBEDDING_DIMENSION", "64")
    image = tmp_path / "red_car.jpg"
    Image.new("RGB", (40, 30), "red").save(image)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        with image.open("rb") as handle:
            upload = client.post(
                "/v1/index/upload", files={"file": ("red_car.jpg", handle, "image/jpeg")}
            )
        assert upload.status_code == 200 and upload.json()["indexed"] == 1
        assert client.get("/v1/images").json()[0]["filename"] == "red_car.jpg"
        result = client.post("/v1/search", json={"query": "red car", "top_k": 5})
        assert result.status_code == 200 and result.json()["hits"]
        image_id = result.json()["hits"][0]["image_id"]
        assert client.get(f"/v1/images/{image_id}/content?annotated=true").status_code == 200
        assert client.get("/v1/classes").status_code == 200


def test_cli_index_search_and_benchmark(tmp_path: Path, image_dir: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISUAL_SEARCH_DATA_DIR", str(tmp_path / "cli-images"))
    monkeypatch.setenv("VISUAL_SEARCH_DATABASE_PATH", str(tmp_path / "cli.db"))
    monkeypatch.setenv("VISUAL_SEARCH_INDEX_PATH", str(tmp_path / "cli.index"))
    monkeypatch.setenv("VISUAL_SEARCH_INDEX_MAP_PATH", str(tmp_path / "cli.json"))
    monkeypatch.setenv("VISUAL_SEARCH_DETECTOR_BACKEND", "hash")
    monkeypatch.setenv("VISUAL_SEARCH_ENCODER_BACKEND", "hash")
    monkeypatch.setenv("VISUAL_SEARCH_EMBEDDING_DIMENSION", "64")
    assert index_directory(image_dir) == 0
    assert search("red car", None, 1, 3) == 0
    output = tmp_path / "benchmark.json"
    assert benchmark(None, output) == 0
    assert output.exists()


def test_cli_parser_has_all_commands() -> None:
    root = parser()
    assert root.parse_args(["search", "dog"]).command == "search"
    assert root.parse_args(["benchmark"]).command == "benchmark"
