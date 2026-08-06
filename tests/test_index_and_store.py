from __future__ import annotations

import json


def test_batch_index_persists_metadata_and_vectors(service, image_dir) -> None:
    result = service.index_paths(list(image_dir.iterdir()))
    assert result.indexed == 4
    assert result.failed == 0
    assert service.store.count() == 4
    assert len(service.index) == 4
    assert service.index.map_path.exists()


def test_duplicate_content_is_skipped(service, image_dir) -> None:
    paths = list(image_dir.iterdir())
    service.index_paths(paths)
    second = service.index_paths(paths)
    assert second.indexed == 0
    assert second.skipped_duplicates == 4


def test_vector_index_reloads(service, image_dir, settings) -> None:
    service.index_paths(list(image_dir.iterdir()))
    from visual_search.engines import HashDetector, HashEncoder
    from visual_search.service import VisualSearchService

    loaded = VisualSearchService(settings, HashDetector(), HashEncoder(64))
    assert len(loaded.index) == 4
    assert json.loads(settings.index_map_path.read_text())


def test_store_has_normalized_detection_rows(service, image_dir) -> None:
    service.index_paths([image_dir / "red_car.jpg"])
    record = service.store.list_images()[0]
    assert record.width == 80 and record.height == 60
    assert record.sha256
    assert record.path.exists()
