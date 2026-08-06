from __future__ import annotations

import pytest

from visual_search.filters import object_filter_score
from visual_search.models import ObjectConstraint, SearchRequest


def test_minimum_and_maximum_object_filters() -> None:
    constraints = [ObjectConstraint(label="person", min_count=2, max_count=3)]
    assert object_filter_score({"person": 2}, constraints, "all") == (True, 1.0)
    assert object_filter_score({"person": 4}, constraints, "all") == (False, 0.0)


def test_all_and_any_semantics() -> None:
    constraints = [
        ObjectConstraint(label="person", min_count=1),
        ObjectConstraint(label="bicycle", min_count=1),
    ]
    counts = {"person": 2}
    assert object_filter_score(counts, constraints, "all")[0] is False
    assert object_filter_score(counts, constraints, "any")[0] is True


def test_semantic_search_returns_ranked_results(service, image_dir) -> None:
    service.index_paths(list(image_dir.iterdir()))
    response = service.search(SearchRequest(query="red car", top_k=3))
    assert response.hits
    assert response.hits[0].filename == "red_car.jpg"
    assert response.hits == sorted(response.hits, key=lambda item: item.score, reverse=True)


def test_hybrid_search_applies_filter(service, image_dir) -> None:
    service.index_paths(list(image_dir.iterdir()))
    records = service.store.list_images()
    target = next(record for record in records if record.detections)
    label = target.detections[0].label
    response = service.search(
        SearchRequest(
            query="image",
            objects=[ObjectConstraint(label=label, min_count=1)],
            top_k=10,
        )
    )
    assert all(hit.object_counts.get(label, 0) >= 1 for hit in response.hits)


def test_search_requires_query_or_filter(service) -> None:
    with pytest.raises(ValueError, match="Provide"):
        service.search(SearchRequest())


def test_benchmark_has_percentiles(service, image_dir) -> None:
    indexing = service.index_paths(list(image_dir.iterdir()))
    report = service.benchmark(["red car", "dog beach", "food table"], indexing)
    assert report.search_p95_ms >= report.search_p50_ms
    assert report.search_samples == 3
