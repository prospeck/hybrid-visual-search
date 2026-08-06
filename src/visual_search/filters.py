from __future__ import annotations

from visual_search.models import ObjectConstraint


def object_filter_score(
    counts: dict[str, int], constraints: list[ObjectConstraint], mode: str
) -> tuple[bool, float]:
    if not constraints:
        return True, 1.0
    matches = []
    for constraint in constraints:
        actual = counts.get(constraint.label, 0)
        matched = actual >= constraint.min_count and (
            constraint.max_count is None or actual <= constraint.max_count
        )
        matches.append(matched)
    passed = all(matches) if mode == "all" else any(matches)
    return passed, sum(matches) / len(matches)
