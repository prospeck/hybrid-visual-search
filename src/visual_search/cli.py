from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from visual_search.config import Settings
from visual_search.models import SearchRequest
from visual_search.service import VisualSearchService


def import_zip(zip_path: Path, output: Path, limit: int) -> int:
    output.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            member
            for member in archive.infolist()
            if Path(member.filename).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            and not member.is_dir()
        ]
        for member in members[:limit]:
            filename = Path(member.filename).name
            destination = output / filename
            with archive.open(member) as source, destination.open("wb") as target:
                target.write(source.read())
            extracted += 1
    print(f"Imported {extracted} images into {output}")
    return 0


def index_directory(directory: Path) -> int:
    service = VisualSearchService(Settings())
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in service.settings.allowed_extensions
    )
    result = service.index_paths(paths)
    print(result.model_dump_json(indent=2))
    return 0 if result.failed == 0 else 2


def benchmark(directory: Path | None, output: Path) -> int:
    service = VisualSearchService(Settings())
    if directory:
        paths = sorted(
            path
            for path in directory.iterdir()
            if path.suffix.lower() in service.settings.allowed_extensions
        )
        indexing = service.index_paths(paths)
    else:
        from visual_search.models import IndexResponse

        indexing = IndexResponse(
            indexed=0,
            skipped_duplicates=0,
            failed=0,
            duration_ms=0,
            images_per_second=0,
            detector_device=service.detector.device,
            vector_backend=service.index.backend,
        )
    queries = [
        "people outdoors",
        "a city street with vehicles",
        "animals in nature",
        "food on a table",
        "sports activity",
    ] * 4
    report = service.benchmark(queries, indexing)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(report.model_dump_json(indent=2))
    return 0


def search(query: str, label: str | None, minimum: int, top_k: int) -> int:
    from visual_search.models import ObjectConstraint

    service = VisualSearchService(Settings())
    constraints = [ObjectConstraint(label=label, min_count=minimum)] if label else []
    result = service.search(SearchRequest(query=query, objects=constraints, top_k=top_k))
    print(result.model_dump_json(indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Hybrid visual search operations")
    commands = root.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import-zip")
    importer.add_argument("zip_path", type=Path)
    importer.add_argument("--output", type=Path, default=Path("runtime/source-images"))
    importer.add_argument("--limit", type=int, default=50)
    indexer = commands.add_parser("index")
    indexer.add_argument("directory", type=Path)
    benchmarker = commands.add_parser("benchmark")
    benchmarker.add_argument("--directory", type=Path)
    benchmarker.add_argument("--output", type=Path, default=Path("reports/benchmark.json"))
    searcher = commands.add_parser("search")
    searcher.add_argument("query")
    searcher.add_argument("--object", dest="label")
    searcher.add_argument("--min-count", type=int, default=1)
    searcher.add_argument("--top-k", type=int, default=10)
    return root


def main() -> None:
    arguments = parser().parse_args()
    if arguments.command == "import-zip":
        code = import_zip(arguments.zip_path, arguments.output, arguments.limit)
    elif arguments.command == "index":
        code = index_directory(arguments.directory)
    elif arguments.command == "benchmark":
        code = benchmark(arguments.directory, arguments.output)
    else:
        code = search(arguments.query, arguments.label, arguments.min_count, arguments.top_k)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
