from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from visual_search.models import BoundingBox, Detection, ImageRecord


class MetadataStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS images (
                    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    path TEXT NOT NULL UNIQUE,
                    sha256 TEXT NOT NULL UNIQUE,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL,
                    detector_model TEXT NOT NULL,
                    embedding_model TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS detections (
                    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_detections_label ON detections(label, image_id);
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )

    def contains_hash(self, sha256: str) -> bool:
        with self._connect() as connection:
            row = connection.execute("SELECT 1 FROM images WHERE sha256 = ?", (sha256,)).fetchone()
        return row is not None

    def add_image(
        self,
        *,
        filename: str,
        path: Path,
        sha256: str,
        width: int,
        height: int,
        detector_model: str,
        embedding_model: str,
        detections: list[Detection],
    ) -> int:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO images
                   (filename, path, sha256, width, height, indexed_at,
                    detector_model, embedding_model)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    filename,
                    str(path.resolve()),
                    sha256,
                    width,
                    height,
                    now,
                    detector_model,
                    embedding_model,
                ),
            )
            image_id = int(cursor.lastrowid or 0)
            connection.executemany(
                """INSERT INTO detections
                   (image_id, label, confidence, x1, y1, x2, y2)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        image_id,
                        item.label,
                        item.confidence,
                        item.bbox.x1,
                        item.bbox.y1,
                        item.bbox.x2,
                        item.bbox.y2,
                    )
                    for item in detections
                ],
            )
        return image_id

    def get(self, image_id: int) -> ImageRecord | None:
        with self._connect() as connection:
            image = connection.execute(
                "SELECT * FROM images WHERE image_id = ?", (image_id,)
            ).fetchone()
            if image is None:
                return None
            detections = connection.execute(
                "SELECT * FROM detections WHERE image_id = ?", (image_id,)
            ).fetchall()
        return self._record(image, detections)

    def list_images(self, limit: int = 10000) -> list[ImageRecord]:
        with self._connect() as connection:
            images = connection.execute(
                "SELECT * FROM images ORDER BY image_id LIMIT ?", (limit,)
            ).fetchall()
            detections = connection.execute(
                "SELECT * FROM detections ORDER BY image_id, detection_id"
            ).fetchall()
        grouped: dict[int, list[sqlite3.Row]] = {}
        for detection in detections:
            grouped.setdefault(int(detection["image_id"]), []).append(detection)
        return [self._record(row, grouped.get(int(row["image_id"]), [])) for row in images]

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT count(*) FROM images").fetchone()
        return int(row[0]) if row else 0

    def classes(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT label FROM detections ORDER BY label"
            ).fetchall()
        return [str(row[0]) for row in rows]

    def event(self, event_type: str, payload: dict[str, object]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO events(created_at, event_type, payload_json) VALUES (?, ?, ?)",
                (datetime.now(UTC).isoformat(), event_type, json.dumps(payload)),
            )

    @staticmethod
    def _record(image: sqlite3.Row, detections: list[sqlite3.Row]) -> ImageRecord:
        return ImageRecord(
            image_id=int(image["image_id"]),
            filename=str(image["filename"]),
            path=Path(str(image["path"])),
            sha256=str(image["sha256"]),
            width=int(image["width"]),
            height=int(image["height"]),
            indexed_at=datetime.fromisoformat(str(image["indexed_at"])),
            detector_model=str(image["detector_model"]),
            embedding_model=str(image["embedding_model"]),
            detections=[
                Detection(
                    label=str(item["label"]),
                    confidence=float(item["confidence"]),
                    bbox=BoundingBox(
                        x1=float(item["x1"]),
                        y1=float(item["y1"]),
                        x2=float(item["x2"]),
                        y2=float(item["y2"]),
                    ),
                )
                for item in detections
            ],
        )
