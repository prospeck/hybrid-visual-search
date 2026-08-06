from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class VectorIndex:
    """Persistent cosine index using FAISS when installed and NumPy otherwise."""

    def __init__(self, path: Path, map_path: Path, dimension: int) -> None:
        self.path = path
        self.map_path = map_path
        self.dimension = dimension
        self.image_ids: list[int] = []
        self._faiss = None
        self._index = None
        self._vectors = np.empty((0, dimension), dtype=np.float32)
        try:
            import faiss

            self._faiss = faiss
            self.backend = "faiss.IndexFlatIP"
        except ImportError:
            self.backend = "numpy.cosine"
        self._load()

    def _load(self) -> None:
        if not self.map_path.exists():
            self._new()
            return
        self.image_ids = [
            int(value) for value in json.loads(self.map_path.read_text(encoding="utf-8"))
        ]
        if self._faiss is not None and self.path.exists():
            self._index = self._faiss.read_index(str(self.path))
        elif self.path.with_suffix(".npz").exists():
            self._vectors = np.load(self.path.with_suffix(".npz"))["vectors"]
        else:
            self._new()

    def _new(self) -> None:
        self.image_ids = []
        self._vectors = np.empty((0, self.dimension), dtype=np.float32)
        if self._faiss is not None:
            self._index = self._faiss.IndexFlatIP(self.dimension)

    def add(self, image_id: int, vector: np.ndarray) -> None:
        normalized = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if normalized.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {normalized.shape[1]} does not match index {self.dimension}"
            )
        norm = np.linalg.norm(normalized, axis=1, keepdims=True)
        normalized = normalized / np.maximum(norm, 1e-12)
        if self._index is not None:
            self._index.add(normalized)
        else:
            self._vectors = np.vstack([self._vectors, normalized])
        self.image_ids.append(image_id)

    def search(self, vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if not self.image_ids:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
        count = min(top_k, len(self.image_ids))
        if self._index is not None:
            scores, positions = self._index.search(query, count)
            return [
                (self.image_ids[int(position)], float(score))
                for position, score in zip(positions[0], scores[0], strict=True)
                if position >= 0
            ]
        scores = self._vectors @ query[0]
        positions = np.argsort(-scores)[:count]
        return [(self.image_ids[int(position)], float(scores[position])) for position in positions]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.map_path.write_text(json.dumps(self.image_ids), encoding="utf-8")
        if self._index is not None:
            self._faiss.write_index(self._index, str(self.path))
        else:
            np.savez_compressed(self.path.with_suffix(".npz"), vectors=self._vectors)

    def __len__(self) -> int:
        return len(self.image_ids)
