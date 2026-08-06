from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from PIL import Image

from visual_search.models import BoundingBox, Detection


class Detector(Protocol):
    name: str
    device: str

    def detect(self, image_path: Path) -> list[Detection]: ...


class Encoder(Protocol):
    name: str
    dimension: int
    device: str

    def encode_image(self, image_path: Path) -> np.ndarray: ...

    def encode_text(self, text: str) -> np.ndarray: ...


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class HashDetector:
    """Deterministic dependency-free detector for CI; not a CV quality claim."""

    name = "hash-detector-v1"
    device = "cpu"

    def detect(self, image_path: Path) -> list[Detection]:
        with Image.open(image_path) as image:
            width, height = image.size
        labels = ["person", "car", "dog", "bicycle", "chair"]
        digest = hashlib.sha256(image_path.read_bytes()).digest()
        count = digest[0] % 3
        return [
            Detection(
                label=labels[digest[index + 1] % len(labels)],
                confidence=round(0.5 + digest[index + 4] / 510, 4),
                bbox=BoundingBox(
                    x1=width * 0.1 * (index + 1),
                    y1=height * 0.1,
                    x2=width * (0.55 + 0.1 * index),
                    y2=height * 0.8,
                ),
            )
            for index in range(count)
        ]


class HashEncoder:
    """Feature-hash test encoder sharing a space for labels and text."""

    name = "hash-embedding-v1"
    device = "cpu"

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def _encode(self, tokens: list[str]) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in tokens:
            index = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % self.dimension
            vector[index] += 1
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def encode_image(self, image_path: Path) -> np.ndarray:
        tokens = re.findall(r"[a-z0-9]+", image_path.stem.lower())
        tokens.append(hashlib.sha256(image_path.read_bytes()).hexdigest()[:8])
        return self._encode(tokens)

    def encode_text(self, text: str) -> np.ndarray:
        return self._encode(re.findall(r"[a-z0-9]+", text.lower()))


class YOLODetector:
    def __init__(self, model: str, confidence: float, device: str) -> None:
        try:
            from ultralytics import YOLO  # type: ignore[attr-defined]
        except ImportError as error:
            raise RuntimeError("Install the ml extra to use YOLO: uv sync --extra ml") from error
        self.name = model
        self.device = resolve_device(device)
        self.confidence = confidence
        self.model = YOLO(model)

    def detect(self, image_path: Path) -> list[Detection]:
        results = cast(
            list[Any],
            self.model.predict(
                source=str(image_path),
                conf=self.confidence,
                device=self.device,
                verbose=False,
            ),
        )
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                coordinates = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        label=str(result.names[int(box.cls.item())]),
                        confidence=float(box.conf.item()),
                        bbox=BoundingBox(
                            x1=coordinates[0],
                            y1=coordinates[1],
                            x2=coordinates[2],
                            y2=coordinates[3],
                        ),
                    )
                )
        return detections


class OpenCLIPEncoder:
    def __init__(self, model_name: str, pretrained: str, device: str) -> None:
        try:
            import open_clip
            import torch
        except ImportError as error:
            raise RuntimeError("Install the ml extra to use CLIP: uv sync --extra ml") from error
        self.device = resolve_device(device)
        self.model_name = model_name
        self.name = f"{model_name}/{pretrained}"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        with torch.inference_mode():
            sample = self.tokenizer(["test"]).to(self.device)
            self.dimension = int(self.model.encode_text(sample).shape[-1])

    def encode_image(self, image_path: Path) -> np.ndarray:
        import torch

        with Image.open(image_path) as image:
            tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            vector = self.model.encode_image(tensor)
            vector /= vector.norm(dim=-1, keepdim=True)
        return cast(np.ndarray, vector[0].detach().cpu().numpy().astype(np.float32))

    def encode_text(self, text: str) -> np.ndarray:
        import torch

        tokens = self.tokenizer([text]).to(self.device)
        with torch.inference_mode():
            vector = self.model.encode_text(tokens)
            vector /= vector.norm(dim=-1, keepdim=True)
        return cast(np.ndarray, vector[0].detach().cpu().numpy().astype(np.float32))
