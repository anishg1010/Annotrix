"""Computer-vision models that assist the annotator.

* Detector (YOLO, Ultralytics): proposes labelled bounding boxes for a whole image.
* Segmenter (MobileSAM / SAM 2, Ultralytics): turns a click or a box into a mask,
  which we convert into a polygon plus a tight bounding box.

Both are loaded lazily on first use and kept in memory. The segmenter caches the
image embedding of the last image so repeated clicks on the same image are fast.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config

_lock = threading.Lock()
# Model calls are not thread-safe (the segmenter keeps per-image state), so the API
# serialises inference through this lock.
infer_lock = threading.Lock()


def resolve_device() -> str:
    if config.DEVICE != "auto":
        return config.DEVICE
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        return "cpu"


def _weights_path(name: str) -> str:
    """Prefer a file in the weights dir; otherwise let Ultralytics download it there."""
    p = Path(name)
    if p.is_file():
        return str(p)
    local = config.WEIGHTS_DIR / p.name
    if local.is_file():
        return str(local)
    return str(local) if "/" not in name else name


# --------------------------------------------------------------------------- detector


@dataclass
class Detection:
    label: str
    score: float
    x: float
    y: float
    w: float
    h: float
    model: str = ""


class Detector:
    def __init__(self, weights: str):
        from ultralytics import YOLO

        self.weights = weights
        self.model = YOLO(_weights_path(weights))
        self.names: dict[int, str] = dict(self.model.names)

    def predict(
        self,
        image_path: Path,
        conf: float = 0.25,
        labels: Optional[list[str]] = None,
    ) -> list[Detection]:
        """Run detection. If `labels` is given, keep only classes whose name matches
        a project label (case-insensitive) and return them with the project's spelling."""
        res = self.model.predict(
            str(image_path), conf=conf, device=resolve_device(), verbose=False
        )[0]
        wanted = {l.lower(): l for l in labels} if labels else None
        out: list[Detection] = []
        for xyxy, c, s in zip(
            res.boxes.xyxy.cpu().numpy(),
            res.boxes.cls.cpu().numpy().astype(int),
            res.boxes.conf.cpu().numpy(),
        ):
            name = self.names.get(int(c), str(c))
            if wanted is not None:
                if name.lower() not in wanted:
                    continue
                name = wanted[name.lower()]
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            out.append(Detection(name, float(s), x1, y1, x2 - x1, y2 - y1, self.weights))
        return out


class OpenVocabDetector:
    """Zero-shot detector prompted with the project's label names (YOLOE).

    A MobileCLIP text encoder turns each label into an embedding once; the
    detector then finds objects matching those embeddings. Used only for labels
    the main detector was not trained on.
    """

    def __init__(self, weights: str):
        from ultralytics import YOLOE

        self.weights = weights
        self.model = YOLOE(_weights_path(weights))
        self._classes: tuple[str, ...] = ()

    def _set_classes(self, labels: list[str]) -> None:
        key = tuple(labels)
        if key == self._classes:
            return
        # the text encoder is downloaded next to the other weights on first use
        cwd = os.getcwd()
        config.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        os.chdir(config.WEIGHTS_DIR)
        try:
            self.model.set_classes(list(labels), self.model.get_text_pe(list(labels)))
        finally:
            os.chdir(cwd)
        self._classes = key

    def predict(self, image_path: Path, conf: float = 0.25, labels: Optional[list[str]] = None) -> list[Detection]:
        if not labels:
            return []
        self._set_classes(labels)
        res = self.model.predict(str(image_path), conf=conf, device=resolve_device(), verbose=False)[0]
        out: list[Detection] = []
        for xyxy, c, sc in zip(
            res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int), res.boxes.conf.cpu().numpy()
        ):
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            out.append(Detection(labels[int(c)], float(sc), x1, y1, x2 - x1, y2 - y1, self.weights))
        return out


# --------------------------------------------------------------------------- segmenter


@dataclass
class SegResult:
    polygon: list[list[float]]
    x: float
    y: float
    w: float
    h: float
    score: Optional[float]
    area: float


def mask_to_polygon(mask: np.ndarray, epsilon_frac: float = 0.002) -> list[list[float]]:
    """Largest external contour of a binary mask, lightly simplified."""
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    eps = epsilon_frac * cv2.arcLength(c, True)
    c = cv2.approxPolyDP(c, eps, True)
    return [[float(p[0][0]), float(p[0][1])] for p in c]


class Segmenter:
    def __init__(self, weights: str):
        from ultralytics.models.sam import Predictor as SAMPredictor
        from ultralytics.models.sam import SAM2Predictor

        self.weights = weights
        # SAM 2 / 2.1 checkpoints (sam2*.pt) need their own predictor class
        cls = SAM2Predictor if Path(weights).name.startswith("sam2") else SAMPredictor
        self.predictor = cls(
            overrides=dict(
                task="segment",
                mode="predict",
                imgsz=1024,
                model=_weights_path(weights),
                save=False,
                verbose=False,
                device=resolve_device(),
            )
        )
        self._current: Optional[str] = None

    def _set_image(self, image_path: Path) -> None:
        key = str(image_path)
        if self._current != key:
            self.predictor.set_image(key)
            self._current = key

    def segment(
        self,
        image_path: Path,
        points: Optional[list[list[float]]] = None,
        point_labels: Optional[list[int]] = None,
        box: Optional[list[float]] = None,
    ) -> Optional[SegResult]:
        """points: [[x, y], ...] with point_labels 1 = object, 0 = background.
        box: [x1, y1, x2, y2]. Any combination of the two is allowed."""
        self._set_image(image_path)
        kwargs: dict = {}
        if box is not None:
            kwargs["bboxes"] = [box]
        if points:
            kwargs["points"] = [points]
            kwargs["labels"] = [point_labels or [1] * len(points)]
        if not kwargs:
            return None
        res = self.predictor(**kwargs)[0]
        if res.masks is None or len(res.masks.data) == 0:
            return None
        mask = res.masks.data[0].cpu().numpy()
        poly = mask_to_polygon(mask)
        if len(poly) < 3:
            return None
        ys, xs = np.nonzero(mask > 0)
        x1, x2, y1, y2 = float(xs.min()), float(xs.max()) + 1, float(ys.min()), float(ys.max()) + 1
        score = None
        if res.boxes is not None and res.boxes.conf is not None and len(res.boxes.conf):
            score = float(res.boxes.conf[0])
        return SegResult(poly, x1, y1, x2 - x1, y2 - y1, score, float((mask > 0).sum()))


# --------------------------------------------------------------------------- registry

_detector: Optional[Detector] = None
_segmenter: Optional[Segmenter] = None
_open_vocab: Optional[OpenVocabDetector] = None


def detector() -> Detector:
    global _detector
    with _lock:
        if _detector is None:
            _detector = Detector(config.DETECTOR_WEIGHTS)
        return _detector


def open_vocab() -> Optional[OpenVocabDetector]:
    global _open_vocab
    if not config.OPEN_VOCAB_WEIGHTS:
        return None
    with _lock:
        if _open_vocab is None:
            _open_vocab = OpenVocabDetector(config.OPEN_VOCAB_WEIGHTS)
        return _open_vocab


def detect_labels(image_path: Path, conf: float, labels: Optional[list[str]]) -> tuple[list[Detection], list[str]]:
    """Hybrid detection: labels the trained detector knows go to it; any other
    label is detected zero-shot by the open-vocabulary model.
    Returns the detections and the names of the models that were used."""
    det = detector()
    if not labels:
        return det.predict(image_path, conf=conf), [det.weights]
    known = {n.lower() for n in det.names.values()}
    closed = [l for l in labels if l.lower() in known]
    other = [l for l in labels if l.lower() not in known]
    out: list[Detection] = []
    used: list[str] = []
    if closed:
        out += det.predict(image_path, conf=conf, labels=closed)
        used.append(det.weights)
    if other:
        ov = open_vocab()
        if ov is not None:
            out += ov.predict(image_path, conf=conf, labels=other)
            used.append(ov.weights)
    return out, used


def segmenter() -> Segmenter:
    global _segmenter
    with _lock:
        if _segmenter is None:
            _segmenter = Segmenter(config.SEGMENTER_WEIGHTS)
        return _segmenter


def set_detector_weights(weights: str) -> Detector:
    """Swap the detector at runtime (e.g. after fine-tuning)."""
    global _detector
    with _lock:
        _detector = Detector(weights)
        config.DETECTOR_WEIGHTS = weights
        return _detector
