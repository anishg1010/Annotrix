"""Runtime configuration, read from environment variables."""
from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.getenv("LA_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
IMAGES_DIR = DATA_DIR / "images"
DB_PATH = DATA_DIR / "annotrix.db"
WEIGHTS_DIR = Path(os.getenv("LA_WEIGHTS_DIR", DATA_DIR / "weights"))

# Detector used for automatic pre-labelling. Can be a stock Ultralytics name
# (downloaded on first use) or a path to fine-tuned weights, e.g. weights/wildlife-yolo11s.pt
DETECTOR_WEIGHTS = os.getenv("LA_DETECTOR", "yolo11s.pt")
# Open-vocabulary detector for labels the main detector does not know (text prompts).
# Set LA_OPEN_VOCAB="" to disable.
OPEN_VOCAB_WEIGHTS = os.getenv("LA_OPEN_VOCAB", "yoloe-11s-seg.pt")
# Promptable segmenter used for click-to-segment and box-to-mask.
SEGMENTER_WEIGHTS = os.getenv("LA_SEGMENTER", "mobile_sam.pt")
# "auto" picks CUDA when available, otherwise CPU.
DEVICE = os.getenv("LA_DEVICE", "auto")

STATIC_DIR = Path(os.getenv("LA_STATIC_DIR", Path(__file__).resolve().parents[1] / "static"))

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def ensure_dirs() -> None:
    for d in (DATA_DIR, IMAGES_DIR, WEIGHTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
