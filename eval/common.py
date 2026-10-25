"""Shared helpers for the evaluation scripts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATASETS = DATA / "datasets"
RESULTS = ROOT / "eval" / "results"

# Let the scripts reuse the exact model code the app runs (backend/app/ml.py).
sys.path.insert(0, str(ROOT / "backend"))

AW_NAMES = ["buffalo", "elephant", "rhino", "zebra"]
AW_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip"
COCO128_SEG_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip"


def download_and_unzip(url: str, dest_dir: Path, marker: str) -> Path:
    """Download a zip once and extract it into dest_dir. Returns dest_dir/marker."""
    import urllib.request
    import zipfile

    target = dest_dir / marker
    if target.exists():
        return target
    dest_dir.mkdir(parents=True, exist_ok=True)
    zpath = dest_dir / Path(url).name
    if not zpath.exists():
        print(f"Downloading {url} …")
        urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest_dir)
    if not target.exists():
        raise SystemExit(f"Expected {target} after extracting {zpath}")
    return target


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.width, im.height


def yolo_split_to_coco(root: Path, split: str, names: list[str]) -> dict:
    """Convert a YOLO-format split (images/<split>, labels/<split>) to a COCO dict."""
    img_dir, lbl_dir = root / "images" / split, root / "labels" / split
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)],
    }
    aid = 1
    for iid, p in enumerate(sorted(img_dir.iterdir()), start=1):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        w, h = image_size(p)
        coco["images"].append({"id": iid, "file_name": p.name, "width": w, "height": h})
        lbl = lbl_dir / (p.stem + ".txt")
        if not lbl.exists():
            continue
        for line in lbl.read_text().splitlines():
            v = line.split()
            if len(v) < 5:
                continue
            c = int(v[0])
            if len(v) == 5:
                cx, cy, bw, bh = (float(t) for t in v[1:])
                x, y, bw, bh = (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h
                seg = []
            else:  # polygon label
                pts = np.array([float(t) for t in v[1:]]).reshape(-1, 2) * [w, h]
                x, y = pts.min(0)
                bw, bh = pts.max(0) - pts.min(0)
                seg = [pts.flatten().round(2).tolist()]
            coco["annotations"].append(
                {
                    "id": aid,
                    "image_id": iid,
                    "category_id": c + 1,
                    "bbox": [float(x), float(y), float(bw), float(bh)],
                    "area": float(bw * bh),
                    "segmentation": seg,
                    "iscrowd": 0,
                }
            )
            aid += 1
    return coco


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes in xywh format. a: (N,4), b: (M,4) -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a1, a2 = a[:, None, :2], a[:, None, :2] + a[:, None, 2:]
    b1, b2 = b[None, :, :2], b[None, :, :2] + b[None, :, 2:]
    inter = np.clip(np.minimum(a2, b2) - np.maximum(a1, b1), 0, None).prod(-1)
    union = a[:, None, 2:].prod(-1) + b[None, :, 2:].prod(-1) - inter
    return inter / np.maximum(union, 1e-9)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def coco_map(gt: dict, dets: list[dict]) -> dict:
    """mAP@0.5, mAP@0.5:0.95 and per-class AP@0.5 with pycocotools."""
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not dets:
        return {"mAP50": 0.0, "mAP50_95": 0.0, "per_class_AP50": {c["name"]: 0.0 for c in gt["categories"]}}
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = json.loads(json.dumps(gt))
        cg.createIndex()
        cd = cg.loadRes(dets)
        ev = COCOeval(cg, cd, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    prec = ev.eval["precision"]  # [T, R, K, A, M]
    per_class = {}
    for k, cid in enumerate(ev.params.catIds):
        p = prec[0, :, k, 0, -1]
        p = p[p > -1]
        per_class[cg.cats[cid]["name"]] = float(p.mean()) if p.size else 0.0
    return {"mAP50": float(ev.stats[1]), "mAP50_95": float(ev.stats[0]), "per_class_AP50": per_class}
