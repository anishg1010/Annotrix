"""Export a project as a zip in COCO or YOLO format (images included)."""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from . import config
from .db import Annotation, Image, Project


def _unique_names(images: list[Image]) -> dict[int, str]:
    """Map image id -> export file name, de-duplicating clashing original names."""
    seen: dict[str, int] = {}
    out: dict[int, str] = {}
    for im in images:
        name = PurePosixPath(im.filename).name
        if name in seen:
            seen[name] += 1
            stem, suf = PurePosixPath(name).stem, PurePosixPath(name).suffix
            name = f"{stem}_{seen[name]}{suf}"
        else:
            seen[name] = 0
        out[im.id] = name
    return out


def polygon_area(poly: list[list[float]]) -> float:
    if not poly or len(poly) < 3:
        return 0.0
    s = 0.0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def build_coco(project: Project, images: list[Image], anns: dict[int, list[Annotation]]) -> dict:
    names = _unique_names(images)
    labels = list(project.labels)
    for a in (a for lst in anns.values() for a in lst):
        if a.label not in labels:
            labels.append(a.label)
    cat_ids = {l: i + 1 for i, l in enumerate(labels)}
    coco = {
        "info": {
            "description": f"{project.name} — exported from LabelAssist",
            "date_created": datetime.now(timezone.utc).isoformat(),
        },
        "licenses": [],
        "categories": [{"id": cid, "name": l, "supercategory": "object"} for l, cid in cat_ids.items()],
        "images": [],
        "annotations": [],
    }
    aid = 1
    for im in images:
        coco["images"].append(
            {"id": im.id, "file_name": f"images/{names[im.id]}", "width": im.width, "height": im.height}
        )
        for a in anns.get(im.id, []):
            seg = []
            area = a.w * a.h
            if a.polygon and len(a.polygon) >= 3:
                seg = [[round(v, 2) for pt in a.polygon for v in pt]]
                area = polygon_area(a.polygon)
            coco["annotations"].append(
                {
                    "id": aid,
                    "image_id": im.id,
                    "category_id": cat_ids[a.label],
                    "bbox": [round(a.x, 2), round(a.y, 2), round(a.w, 2), round(a.h, 2)],
                    "area": round(area, 2),
                    "segmentation": seg,
                    "iscrowd": 0,
                    "attributes": {"source": a.source, "edited": a.edited},
                }
            )
            aid += 1
    return coco


def _yolo_lines(im: Image, alist: list[Annotation], cat_index: dict[str, int], seg: bool) -> list[str]:
    lines = []
    for a in alist:
        c = cat_index[a.label]
        if seg:
            if not a.polygon or len(a.polygon) < 3:
                pts = [[a.x, a.y], [a.x + a.w, a.y], [a.x + a.w, a.y + a.h], [a.x, a.y + a.h]]
            else:
                pts = a.polygon
            coords = " ".join(
                f"{min(max(x / im.width, 0), 1):.6f} {min(max(y / im.height, 0), 1):.6f}" for x, y in pts
            )
            lines.append(f"{c} {coords}")
        else:
            cx = (a.x + a.w / 2) / im.width
            cy = (a.y + a.h / 2) / im.height
            lines.append(f"{c} {cx:.6f} {cy:.6f} {a.w / im.width:.6f} {a.h / im.height:.6f}")
    return lines


def export_zip(
    project: Project,
    images: list[Image],
    anns: dict[int, list[Annotation]],
    fmt: str = "coco",
    only_done: bool = False,
) -> bytes:
    if only_done:
        images = [im for im in images if im.status == "done"]
    names = _unique_names(images)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for im in images:
            z.write(config.IMAGES_DIR / im.stored_name, f"images/{names[im.id]}")
        if fmt == "coco":
            z.writestr("annotations.json", json.dumps(build_coco(project, images, anns), indent=2))
        elif fmt in ("yolo", "yolo-seg"):
            labels = list(project.labels)
            for a in (a for lst in anns.values() for a in lst):
                if a.label not in labels:
                    labels.append(a.label)
            idx = {l: i for i, l in enumerate(labels)}
            for im in images:
                stem = PurePosixPath(names[im.id]).stem
                z.writestr(
                    f"labels/{stem}.txt",
                    "\n".join(_yolo_lines(im, anns.get(im.id, []), idx, fmt == "yolo-seg")),
                )
            yaml = "path: .\ntrain: images\nval: images\nnames:\n" + "".join(
                f"  {i}: {l}\n" for l, i in idx.items()
            )
            z.writestr("data.yaml", yaml)
        else:
            raise ValueError(f"unknown format {fmt}")
        z.writestr(
            "README.txt",
            f"Exported from LabelAssist on {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC\n"
            f"Project: {project.name}\nFormat: {fmt}\nImages: {len(images)}\n",
        )
    return buf.getvalue()
