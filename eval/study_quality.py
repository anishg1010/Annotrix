"""Score a COCO export from Annotrix against the African Wildlife ground truth.

Used for the user study (set A manual, set B assisted; test split) and for the
deliverable dataset (validation split). Needs `python scripts/prepare_data.py` first.

    python eval/study_quality.py eval/results/effort/study_manual_setA_coco.zip --split test
    python eval/study_quality.py eval/results/effort/study_assisted_setB_coco.zip --split test
    python eval/study_quality.py Wildlife_dataset_coco.zip --split val

A saved box matches a ground-truth box when the class is the same and IoU >= 0.5
(greedy matching, highest IoU first). Reports precision, recall and mean IoU of matches.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from common import AW_NAMES, DATASETS, image_size


def gt_boxes(split: str, fname: str):
    root = DATASETS / "african-wildlife"
    w, h = image_size(root / "images" / split / fname)
    out = []
    lbl = root / "labels" / split / (Path(fname).stem + ".txt")
    for line in lbl.read_text().splitlines() if lbl.exists() else []:
        v = line.split()
        if len(v) < 5:
            continue
        c, cx, cy, bw, bh = int(v[0]), *map(float, v[1:5])
        out.append((AW_NAMES[c], [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h]))
    return out


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    i = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0


def score(preds: dict, split: str) -> dict:
    tp = fp = fn = n_gt = 0
    ious = []
    for f, P in preds.items():
        G = gt_boxes(split, f)
        n_gt += len(G)
        used_g, used_p = set(), set()
        pairs = sorted(((iou(p[1], g[1]), i, j) for i, p in enumerate(P) for j, g in enumerate(G)), reverse=True)
        for v, i, j in pairs:
            if v < 0.5 or i in used_p or j in used_g or P[i][0] != G[j][0]:
                continue
            used_p.add(i)
            used_g.add(j)
            tp += 1
            ious.append(v)
        fp += len(P) - len(used_p)
        fn += len(G) - len(used_g)
    return {
        "images": len(preds), "gt_objects": n_gt, "saved_boxes": sum(map(len, preds.values())),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3),
        "mean_iou": round(float(np.mean(ious)), 3) if ious else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export", help="COCO zip exported from Annotrix")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()
    d = json.loads(zipfile.ZipFile(args.export).read("annotations.json"))
    cat = {c["id"]: c["name"].lower() for c in d["categories"]}
    fname = {i["id"]: Path(i["file_name"]).name for i in d["images"]}
    preds = {f: [] for f in fname.values()}
    src = {}
    for a in d["annotations"]:
        preds[fname[a["image_id"]]].append((cat[a["category_id"]], a["bbox"]))
        s = a.get("attributes", {}).get("source", "?")
        src[s] = src.get(s, 0) + 1
    res = score(preds, args.split)
    res["box_sources"] = src
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
