"""Experiment 1: how does the detector confidence threshold change the work left for the annotator?

Runs the detector once at a very low threshold on the test split, then for each
threshold counts, at IoU >= 0.5 with matching class:
  TP (good suggestions), FP (junk boxes to delete), FN (boxes still to draw).
Also splits recall by object size, to test whether low thresholds rescue small animals.

    python eval/threshold_sweep.py --weights wildlife-yolo11n-cpu.pt
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from common import AW_NAMES, DATASETS, RESULTS

THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def size_bucket(box, img_area):
    frac = box[2] * box[3] / img_area
    return "small" if frac < 0.01 else "medium" if frac < 0.10 else "large"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="wildlife-yolo11n-cpu.pt")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    from app import ml
    from common import yolo_split_to_coco

    root = DATASETS / "african-wildlife"
    gt = yolo_split_to_coco(root, args.split, AW_NAMES)
    cat = {c["id"]: c["name"] for c in gt["categories"]}
    gts = {}
    for a in gt["annotations"]:
        gts.setdefault(a["image_id"], []).append((cat[a["category_id"]], a["bbox"]))

    det = ml.Detector(args.weights)
    preds = {}
    for im in gt["images"]:
        p = root / "images" / args.split / im["file_name"]
        ds = det.predict(str(p), conf=min(THRESHOLDS), labels=AW_NAMES)
        preds[im["id"]] = [(d.label, [d.x, d.y, d.w, d.h], d.score) for d in ds]

    rows = []
    for t in THRESHOLDS:
        tp = fp = fn = 0
        size_tot = {"small": 0, "medium": 0, "large": 0}
        size_hit = {"small": 0, "medium": 0, "large": 0}
        for im in gt["images"]:
            area = im["width"] * im["height"]
            g = gts.get(im["id"], [])
            p = sorted([x for x in preds[im["id"]] if x[2] >= t], key=lambda x: -x[2])
            used = set()
            for lab, box, _ in p:
                best, bi = 0.5, -1
                for j, (gl, gb) in enumerate(g):
                    if j in used or gl != lab:
                        continue
                    v = iou(box, gb)
                    if v >= best:
                        best, bi = v, j
                if bi >= 0:
                    used.add(bi)
                    tp += 1
                else:
                    fp += 1
            fn += len(g) - len(used)
            for j, (_, gb) in enumerate(g):
                b = size_bucket(gb, area)
                size_tot[b] += 1
                size_hit[b] += j in used
        rows.append({
            "threshold": t, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(tp / max(tp + fp, 1), 3),
            "recall": round(tp / max(tp + fn, 1), 3),
            "clicks": fp + fn,
            **{f"recall_{k}": round(size_hit[k] / max(size_tot[k], 1), 3) for k in size_tot},
        })
    sizes = size_tot

    out = RESULTS / "detector"
    out.mkdir(parents=True, exist_ok=True)
    (out / "threshold_sweep.json").write_text(json.dumps({"weights": args.weights, "split": args.split,
                                                           "gt_boxes_by_size": sizes, "rows": rows}, indent=2))
    lines = [f"# Threshold sweep — {args.weights} on {args.split} ({len(gt['images'])} images, "
             f"{sum(sizes.values())} boxes: {sizes['small']} small / {sizes['medium']} medium / {sizes['large']} large)",
             "", "Small = under 1% of the image area, large = over 10%. Clicks = junk to delete + boxes to draw.", "",
             "| conf | good | junk | missed | precision | recall | clicks | recall small | recall medium | recall large |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['threshold']:.2f} | {r['tp']} | {r['fp']} | {r['fn']} | {r['precision']:.3f} | "
                     f"{r['recall']:.3f} | {r['clicks']} | {r['recall_small']:.3f} | {r['recall_medium']:.3f} | "
                     f"{r['recall_large']:.3f} |")
    (out / "threshold_sweep.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ts, [r["fp"] for r in rows], "o-", label="junk boxes to delete")
    ax[0].plot(ts, [r["fn"] for r in rows], "s-", label="boxes still to draw")
    ax[0].plot(ts, [r["clicks"] for r in rows], "^-", label="total corrections")
    ax[0].set_xlabel("confidence threshold"); ax[0].set_ylabel("count (test split)")
    ax[0].set_title("Work left for the annotator"); ax[0].legend(); ax[0].grid(alpha=.3)
    for k in ("small", "medium", "large"):
        ax[1].plot(ts, [r[f"recall_{k}"] for r in rows], "o-", label=f"{k} ({sizes[k]})")
    ax[1].set_xlabel("confidence threshold"); ax[1].set_ylabel("recall")
    ax[1].set_ylim(0, 1.02); ax[1].set_title("Recall by object size"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "threshold_sweep.png", dpi=150)


if __name__ == "__main__":
    main()
