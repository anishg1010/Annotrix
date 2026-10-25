"""Evaluate click-to-segment on COCO128-seg instance masks.

Simulates an annotator using the Smart Segment tool, the standard way interactive
segmentation is evaluated (e.g. NoC@IoU in RITM / SAM papers):
  * box prompt: the ground-truth box (what "fit mask to box" does in the app)
  * click 1: a positive click at the point deepest inside the object
  * clicks 2..K: each next click goes to the centre of the largest error region
    (positive if the model missed object pixels there, negative if it over-segmented)

Reports mean IoU after each click, % of objects reaching IoU >= 0.85, and NoC@85
(mean number of clicks needed to reach 0.85, capped at K). Also a gallery of failures.

    python eval/eval_segmenter.py --max-objects 300 --clicks 5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from common import DATASETS, RESULTS, save_json


def load_instances(root: Path, min_area: int):
    from PIL import Image

    inst = []
    for img in sorted((root / "images" / "train2017").glob("*.jpg")):
        lbl = root / "labels" / "train2017" / (img.stem + ".txt")
        if not lbl.exists():
            continue
        w, h = Image.open(img).size
        for line in lbl.read_text().splitlines():
            v = line.split()
            if len(v) < 7:
                continue
            pts = (np.array(v[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [pts], 1)
            if m.sum() >= min_area:
                inst.append((img, int(v[0]), m.astype(bool)))
    return inst


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def deepest_point(region: np.ndarray) -> tuple[int, int]:
    d = cv2.distanceTransform(np.pad(region.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(int(d.argmax()), d.shape)
    return int(x), int(y)


def next_click(gt: np.ndarray, pred: np.ndarray) -> tuple[int, int, int]:
    fn = np.logical_and(gt, ~pred)
    fp = np.logical_and(~gt, pred)
    best = None
    for region, label in ((fn, 1), (fp, 0)):
        n, lab, stats, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), 8)
        if n > 1:
            k = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
            area = stats[k, cv2.CC_STAT_AREA]
            if best is None or area > best[0]:
                best = (area, lab == k, label)
    if best is None:
        return (*deepest_point(gt), 1)
    x, y = deepest_point(best[1])
    return x, y, best[2]


def polygon_mask(poly, h, w) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    if poly:
        cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
    return m.astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="segmenter weights (default: app setting)")
    ap.add_argument("--max-objects", type=int, default=300)
    ap.add_argument("--clicks", type=int, default=5)
    ap.add_argument("--min-area", type=int, default=32 * 32, help="skip tiny objects (pixels)")
    ap.add_argument("--target", type=float, default=0.85)
    args = ap.parse_args()

    from app import config
    from app.ml import Segmenter

    seg = Segmenter(args.weights or config.SEGMENTER_WEIGHTS)
    inst = load_instances(DATASETS / "coco128-seg", args.min_area)
    rng = np.random.default_rng(0)
    if len(inst) > args.max_objects:
        inst = [inst[i] for i in sorted(rng.choice(len(inst), args.max_objects, replace=False))]
    print(f"{len(inst)} objects, segmenter {seg.weights}")

    box_ious, click_ious, noc, times, rows = [], [], [], [], []
    for n, (img, cls, gt) in enumerate(inst):
        h, w = gt.shape
        ys, xs = np.nonzero(gt)
        box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        t0 = time.perf_counter()
        r = seg.segment(img, box=box)
        times.append(time.perf_counter() - t0)
        b_iou = iou(gt, polygon_mask(r.polygon, h, w)) if r else 0.0
        box_ious.append(b_iou)

        pts, labels, curve = [], [], []
        pred = np.zeros_like(gt)
        for _ in range(args.clicks):
            x, y, l = next_click(gt, pred) if pts else (*deepest_point(gt), 1)
            pts.append([float(x), float(y)])
            labels.append(l)
            r = seg.segment(img, points=pts, point_labels=labels)
            pred = polygon_mask(r.polygon, h, w) if r else np.zeros_like(gt)
            curve.append(iou(gt, pred))
        click_ious.append(curve)
        reached = next((i + 1 for i, v in enumerate(curve) if v >= args.target), args.clicks)
        noc.append(reached)
        rows.append({"image": img.name, "class_id": cls, "area": int(gt.sum()), "box_iou": b_iou, "click_iou": curve})
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(inst)}  box mIoU {np.mean(box_ious):.3f}  1-click mIoU {np.mean([c[0] for c in click_ious]):.3f}")

    ci = np.array(click_ious)
    res = {
        "segmenter": seg.weights,
        "objects": len(inst),
        "min_area_px": args.min_area,
        "box_prompt_mIoU": float(np.mean(box_ious)),
        "box_prompt_pct_ge_target": float(np.mean(np.array(box_ious) >= args.target)),
        "click_mIoU": [float(v) for v in ci.mean(0)],
        "click_pct_ge_target": [float(v) for v in (ci >= args.target).mean(0)],
        f"NoC@{int(args.target * 100)}": float(np.mean(noc)),
        "latency_ms_median_per_prompt": round(1000 * float(np.median(times)), 1),
    }
    out = RESULTS / "segmenter"
    save_json(res, out / "summary.json")
    save_json(rows, out / "per_object.json")

    md = [
        f"Segmenter `{Path(seg.weights).name}` on {len(inst)} COCO128-seg objects (area ≥ {args.min_area} px)\n",
        f"| prompt | mIoU | % objects IoU ≥ {args.target:.2f} |",
        "|---|---|---|",
        f"| GT box | {res['box_prompt_mIoU']:.3f} | {100 * res['box_prompt_pct_ge_target']:.0f}% |",
    ]
    for k in range(args.clicks):
        md.append(f"| {k + 1} click{'s' if k else ''} | {res['click_mIoU'][k]:.3f} | {100 * res['click_pct_ge_target'][k]:.0f}% |")
    md.append(f"\nNoC@{int(args.target * 100)} = {np.mean(noc):.2f} clicks (capped at {args.clicks})")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # curve + failure gallery
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(range(1, args.clicks + 1), ci.mean(0), marker="o", label="clicks", color="#ff6b35")
    ax.axhline(res["box_prompt_mIoU"], ls="--", color="#2ec4b6", label="box prompt")
    ax.set_xlabel("number of clicks")
    ax.set_ylabel("mean IoU")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "click_curve.png", dpi=130)
    plt.close(fig)

    worst = sorted(range(len(rows)), key=lambda i: rows[i]["click_iou"][-1])[:8]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, i in zip(axes.flat, worst):
        img, _, gt = inst[i]
        ax.imshow(Image.open(img))
        ax.contour(gt, levels=[0.5], colors="#2ec4b6", linewidths=1.5)
        ax.set_title(f"{img.name}  IoU after {args.clicks} clicks: {rows[i]['click_iou'][-1]:.2f}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Hardest objects for click-to-segment (teal = ground-truth outline)")
    fig.tight_layout()
    fig.savefig(out / "failures.png", dpi=100)
    plt.close(fig)
    print(f"\nResults in {out}")


if __name__ == "__main__":
    main()
