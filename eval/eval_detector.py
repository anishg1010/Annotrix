"""Evaluate the pre-labelling detector on the African Wildlife test split.

Reports, for one or more detector weights:
  * model quality: COCO mAP@0.5, mAP@0.5:0.95, per-class AP@0.5
  * suggestion usefulness: what an annotator would have to do with each suggestion
    (accept as is / adjust box / relabel / delete false positive / draw missed object),
    i.e. a simulated count of editing actions, assisted vs fully manual
  * failure cases: an image gallery of the worst images

    python eval/eval_detector.py --weights yolo11s.pt data/weights/wildlife-yolo11s.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import DATASETS, RESULTS, box_iou, coco_map, save_json

ACCEPT_IOU = 0.75  # good enough to keep unchanged
MATCH_IOU = 0.5  # same object, but needs a box adjustment below ACCEPT_IOU


def simulate(gts: list[dict], preds: list[dict]) -> dict:
    """Greedy match suggestions to ground truth and count the edit actions needed."""
    out = dict(accept=0, adjust=0, relabel=0, delete=0, draw=0, matched_same_class=0)
    g = np.array([a["bbox"] for a in gts]) if gts else np.zeros((0, 4))
    p = np.array([d["bbox"] for d in preds]) if preds else np.zeros((0, 4))
    iou = box_iou(p, g)
    used_g, used_p = set(), set()
    pairs = sorted(((iou[i, j], i, j) for i in range(len(p)) for j in range(len(g))), reverse=True)
    for v, i, j in pairs:
        if v < MATCH_IOU or i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        same = preds[i]["category_id"] == gts[j]["category_id"]
        if same:
            out["matched_same_class"] += 1
        else:
            out["relabel"] += 1  # right object, wrong class
        if v < ACCEPT_IOU:
            out["adjust"] += 1  # box needs dragging
        elif same:
            out["accept"] += 1  # keep as is (one click / bulk accept)
    out["delete"] = len(p) - len(used_p)
    out["draw"] = len(g) - len(used_g)
    return out


def gallery(cases: list[tuple], img_dir: Path, names: dict[int, str], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    n = len(cases)
    if not n:
        return
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.4 * rows))
    for ax in np.array(axes).flat:
        ax.axis("off")
    for ax, (fname, gts, preds, why) in zip(np.array(axes).flat, cases):
        ax.imshow(Image.open(img_dir / fname))
        for a in gts:
            x, y, w, h = a["bbox"]
            ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="#2ec4b6", lw=2))
        for d in preds:
            x, y, w, h = d["bbox"]
            ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="#ff6b35", lw=2, ls="--"))
            ax.text(x, y, f"{names.get(d['category_id'], d.get('raw', '?'))} {d['score']:.2f}",
                    color="white", fontsize=7, bbox=dict(fc="#ff6b35", ec="none", pad=1))
        ax.set_title(f"{fname}\n{why}", fontsize=8)
    fig.suptitle(f"{title}   (teal = ground truth, orange dashed = suggestion)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def evaluate(weights: str, gt: dict, img_dir: Path, conf: float, out_dir: Path) -> dict:
    from app.ml import Detector

    det = Detector(weights)
    name_to_cat = {c["name"].lower(): c["id"] for c in gt["categories"]}
    cat_names = {c["id"]: c["name"] for c in gt["categories"]}
    gts_by_img: dict[int, list] = {}
    for a in gt["annotations"]:
        gts_by_img.setdefault(a["image_id"], []).append(a)

    dets, per_image, times, unmapped = [], [], [], {}
    for im in gt["images"]:
        t0 = time.perf_counter()
        preds = det.predict(img_dir / im["file_name"], conf=0.001)  # low conf for mAP
        times.append(time.perf_counter() - t0)
        shown = []
        for d in preds:
            cid = name_to_cat.get(d.label.lower())
            if cid is None:
                unmapped[d.label] = unmapped.get(d.label, 0) + (d.score >= conf)
                continue
            rec = {"image_id": im["id"], "category_id": cid, "bbox": [d.x, d.y, d.w, d.h], "score": d.score}
            dets.append(rec)
            if d.score >= conf:
                shown.append(rec)
        g = gts_by_img.get(im["id"], [])
        sim = simulate(g, shown)
        per_image.append({"file": im["file_name"], "n_gt": len(g), "n_suggested": len(shown), **sim,
                          "_gts": g, "_preds": shown})

    m = coco_map(gt, dets)
    tot = {k: int(sum(r[k] for r in per_image))
           for k in ("accept", "adjust", "relabel", "delete", "draw", "matched_same_class", "n_gt", "n_suggested")}
    # Every assisted action is cheaper than drawing a box, but they are not free; the
    # time model is fitted from the user study in eval/effort.py.
    assisted_actions = tot["accept"] + tot["adjust"] + tot["relabel"] + tot["delete"] + tot["draw"]
    res = {
        "weights": weights,
        "conf_threshold_for_suggestions": conf,
        "images": len(gt["images"]),
        **m,
        "latency_ms_median": round(1000 * float(np.median(times)), 1),
        "actions": tot,
        "gt_objects_covered_by_correct_suggestion": round(tot["matched_same_class"] / max(tot["n_gt"], 1), 3),
        "boxes_to_draw_manual": tot["n_gt"],
        "boxes_to_draw_assisted": tot["draw"],
        "total_actions_manual": tot["n_gt"],
        "total_actions_assisted": assisted_actions,
        "unmapped_model_classes_above_conf": dict(sorted(unmapped.items(), key=lambda kv: -kv[1])[:10]),
    }

    # failure gallery: images needing the most corrections
    worst = sorted(per_image, key=lambda r: -(r["draw"] * 2 + r["relabel"] + r["delete"] + r["adjust"] * 0.5))[:8]
    cases = []
    for r in worst:
        why = ", ".join(f"{k} {r[k]}" for k in ("draw", "relabel", "delete", "adjust") if r[k])
        cases.append((r["file"], r["_gts"], r["_preds"], why or "ok"))
    tag = Path(weights).stem
    gallery(cases, img_dir, cat_names, out_dir / f"failures_{tag}.png", f"Hardest test images for {tag}")
    save_json([{k: v for k, v in r.items() if not k.startswith("_")} for r in per_image], out_dir / f"per_image_{tag}.json")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", default=["yolo11s.pt"])
    ap.add_argument("--conf", type=float, default=0.3, help="threshold at which suggestions are shown in the app")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N test images (quick check)")
    args = ap.parse_args()

    aw = DATASETS / "african-wildlife"
    gt = json.loads((aw / "test_gt_coco.json").read_text())
    if args.limit:
        keep = {im["id"] for im in gt["images"][: args.limit]}
        gt["images"] = [im for im in gt["images"] if im["id"] in keep]
        gt["annotations"] = [a for a in gt["annotations"] if a["image_id"] in keep]
    out = RESULTS / "detector"
    results = [evaluate(w, gt, aw / "images" / "test", args.conf, out) for w in args.weights]
    save_json(results, out / "summary.json")

    lines = ["| detector | mAP@0.5 | mAP@0.5:0.95 | " + " | ".join(f"AP50 {c['name']}" for c in gt["categories"])
             + " | GT found (right class) | boxes drawn by hand: manual → assisted | corrections (adjust / relabel / delete) | ms/img |",
             "|---" * (7 + len(gt["categories"])) + "|"]
    for r in results:
        lines.append(
            f"| {Path(r['weights']).name} | {r['mAP50']:.3f} | {r['mAP50_95']:.3f} | "
            + " | ".join(f"{r['per_class_AP50'][c['name']]:.3f}" for c in gt["categories"])
            + f" | {100 * r['gt_objects_covered_by_correct_suggestion']:.0f}% | "
            f"{r['boxes_to_draw_manual']} → {r['boxes_to_draw_assisted']} | "
            f"{r['actions']['adjust']} / {r['actions']['relabel']} / {r['actions']['delete']} | {r['latency_ms_median']} |"
        )
    md = "\n".join(lines)
    (out / "summary.md").write_text(md + "\n")
    print(md)
    print(f"\nResults and failure galleries in {out}")


if __name__ == "__main__":
    main()
