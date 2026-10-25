"""Manual vs AI-assisted labelling: effort and quality from the user study.

Protocol (see docs/EVALUATION.md):
  1. python scripts/prepare_data.py  -> data/study/A and data/study/B
  2. In the app, project "Study manual" (mode manual) <- upload A; label everything.
     Project "Study assisted" (mode assisted) <- upload B; label everything.
  3. python eval/effort.py --manual "Study manual" --assisted "Study assisted"

The app logs active time (idle > 30 s excluded) and every edit for each image. This script
  * compares time per image / per object and number of actions (Mann-Whitney U test)
  * checks the labels are still correct: precision, recall and mean IoU of the saved boxes
    against the African Wildlife ground truth (so speed is not bought with quality)
  * fits a simple time model  t = c_draw*drawn + c_accept*accepted + c_adjust*adjusted + ...
    and uses it with eval_detector.py's simulated action counts to estimate the time to label
    the whole test split manually vs with each detector.
"""
from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from common import DATASETS, RESULTS, box_iou, save_json


def get(base: str, path: str):
    with urllib.request.urlopen(base + path) as r:
        data = r.read()
    return data if path.endswith("zip") or "export" in path else json.loads(data)


def find_project(base: str, name_or_id: str) -> dict:
    projects = get(base, "/api/projects")
    for p in projects:
        if str(p["id"]) == name_or_id or p["name"] == name_or_id:
            return p
    raise SystemExit(f"project {name_or_id!r} not found; have: {[p['name'] for p in projects]}")


def quality(base: str, pid: int, gt: dict) -> dict:
    """Compare the project's saved boxes to ground truth (IoU >= 0.5, same class)."""
    z = zipfile.ZipFile(io.BytesIO(get(base, f"/api/projects/{pid}/export?format=coco&only_done=true")))
    coco = json.loads(z.read("annotations.json"))
    cat_name = {c["id"]: c["name"].lower() for c in coco["categories"]}
    gt_cat = {c["id"]: c["name"].lower() for c in gt["categories"]}
    gt_by_file: dict[str, list] = {}
    gt_img = {im["id"]: im["file_name"] for im in gt["images"]}
    for a in gt["annotations"]:
        gt_by_file.setdefault(gt_img[a["image_id"]], []).append(a)
    img_file = {im["id"]: Path(im["file_name"]).name for im in coco["images"]}
    pred_by_file: dict[str, list] = {}
    for a in coco["annotations"]:
        pred_by_file.setdefault(img_file[a["image_id"]], []).append(a)

    tp = fp = fn = 0
    ious = []
    for f in img_file.values():
        g = gt_by_file.get(f, [])
        p = pred_by_file.get(f, [])
        m = box_iou(np.array([a["bbox"] for a in p]).reshape(-1, 4), np.array([a["bbox"] for a in g]).reshape(-1, 4))
        used = set()
        for i in np.argsort(-m.max(1) if m.size else np.array([], int)):
            j = int(m[i].argmax()) if m.shape[1] else -1
            if j >= 0 and m[i, j] >= 0.5 and j not in used and cat_name[p[i]["category_id"]] == gt_cat[g[j]["category_id"]]:
                used.add(j)
                tp += 1
                ious.append(m[i, j])
            else:
                fp += 1
        fn += len(g) - len(used)
    return {
        "images": len(img_file),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "mean_box_iou_matched": float(np.mean(ious)) if ious else 0.0,
    }


def summarise(sessions: list[dict]) -> dict:
    per_img: dict[int, dict] = {}
    for s in sessions:
        d = per_img.setdefault(s["image_id"], {k: 0 for k in ("active_seconds", "n_actions", "n_drawn", "n_accepted", "n_adjusted", "n_rejected", "n_relabelled", "n_ai_calls")})
        for k in d:
            d[k] += s[k]
        d["n_final"] = s["n_final"]
    rows = list(per_img.values())
    t = np.array([r["active_seconds"] for r in rows])
    objs = np.array([max(r["n_final"], 1) for r in rows])
    return {
        "images": len(rows),
        "objects": int(sum(r["n_final"] for r in rows)),
        "seconds_per_image_mean": float(t.mean()) if len(t) else 0,
        "seconds_per_image_median": float(np.median(t)) if len(t) else 0,
        "seconds_per_object": float(t.sum() / max(objs.sum(), 1)),
        "actions_per_image": float(np.mean([r["n_actions"] for r in rows])) if rows else 0,
        "_times": t.tolist(),
        "_rows": rows,
    }


def fit_time_model(rows: list[dict]) -> dict:
    """Non-negative least squares: seconds ~ per-action costs (+ per-image overhead)."""
    from scipy.optimize import nnls

    feats = ["n_drawn", "n_accepted", "n_adjusted", "n_rejected", "n_relabelled"]
    X = np.array([[r[f] for f in feats] + [1.0] for r in rows], float)
    y = np.array([r["active_seconds"] for r in rows], float)
    coef, _ = nnls(X, y)
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
    return {**{f: float(c) for f, c in zip(feats, coef)}, "per_image_overhead": float(coef[-1]), "r2": float(r2), "n": len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--manual", required=True, help="name or id of the manual-mode project")
    ap.add_argument("--assisted", required=True, help="name or id of the assisted-mode project")
    args = ap.parse_args()

    gt = json.loads((DATASETS / "african-wildlife" / "test_gt_coco.json").read_text())
    out = RESULTS / "effort"
    res = {}
    for mode, name in (("manual", args.manual), ("assisted", args.assisted)):
        p = find_project(args.api, name)
        s = summarise(get(args.api, f"/api/projects/{p['id']}/sessions"))
        s["quality"] = quality(args.api, p["id"], gt)
        s["project"] = p["name"]
        res[mode] = s

    from scipy.stats import mannwhitneyu

    tm, ta = res["manual"]["_times"], res["assisted"]["_times"]
    if len(tm) > 1 and len(ta) > 1:
        res["mannwhitney_p_time_per_image"] = float(mannwhitneyu(tm, ta, alternative="greater").pvalue)
    res["speedup_seconds_per_object"] = res["manual"]["seconds_per_object"] / max(res["assisted"]["seconds_per_object"], 1e-9)
    model = fit_time_model(res["manual"]["_rows"] + res["assisted"]["_rows"])
    res["time_model"] = model

    # project to the full test split with each evaluated detector
    proj = []
    det_summary = RESULTS / "detector" / "summary.json"
    if det_summary.exists():
        for d in json.loads(det_summary.read_text()):
            a = d["actions"]
            n_img = d["images"]
            manual = model["n_drawn"] * a["n_gt"] + model["per_image_overhead"] * n_img
            assisted = (model["n_drawn"] * a["draw"] + model["n_accepted"] * a["accept"] + model["n_adjusted"] * a["adjust"]
                        + model["n_rejected"] * a["delete"] + model["n_relabelled"] * a["relabel"] + model["per_image_overhead"] * n_img)
            proj.append({"detector": Path(d["weights"]).name, "manual_min": manual / 60, "assisted_min": assisted / 60})
    res["projection_full_test_split"] = proj

    save_json({k: ({kk: vv for kk, vv in v.items() if not kk.startswith("_")} if isinstance(v, dict) else v) for k, v in res.items()},
              out / "summary.json")

    md = ["| | manual | assisted |", "|---|---|---|"]
    for label, key in (("images", "images"), ("objects", "objects"), ("s / image (mean)", "seconds_per_image_mean"),
                       ("s / image (median)", "seconds_per_image_median"), ("s / object", "seconds_per_object"),
                       ("actions / image", "actions_per_image")):
        md.append(f"| {label} | {res['manual'][key]:.1f} | {res['assisted'][key]:.1f} |")
    for label, key in (("precision vs GT", "precision"), ("recall vs GT", "recall"), ("mean IoU vs GT", "mean_box_iou_matched")):
        md.append(f"| {label} | {res['manual']['quality'][key]:.3f} | {res['assisted']['quality'][key]:.3f} |")
    md.append(f"\nSpeed-up per object: **{res['speedup_seconds_per_object']:.2f}×**")
    if "mannwhitney_p_time_per_image" in res:
        md.append(f"Mann-Whitney U (manual slower than assisted), p = {res['mannwhitney_p_time_per_image']:.4f}")
    md.append("\nFitted time model (seconds): " + ", ".join(f"{k} {v:.2f}" for k, v in model.items() if k not in ("r2", "n"))
              + f"  (R² {model['r2']:.2f}, n={model['n']})")
    for p in proj:
        md.append(f"Projected time for the full test split with `{p['detector']}`: manual {p['manual_min']:.0f} min → assisted {p['assisted_min']:.0f} min")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.boxplot([tm, ta], tick_labels=["manual", "assisted"], widths=0.5)
    ax.set_ylabel("active seconds per image")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "time_per_image.png", dpi=130)
    print(f"\nResults in {out}")


if __name__ == "__main__":
    main()
