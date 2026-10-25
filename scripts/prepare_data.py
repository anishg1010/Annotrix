"""Download the evaluation data and create the user-study image sets.

    python scripts/prepare_data.py            # African Wildlife + COCO128-seg
    python scripts/prepare_data.py --study 20 # images per study set (default 20)

Creates
  data/datasets/african-wildlife/        YOLO dataset (train/val/test) for fine-tuning + detector eval
  data/datasets/african-wildlife/test_gt_coco.json   ground truth of the test split as COCO
  data/datasets/coco128-seg/             instance masks for the segmenter eval
  data/study/A, data/study/B             two disjoint image sets from the test split, for the
                                         manual-vs-assisted timing study (upload one set per mode)
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from common import (  # noqa: E402
    AW_NAMES,
    AW_URL,
    COCO128_SEG_URL,
    DATA,
    DATASETS,
    download_and_unzip,
    save_json,
    yolo_split_to_coco,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", type=int, default=20, help="images per study set")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    aw = DATASETS / "african-wildlife"
    download_and_unzip(AW_URL, aw, "images/test")
    yaml = aw / "african-wildlife.yaml"
    yaml.write_text(
        f"path: {aw.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(AW_NAMES))
    )
    gt = yolo_split_to_coco(aw, "test", AW_NAMES)
    save_json(gt, aw / "test_gt_coco.json")
    print(f"African Wildlife test: {len(gt['images'])} images, {len(gt['annotations'])} boxes")

    download_and_unzip(COCO128_SEG_URL, DATASETS, "coco128-seg/images")
    print("COCO128-seg ready")

    # Study sets: stratify by dominant class so A and B are comparable.
    by_class: dict[int, list[dict]] = {}
    counts: dict[int, list[int]] = {}
    for a in gt["annotations"]:
        counts.setdefault(a["image_id"], []).append(a["category_id"])
    for im in gt["images"]:
        cs = counts.get(im["id"])
        if cs:
            by_class.setdefault(max(set(cs), key=cs.count), []).append(im)
    rng = random.Random(args.seed)
    A, B = [], []
    per_class = max(1, args.study // len(by_class))
    for c in sorted(by_class):
        ims = by_class[c][:]
        rng.shuffle(ims)
        A += ims[:per_class]
        B += ims[per_class : 2 * per_class]
    for name, ims in (("A", A), ("B", B)):
        d = DATA / "study" / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        for im in ims:
            shutil.copy(aw / "images" / "test" / im["file_name"], d / im["file_name"])
        print(f"study set {name}: {len(ims)} images -> {d}")
    print("\nNext: create a 'manual' project and upload data/study/A, an 'assisted' project and upload data/study/B.")


if __name__ == "__main__":
    main()
