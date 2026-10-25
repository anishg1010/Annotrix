"""Fine-tune the pre-labelling detector on African Wildlife (run on a GPU machine).

    python scripts/finetune.py                         # yolo11s, 60 epochs, GPU 0
    python scripts/finetune.py --model yolo11n.pt --epochs 3 --device cpu   # smoke test
    python scripts/finetune.py --data path/to/export/data.yaml               # train on a dataset exported from the app

The best weights are copied to data/weights/wildlife-<model>.pt. Point the app at them with
LA_DETECTOR=wildlife-yolo11s.pt (docker compose reads it from .env) or from the API:
    curl -X POST localhost:8000/api/models/detector -H 'content-type: application/json' \
         -d '{"weights": "wildlife-yolo11s.pt"}'
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt", help="COCO-pretrained starting point")
    ap.add_argument("--data", default=str(ROOT / "data/datasets/african-wildlife/african-wildlife.yaml"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="'0' for the first GPU, 'cpu' otherwise")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    weights_dir = ROOT / "data" / "weights"
    start = weights_dir / args.model
    model = YOLO(str(start) if start.exists() else args.model)
    name = args.name or f"wildlife-{Path(args.model).stem}"
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "runs"),
        name=name,
        exist_ok=True,
        patience=15,
        seed=0,
        plots=True,
    )
    best = ROOT / "runs" / name / "weights" / "best.pt"
    weights_dir.mkdir(parents=True, exist_ok=True)
    dest = weights_dir / f"{name}.pt"
    shutil.copy(best, dest)
    print(f"\nSaved fine-tuned detector to {dest}")
    print(f"Evaluate it:  python eval/eval_detector.py --weights {args.model} {dest}")


if __name__ == "__main__":
    main()
