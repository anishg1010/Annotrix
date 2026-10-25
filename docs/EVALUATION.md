# Evaluation protocol

The brief asks for *evidence that it works compared to manual labeling*, with model quality
(e.g. mAP, IoU), effort saved, and failure cases. We measure four things.

## 1. Detector quality — `eval/eval_detector.py`

* **Data:** Ultralytics African Wildlife, test split (227 images, 375 boxes; buffalo, elephant, rhino, zebra).
* **Metrics:** COCO mAP@0.5 and mAP@0.5:0.95 (pycocotools), per-class AP@0.5, median latency.
* **Compared:** stock COCO YOLO11 vs YOLO11 fine-tuned on the African Wildlife train split.
* **Failure cases:** `failures_<model>.png`, the 8 test images needing the most corrections.

## 2. Suggestion usefulness (simulated annotator) — same script

Each suggestion shown at the app's confidence threshold (default 0.30) is matched greedily to ground truth:

| Outcome | Rule | User action in the app |
|---|---|---|
| accept | same class, IoU ≥ 0.75 | none (bulk accept `A`) |
| adjust | IoU 0.5–0.75 | drag box handles |
| relabel | IoU ≥ 0.5, wrong class | press `1–9` |
| delete | no match (false positive) | `Del` |
| draw | ground-truth object with no suggestion | draw by hand |

Manual labelling = drawing every object. This turns mAP into "how much work is left".

## 3. Click-to-segment quality — `eval/eval_segmenter.py`

* **Data:** COCO128-seg instance masks (objects ≥ 32×32 px).
* **Protocol:** standard interactive-segmentation simulation (as in RITM / SAM): first click at the
  object's deepest interior point; each next click at the centre of the largest error region
  (positive if under-segmented, negative if over-segmented). Also the GT box as a prompt.
* **Metrics:** mIoU after 1…5 clicks, % objects with IoU ≥ 0.85, NoC@85.

Baseline (MobileSAM, CPU, 150 objects): box prompt mIoU 0.74; 1 click 0.50 → 5 clicks 0.76; NoC@85 = 3.6.
Converting the mask to a polygon costs ≈ 0.01 IoU (measured); COCO's own polygons are coarse,
which caps the attainable IoU.

## 4. User study: manual vs assisted — `eval/effort.py`

1. `python scripts/prepare_data.py` creates two disjoint, class-stratified sets `data/study/A` and `data/study/B`
   (20 test images each, never used for fine-tuning).
2. In the app create **Study manual** (mode *manual*) and upload A; create **Study assisted** (mode *assisted*) and upload B.
3. Label each set completely. Do one warm-up image first in each mode (delete its session or keep note).
   Ideally have a second person do the opposite assignment (B manual, A assisted) to cancel out set difficulty.
4. `python eval/effort.py --manual "Study manual" --assisted "Study assisted"`

The app logs per image: active seconds (time with no input for 30 s is not counted), and counts of boxes drawn,
suggestions accepted / adjusted / rejected, relabels and AI calls.

The script reports:
* seconds per image and per object, actions per image, Mann-Whitney U test on time per image
* **quality vs ground truth** for both sets (precision, recall, mean IoU of saved boxes), showing speed is not bought with errors
* a **time model** fitted on the logged sessions (seconds per drawn box, per accepted suggestion, …), applied to the
  simulated action counts from §2 to estimate the time to label the full test split with each detector.

## Threats to validity (for the report's limitations section)

* Single annotator, small sample; the learning effect is reduced by the warm-up image.
* Test images are from the same distribution as the fine-tuning data; real deployments meet new domains.
* COCO128-seg is a small, generic set; mask quality on wildlife may differ.
* The simulated annotator assumes perfect corrections; real users sometimes accept wrong suggestions,
  which is what the quality-vs-GT check in §4 catches.
