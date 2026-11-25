# User study: manual vs assisted (24 Nov 2026, one annotator, laptop CPU)

Two disjoint sets of 20 African Wildlife test images (created by `scripts/prepare_data.py`):
set A labelled in a manual-mode project, set B in an assisted project with `wildlife-yolo11n-cpu.pt`.
Active time and edit counts come from the app's effort log; quality from
`python eval/study_quality.py <export> --split test` on the two exports in this folder.

| | Manual (set A) | Assisted (set B) |
|---|---|---|
| Images | 20 | 20 |
| Animals in ground truth | 47 | 38 |
| Boxes saved | 51 | 43 |
| Boxes drawn by hand | 58 (incl. redrawn) | 6 |
| AI suggestions accepted (unchanged / adjusted) | – | 36 / 1 |
| Total active time | 351 s | 235 s |
| Seconds per image | 17.6 | 11.8 (median 8.3) |
| Seconds per saved box | 6.9 | 5.5 |
| Precision vs ground truth | 0.90 | 0.84 |
| Recall vs ground truth | 0.98 | 0.95 |
| Mean box IoU vs ground truth | 0.80 | 0.90 |

Caveats: one participant; set B had been opened once in a practice run; per-image manual times
were not retained (the manual project was deleted after its totals were recorded), so no significance
test; three assisted images were first saved before accepting suggestions and then corrected (time included).

Deliverable dataset (100 validation images, assisted): 156 boxes (140 AI, 16 manual), 364 s active
(3.6 s per image); vs ground truth precision 0.93, recall 0.94, mean IoU 0.90
(`python eval/study_quality.py Wildlife_dataset_coco.zip --split val`).
