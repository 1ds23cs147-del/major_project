# Forest Fire Detector — Classification Metrics Report

> **Model:** YOLO11n (Ultralytics) · **Task:** Fire/Smoke object detection + binary fire-alert classification
> **Test set:** 5,203 images · **Alert threshold:** 0.40 confidence
> **Generated:** 2026-08-30

---

## 1. Binary Fire-Alert Classification

The model is treated as a binary classifier: an image is flagged as **fire** if the detector outputs at least one bounding box above the confidence threshold.

| Metric | Value |
|--------|-------|
| **Accuracy** | **92.16%** |
| **Precision** | 99.32% |
| **Recall** | 87.76% |
| **F1-Score** | 93.19% |
| **Specificity** | 99.06% |
| **Balanced Accuracy** | 93.41% |
| **False Positive Rate** | 0.94% |
| **False Negative Rate** | 12.24% |

### Confusion Matrix

| | Predicted: No Fire | Predicted: Fire |
|---|---|---|
| **Actual: No Fire** | 2,005 (TN) | 19 (FP) |
| **Actual: Fire** | 389 (FN) | 2,790 (TP) |

- **True Negatives (TN):** 2,005
- **False Positives (FP):** 19
- **False Negatives (FN):** 389
- **True Positives (TP):** 2,790

### Derived Metrics

| Metric | Formula | Value |
|--------|---------|-------|
| Accuracy | (TP + TN) / Total | 92.16% |
| Precision | TP / (TP + FP) | 99.32% |
| Recall (Sensitivity) | TP / (TP + FN) | 87.76% |
| Specificity | TN / (TN + FP) | 99.06% |
| F1-Score | 2·P·R / (P + R) | 93.19% |
| Balanced Accuracy | (Sens + Spec) / 2 | 93.41% |

---

## 2. Object Detection Metrics (YOLO)

| Metric | Value |
|--------|-------|
| **Precision** | 78.0% |
| **Recall** | 69.6% |
| **mAP@50** | 78.1% |
| **mAP@50-95** | 46.5% |

### Per-Class Detection Metrics

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|--------|-----------|-----------|--------|--------|-----------|
| **fire** | 1,752 | 3,872 | 75.1% | 64.0% | 74.1% | 41.7% |
| **smoke** | 2,812 | 3,196 | 80.9% | 75.2% | 82.1% | 51.4% |
| **all** | 5,203 | 7,068 | 78.0% | 69.6% | 78.1% | 46.5% |

---

## 3. Inference Performance

| Stage | Time per image |
|-------|----------------|
| Preprocess | 2.4 ms |
| Inference | 2.9 ms |
| Postprocess | 1.6 ms |
| **Total** | **~6.9 ms** |

---

## 4. Interpretation

- **Excellent binary classification:** 92% accuracy with 99% precision means false alarms are extremely rare (only 19 in 5,203 images).
- **Conservative fire detection:** Recall of 87.8% means the model misses ~12% of actual fire cases (389 false negatives). This is a deliberate trade-off favoring precision to avoid nuisance alerts.
- **Smoke is detected better than fire** (82.1% vs 74.1% mAP@50) — smoke plumes are typically larger and more distinctive than small flame regions.
- **Real-time capable:** ~7 ms per image supports live video monitoring at high frame rates.

---

## 5. Raw Data Source

The raw JSON output is stored in `runs/forest_fire/fire_smoke/test_metrics.json` (local, not versioned).

```json
{
  "test_images": 5203,
  "alert_threshold": 0.4,
  "binary_alert": {
    "accuracy": 0.9215837017105516,
    "precision": 0.9932360270558918,
    "recall": 0.8776344762503933,
    "f1": 0.9318637274549099,
    "confusion_matrix": [[2005, 19], [389, 2790]]
  },
  "detection": {
    "precision": 0.7799302077682506,
    "recall": 0.6960685096838155,
    "map50": 0.7809149259496695,
    "map50_95": 0.4654314042311027
  }
}
```

## 6. Reproducing These Metrics

Run the generator script to recompute or extend these metrics:

```powershell
# Offline mode — derives all metrics from the existing test_metrics.json (no model needed)
python generate_classification_report.py --from-json

# Full mode — re-runs the model over the test set (requires venv with ultralytics + scikit-learn)
.\.venv\Scripts\python.exe generate_classification_report.py
```

The full JSON output (including specificity, balanced accuracy, FPR/FNR, and the
sklearn classification report) is written to
`runs/forest_fire/fire_smoke/classification_metrics.json`.
