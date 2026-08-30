"""Generate a detailed binary fire-alert classification report for the trained detector.

Two modes:

1. **Full mode (default):** runs the trained YOLO model over the test set and computes
   metrics including ROC/AUC from raw confidence scores. Requires the venv with
   ultralytics + scikit-learn installed.

2. **Offline mode (--from-json):** reads an existing ``test_metrics.json`` (produced by
   ``evaluate_forest_fire_detector.py``) and derives all classification metrics from the
   stored confusion matrix. Pure Python — no model or heavy dependencies required.

Usage:
    .\\.venv\\Scripts\\python.exe generate_classification_report.py [--conf 0.40] [--device 0]
    python generate_classification_report.py --from-json [--metrics runs/forest_fire/fire_smoke/test_metrics.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def derive_metrics(tn: int, fp: int, fn: int, tp: int) -> dict:
    """Compute all binary classification metrics from a confusion matrix (pure Python)."""
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }


def from_json(metrics_path: Path, conf: float) -> dict:
    """Derive classification metrics from an existing test_metrics.json."""
    if not metrics_path.exists():
        raise SystemExit(f"Metrics file not found: {metrics_path}")
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    matrix = data["binary_alert"]["confusion_matrix"]
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    derived = derive_metrics(tn, fp, fn, tp)
    report = {
        "test_images": int(data["test_images"]),
        "alert_threshold": float(data.get("alert_threshold", conf)),
        "source": str(metrics_path),
        "binary_alert": {
            **derived,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "matrix": matrix},
        },
        "detection": data.get("detection", {}),
    }
    return report


def full_mode(args: argparse.Namespace) -> dict:
    """Run the model over the test set and compute metrics including ROC/AUC."""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        auc,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
        roc_curve,
    )
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    data = args.data.resolve()
    root = data.parent
    image_dir, label_dir = root / "test" / "images", root / "test" / "labels"

    if not data.exists() or not image_dir.is_dir() or not label_dir.is_dir():
        raise SystemExit(
            "The prepared test dataset is missing or incomplete. Rebuild it first with: "
            ".\\.venv\\Scripts\\python.exe prepare_forest_fire_dataset.py --overwrite"
        )

    # Ground-truth labels: 1 if a label file exists and is non-empty, else 0.
    actual: list[int] = []
    for image in sorted(image_dir.iterdir()):
        if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        label = label_dir / f"{image.stem}.txt"
        actual.append(int(label.exists() and bool(label.read_text(encoding="utf-8").strip())))

    # Predictions: binary alert (any box) plus max confidence score for ROC/AUC.
    predicted: list[int] = []
    scores: list[float] = []
    for result in model.predict(source=str(image_dir), conf=args.conf, device=args.device, verbose=False, stream=True):
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            predicted.append(0)
            scores.append(0.0)
            continue
        predicted.append(1)
        scores.append(float(boxes.conf.max().item()))

    if len(actual) != len(predicted):
        raise RuntimeError(
            f"Test set changed during evaluation: found {len(actual)} labels but received {len(predicted)} predictions."
        )

    y_true = np.array(actual)
    y_pred = np.array(predicted)
    y_score = np.array(scores)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    roc_auc_sk = roc_auc_score(y_true, y_score)

    return {
        "test_images": int(len(y_true)),
        "alert_threshold": args.conf,
        "binary_alert": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "specificity": float(specificity),
            "balanced_accuracy": float(balanced_accuracy),
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
            "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
            "roc_auc": float(roc_auc),
            "roc_auc_sklearn": float(roc_auc_sk),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]},
            "classification_report": classification_report(
                y_true, y_pred, labels=[0, 1], target_names=["no_fire", "fire"], zero_division=0
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a detailed binary fire-alert classification report.")
    parser.add_argument("--weights", type=Path, default=Path("runs/forest_fire/fire_smoke/weights/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("artifacts/forest_fire_yolo/data.yaml"))
    parser.add_argument("--conf", type=float, default=0.40, help="Alert threshold for binary classification.")
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--from-json",
        action="store_true",
        help="Offline mode: derive metrics from an existing test_metrics.json instead of running the model.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("runs/forest_fire/fire_smoke/test_metrics.json"),
        help="Path to test_metrics.json used in --from-json mode.",
    )
    args = parser.parse_args()

    report = from_json(args.metrics, args.conf) if args.from_json else full_mode(args)

    destination = args.weights.resolve().parent.parent / "classification_metrics.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    if "classification_report" in report["binary_alert"]:
        print("\n--- sklearn classification_report ---")
        print(report["binary_alert"]["classification_report"])
    print(f"\nReport saved to: {destination}")


if __name__ == "__main__":
    main()
