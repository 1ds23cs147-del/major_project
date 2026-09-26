from __future__ import annotations

import argparse
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Report test detection and binary fire-alert metrics.")
    parser.add_argument("--weights", type=Path, default=Path("project_data/models/rgb_fire_smoke_best.pt"))
    parser.add_argument("--data", type=Path, default=Path("project_data/artifacts/forest_fire_yolo/data.yaml"))
    parser.add_argument("--conf", type=float, default=0.40, help="Alert threshold; tune on validation data to control false alerts.")
    parser.add_argument("--device", default="0")
    args = parser.parse_args(); model = YOLO(str(args.weights)); data = args.data.resolve(); root = data.parent
    image_dir, label_dir = root / "test" / "images", root / "test" / "labels"
    if not data.exists() or not image_dir.is_dir() or not label_dir.is_dir():
        raise SystemExit(
            "The prepared test dataset is missing or incomplete. Rebuild it first with: "
            ".\\.venv\\Scripts\\python.exe prepare_forest_fire_dataset.py --overwrite"
        )
    detection = model.val(data=str(data), split="test", device=args.device, plots=True)
    actual, predicted = [], []
    for image in sorted(image_dir.iterdir()):
        if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        label = label_dir / f"{image.stem}.txt"
        actual.append(int(label.exists() and bool(label.read_text(encoding="utf-8").strip())))
    for result in model.predict(source=str(image_dir), conf=args.conf, device=args.device, verbose=False, stream=True):
        predicted.append(int(len(result.boxes) > 0))
    if len(actual) != len(predicted):
        raise RuntimeError(f"Test set changed during evaluation: found {len(actual)} labels but received {len(predicted)} predictions.")
    precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average="binary", zero_division=0)
    report = {"test_images": len(actual), "alert_threshold": args.conf, "binary_alert": {"accuracy": accuracy_score(actual, predicted), "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist()}, "detection": {"precision": float(detection.results_dict["metrics/precision(B)"]), "recall": float(detection.results_dict["metrics/recall(B)" ]), "map50": float(detection.results_dict["metrics/mAP50(B)"]), "map50_95": float(detection.results_dict["metrics/mAP50-95(B)"])}}
    destination = Path("project_data/runs/evaluation/test_metrics.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
