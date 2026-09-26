from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "project_data" / "datasets" / "rgb_thermal" / "flame3_cls_real" / "thermal"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the thermal fire/no-fire classifier.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", default=str(ROOT / "project_data" / "models" / "yolo11n-cls.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--name", default="thermal_fire")
    args = parser.parse_args()

    if not args.data.is_dir():
        raise SystemExit(f"Thermal classification dataset directory not found: {args.data}")

    YOLO(args.weights).train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(ROOT / "project_data" / "runs" / "forest_fire"),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        patience=10,
        seed=42,
        optimizer="AdamW",
    )


if __name__ == "__main__":
    main()
