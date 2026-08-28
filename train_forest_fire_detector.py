from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the forest-camera fire/smoke alert detector.")
    parser.add_argument("--data", type=Path, default=Path("artifacts/forest_fire_yolo/data.yaml"))
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=40, help="Keep below 50 epochs for the requested training budget.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parent / "runs" / "forest_fire"
    YOLO(args.weights).train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers, project=str(project_dir), name="fire_smoke", exist_ok=True, pretrained=True, patience=12, seed=42)


if __name__ == "__main__":
    main()
