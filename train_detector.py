from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the unified fire/smoke/person detector.")
    parser.add_argument("--data", type=Path, default=Path("project_data/artifacts/unified_yolo/data.yaml"))
    parser.add_argument("--weights", default="project_data/models/yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    model = YOLO(args.weights)
    model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers, project="project_data/runs/detect", name="fire_smoke_person", pretrained=True, patience=20, cache=False)
    model.val(data=str(args.data.resolve()), device=args.device)


if __name__ == "__main__":
    main()
