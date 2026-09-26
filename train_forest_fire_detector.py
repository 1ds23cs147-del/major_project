from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the forest-camera fire/smoke alert detector.")
    parser.add_argument("--data", type=Path, default=Path("artifacts/forest_fire_yolo/data.yaml"))
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs (GPU makes 50-60 affordable).")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--optimizer", default="AdamW", help="Optimizer: AdamW (default), SGD, Adam, etc.")
    parser.add_argument("--name", default="fire_smoke", help="Output run name.")
    parser.add_argument("--resume", type=Path, help="Resume an interrupted run from its last.pt checkpoint.")
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parent / "runs" / "forest_fire"
    if args.resume:
        if not args.resume.is_file():
            raise SystemExit(f"Checkpoint not found: {args.resume}")
        YOLO(str(args.resume)).train(resume=True)
        return
    YOLO(args.weights).train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        patience=15,
        seed=42,
        optimizer=args.optimizer,
        cos_lr=True,
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        warmup_bias_lr=0.1,
        weight_decay=0.0005,
        close_mosaic=10,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )


if __name__ == "__main__":
    main()
