from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fire/smoke/person detection on a video.")
    parser.add_argument("video", help="Video path, webcam index such as 0, or an IP Webcam/RTSP URL")
    parser.add_argument("--weights", type=Path, default=Path("runs/detect/fire_smoke_person/weights/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/video_predictions.mp4"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    model = YOLO(str(args.weights))
    capture_source = int(args.video) if args.video.isdigit() else args.video
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video or stream: {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = model.predict(frame, conf=args.conf, device=args.device, verbose=False)[0]
        writer.write(result.plot())
    capture.release()
    writer.release()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
