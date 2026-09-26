from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the forest-camera fire/smoke alert detector on a video.")
    parser.add_argument("video", help="Video path, webcam index such as 0, or an IP Webcam/RTSP URL")
    parser.add_argument("--weights", type=Path, default=Path("runs/forest_fire/fasdd_real_smoke_30ep/weights/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/video_predictions.mp4"))
    parser.add_argument("--conf", type=float, default=0.40, help="Detection confidence threshold.")
    parser.add_argument("--sample-seconds", type=float, default=10.0, help="Run detection once per this many seconds; 0 processes every frame.")
    parser.add_argument("--confirmations", type=int, default=2, help="Consecutive positive samples required before a FIRE ALERT.")
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
    frame_number = 0
    next_sample = 0
    consecutive_alerts = 0
    active_alert = False
    last_result = None
    sample_interval = max(1, round(args.sample_seconds * fps)) if args.sample_seconds else 1
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_number >= next_sample:
            last_result = model.predict(frame, conf=args.conf, device=args.device, verbose=False)[0]
            detected = len(last_result.boxes) > 0
            consecutive_alerts = consecutive_alerts + 1 if detected else 0
            active_alert = consecutive_alerts >= args.confirmations
            next_sample = frame_number + sample_interval
        annotated = last_result.plot() if last_result is not None else frame.copy()
        status = "FIRE ALERT" if active_alert else "Monitoring: no confirmed fire/smoke"
        color = (0, 0, 255) if active_alert else (0, 180, 0)
        cv2.putText(annotated, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        writer.write(annotated)
        frame_number += 1
    capture.release()
    writer.release()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
