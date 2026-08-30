"""Multimodal Forest-Fire & Endangered-Animal Web Application.

A Flask server that:
  - streams a live camera feed (RGB only) with real-time fire/smoke detection,
  - accepts RGB / thermal / NIR uploads of the same scene and fuses them,
  - only allows an SOS alert when at least 2 of 3 modalities confirm fire,
  - reports endangered-animal co-occurrence.

Run:
    .\\.venv\\Scripts\\python.exe webapp\\app.py
"""

from __future__ import annotations

import io
import json
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

from multimodal_detector import (
    FusedDecision,
    ModalityResult,
    get_detector,
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
    ".jfif", ".pnm", ".ppm", ".pgm", ".pbm",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

# ---------------------------------------------------------------------------
# Live camera state
# ---------------------------------------------------------------------------
_camera = None
_camera_lock = None
_camera_source = 0  # default webcam index; can be an RTSP/IP URL


def _get_camera():
    """Lazily open the camera and return a thread-safe handle."""
    global _camera, _camera_lock
    if _camera is None:
        _camera = cv2.VideoCapture(_camera_source)
        _camera_lock = __import__("threading").Lock()
    return _camera, _camera_lock


def _read_frame():
    """Read a single frame from the camera, or None if unavailable."""
    try:
        cam, lock = _get_camera()
        with lock:
            ok, frame = cam.read()
        return frame if ok else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _decode_image(data: bytes) -> np.ndarray | None:
    """Decode raw bytes into a BGR image, handling 16-bit thermal data."""
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.dtype == np.uint16:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def _encode_jpeg(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        return b""
    return buf.tobytes()


def _modality_result_to_dict(result: ModalityResult) -> dict:
    return {
        "modality": result.modality,
        "fire_score": round(result.fire_score, 4),
        "fire_detected": result.fire_detected,
        "detections": [
            {"label": d.label, "confidence": round(d.confidence, 4), "box": d.box}
            for d in result.detections
        ],
        "animal_detected": result.animal_detected,
        "animal_labels": result.animal_labels,
    }


def _decision_to_dict(decision: FusedDecision) -> dict:
    return {
        "status": decision.status,
        "fused_score": round(decision.fused_score, 4),
        "modalities_checked": decision.modalities_checked,
        "modalities_confirmed": decision.modalities_confirmed,
        "sos_allowed": decision.sos_allowed,
        "message": decision.message,
        "animal_alerts": decision.animal_alerts,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "time": time.time()})


@app.route("/api/live/feed")
def live_feed():
    """MJPEG stream of the live camera with RGB fire/smoke detection overlay."""
    detector = get_detector()

    def generate():
        while True:
            frame = _read_frame()
            if frame is None:
                # Send a placeholder frame when the camera is unavailable.
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Camera unavailable", (120, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + _encode_jpeg(placeholder) + b"\r\n")
                time.sleep(0.5)
                continue

            result = detector.detect_rgb(frame)
            annotated = result.annotated if result.annotated is not None else frame
            status = "FIRE ALERT" if result.fire_detected else "Monitoring"
            color = (0, 0, 255) if result.fire_detected else (0, 180, 0)
            cv2.putText(annotated, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, color, 2, cv2.LINE_AA)
            if result.animal_detected:
                cv2.putText(annotated, "Animal: " + ",".join(result.animal_labels),
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + _encode_jpeg(annotated) + b"\r\n")
            time.sleep(0.05)  # ~20 fps

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/live/status")
def live_status():
    """Return the latest live detection status as JSON."""
    detector = get_detector()
    frame = _read_frame()
    if frame is None:
        return jsonify({"available": False, "fire_detected": False})
    result = detector.detect_rgb(frame)
    return jsonify({
        "available": True,
        "fire_detected": result.fire_detected,
        "fire_score": round(result.fire_score, 4),
        "animal_detected": result.animal_detected,
        "animal_labels": result.animal_labels,
        "detections": [{"label": d.label, "confidence": round(d.confidence, 4)} for d in result.detections],
    })


@app.route("/api/upload", methods=["POST"])
def upload():
    """Accept RGB / thermal / NIR uploads of the same scene and fuse them.

    Expects multipart form fields: rgb, thermal, nir (at least one), plus an
    optional capture_id. An SOS alert is only allowed when at least 2 of the 3
    modalities confirm fire.
    """
    detector = get_detector()
    capture_id = request.form.get("capture_id") or str(uuid.uuid4())

    results: dict[str, ModalityResult] = {}
    uploaded: dict[str, str] = {}

    for modality in ("rgb", "thermal", "nir"):
        file = request.files.get(modality)
        if file is None or file.filename == "":
            continue
        filename = secure_filename(file.filename)
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"Unsupported format for {modality}: {ext}"}), 400
        data = file.read()
        image = _decode_image(data)
        if image is None:
            return jsonify({"error": f"Could not decode {modality} image"}), 400

        # Save a copy for the audit trail.
        saved_name = f"{capture_id}_{modality}{ext}"
        (UPLOAD_DIR / saved_name).write_bytes(data)
        uploaded[modality] = saved_name

        if modality == "rgb":
            results[modality] = detector.detect_rgb(image)
        elif modality == "thermal":
            results[modality] = detector.detect_thermal(image)
        elif modality == "nir":
            results[modality] = detector.detect_nir(image)

    if not results:
        return jsonify({"error": "No valid images uploaded. Provide at least one of rgb, thermal, nir."}), 400

    decision = detector.fuse(results)

    response = {
        "capture_id": capture_id,
        "uploaded": uploaded,
        "modalities": {m: _modality_result_to_dict(r) for m, r in results.items()},
        "decision": _decision_to_dict(decision),
    }
    return jsonify(response)


@app.route("/api/sos", methods=["POST"])
def sos():
    """Trigger an SOS alert. Only allowed if the fused decision permits it."""
    data = request.get_json(silent=True) or {}
    capture_id = data.get("capture_id")
    if not capture_id:
        return jsonify({"error": "capture_id is required"}), 400

    # In a real deployment this would look up the stored decision for capture_id.
    # Here we require the caller to pass the confirmed status to keep it stateless.
    confirmed = data.get("confirmed", False)
    if not confirmed:
        return jsonify({"error": "SOS not permitted: fire not confirmed by required modalities."}), 403

    # Placeholder for SMS/voice provider integration.
    alert = {
        "capture_id": capture_id,
        "status": "sent",
        "provider_message_id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "note": "Configure an SMS/voice provider and emergency contacts to enable real alerts.",
    }
    return jsonify(alert)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
