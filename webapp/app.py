"""Multimodal Forest-Fire Web Application.

A Flask server that:
  - streams a live camera feed (RGB only) with real-time fire/smoke detection,
  - accepts RGB / thermal / NIR uploads of the same scene and fuses them,
  - only allows an SOS alert when at least 2 of 3 modalities confirm fire,
  - captures a snapshot when fire is detected on the live feed and asks the
    user to confirm before dispatching the rescue team.

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
    LIVE_FIRE_THRESHOLD,
    ModalityResult,
    get_detector,
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
SNAPSHOT_DIR = BASE_DIR / "snapshots"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

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

# Live-fire confirmation state: when the live feed crosses the alert
# threshold we capture a snapshot and wait for the user to confirm.
_live_state = {
    "alert_active": False,
    "snapshot_id": None,
    "snapshot_path": None,
    "fire_score": 0.0,
    "confirmed": False,
    "last_alert_time": 0.0,
}
_live_state_lock = __import__("threading").Lock()


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


def _set_camera_source(source):
    """Switch the live camera source (webcam index, RTSP URL, or IP cam URL)."""
    global _camera, _camera_source
    with _live_state_lock:
        _camera_source = source
        if _camera is not None:
            _camera.release()
            _camera = None
        _live_state["alert_active"] = False
        _live_state["snapshot_id"] = None
        _live_state["snapshot_path"] = None
        _live_state["confirmed"] = False


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
    }


def _decision_to_dict(decision: FusedDecision) -> dict:
    return {
        "status": decision.status,
        "fused_score": round(decision.fused_score, 4),
        "modalities_checked": decision.modalities_checked,
        "modalities_confirmed": decision.modalities_confirmed,
        "sos_allowed": decision.sos_allowed,
        "message": decision.message,
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

            result = detector.detect_rgb(frame, conf=LIVE_FIRE_THRESHOLD)
            annotated = result.annotated if result.annotated is not None else frame
            status = "FIRE ALERT" if result.fire_detected else "Monitoring"
            color = (0, 0, 255) if result.fire_detected else (0, 180, 0)
            cv2.putText(annotated, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, color, 2, cv2.LINE_AA)

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

    result = detector.detect_rgb(frame, conf=LIVE_FIRE_THRESHOLD)

    # Snapshot + confirmation flow: when fire crosses the alert threshold,
    # capture a snapshot once and hold the alert until the user confirms.
    with _live_state_lock:
        if result.fire_detected and not _live_state["alert_active"]:
            now = time.time()
            if now - _live_state["last_alert_time"] > 10.0:  # debounce 10s
                snap_id = f"snap_{int(now)}_{uuid.uuid4().hex[:6]}"
                snap_path = SNAPSHOT_DIR / f"{snap_id}.jpg"
                snap_path.write_bytes(_encode_jpeg(frame))
                _live_state.update({
                    "alert_active": True,
                    "snapshot_id": snap_id,
                    "snapshot_path": str(snap_path),
                    "fire_score": result.fire_score,
                    "confirmed": False,
                    "last_alert_time": now,
                })
        elif not result.fire_detected:
            # Auto-clear after the fire leaves the frame (unless confirmed).
            if not _live_state["confirmed"]:
                _live_state["alert_active"] = False

        state = dict(_live_state)

    return jsonify({
        "available": True,
        "fire_detected": result.fire_detected,
        "fire_score": round(result.fire_score, 4),
        "detections": [{"label": d.label, "confidence": round(d.confidence, 4)} for d in result.detections],
        "alert_active": state["alert_active"],
        "snapshot_id": state["snapshot_id"],
        "snapshot_url": f"/api/snapshots/{state['snapshot_id']}.jpg" if state["snapshot_id"] else None,
        "confirmed": state["confirmed"],
    })


@app.route("/api/snapshots/<path:filename>")
def snapshot_file(filename):
    """Serve a captured snapshot image."""
    safe = Path(filename).name
    path = SNAPSHOT_DIR / safe
    if not path.exists():
        return jsonify({"error": "Snapshot not found"}), 404
    return Response(path.read_bytes(), mimetype="image/jpeg")


@app.route("/api/live/confirm", methods=["POST"])
def live_confirm():
    """User confirms the live-fire snapshot — dispatch the rescue team."""
    data = request.get_json(silent=True) or {}
    snapshot_id = data.get("snapshot_id")
    with _live_state_lock:
        if not _live_state["alert_active"] or _live_state["snapshot_id"] != snapshot_id:
            return jsonify({"error": "No active alert for this snapshot."}), 400
        _live_state["confirmed"] = True

    alert = {
        "snapshot_id": snapshot_id,
        "status": "dispatched",
        "provider_message_id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "note": "Rescue team dispatch confirmed by operator. Configure SMS/voice provider for real alerts.",
    }
    return jsonify(alert)


@app.route("/api/live/dismiss", methods=["POST"])
def live_dismiss():
    """User dismisses the live-fire alert (false alarm)."""
    with _live_state_lock:
        _live_state["alert_active"] = False
        _live_state["snapshot_id"] = None
        _live_state["snapshot_path"] = None
        _live_state["confirmed"] = False
    return jsonify({"status": "dismissed"})


@app.route("/api/camera/source", methods=["POST"])
def camera_source():
    """Switch the live camera source (webcam index, RTSP, or IP cam URL)."""
    data = request.get_json(silent=True) or {}
    source = data.get("source")
    if source is None or str(source).strip() == "":
        return jsonify({"error": "source is required"}), 400
    try:
        _set_camera_source(int(source))
    except ValueError:
        _set_camera_source(str(source).strip())
    return jsonify({"status": "ok", "source": _camera_source})


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
