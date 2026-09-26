"""
FireWatch -- multi-camera fire detection console with human-confirmed dispatch.

    pip install -r requirements.txt
    python app.py
    open http://127.0.0.1:5000

Detection is deliberately advisory: the model raises an alert, a person decides
whether a response team goes out. Nothing is dispatched automatically.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from core.cameras import CameraManager, resolve_source, CV_OK
from core.detector import FireDetector, FusionConfig, VERDICT_META
from core.store import Store

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover
    cv2 = None
    np = None

BASE = Path(__file__).parent
SNAPS = BASE / "static" / "snapshots"
SNAPS.mkdir(parents=True, exist_ok=True)
DEMO_DIR = BASE.parent / "webapp" / "uploads"
DEMO_PAIRS = (
    {"id": "cap_1788092699954_62u3d0", "label": "Demo 01", "rgb": "cap_1788092699954_62u3d0_rgb.jpg", "thermal": "cap_1788092699954_62u3d0_thermal.jpg"},
    {"id": "cap_1788093931850_3jinv5", "label": "Demo 02", "rgb": "cap_1788093931850_3jinv5_rgb.jpg", "thermal": "cap_1788093931850_3jinv5_thermal.jpg"},
    {"id": "test_cap_1", "label": "Demo 03", "rgb": "test_cap_1_rgb.jpg", "thermal": "test_cap_1_thermal.jpg"},
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB per upload

store = Store(str(BASE / "firewatch.db"))
detector = FireDetector(FusionConfig().update(store.get_setting("fusion", {}) or {}))

# --- swap in your trained model here -------------------------------------
# from core.detector import load_keras_example
# load_keras_example(detector)
# -------------------------------------------------------------------------

trackers: dict = {}
ANALYSE_EVERY = 0.4  # seconds between detector runs per camera (not every frame)
_last_run: dict = {}


def on_frame(worker):
    """Called by each camera thread. Throttled -- the model is the slow part."""
    cid = worker.spec.id
    if worker.spec.modality == "thermal":
        return  # thermal cams are analysed through their RGB partner
    now = time.time()
    if now - _last_run.get(cid, 0) < ANALYSE_EVERY:
        return
    _last_run[cid] = now

    tracker = trackers.setdefault(cid, detector.make_tracker())
    rgb = worker.snapshot()
    thermal = cameras.partner_frame(cid)

    alert = tracker.push(rgb, thermal)
    worker.risk = tracker.risk
    if not alert:
        return

    rgb_path = _save_snapshot(rgb, cid, "rgb")
    thermal_path = _save_snapshot(thermal, cid, "thermal") if thermal is not None else None
    store.add_event(alert, camera_id=cid, camera_name=worker.spec.name,
                    location=worker.spec.location, source="live",
                    rgb_image=rgb_path, thermal_image=thermal_path)


def on_camera_issue(worker, message):
    """Queue one named notification for a camera outage until it recovers."""
    store.add_event(
        {
            "verdict": "camera_offline",
            "label": "Camera not working",
            "severity": 2,
            "rgb": None,
            "thermal": None,
            "fused": 0.0,
            "reason": f"{worker.spec.name}: {message}",
            "steps": [],
            "matrix_cell": None,
        },
        camera_id=worker.spec.id,
        camera_name=worker.spec.name,
        location=worker.spec.location,
        source="camera_health",
    )


cameras = CameraManager(store, on_frame=on_frame, on_issue=on_camera_issue)
cameras.boot()


def _save_snapshot(frame, cid, kind) -> str | None:
    if frame is None or cv2 is None:
        return None
    name = f"{int(time.time())}_{cid}_{kind}.jpg"
    cv2.imwrite(str(SNAPS / name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return f"snapshots/{name}"


def _decode(file_storage):
    if not file_storage or not file_storage.filename or cv2 is None:
        return None
    data = np.frombuffer(file_storage.read(), np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# =========================================================================
# Pages
# =========================================================================
NAV = [
    {"slug": "live", "label": "Live wall", "url": "/"},
    {"slug": "classifier", "label": "Classifier", "url": "/classifier"},
    {"slug": "alerts", "label": "Alert wall", "url": "/alerts"},
]


def page(template, slug, **kw):
    return render_template(template, nav=NAV, active=slug, cv_ok=CV_OK,
                           cfg=detector.cfg.to_dict(), **kw)


@app.get("/")
def live_page():
    return page("live.html", "live", cameras=cameras.list())


@app.get("/classifier")
def classifier_page():
    return page("classifier.html", "classifier")


@app.get("/alerts")
def alerts_page():
    return page("review.html", "alerts", events=store.list_events("pending"))


@app.get("/history")
def history_page():
    # `counts` already arrives via the context processor.
    return page("history.html", "history", events=store.list_events(limit=200),
                stats=store.stats())


@app.get("/cameras")
def cameras_page():
    return page("cameras.html", "cameras", cameras=cameras.list())


# =========================================================================
# Video
# =========================================================================
BOUNDARY = b"--frame\r\n"
PLACEHOLDER_WAIT = 0.05


@app.get("/stream/<cam_id>")
def stream(cam_id):
    worker = cameras.get(cam_id)
    if not worker:
        return "Unknown camera", 404

    def gen():
        blank_since = time.time()
        while True:
            jpg = worker.snapshot_jpeg()
            if jpg:
                blank_since = time.time()
                yield BOUNDARY + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            elif time.time() - blank_since > 20:
                break  # nothing for 20s -- let the browser retry cleanly
            time.sleep(PLACEHOLDER_WAIT)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/snapshot/<cam_id>.jpg")
def snapshot(cam_id):
    worker = cameras.get(cam_id)
    jpg = worker.snapshot_jpeg() if worker else None
    if not jpg:
        return "No frame", 404
    return Response(jpg, mimetype="image/jpeg")


# =========================================================================
# Camera API
# =========================================================================
@app.get("/api/cameras")
def api_cameras():
    return jsonify(cameras=cameras.list())


@app.post("/api/cameras")
def api_add_camera():
    d = request.get_json(silent=True) or {}
    source = (d.get("source") or "").strip()
    name = (d.get("name") or "").strip() or f"Camera {len(cameras.workers) + 1}"
    if not source:
        return jsonify(error="Enter a camera source."), 400
    cam = cameras.add(name=name, source=source, location=(d.get("location") or "").strip(),
                      modality=d.get("modality") or "rgb",
                      pair_id=d.get("pair_id") or None)
    return jsonify(camera=cam), 201


@app.post("/api/cameras/<cam_id>/edit")
def api_update_camera(cam_id):
    """
    Rename, move, re-point or pair an existing camera. Deliberately not
    POST /api/cameras/<id>, which would sit ambiguously beside
    POST /api/cameras/test.
    """
    d = request.get_json(silent=True) or {}
    fields = {k: v for k, v in d.items()
              if k in ("name", "source", "location", "modality", "pair_id")}
    if "source" in fields and not str(fields["source"]).strip():
        return jsonify(error="Enter a camera source."), 400
    if fields.get("pair_id") == cam_id:
        return jsonify(error="A camera cannot be its own thermal pair."), 400
    cam = cameras.update(cam_id, **fields)
    if not cam:
        return jsonify(error="Unknown camera"), 404
    trackers.pop(cam_id, None)  # thresholds or channel may have changed
    return jsonify(camera=cam)


@app.delete("/api/cameras/<cam_id>")
def api_remove_camera(cam_id):
    trackers.pop(cam_id, None)
    return jsonify(removed=cameras.remove(cam_id))


@app.post("/api/cameras/<cam_id>/toggle")
def api_toggle_camera(cam_id):
    d = request.get_json(silent=True) or {}
    cam = cameras.toggle(cam_id, d.get("enabled", True))
    return (jsonify(camera=cam), 200) if cam else (jsonify(error="Unknown camera"), 404)


@app.post("/api/cameras/test")
def api_test_camera():
    d = request.get_json(silent=True) or {}
    kind, candidates = resolve_source(d.get("source") or "")
    if kind == "empty":
        return jsonify(ok=False, error="Enter a camera source."), 400

    # A device index, or a URL that already names a stream path: the caller knows
    # what they want opened, so open exactly that.
    if kind in ("device", "exact"):
        res = CameraManager.probe(candidates[0])
        res["source"] = candidates[0]
        return jsonify(res)

    # Just a host, which is what the phone apps display on screen. Try the stream
    # paths those apps are known to serve.
    tried = []
    for url in candidates:
        res = CameraManager.probe(url)
        tried.append({"url": url, "ok": res.get("ok")})
        if res.get("ok"):
            res.update(source=url, tried=tried)
            return jsonify(res)
    return jsonify(ok=False, tried=tried,
                   error="No stream found on that address. Check that the phone is "
                         "on this wifi with its server running, or paste the full "
                         "URL including the stream path.")


# =========================================================================
# Classifier API
# =========================================================================
@app.get("/api/demo-pairs")
def api_demo_pairs():
    return jsonify(pairs=[
        {**pair, "rgb_url": f"/demo-assets/{pair['rgb']}", "thermal_url": f"/demo-assets/{pair['thermal']}"}
        for pair in DEMO_PAIRS
        if (DEMO_DIR / pair["rgb"]).is_file() and (DEMO_DIR / pair["thermal"]).is_file()
    ])


@app.get("/demo-assets/<path:filename>")
def demo_asset(filename):
    return send_from_directory(DEMO_DIR, filename)


@app.post("/api/classify")
def api_classify():
    if cv2 is None:
        return jsonify(error="OpenCV is not installed on the server. "
                             "Run: pip install opencv-python"), 503

    rgb = _decode(request.files.get("rgb"))
    thermal = _decode(request.files.get("thermal"))
    if rgb is None and thermal is None:
        return jsonify(error="Add at least one image. Both channels together give "
                             "a confirmed result."), 400

    result = detector.classify_pair(rgb, thermal)
    result["channels"] = {
        "rgb": {"present": rgb is not None, "backend": detector.rgb.backend},
        "thermal": {"present": thermal is not None, "backend": detector.thermal.backend},
    }

    event = None
    if result["actionable"] and (request.form.get("queue", "1") != "0"):
        rgb_path = _save_snapshot(rgb, "upload", "rgb")
        thermal_path = _save_snapshot(thermal, "upload", "thermal")
        event = store.add_event(result, camera_id=None,
                                camera_name=request.form.get("label") or "Manual upload",
                                source="upload", rgb_image=rgb_path,
                                thermal_image=thermal_path)
    return jsonify(result=result, event=event)


@app.get("/api/matrix")
def api_matrix():
    """The 2x2 that explains why one channel is not enough."""
    return jsonify(matrix=[
        {"cell": "cold_noflame", "rgb": False, "thermal": False,
         "verdict": "clear", "title": "No fire",
         "note": "Neither channel sees anything."},
        {"cell": "cold_flame", "rgb": True, "thermal": False,
         "verdict": "flame_no_heat", "title": "Likely false alarm",
         "note": "Fire-coloured but cold: a red object, a sunset, a bright screen."},
        {"cell": "hot_noflame", "rgb": False, "thermal": True,
         "verdict": "heat_no_flame", "title": "Heat, no flame",
         "note": "Smouldering, or a source hidden behind an obstruction."},
        {"cell": "hot_flame", "rgb": True, "thermal": True,
         "verdict": "fire", "title": "Fire confirmed",
         "note": "Visible flame with a matching heat signature."},
    ])


# =========================================================================
# Events / review queue
# =========================================================================
@app.get("/api/events")
def api_events():
    return jsonify(events=store.list_events(request.args.get("status"),
                                            int(request.args.get("limit", 200))),
                   counts=store.counts())


@app.post("/api/events/<ev_id>/<decision>")
def api_decide(ev_id, decision):
    mapped = {"dispatch": "dispatched", "dismiss": "dismissed",
              "reopen": "pending"}.get(decision)
    if not mapped:
        return jsonify(error="Unknown decision."), 400
    note = (request.get_json(silent=True) or {}).get("note", "")
    ev = store.decide(ev_id, mapped, note)
    if not ev:
        return jsonify(error="Unknown alert."), 404
    return jsonify(event=ev, counts=store.counts())


@app.post("/api/test-alert")
def api_test_alert():
    """Puts a synthetic alert in the queue so you can rehearse the demo."""
    worker = next((w for w in cameras.workers.values() if w.status == "online"), None)
    frame = worker.snapshot() if worker else None
    result = detector.fuse(0.87, 0.91)
    ev = store.add_event(result,
                         camera_id=worker.spec.id if worker else None,
                         camera_name=(worker.spec.name if worker else "Test trigger"),
                         location=(worker.spec.location if worker else ""),
                         source="test",
                         rgb_image=_save_snapshot(frame, "test", "rgb"))
    return jsonify(event=ev, counts=store.counts()), 201


# =========================================================================
# State polling
# =========================================================================
# Polling rather than SSE on purpose: browsers cap ~6 connections per host and
# every live MJPEG feed already holds one open. A 2s poll leaves them free.
@app.get("/api/state")
def api_state():
    cams = cameras.list()
    counts = store.counts()
    return jsonify(
        cameras=cams,
        counts=counts,
        online=sum(1 for c in cams if c["status"] == "online"),
        total=len(cams),
        peak_risk=max([c["risk"] for c in cams], default=0.0),
        armed=any(c["status"] == "online" for c in cams),
        cv_ok=CV_OK,
        server_time=time.time(),
    )


@app.get("/api/settings")
def api_get_settings():
    return jsonify(fusion=detector.cfg.to_dict(),
                   backends={"rgb": detector.rgb.backend,
                             "thermal": detector.thermal.backend})


@app.post("/api/settings")
def api_set_settings():
    detector.cfg.update(request.get_json(silent=True) or {})
    store.set_setting("fusion", detector.cfg.to_dict())
    return jsonify(fusion=detector.cfg.to_dict())


@app.template_filter("clock")
def clock(ts):
    return time.strftime("%H:%M:%S", time.localtime(ts or 0))


@app.template_filter("stamp")
def stamp(ts):
    return time.strftime("%d %b, %H:%M:%S", time.localtime(ts or 0))


@app.context_processor
def inject():
    return {"verdict_meta": VERDICT_META, "counts": store.counts()}


if __name__ == "__main__":
    print("\n  FireWatch console  ->  http://127.0.0.1:5000")
    if not CV_OK:
        print("  OpenCV missing: feeds and detection are off until you run"
              " 'pip install opencv-python'\n")
    try:
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)),
                debug=False, threaded=True)
    finally:
        cameras.shutdown()
