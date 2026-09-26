"""
Camera plumbing: one background thread per camera, newest-frame-wins.

Why a thread per camera: cv2.VideoCapture.read() blocks. If Flask read frames
inside the request handler, one stalled phone on flaky wifi would freeze every
other feed in the browser. Instead each worker loops on its own, keeps only the
most recent JPEG, and the MJPEG route just hands out whatever is current.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import cv2
    import numpy as np

    CV_OK = True
except Exception:  # pragma: no cover
    CV_OK = False

RECONNECT_BACKOFF = [1, 2, 4, 8, 15, 30]  # seconds, then holds at 30
BLANK_FRAME_LIMIT = 3


@dataclass
class CameraSpec:
    id: str
    name: str
    source: str                 # "0" for a USB/built-in cam, or an http:// URL
    location: str = ""
    modality: str = "rgb"       # "rgb" | "thermal"
    pair_id: Optional[str] = None   # thermal twin, if this camera has one
    enabled: bool = True

    def cv_source(self):
        """A bare number means a local device index, anything else is a URL."""
        s = (self.source or "").strip()
        return int(s) if s.isdigit() else s


class CameraWorker(threading.Thread):
    daemon = True

    def __init__(self, spec: CameraSpec, on_frame=None, on_issue=None):
        super().__init__(name=f"cam-{spec.id}")
        self.spec = spec
        self.on_frame = on_frame
        self.on_issue = on_issue
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._capture = None

        self.frame = None            # newest BGR frame
        self.jpeg: Optional[bytes] = None
        self.status = "connecting"   # connecting | online | offline | disabled
        self.error = ""
        self.fps = 0.0
        self.frames_seen = 0
        self.connected_at: Optional[float] = None
        self.last_frame_at = 0.0
        self.risk = 0.0
        self._blank_frames = 0
        self._issue_reported = False

    def _report_issue(self, message):
        self.error = message
        if self._issue_reported or not self.on_issue:
            return
        self._issue_reported = True
        try:
            self.on_issue(self, message)
        except Exception:
            pass

    @staticmethod
    def _is_blank(frame):
        if frame is None or getattr(frame, "size", 0) == 0:
            return True
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return float(grey.mean()) < 8.0 and float(grey.std()) < 4.0

    # ---------------------------------------------------------------- run
    def run(self):
        if not CV_OK:
            self.status = "offline"
            self.error = "OpenCV is not installed. Run: pip install opencv-python"
            return

        attempt = 0
        while not self._stop_event.is_set():
            if not self.spec.enabled:
                self.status = "disabled"
                time.sleep(0.5)
                continue

            cap = self._open()
            if cap is None or not cap.isOpened():
                self.status = "offline"
                delay = RECONNECT_BACKOFF[min(attempt, len(RECONNECT_BACKOFF) - 1)]
                attempt += 1
                self._report_issue(self.error or f"Could not open {self.spec.source}")
                if self._stop_event.wait(delay):
                    break
                continue

            attempt = 0
            self.status = "connecting"
            self.error = ""
            self.connected_at = time.time()
            tick = time.time()
            count = 0

            while not self._stop_event.is_set() and self.spec.enabled:
                ok, frame = cap.read()
                if not ok or frame is None:
                    self._report_issue("Stream dropped or no frame received")
                    break

                if self._is_blank(frame):
                    self._blank_frames += 1
                    if self._blank_frames >= BLANK_FRAME_LIMIT:
                        self.status = "offline"
                        self._report_issue("Camera is connected but sending blank frames")
                        break
                    continue

                self._blank_frames = 0
                self._issue_reported = False
                self.status = "online"

                ok_enc, buf = cv2.imencode(".jpg", frame,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                with self._lock:
                    self.frame = frame
                    if ok_enc:
                        self.jpeg = buf.tobytes()
                    self.last_frame_at = time.time()
                    self.frames_seen += 1

                count += 1
                if time.time() - tick >= 1.0:
                    self.fps = round(count / (time.time() - tick), 1)
                    tick, count = time.time(), 0

                if self.on_frame:
                    try:
                        self.on_frame(self)
                    except Exception:
                        pass  # a detector failure must never kill the feed

            cap.release()
            with self._lock:
                if self._capture is cap:
                    self._capture = None
            self.status = "offline"

    def _open(self):
        sources = [self.spec.cv_source()]
        raw = (self.spec.source or "").strip()
        if raw and not raw.isdigit():
            url = raw if "://" in raw else f"http://{raw}"
            address = url.split("://", 1)[1].rstrip("/")
            if "/" not in address:
                sources = suggest_urls(raw)

        for source in sources:
            try:
                cap = cv2.VideoCapture(source)
                # Small buffer keeps latency down; without it the feed lags behind.
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                except Exception:
                    pass
                if cap.isOpened():
                    with self._lock:
                        self._capture = cap
                    # Keep device sources serializable as strings for the API/UI;
                    # OpenCV still receives the integer device index above.
                    self.spec.source = raw if isinstance(source, int) else source
                    return cap
                cap.release()
            except Exception as exc:
                self.error = str(exc)
        self.error = f"Could not open {raw or self.spec.source}"
        return None

    # ------------------------------------------------------------- access
    def snapshot(self):
        with self._lock:
            return None if self.frame is None else self.frame.copy()

    def snapshot_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self.jpeg

    def stop(self):
        self._stop_event.set()
        with self._lock:
            cap = self._capture
        if cap is not None:
            cap.release()

    def to_dict(self) -> dict:
        d = {
            "id": self.spec.id,
            "name": self.spec.name,
            "source": self.spec.source,
            "location": self.spec.location,
            "modality": self.spec.modality,
            "pair_id": self.spec.pair_id,
            "enabled": self.spec.enabled,
            "status": self.status,
            "error": self.error,
            "fps": self.fps,
            "risk": round(self.risk, 4),
            "frames": self.frames_seen,
            "uptime": int(time.time() - self.connected_at) if self.connected_at and self.status == "online" else 0,
        }
        return d


class CameraManager:
    def __init__(self, store, on_frame=None, on_issue=None):
        self.store = store
        self.on_frame = on_frame
        self.on_issue = on_issue
        self.workers: Dict[str, CameraWorker] = {}
        self._lock = threading.Lock()

    def boot(self):
        for spec in self.store.list_cameras():
            self._spawn(spec)

    def _spawn(self, spec: CameraSpec):
        w = CameraWorker(spec, on_frame=self.on_frame, on_issue=self.on_issue)
        with self._lock:
            self.workers[spec.id] = w
        w.start()
        return w

    def add(self, name, source, location="", modality="rgb", pair_id=None) -> dict:
        spec = self.store.add_camera(name=name, source=source, location=location,
                                     modality=modality, pair_id=pair_id)
        return self._spawn(spec).to_dict()

    def remove(self, cam_id) -> bool:
        with self._lock:
            w = self.workers.pop(cam_id, None)
        if w:
            w.stop()
            if w.is_alive():
                w.join(timeout=2)
        self.store.remove_camera(cam_id)
        # Any camera that pointed at this one as its thermal twin no longer does.
        for other in self.workers.values():
            if other.spec.pair_id == cam_id:
                other.spec.pair_id = None
        return bool(w)

    def update(self, cam_id, **fields) -> Optional[dict]:
        """
        Edit a camera in place. Only a changed source needs the capture thread
        restarted; renaming or re-pairing must not interrupt the feed.
        """
        w = self.workers.get(cam_id)
        if not w:
            return None
        spec = self.store.update_camera(cam_id, **fields)
        if spec is None:
            return None

        reopen = spec.source != w.spec.source
        spec.enabled = w.spec.enabled  # enabling is the toggle's job, not this
        if reopen:
            w.stop()
            if w.is_alive():
                w.join(timeout=2)
            with self._lock:
                self.workers.pop(cam_id, None)
            new = self._spawn(spec)
        else:
            w.spec = spec
            new = w

        # Keep the in-memory pairings consistent with what the database now says.
        for other in self.workers.values():
            if other.spec.id != cam_id and other.spec.pair_id == cam_id \
                    and spec.modality != "thermal":
                other.spec.pair_id = None
        return new.to_dict()

    def toggle(self, cam_id, enabled) -> Optional[dict]:
        w = self.workers.get(cam_id)
        if not w:
            return None
        w.spec.enabled = bool(enabled)
        if enabled:
            w.status = "connecting"
        self.store.set_camera_enabled(cam_id, bool(enabled))
        return w.to_dict()

    def get(self, cam_id) -> Optional[CameraWorker]:
        return self.workers.get(cam_id)

    def list(self) -> List[dict]:
        return [w.to_dict() for w in list(self.workers.values())]

    def partner_frame(self, cam_id):
        """The paired thermal frame for a camera, if one is configured."""
        w = self.workers.get(cam_id)
        if not w or not w.spec.pair_id:
            return None
        p = self.workers.get(w.spec.pair_id)
        return p.snapshot() if p else None

    def shutdown(self):
        for w in list(self.workers.values()):
            w.stop()

    # ------------------------------------------------------------ helpers
    @staticmethod
    def probe(source: str, timeout=6.0) -> dict:
        """Try a source once, without saving it. Powers the Test button."""
        if not CV_OK:
            return {"ok": False, "error": "OpenCV is not installed on the server."}

        started = time.time()
        try:
            src = int(source) if str(source).strip().isdigit() else source
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                cap.release()
                return {"ok": False, "error": "Could not open the stream. "
                                              "Check the URL and that both devices are on the same network."}
            ok, frame = cap.read()
            elapsed = round((time.time() - started) * 1000)
            h, w = (frame.shape[:2] if ok and frame is not None else (0, 0))
            cap.release()
            if not ok or frame is None:
                return {"ok": False, "error": "Connected, but no frames arrived.",
                        "latency_ms": elapsed}
            return {"ok": True, "latency_ms": elapsed, "width": int(w), "height": int(h)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


# --------------------------------------------------------------------------
# Guessing the right URL for the Android "IP Webcam" app
# --------------------------------------------------------------------------
IP_WEBCAM_PATHS = ["/video", "/videofeed", "/shot.jpg", "/mjpegfeed?640x480"]


def suggest_urls(host: str) -> List[str]:
    """
    People type '192.168.1.7:8080' and expect it to work. Build the candidate
    stream URLs the common phone webcam apps expose.
    """
    host = (host or "").strip().rstrip("/")
    if not host:
        return []
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    if ":" not in host.split("//", 1)[1]:
        host += ":8080"
    return [host + p for p in IP_WEBCAM_PATHS]


def resolve_source(raw: str):
    """
    Work out what someone actually typed into the add form, without touching the
    network. Kept out of the route so it can be tested on its own.

    Returns (kind, candidates):
        "device"  a local camera index -- open it as given
        "exact"   a URL that already names a stream path, or any RTSP URL
        "guess"   a bare host: candidate URLs to try in order
        "empty"   nothing usable
    """
    raw = (raw or "").strip()
    if not raw:
        return "empty", []
    if raw.isdigit():
        return "device", [raw]
    url = raw if "://" in raw else "http://" + raw
    if url.startswith("rtsp://") or "/" in url.split("://", 1)[1].rstrip("/"):
        return "exact", [url]
    return "guess", suggest_urls(raw)
