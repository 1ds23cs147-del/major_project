"""
Two-channel fire detector: RGB + thermal, combined by late fusion.

=============================================================================
PLUGGING IN YOUR TRAINED MODEL
=============================================================================
There are exactly two functions to replace. Everything else -- the fusion
logic, the live alerting, the whole UI -- keeps working unchanged.

    RgbChannel.score(image)      -> float in 0..1   "how flame-like is this?"
    ThermalChannel.score(image)  -> float in 0..1   "how hot is the hotspot?"

`image` is a BGR numpy array (what cv2.imread / VideoCapture.read gives you).

Until you swap them out, both channels run a colour/intensity heuristic so the
app is fully usable end to end. The heuristic is NOT your project's
contribution -- it is scaffolding so the UI has something to display.

See load_keras_example() at the bottom for a copy-paste starting point.
=============================================================================
"""

from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

try:
    import cv2
    import numpy as np

    CV_OK = True
except Exception:  # pragma: no cover - lets the UI boot without OpenCV
    CV_OK = False

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - keeps the UI usable without the model package
    YOLO = None


MODEL_ROOT = Path(__file__).resolve().parents[2] / "runs" / "forest_fire"
MODEL_CANDIDATES = (
    MODEL_ROOT / "fasdd_real_smoke_30ep" / "weights" / "best.pt",
    MODEL_ROOT / "fire_smoke" / "weights" / "best.pt",
)
MODEL_PATH = next((path for path in MODEL_CANDIDATES if path.is_file()), MODEL_CANDIDATES[-1])
THERMAL_MODEL_PATH = Path(__file__).resolve().parents[2] / "runs" / "forest_fire" / "thermal_fire" / "weights" / "best.pt"


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------
# These four are the whole point of the project: a flame-coloured object with
# no heat behind it is NOT a fire, and a heat signature with no visible flame
# is still worth someone's attention.

FIRE = "fire"
HEAT_NO_FLAME = "heat_no_flame"
FLAME_NO_HEAT = "flame_no_heat"
NEEDS_REVIEW = "needs_review"
CLEAR = "clear"
RGB_ONLY = "rgb_only"

VERDICT_META = {
    FIRE:           {"label": "Fire confirmed",      "severity": 4, "act": True},
    HEAT_NO_FLAME:  {"label": "Heat, no flame",      "severity": 3, "act": True},
    NEEDS_REVIEW:   {"label": "Needs review",        "severity": 2, "act": True},
    RGB_ONLY:       {"label": "Unverified by heat",   "severity": 2, "act": True},
    FLAME_NO_HEAT:  {"label": "Likely false alarm",  "severity": 1, "act": False},
    CLEAR:          {"label": "No fire",             "severity": 0, "act": False},
}


@dataclass
class FusionConfig:
    """Tunable during the demo from the Settings drawer."""

    rgb_trigger: float = 0.55       # RGB score that counts as "flame seen"
    thermal_trigger: float = 0.60   # thermal score that counts as "hotspot"
    thermal_floor: float = 0.35     # below this, thermal vetoes an RGB claim
    w_rgb: float = 0.45             # thermal is weighted higher on purpose:
    w_thermal: float = 0.55         # it is physical evidence, not appearance
    alert_threshold: float = 0.62   # fused score needed to raise an alert
    consecutive: int = 4            # live: frames of agreement before alerting
    window: int = 6                 # live: out of this many recent frames
    cooldown_s: int = 45            # live: min gap between alerts per camera

    def to_dict(self):
        return asdict(self)

    def update(self, data: dict):
        for k, v in (data or {}).items():
            if hasattr(self, k) and v is not None:
                cur = getattr(self, k)
                try:
                    setattr(self, k, type(cur)(v))
                except (TypeError, ValueError):
                    pass
        # A camera can never gather more agreeing frames than the window holds,
        # so asking for more would silently switch live alerting off entirely.
        self.window = max(1, self.window)
        self.consecutive = max(1, min(self.consecutive, self.window))
        self.cooldown_s = max(0, self.cooldown_s)
        return self


# --------------------------------------------------------------------------
# Channel 1 -- RGB
# --------------------------------------------------------------------------
class RgbChannel:
    """Does this frame contain something that looks like flame?"""

    name = "rgb"
    backend = "heuristic"
    model = None

    def __init__(self, model_path: Path = MODEL_PATH):
        self._lock = threading.Lock()
        if YOLO is not None and model_path.is_file():
            try:
                self.model = YOLO(str(model_path))
                self.backend = f"YOLO11n ({model_path.name})"
            except Exception:
                self.model = None

    def score(self, image) -> float:
        if self.model is not None and image is not None:
            with self._lock:
                result = self.model.predict(image, conf=0.01, verbose=False)[0]
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None)
            scores = []
            if boxes is not None:
                for box in boxes:
                    label = str(names.get(int(box.cls.item()), "")).lower()
                    if label in {"fire", "smoke"}:
                        scores.append(float(box.conf.item()))
            return round(max(scores, default=0.0), 4)

        if not CV_OK or image is None:
            return 0.0

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Fire hues sit at both ends of the hue wheel (red wraps around).
        lower = cv2.inRange(hsv, np.array([0, 90, 150]), np.array([32, 255, 255]))
        upper = cv2.inRange(hsv, np.array([160, 90, 150]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        area = float(mask.mean()) / 255.0  # fraction of frame that is fire-coloured
        if area < 0.0015:
            return 0.0

        # A real flame is bright and saturated in its core, not just orange-ish.
        v = hsv[:, :, 2][mask > 0]
        brightness = float(v.mean()) / 255.0 if v.size else 0.0

        # Largest connected blob -- scattered orange pixels are usually texture.
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        biggest = 0.0
        if n > 1:
            biggest = float(stats[1:, cv2.CC_STAT_AREA].max()) / mask.size

        raw = 0.45 * min(area * 14.0, 1.0) + 0.30 * brightness + 0.25 * min(biggest * 20.0, 1.0)
        return round(_clamp(raw), 4)


# --------------------------------------------------------------------------
# Channel 2 -- Thermal
# --------------------------------------------------------------------------
class ThermalChannel:
    """Is there a hotspot? Works on greyscale or pseudo-coloured thermal."""

    name = "thermal"
    backend = "heuristic"
    model = None

    def __init__(self, model_path: Path = THERMAL_MODEL_PATH):
        self._lock = threading.Lock()
        if YOLO is not None and model_path.is_file():
            try:
                self.model = YOLO(str(model_path))
                self.backend = f"YOLO11n-cls ({model_path.name})"
            except Exception:
                self.model = None

    def score(self, image) -> float:
        if self.model is not None and image is not None:
            with self._lock:
                result = self.model.predict(image, imgsz=224, verbose=False)[0]
            names = getattr(result, "names", {})
            probs = getattr(result, "probs", None)
            if probs is not None:
                fire_index = next((index for index, name in names.items()
                                   if str(name).lower() == "fire"), None)
                if fire_index is not None:
                    return round(float(probs.data[fire_index].item()), 4)

        # >>> REPLACE FROM HERE ------------------------------------------
        # if self.model is not None:
        #     x = cv2.resize(image, (224, 224)) / 255.0
        #     return float(self.model.predict(x[None, ...], verbose=0)[0][0])
        # >>> TO HERE ----------------------------------------------------
        if not CV_OK or image is None:
            return 0.0

        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        grey = cv2.GaussianBlur(grey, (5, 5), 0)

        # In every common thermal palette -- ironbow, rainbow, white-hot --
        # the hottest pixels end up brightest once flattened to greyscale.
        hot = (grey > 205).astype(np.uint8) * 255
        frac = float(hot.mean()) / 255.0
        if frac < 0.0008:
            return 0.0

        peak = float(grey.max()) / 255.0
        # Contrast against the scene matters more than absolute brightness:
        # a uniformly bright image is an exposure problem, not a fire.
        contrast = _clamp((float(grey[hot > 0].mean()) - float(grey.mean())) / 90.0)

        n, _, stats, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)
        biggest = 0.0
        if n > 1:
            biggest = float(stats[1:, cv2.CC_STAT_AREA].max()) / hot.size

        raw = 0.35 * min(frac * 22.0, 1.0) + 0.25 * peak + 0.25 * contrast + 0.15 * min(biggest * 25.0, 1.0)
        return round(_clamp(raw), 4)


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))


# --------------------------------------------------------------------------
# Fusion -- where the two channels actually decide together
# --------------------------------------------------------------------------
class FireDetector:
    def __init__(self, cfg: Optional[FusionConfig] = None):
        self.cfg = cfg or FusionConfig()
        self.rgb = RgbChannel()
        self.thermal = ThermalChannel()

    # -- single instance (the upload screen) -------------------------------
    def classify_pair(self, rgb_img=None, thermal_img=None) -> dict:
        r = self.rgb.score(rgb_img) if rgb_img is not None else None
        t = self.thermal.score(thermal_img) if thermal_img is not None else None
        return self.fuse(r, t)

    def fuse(self, r: Optional[float], t: Optional[float]) -> dict:
        c = self.cfg
        steps = []

        if r is None and t is None:
            return _result(CLEAR, 0.0, None, None, "Nothing to analyse.", steps, None)

        # ---- RGB only: no heat evidence, so nothing can be confirmed ----
        if t is None:
            hit = r >= c.rgb_trigger
            steps.append(_step("Flame-like appearance", r, c.rgb_trigger, hit))
            steps.append(_step("Thermal channel", None, None, False,
                               "No thermal image for this instance"))
            if hit:
                return _result(
                    RGB_ONLY, round(r * 0.8, 4), r, None,
                    "Flame-like appearance, but with no thermal image this cannot be "
                    "confirmed. Sent for human review.", steps, "unknown_flame")
            return _result(CLEAR, round(r * 0.8, 4), r, None,
                           "Nothing flame-like in the visible image.", steps, "unknown_noflame")

        # ---- Thermal only ------------------------------------------------
        if r is None:
            hit = t >= c.thermal_trigger
            steps.append(_step("Visible channel", None, None, False,
                               "No RGB image for this instance"))
            steps.append(_step("Hotspot present", t, c.thermal_trigger, hit))
            if hit:
                return _result(HEAT_NO_FLAME, round(t * 0.85, 4), None, t,
                               "Strong hotspot with no visible image to corroborate it.",
                               steps, "hot_unknown")
            return _result(CLEAR, round(t * 0.85, 4), None, t,
                           "No significant hotspot.", steps, "cold_unknown")

        # ---- Both channels present: the real case ------------------------
        fused = c.w_rgb * r + c.w_thermal * t
        flame = r >= c.rgb_trigger
        hot = t >= c.thermal_trigger
        cold = t < c.thermal_floor
        reason = None

        steps.append(_step("Flame-like appearance", r, c.rgb_trigger, flame))
        steps.append(_step("Hotspot present", t, c.thermal_trigger, hot))

        if flame and hot:
            fused = _clamp(fused * 1.08)  # channels agree -> slight boost
            steps.append(_step("Channels agree", fused, c.alert_threshold, True,
                               "Visible flame backed by a heat signature"))
            cell = "hot_flame"
            verdict = FIRE if fused >= c.alert_threshold else NEEDS_REVIEW
            if verdict == NEEDS_REVIEW:
                reason = ("Both channels are firing, but the fused score is still "
                          "short of the alert threshold. Worth a person's eyes.")

        elif flame and cold:
            # The case this project exists for. A red shirt, a sunset through a
            # window, a bright poster -- fire-coloured, but physically cold.
            fused = min(fused, 0.45)
            steps.append(_step("Thermal veto", t, c.thermal_floor, False,
                               "Flame-coloured but cold -- no fire can be this cold"))
            verdict, cell = FLAME_NO_HEAT, "cold_flame"

        elif flame and not hot:
            steps.append(_step("Heat inconclusive", t, c.thermal_trigger, False,
                               "Warm, but under the hotspot threshold"))
            verdict, cell = NEEDS_REVIEW, "warm_flame"

        elif hot and not flame:
            # Smouldering, or fire hidden behind something. Still actionable.
            steps.append(_step("No visible flame", r, c.rgb_trigger, False,
                               "Heat without flame -- possible smouldering or "
                               "an obscured source"))
            verdict, cell = HEAT_NO_FLAME, "hot_noflame"

        else:
            verdict, cell = CLEAR, "cold_noflame"

        return _result(verdict, round(fused, 4), r, t,
                       reason or _reason(verdict), steps, cell)

    # -- live stream (per camera, with temporal agreement) -----------------
    def make_tracker(self):
        return CameraTracker(self)


def _step(test, value, threshold, passed, note=None):
    return {"test": test, "value": value, "threshold": threshold,
            "passed": bool(passed), "note": note}


def _result(verdict, fused, r, t, reason, steps, cell):
    meta = VERDICT_META[verdict]
    return {
        "verdict": verdict,
        "label": meta["label"],
        "severity": meta["severity"],
        "actionable": meta["act"],
        "fused": fused,
        "rgb": r,
        "thermal": t,
        "reason": reason,
        "steps": steps,
        "matrix_cell": cell,
    }


def _reason(v):
    return {
        FIRE: "Both channels agree: visible flame with a matching heat signature.",
        HEAT_NO_FLAME: "A hotspot with no visible flame. Could be smouldering, or a "
                       "source hidden behind an obstruction.",
        FLAME_NO_HEAT: "Flame-coloured, but cold. Almost certainly a fire-coloured "
                       "object rather than a fire.",
        NEEDS_REVIEW: "Flame-like appearance with heat that is warm but under the "
                      "hotspot threshold. Too close to call automatically.",
        CLEAR: "No flame and no hotspot.",
        RGB_ONLY: "Visible channel only -- unconfirmed.",
    }[v]


class CameraTracker:
    """
    Stops a single flickering frame from setting off the siren: an alert needs
    `consecutive` convincing frames out of the last `window`, and the frame in
    hand must be one of them, then the camera goes quiet for `cooldown_s`.

    "Convincing" is stricter than "actionable": the frame has to be actionable
    *and* score at least 80% of `alert_threshold`, so a stream of barely-warm
    Needs-review frames never accumulates into a full alert.
    """

    def __init__(self, detector: FireDetector):
        self.det = detector
        self.recent = deque(maxlen=detector.cfg.window)
        self.last_alert = 0.0
        self.last: dict = _result(CLEAR, 0.0, None, None, "Waiting for frames.", [], None)

    def push(self, rgb_img=None, thermal_img=None) -> Optional[dict]:
        """Returns the result dict if this frame should raise an alert."""
        # Thresholds are editable at runtime, so resize the window if it changed.
        if self.recent.maxlen != self.det.cfg.window:
            self.recent = deque(self.recent, maxlen=self.det.cfg.window)

        res = self.det.classify_pair(rgb_img, thermal_img)
        self.last = res
        hit = res["actionable"] and res["fused"] >= self.det.cfg.alert_threshold * 0.8
        self.recent.append(1 if hit else 0)

        # The alert carries this frame's images and scores, so a quiet frame must
        # never be the one that raises it, however much agreement is banked.
        if not hit:
            return None
        if sum(self.recent) < self.det.cfg.consecutive:
            return None
        if time.time() - self.last_alert < self.det.cfg.cooldown_s:
            return None

        self.last_alert = time.time()
        self.recent.clear()
        return res

    @property
    def risk(self) -> float:
        return float(self.last.get("fused") or 0.0)


# --------------------------------------------------------------------------
# Copy-paste starting point for your real model
# --------------------------------------------------------------------------
def load_keras_example(detector: FireDetector,
                       rgb_path="models/rgb_fire.h5",
                       thermal_path="models/thermal_fire.h5"):
    """
    Call this once at startup in app.py:

        from core.detector import load_keras_example
        load_keras_example(detector)

    Assumes each model outputs a single sigmoid unit where 1 == fire. If yours
    is 2-class softmax, change the trailing `[0][0]` to `[0][1]` in both
    rgb_score and thermal_score below.
    """
    from tensorflow import keras  # noqa: local import, optional dependency

    detector.rgb.model = keras.models.load_model(rgb_path)
    detector.rgb.backend = "keras"
    detector.thermal.model = keras.models.load_model(thermal_path)
    detector.thermal.backend = "keras"

    def rgb_score(image, _m=detector.rgb.model):
        x = cv2.resize(image, (224, 224))[:, :, ::-1].astype("float32") / 255.0
        return _clamp(_m.predict(x[None, ...], verbose=0)[0][0])

    def thermal_score(image, _m=detector.thermal.model):
        x = cv2.resize(image, (224, 224)).astype("float32") / 255.0
        return _clamp(_m.predict(x[None, ...], verbose=0)[0][0])

    detector.rgb.score = rgb_score
    detector.thermal.score = thermal_score
    return detector
