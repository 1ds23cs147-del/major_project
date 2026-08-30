"""Multimodal forest-fire and endangered-animal detection engine.

Loads the trained YOLO fire/smoke detector and provides modality-aware inference
for RGB, thermal/IR, and NIR images. A fusion service combines the per-modality
scores and only allows an SOS alert when at least 2 of the 3 modalities agree.

Design notes
------------
- The current project ships a single RGB fire/smoke model. Thermal and NIR inputs
  are preprocessed (normalised / channel-mapped) so the same detector can be used
  as a base, and their scores are calibrated with modality-specific thresholds.
- A dedicated thermal/NIR model can be dropped in later by pointing
  ``THERMAL_WEIGHTS`` / ``NIR_WEIGHTS`` at the trained checkpoints.
- Endangered-animal detection is reported as a co-occurrence signal. A dedicated
  animal model can be supplied via ``ANIMAL_WEIGHTS``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RGB_WEIGHTS = BASE_DIR / "runs" / "forest_fire" / "fire_smoke" / "weights" / "best.pt"
# Optional dedicated models (set to None to fall back to the RGB model).
THERMAL_WEIGHTS: Optional[Path] = None
NIR_WEIGHTS: Optional[Path] = None
ANIMAL_WEIGHTS: Optional[Path] = None

# Modality-specific confirmation thresholds (tune on validation data).
RGB_REVIEW_THRESHOLD = 0.40
THERMAL_CONFIRM_THRESHOLD = 0.50
NIR_CONFIRM_THRESHOLD = 0.50
SOS_THRESHOLD = 0.75

# Fusion weights (calibrate on a held-out multimodal validation set).
FUSION_WEIGHTS = {"rgb": 0.40, "thermal": 0.40, "nir": 0.20}

# Minimum number of modalities that must agree before an SOS alert is allowed.
MIN_MODALITIES_FOR_SOS = 2

# Endangered animal classes (COCO-style ids mapped to names). A dedicated model
# can replace this mapping.
ANIMAL_CLASSES = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "deer",
    25: "fox",
    26: "wolf",
    27: "rabbit",
    28: "squirrel",
    29: "raccoon",
    30: "moose",
    31: "elk",
    32: "bison",
    33: "mountain goat",
    34: "mountain lion",
    35: "wild boar",
    36: "coyote",
    37: "bobcat",
    38: "lynx",
    39: "puma",
    40: "panther",
    41: "jaguar",
    42: "leopard",
    43: "tiger",
    44: "lion",
    45: "monkey",
    46: "gorilla",
    47: "chimpanzee",
    48: "orangutan",
    49: "panda",
    50: "koala",
    51: "kangaroo",
    52: "wallaby",
    53: "platypus",
    54: "echidna",
    55: "wombat",
    56: "tasmanian devil",
    57: "quokka",
    58: "bandicoot",
    59: "bilby",
    60: "numbat",
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """A single detected object."""

    label: str
    confidence: float
    box: list  # [x1, y1, x2, y2]
    modality: str


@dataclass
class ModalityResult:
    """Result of running one modality through the detector."""

    modality: str
    fire_score: float  # max confidence of fire/smoke detections
    fire_detected: bool
    detections: list = field(default_factory=list)
    animal_detected: bool = False
    animal_labels: list = field(default_factory=list)
    annotated: Optional[np.ndarray] = None


@dataclass
class FusedDecision:
    """Final fused decision across modalities."""

    status: str  # "no_fire" | "needs_review" | "confirmed_fire"
    fused_score: float
    modalities_checked: list
    modalities_confirmed: list
    sos_allowed: bool
    message: str
    animal_alerts: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class MultimodalDetector:
    """Loads models once and provides thread-safe inference."""

    def __init__(self, device: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self.device = self._resolve_device(device)
        self.rgb_model = self._load(RGB_WEIGHTS)
        self.thermal_model = self._load(THERMAL_WEIGHTS) if THERMAL_WEIGHTS else self.rgb_model
        self.nir_model = self._load(NIR_WEIGHTS) if NIR_WEIGHTS else self.rgb_model
        self.animal_model = self._load(ANIMAL_WEIGHTS) if ANIMAL_WEIGHTS else None

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        """Use CUDA when available, otherwise fall back to CPU."""
        if device:
            return device
        try:
            import torch

            if torch.cuda.is_available():
                return "0"
        except Exception:
            pass
        return "cpu"

    @staticmethod
    def _load(path: Optional[Path]) -> Optional[YOLO]:
        if path is None or not Path(path).exists():
            return None
        return YOLO(str(path))

    # -- modality preprocessing --------------------------------------------
    @staticmethod
    def _to_rgb(image: np.ndarray) -> np.ndarray:
        """Normalise any input to a 3-channel BGR image for the detector."""
        if image is None:
            return np.zeros((640, 640, 3), dtype=np.uint8)
        if image.ndim == 2:  # single channel (grayscale / 8-bit thermal)
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:  # RGBA
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        return image

    @staticmethod
    def _normalise_thermal(image: np.ndarray) -> np.ndarray:
        """Map 16-bit thermal data to an 8-bit BGR representation."""
        if image.dtype == np.uint16:
            dst = np.zeros(image.shape, dtype=np.uint8)
            image = cv2.normalize(image, dst, 0, 255, cv2.NORM_MINMAX)
        return MultimodalDetector._to_rgb(image)

    @staticmethod
    def _normalise_nir(image: np.ndarray) -> np.ndarray:
        """Map NIR (often single-channel) data to an 8-bit BGR representation."""
        return MultimodalDetector._normalise_thermal(image)

    # -- inference ----------------------------------------------------------
    def _run_model(
        self,
        model: Optional[YOLO],
        image: np.ndarray,
        conf: float,
        modality: str,
    ) -> ModalityResult:
        if model is None:
            return ModalityResult(modality=modality, fire_score=0.0, fire_detected=False)

        with self._lock:
            results = model.predict(image, conf=conf, device=self.device, verbose=False)
        result = next(iter(results))

        detections: list[Detection] = []
        fire_score = 0.0
        fire_detected = False
        animal_detected = False
        animal_labels: list[str] = []

        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {})
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls.item())
                label = names.get(cls_id, str(cls_id))
                conf_val = float(box.conf.item())
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                detections.append(Detection(label=label, confidence=conf_val, box=xyxy, modality=modality))
                if label.lower() in {"fire", "smoke"}:
                    fire_score = max(fire_score, conf_val)
                    fire_detected = True
                if label.lower() in {v.lower() for v in ANIMAL_CLASSES.values()}:
                    animal_detected = True
                    animal_labels.append(label)

        plot = getattr(result, "plot", None)
        annotated = plot() if boxes is not None and len(boxes) > 0 and plot is not None else image.copy()
        return ModalityResult(
            modality=modality,
            fire_score=fire_score,
            fire_detected=fire_detected,
            detections=detections,
            animal_detected=animal_detected,
            animal_labels=animal_labels,
            annotated=annotated,
        )

    def detect_rgb(self, image: np.ndarray, conf: float = RGB_REVIEW_THRESHOLD) -> ModalityResult:
        return self._run_model(self.rgb_model, self._to_rgb(image), conf, "rgb")

    def detect_thermal(self, image: np.ndarray, conf: float = THERMAL_CONFIRM_THRESHOLD) -> ModalityResult:
        return self._run_model(self.thermal_model, self._normalise_thermal(image), conf, "thermal")

    def detect_nir(self, image: np.ndarray, conf: float = NIR_CONFIRM_THRESHOLD) -> ModalityResult:
        return self._run_model(self.nir_model, self._normalise_nir(image), conf, "nir")

    # -- fusion -------------------------------------------------------------
    def fuse(self, results: dict[str, ModalityResult]) -> FusedDecision:
        """Fuse per-modality results and decide whether an SOS alert is allowed."""
        checked = [m for m in ("rgb", "thermal", "nir") if m in results and results[m] is not None]
        confirmed = [m for m in checked if results[m].fire_detected]

        # Weighted fusion over available modalities.
        weights = {m: FUSION_WEIGHTS.get(m, 0.0) for m in checked}
        total_w = sum(weights.values()) or 1.0
        fused = sum(results[m].fire_score * weights[m] for m in checked) / total_w

        # SOS requires at least MIN_MODALITIES_FOR_SOS confirmed modalities.
        sos_allowed = len(confirmed) >= MIN_MODALITIES_FOR_SOS and fused >= SOS_THRESHOLD

        if sos_allowed:
            status = "confirmed_fire"
            message = (
                f"Confirmed fire: {len(confirmed)}/{len(checked)} modalities agree "
                f"(fused score {fused:.2f}). SOS alert permitted."
            )
        elif len(confirmed) >= 1:
            status = "needs_review"
            message = (
                f"Possible fire detected in {len(confirmed)} modality/ies "
                f"(fused score {fused:.2f}). Needs review before alert."
            )
        else:
            status = "no_fire"
            message = f"No fire detected across {len(checked)} modality/ies (fused score {fused:.2f})."

        animal_alerts = []
        for m in checked:
            if results[m].animal_detected:
                animal_alerts.extend(results[m].animal_labels)

        return FusedDecision(
            status=status,
            fused_score=fused,
            modalities_checked=checked,
            modalities_confirmed=confirmed,
            sos_allowed=sos_allowed,
            message=message,
            animal_alerts=list(dict.fromkeys(animal_alerts)),
        )


# Singleton for reuse across requests.
_detector: Optional[MultimodalDetector] = None
_detector_lock = threading.Lock()


def get_detector(device: Optional[str] = None) -> MultimodalDetector:
    global _detector
    with _detector_lock:
        if _detector is None:
            _detector = MultimodalDetector(device=device)
    return _detector
