# Forest Fire AI Pipeline

This project trains an RGB fire/smoke detector for forest-camera alerts. It returns a binary operational decision: **FIRE ALERT** when visible fire or smoke is detected, otherwise **no alert**. The supplied data has no separate forest-scene label, so deploy it on a forest-facing camera rather than interpreting it as a forest classifier.

## Install on Windows with the RTX 2070

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Prepare, train, and evaluate

```powershell
python prepare_forest_fire_dataset.py
python train_forest_fire_detector.py --epochs 40 --batch 8 --device 0
python evaluate_forest_fire_detector.py --device 0
```

To watch training progress in a separate PowerShell terminal:

```powershell
python monitor_training_status.py
```

Use `python monitor_training_status.py --once` for a single update. The monitor refreshes every 10 seconds and reports whether the run is still running, the completed epoch count, and the latest metrics.

The training classes are `fire` and `smoke`. The builder excludes unrelated FLIR person labels, retains valid negative images, and removes byte-identical duplicates across splits to prevent leakage. It writes split/class counts to `project_data/artifacts/forest_fire_yolo/dataset_summary.json`.

The evaluation script reports object-detection metrics (precision, recall, mAP) and binary fire-alert metrics (accuracy, precision, recall, F1, and confusion matrix) on the held-out test split.

## Video inference

For a saved video:

```powershell
python infer_video.py "project_data/original_datasets/Videos/FP1.mp4"
```

For Android IP Webcam, use the stream URL shown by the app, for example:

```powershell
python infer_video.py "http://PHONE_IP:8080/video"
```

The phone and computer must be on the same network. RTSP URLs and a local webcam index such as `0` are also accepted.

The live pipeline detects fire, smoke, and people. A fire plus a person in the same frame is reported as a co-occurrence, not proof that the person started the fire.

This can report a person near a detected fire. It cannot determine whether a person intentionally started a fire because the supplied datasets have no intent labels. `Images`/`Masks` is retained for a future fire-boundary segmentation stage rather than mixing masks into bounding-box training.

## FireWatch web application

Run the connected-camera dashboard from the repository root:

```powershell
Set-Location .\firewatch
..\.venv\Scripts\python.exe .\app.py
```

Open `http://127.0.0.1:5000`. The RGB detector loads
`models/rgb_fire_smoke_best.pt`; the thermal
fire/no-fire classifier loads `models/thermal_fire_no_fire_best.pt`.
The classifier page includes three curated RGB/thermal demo pairs. Matching NIR
images are retained with those capture sets for the multimodal demo assets.

## Repository layout

### Frontend

- `firewatch/templates/` - FireWatch pages (live wall, cameras, classifier, alerts, history)
- `firewatch/static/css/app.css` - FireWatch styling
- `firewatch/static/js/core.js` - browser-side camera/status interactions
- `webapp/templates/` - secondary multimodal web interface
- `webapp/static/` - secondary interface CSS and JavaScript

### Backend and AI

- `firewatch/app.py`, `firewatch/core/` - Flask routes, cameras, RGB/thermal inference, fusion, SQLite event store
- `webapp/app.py`, `webapp/multimodal_detector.py` - secondary Flask API and modality inference
- `train_forest_fire_detector.py`, `train_thermal_classifier.py` - model training
- `infer_video.py`, `evaluate_forest_fire_detector.py` - video inference and evaluation
- `prepare_*.py`, `audit_*.py`, `build_clean_candidate_splits.py` - dataset preparation and audit utilities
- `models/` - included trained checkpoints and YOLO starter weights required for inference

## Data and runtime files

Local data and outputs are grouped under `project_data/`:

- `project_data/datasets/` - prepared/source datasets used by the project
- `project_data/original_datasets/` - source datasets organized by dataset provider
- `project_data/artifacts/` - prepared datasets, audit outputs, and generated reports
- `project_data/runs/` - training runs, logs, and intermediate checkpoints
- `project_data/weights/` - archived/backup weights

`project_data/` is not required for live inference after the models are in
`models/`. It is required for dataset preparation, retraining, and reproducing
dataset evaluation.

All contents of `project_data/` are local-only and excluded from Git; required
model weights are stored separately in `models/` and tracked with Git LFS. The Python virtual environment, camera
snapshots, and local FireWatch SQLite database are also local-only. Obtain the
datasets separately before retraining. The matched demo captures remain in
`webapp/uploads/` because they are application demo assets.
