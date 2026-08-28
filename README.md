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

The training classes are `fire` and `smoke`. The builder excludes unrelated FLIR person labels, retains valid negative images, and removes byte-identical duplicates across splits to prevent leakage. It writes split/class counts to `artifacts/forest_fire_yolo/dataset_summary.json`.

The evaluation script reports object-detection metrics (precision, recall, mAP) and binary fire-alert metrics (accuracy, precision, recall, F1, and confusion matrix) on the held-out test split.

## Video inference

For a saved video:

```powershell
python infer_video.py "original datasets/Videos/FP1.mp4"
```

For Android IP Webcam, use the stream URL shown by the app, for example:

```powershell
python infer_video.py "http://PHONE_IP:8080/video"
```

The phone and computer must be on the same network. RTSP URLs and a local webcam index such as `0` are also accepted.

The live pipeline detects fire, smoke, and people. A fire plus a person in the same frame is reported as a co-occurrence, not proof that the person started the fire.

This can report a person near a detected fire. It cannot determine whether a person intentionally started a fire because the supplied datasets have no intent labels. `Images`/`Masks` is retained for a future fire-boundary segmentation stage rather than mixing masks into bounding-box training.
