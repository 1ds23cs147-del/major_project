# Forest Fire AI Pipeline

This project prepares a unified detector from the available fire/smoke YOLO datasets and FLIR person annotations, then supports RGB, thermal, and video inference.

## Install on Windows with the RTX 2070

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Prepare and train

```powershell
python prepare_multimodal_dataset.py
python train_detector.py --epochs 1 --batch 2 --device 0
python train_detector.py --epochs 80 --batch 8 --device 0
```

The unified classes are `fire`, `smoke`, `other`, and `person`. The preparation script combines both YOLO sources and converts FLIR thermal COCO person annotations to YOLO format. It writes source counts to `artifacts/unified_yolo/source_counts.json`.

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
