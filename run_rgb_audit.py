from pathlib import Path

from audit_dataset_leakage import audit_modality


audit_modality(
    "RGB",
    {
        "train": Path("project_data/datasets/rgb_thermal/fasdd_yolo_real/train/images"),
        "val": Path("project_data/datasets/rgb_thermal/fasdd_yolo_real/val/images"),
    },
    Path("project_data/artifacts/leakage_audit/rgb_duplicate_and_leakage_report.xlsx"),
    4,
)