from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from tqdm import tqdm


CLASS_NAMES = ["fire", "smoke", "other", "person"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def valid_yolo_rows(label_path: Path, mapping: dict[int, int]) -> list[str]:
    rows = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        try:
            source_class = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError:
            continue
        if source_class not in mapping or not all(0 <= value <= 1 for value in values) or values[2] <= 0 or values[3] <= 0:
            continue
        fields[0] = str(mapping[source_class])
        rows.append(" ".join(fields))
    return rows


def copy_yolo_split(source_root: Path, split: str, mapping: dict[int, int], output_root: Path, prefix: str) -> int:
    image_dir = source_root / split / "images"
    label_dir = source_root / split / "labels"
    destination_images = output_root / split / "images"
    destination_labels = output_root / split / "labels"
    destination_images.mkdir(parents=True, exist_ok=True)
    destination_labels.mkdir(parents=True, exist_ok=True)
    copied = 0
    for image_path in tqdm(sorted(image_dir.glob("*")), desc=f"{prefix} {split}"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        output_stem = f"{prefix}_{image_path.stem}"
        shutil.copy2(image_path, destination_images / f"{output_stem}{image_path.suffix.lower()}")
        rows = valid_yolo_rows(label_dir / f"{image_path.stem}.txt", mapping)
        (destination_labels / f"{output_stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        copied += 1
    return copied


def copy_flir_person_split(flir_root: Path, source_split: str, output_split: str, output_root: Path) -> int:
    payload = json.loads((flir_root / source_split / "thermal_annotations.json").read_text(encoding="utf-8"))
    images = {item["id"]: item for item in payload["images"]}
    person_ids = {item["id"] for item in payload["categories"] if item["name"] == "person"}
    annotations = {}
    for item in payload["annotations"]:
        if item["category_id"] in person_ids:
            annotations.setdefault(item["image_id"], []).append(item["bbox"])
    source_dir = flir_root / source_split / "thermal_8_bit"
    destination_images = output_root / output_split / "images"
    destination_labels = output_root / output_split / "labels"
    destination_images.mkdir(parents=True, exist_ok=True)
    destination_labels.mkdir(parents=True, exist_ok=True)
    copied = 0
    for image_id, image_info in tqdm(images.items(), desc=f"FLIR {source_split}->{output_split}"):
        source_path = source_dir / Path(image_info["file_name"]).name
        if not source_path.exists():
            continue
        output_stem = f"flir_{source_split}_{image_id:06d}"
        shutil.copy2(source_path, destination_images / f"{output_stem}{source_path.suffix.lower()}")
        width, height = image_info["width"], image_info["height"]
        rows = []
        for x, y, box_width, box_height in annotations.get(image_id, []):
            clipped_x = max(0, min(x, width))
            clipped_y = max(0, min(y, height))
            clipped_width = max(0, min(x + box_width, width) - clipped_x)
            clipped_height = max(0, min(y + box_height, height) - clipped_y)
            if clipped_width and clipped_height:
                rows.append(f"3 {(clipped_x + clipped_width / 2) / width:.8f} {(clipped_y + clipped_height / 2) / height:.8f} {clipped_width / width:.8f} {clipped_height / height:.8f}")
        (destination_labels / f"{output_stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the unified fire, smoke, other, and person YOLO dataset.")
    parser.add_argument("--dataset-root", type=Path, default=Path("project_data/original_datasets"))
    parser.add_argument("--output", type=Path, default=Path("project_data/artifacts/unified_yolo"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    dataset_root = args.dataset_root.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "val", "test"):
        counts[f"dfire_{split}"] = copy_yolo_split(dataset_root / "data", split, {0: 1, 1: 0}, output_root, "dfire")
        source_split = "valid" if split == "val" else split
        counts[f"roboflow_{split}"] = copy_yolo_split(dataset_root, source_split, {0: 0, 1: 2, 2: 1}, output_root, "roboflow")
        flir_source_split = "video" if split == "test" else split
        counts[f"flir_{split}"] = copy_flir_person_split(dataset_root / "FLIR_ADAS_1_3", flir_source_split, split, output_root)
    yaml_root = str(output_root).replace("\\", "/")
    yaml = f"path: {yaml_root}\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 4\nnames: [fire, smoke, other, person]\n"
    (output_root / "data.yaml").write_text(yaml, encoding="utf-8")
    (output_root / "source_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_root), "classes": CLASS_NAMES, "source_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
