from __future__ import annotations

import shutil
import stat
import uuid
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "datasets" / "rgb_thermal"


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for attempt in range(3):
        try:
            for child in path.iterdir():
                if child.is_dir() and not child.is_symlink():
                    _remove_tree(child)
                else:
                    try:
                        child.chmod(child.stat().st_mode | stat.S_IWRITE)
                    except OSError:
                        pass
                    try:
                        child.unlink()
                    except OSError:
                        pass
            path.rmdir()
            return
        except OSError:
            if attempt == 2:
                raise
            shutil.rmtree(path, ignore_errors=True)


def _image_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )


def _first_existing(base: Path, *candidates: str) -> Path:
    for candidate in candidates:
        path = base / candidate
        if path.exists():
            return path
    return base


def _fresh_build_root(final_root: Path) -> Path:
    build_root = final_root.with_name(f"{final_root.name}.building-{uuid.uuid4().hex[:8]}")
    _remove_tree(build_root)
    return build_root


def _publish(build_root: Path, final_root: Path) -> Path:
    _remove_tree(final_root)
    try:
        build_root.rename(final_root)
        return final_root
    except PermissionError:
        return build_root


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def prepare_fasdd_detect_dataset() -> Path:
    src_root = DATA_ROOT / "FASDD_CV" / "FASDD_CV"
    src_images = src_root / "images"
    src_labels = src_root / "annotations" / "YOLO_CV" / "labels"
    final_root = DATA_ROOT / "fasdd_yolo_real"
    out_root = _fresh_build_root(final_root)

    split_files = {}
    for split_name in ("train", "val", "test"):
        split_file = src_root / "annotations" / "YOLO_CV" / f"{split_name}.txt"
        if split_file.exists():
            split_files[split_name] = [
                line.strip().replace("./", "") for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()
            ]

    for split_name, rel_paths in split_files.items():
        dst_images = out_root / split_name / "images"
        dst_labels = out_root / split_name / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)
        for index, rel in enumerate(rel_paths, start=1):
            source_image = src_images / Path(rel).name
            if not source_image.exists():
                continue
            target_image = dst_images / source_image.name
            _link_or_copy(source_image, target_image)
            label_path = src_labels / (source_image.stem + ".txt")
            if label_path.exists():
                _link_or_copy(label_path, dst_labels / label_path.name)
            else:
                (dst_labels / (source_image.stem + ".txt")).write_text("", encoding="utf-8")
            if index % 5000 == 0:
                print(f"FASDD {split_name}: {index}/{len(rel_paths)}", flush=True)

    path_text = str(out_root).replace("\\", "/")
    yaml = (
        f"path: {path_text}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "nc: 2\n"
        "names: ['fire', 'smoke']\n"
    )
    (out_root / "data.yaml").write_text(yaml, encoding="utf-8")
    published_root = _publish(out_root, final_root)
    yaml = yaml.replace(str(out_root).replace("\\", "/"), str(published_root).replace("\\", "/"))
    (published_root / "data.yaml").write_text(yaml, encoding="utf-8")
    return published_root


def prepare_flame3_classification_dataset() -> dict[str, Path]:
    source_root = DATA_ROOT / "FLAME3_CVSubset" / "FLAME 3 CV Dataset (Sycan Marsh)"
    final_root = DATA_ROOT / "flame3_cls_real"
    output_root = _fresh_build_root(final_root)

    prepared: dict[str, Path] = {}
    for modality in ("RGB", "Thermal"):
        modality_root = output_root / modality.lower()
        for class_name, class_label in (("fire", "Fire"), ("no_fire", "No Fire")):
            src_class_dir = source_root / class_label / modality
            candidates = ("Corrected FOV", "Raw") if modality == "RGB" else ("Raw JPG", "Celsius TIFF")
            directory = _first_existing(src_class_dir, *candidates)
            files = _image_files(directory)
            if not files:
                continue
            train_dir = modality_root / "train" / class_name
            val_dir = modality_root / "val" / class_name
            train_dir.mkdir(parents=True, exist_ok=True)
            val_dir.mkdir(parents=True, exist_ok=True)
            split = max(1, int(len(files) * 0.8))
            for file in files[:split]:
                _link_or_copy(file, train_dir / file.name)
            for file in files[split:]:
                _link_or_copy(file, val_dir / file.name)
            prepared[f"{modality.lower()}_{class_name}"] = modality_root

        path_text = str(modality_root).replace("\\", "/")
        yaml = (
            f"path: {path_text}\n"
            "train: train\n"
            "val: val\n"
            "nc: 2\n"
            "names: ['fire', 'no_fire']\n"
        )
        (modality_root / "data.yaml").write_text(yaml, encoding="utf-8")
    published_root = _publish(output_root, final_root)
    for modality in ("rgb", "thermal"):
        yaml_path = published_root / modality / "data.yaml"
        yaml_path.write_text(
            yaml_path.read_text(encoding="utf-8").replace(
                str(output_root).replace("\\", "/"), str(published_root).replace("\\", "/")
            ),
            encoding="utf-8",
        )
    return {key: published_root / path.relative_to(output_root) for key, path in prepared.items()}


if __name__ == "__main__":
    fasdd = prepare_fasdd_detect_dataset()
    flame = prepare_flame3_classification_dataset()
    print({"fasdd_yolo": str(fasdd), "flame3_cls": {k: str(v) for k, v in flame.items()}})
