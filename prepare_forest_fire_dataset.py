from __future__ import annotations

"""Build an RGB fire/smoke dataset for a forest-camera alert system."""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def labels(path: Path, mapping: dict[int, int]) -> list[str]:
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        try:
            source, x, y, width, height = int(fields[0]), *map(float, fields[1:])
        except ValueError:
            continue
        if source in mapping and all(0 <= value <= 1 for value in (x, y, width, height)) and width > 0 and height > 0:
            result.append(f"{mapping[source]} {x:.8f} {y:.8f} {width:.8f} {height:.8f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a clean fire/smoke YOLO dataset.")
    parser.add_argument("--dataset-root", type=Path, default=Path("original datasets"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/forest_fire_yolo"))
    parser.add_argument("--overwrite", action="store_true", help="Replace an incomplete output directory at --output.")
    args = parser.parse_args()
    root, output = args.dataset_root.resolve(), args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"{output} already exists. If it is an incomplete previous build, rerun with --overwrite.")
        shutil.rmtree(output)

    # Test/validation are processed first: duplicate frames can never enter train.
    sources = {
        # Roboflow source: 0=fire, 1=other, 2=smoke.  This detector has
        # only fire/smoke classes, so `other` is deliberately discarded.
        "test": [(root / "data", "test", {0: 1, 1: 0}, "dfire"), (root, "test", {0: 0, 2: 1}, "roboflow")],
        "val": [(root / "data", "val", {0: 1, 1: 0}, "dfire"), (root, "valid", {0: 0, 2: 1}, "roboflow")],
        "train": [(root / "data", "train", {0: 1, 1: 0}, "dfire"), (root, "train", {0: 0, 2: 1}, "roboflow")],
    }
    seen, summary = set(), {}
    for split, split_sources in sources.items():
        images, annotations = output / split / "images", output / split / "labels"
        images.mkdir(parents=True); annotations.mkdir(parents=True)
        counts = Counter()
        for source_root, source_split, mapping, prefix in split_sources:
            image_dir, label_dir = source_root / source_split / "images", source_root / source_split / "labels"
            for image in tqdm(sorted(image_dir.iterdir()), desc=f"{prefix} {split}"):
                if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                fingerprint = digest(image)
                if fingerprint in seen:
                    counts["duplicates_removed"] += 1; continue
                seen.add(fingerprint)
                rows = labels(label_dir / f"{image.stem}.txt", mapping)
                name = f"{prefix}_{image.stem}"
                shutil.copy2(image, images / f"{name}{image.suffix.lower()}")
                (annotations / f"{name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
                counts["images"] += 1
                counts["positive_images" if rows else "negative_images"] += 1
                for row in rows:
                    counts["fire_boxes" if row.startswith("0 ") else "smoke_boxes"] += 1
        summary[split] = dict(counts)
    (output / "data.yaml").write_text(yaml.safe_dump({"path": output.as_posix(), "train": "train/images", "val": "val/images", "test": "test/images", "nc": 2, "names": ["fire", "smoke"]}, sort_keys=False), encoding="utf-8")
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
