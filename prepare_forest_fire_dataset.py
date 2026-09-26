from __future__ import annotations

"""Build an RGB fire/smoke dataset for a forest-camera alert system."""
import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Unlabeled natural-scene / NIR image pools used as negative samples (empty
# label files).  These teach the detector to reject non-fire scenes, which
# directly cuts false alarms in the forest-camera alert system.
NEGATIVE_SOURCES = {
    "seg_pred": ("seg_pred/seg_pred", "segpred"),
    "seg_train": ("seg_train/seg_train", "segtrn"),
    "seg_test": ("seg_test/seg_test", "segtst"),
    "nirscene0": ("nirscene0/jpg1", "nir"),
}


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


def collect_negatives(root: Path, max_per_source: int, rng: random.Random) -> list[tuple[Path, str]]:
    """Return (image_path, prefix) pairs for unlabeled negative images.

    Images are sampled deterministically (seeded RNG) so rebuilds are
    reproducible.  seg_pred/seg_train/seg_test are 150x150 scene crops;
    nirscene0 is high-res NIR imagery.
    """
    candidates: list[tuple[Path, str]] = []
    for rel_dir, prefix in NEGATIVE_SOURCES.values():
        source_dir = root / rel_dir
        if not source_dir.is_dir():
            print(f"  [warn] negative source missing, skipping: {source_dir}")
            continue
        files = [p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        rng.shuffle(files)
        candidates.extend((p, prefix) for p in files[:max_per_source])
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a clean fire/smoke YOLO dataset.")
    parser.add_argument("--dataset-root", type=Path, default=Path("project_data/original_datasets"))
    parser.add_argument("--additional-root", type=Path, help="Optional second YOLO dataset root with fire/smoke labels.")
    parser.add_argument("--output", type=Path, default=Path("project_data/artifacts/forest_fire_yolo"))
    parser.add_argument("--overwrite", action="store_true", help="Replace an incomplete output directory at --output.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for negative-sample selection.")
    parser.add_argument("--neg-train", type=int, default=20000, help="Max negative images added to train (0 disables).")
    parser.add_argument("--neg-val", type=int, default=1500, help="Max negative images added to val (0 disables).")
    parser.add_argument("--neg-test", type=int, default=1500, help="Max negative images added to test (0 disables).")
    args = parser.parse_args()
    root, output = args.dataset_root.resolve(), args.output.resolve()
    categorized_root = root / "trained_datasets"
    if categorized_root.is_dir():
        root = categorized_root
    additional_root = args.additional_root.resolve() if args.additional_root else None
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"{output} already exists. If it is an incomplete previous build, rerun with --overwrite.")
        shutil.rmtree(output)

    # Test/validation are processed first: duplicate frames can never enter train.
    sources = {
        # The local Roboflow fire/smoke source uses 0=fire and 1=smoke.
        # Class 2 is accepted as smoke for older exports that included an
        # additional non-fire class before smoke.
        "test": [(root / "data", "test", {0: 1, 1: 0}, "dfire"), (root, "test", {0: 0, 2: 1}, "roboflow")],
        "val": [(root / "data", "val", {0: 1, 1: 0}, "dfire"), (root, "valid", {0: 0, 2: 1}, "roboflow")],
        "train": [(root / "data", "train", {0: 1, 1: 0}, "dfire"), (root, "train", {0: 0, 2: 1}, "roboflow")],
    }
    if additional_root:
        for split in sources:
            source_split = "valid" if split == "val" else split
            # Prefer the additional/newer export when byte-identical images
            # exist in both sources; it preserves its complete annotations.
            sources[split].insert(0, (additional_root, source_split, {0: 0, 1: 1}, "additional"))

    # The categorized source is the project's fire/smoke export. Its YAML is
    # authoritative: class 0 is fire and class 1 is smoke.
    for split in sources:
        for index, item in enumerate(sources[split]):
            if item[3] == "roboflow":
                sources[split][index] = (item[0], item[1], {0: 0, 1: 1, 2: 1}, item[3])
    seen, summary = set(), {}
    rng = random.Random(args.seed)
    negative_pool = collect_negatives(root, max(args.neg_train, args.neg_val, args.neg_test), rng)
    print(f"Negative pool: {len(negative_pool)} candidate images")
    for split, split_sources in sources.items():
        images, annotations = output / split / "images", output / split / "labels"
        images.mkdir(parents=True); annotations.mkdir(parents=True)
        counts = Counter()
        for source_root, source_split, mapping, prefix in split_sources:
            image_dir, label_dir = source_root / source_split / "images", source_root / source_split / "labels"
            if not image_dir.is_dir():
                print(f"  [warn] source split missing, skipping: {image_dir}")
                counts[f"missing_{prefix}_source"] += 1
                continue
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
        # Add unlabeled negative samples to this split (train gets the most).
        neg_limit = {"train": args.neg_train, "val": args.neg_val, "test": args.neg_test}[split]
        added = 0
        for image, prefix in negative_pool:
            if added >= neg_limit:
                break
            fingerprint = digest(image)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            name = f"neg_{prefix}_{image.stem}"
            shutil.copy2(image, images / f"{name}{image.suffix.lower()}")
            (annotations / f"{name}.txt").write_text("", encoding="utf-8")
            counts["images"] += 1
            counts["negative_images"] += 1
            added += 1
        counts["negatives_added"] = added
        summary[split] = dict(counts)
    if not any(counts.get("smoke_boxes", 0) for counts in summary.values()):
        raise SystemExit(
            "No verified smoke annotations were found. Refusing to build the fire/smoke dataset "
            "with fabricated smoke labels; add a smoke-labeled source or update its class mapping first."
        )
    (output / "data.yaml").write_text(yaml.safe_dump({"path": output.as_posix(), "train": "train/images", "val": "val/images", "test": "test/images", "nc": 2, "names": ["fire", "smoke"]}, sort_keys=False), encoding="utf-8")
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
