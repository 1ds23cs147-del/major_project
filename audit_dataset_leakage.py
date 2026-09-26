from __future__ import annotations

import argparse
import hashlib
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from PIL import Image
import imagehash
from openpyxl import Workbook


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FRAME_PATTERN = re.compile(r"(?i)^(?P<group>.+?)[_-](?:cv|frame)[_-]?\d+$")
SOURCE_PATTERN = re.compile(r"(?i)^(?P<group>.+?)[_-](?:video|event|camera)[_-]?[a-z0-9]+[_-]?\d*$")


def list_images(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def source_group(path: Path) -> str:
    for pattern in (FRAME_PATTERN, SOURCE_PATTERN):
        match = pattern.match(path.stem)
        if match:
            return match.group("group").lower()
    stem = re.sub(r"[_-]?\d{3,}$", "", path.stem.lower())
    return stem.split("_")[0] or path.stem.lower()


def file_record(path: Path, split: str, modality: str) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    with Image.open(path) as image:
        perceptual_hash = str(imagehash.phash(image.convert("RGB")))
    return {
        "modality": modality,
        "split": split,
        "path": str(path.resolve()),
        "filename": path.name,
        "group": source_group(path),
        "sha256": digest.hexdigest(),
        "phash": perceptual_hash,
    }


def audit_split(folder: Path, split: str, modality: str) -> list[dict]:
    files = list_images(folder)
    records = []
    for index, path in enumerate(files, start=1):
        try:
            records.append(file_record(path, split, modality))
        except Exception as exc:
            records.append({
                "modality": modality,
                "split": split,
                "path": str(path.resolve()),
                "filename": path.name,
                "group": source_group(path),
                "error": str(exc),
            })
        if index % 500 == 0:
            print(f"{modality} {split}: {index}/{len(files)}")
    return records


def exact_pairs(records: list[dict], left: str, right: str | None = None) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        if "sha256" in record and record["split"] == left:
            groups[record["sha256"]].append(record)
    if right is None:
        return [
            {"image_1": first["path"], "image_2": second["path"], "sha256": digest, "type": "EXACT_WITHIN"}
            for digest, items in groups.items() if len(items) > 1
            for first, second in combinations(items, 2)
        ]
    right_groups = defaultdict(list)
    for record in records:
        if "sha256" in record and record["split"] == right:
            right_groups[record["sha256"]].append(record)
    return [
        {"image_1": first["path"], "image_2": second["path"], "sha256": digest, "type": "EXACT_CROSS_SPLIT"}
        for digest, left_items in groups.items() if digest in right_groups
        for first in left_items for second in right_groups[digest]
    ]


def near_pairs(records: list[dict], left: str, right: str | None, threshold: int) -> list[dict]:
    left_records = [record for record in records if record.get("split") == left and "phash" in record]
    right_records = [record for record in records if record.get("split") == (right or left) and "phash" in record]
    if right is None:
        right_records = left_records
    buckets = defaultdict(list)
    for index, record in enumerate(right_records):
        hash_bytes = bytes.fromhex(record["phash"])
        for position, value in enumerate(hash_bytes):
            buckets[(position, value)].append(index)
    pairs = []
    seen = set()
    for first_index, first in enumerate(left_records):
        hash_bytes = bytes.fromhex(first["phash"])
        candidate_indices = {
            candidate_index
            for position, value in enumerate(hash_bytes)
            for candidate_index in buckets[(position, value)]
        }
        for second_index in candidate_indices:
            if right is None and second_index <= first_index:
                continue
            second = right_records[second_index]
            pair_key = tuple(sorted((first["path"], second["path"])))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            distance = imagehash.hex_to_hash(first["phash"]).__sub__(imagehash.hex_to_hash(second["phash"]))
            if distance <= threshold and first["sha256"] != second["sha256"]:
                pairs.append({
                    "image_1": first["path"],
                    "image_2": second["path"],
                    "phash_distance": distance,
                    "group_1": first["group"],
                    "group_2": second["group"],
                    "type": "NEAR_WITHIN" if right is None else "NEAR_CROSS_SPLIT",
                    "review_status": "NEEDS_REVIEW",
                })
    return pairs


def group_overlaps(records: list[dict]) -> list[dict]:
    by_group = defaultdict(lambda: defaultdict(int))
    for record in records:
        by_group[record["group"]][record["split"]] += 1
    return [
        {"group": group, **counts, "cross_split": len(counts) > 1}
        for group, counts in sorted(by_group.items())
    ]


def write_sheet(workbook: Workbook, name: str, rows: list[dict]) -> None:
    sheet = workbook.create_sheet(name[:31])
    if not rows:
        sheet.append(["result"])
        sheet.append(["none"])
        return
    columns = sorted({key for row in rows for key in row})
    sheet.append(columns)
    for row in rows:
        sheet.append([row.get(column, "") for column in columns])
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions


def audit_modality(modality: str, split_dirs: dict[str, Path], output: Path, threshold: int) -> dict:
    records = []
    for split, folder in split_dirs.items():
        records.extend(audit_split(folder, split, modality))
    valid_records = [record for record in records if "sha256" in record]
    results = {
        "records": records,
        "Train_Exact": exact_pairs(valid_records, "train"),
        "Val_Exact": exact_pairs(valid_records, "val"),
        "Train_Val_Exact": exact_pairs(valid_records, "train", "val"),
        "Train_Near": near_pairs(valid_records, "train", None, threshold),
        "Val_Near": near_pairs(valid_records, "val", None, threshold),
        "Train_Val_Near": near_pairs(valid_records, "train", "val", threshold),
        "Group_Overlap": group_overlaps(valid_records),
    }
    workbook = Workbook()
    workbook.remove(workbook.active)
    for name in ("Train_Exact", "Train_Near", "Val_Exact", "Val_Near", "Train_Val_Exact", "Train_Val_Near", "Group_Overlap"):
        write_sheet(workbook, f"{modality}_{name}", results[name])
    write_sheet(workbook, f"{modality}_Errors", [record for record in records if "error" in record])
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)
    print(f"Wrote {output}")
    for name in results:
        if name != "records":
            print(f"{modality} {name}: {len(results[name])}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit exact and perceptual duplicate leakage in RGB and thermal data.")
    parser.add_argument("--rgb-yolo", type=Path, required=True)
    parser.add_argument("--thermal", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/leakage_audit"))
    parser.add_argument("--threshold", type=int, default=4)
    args = parser.parse_args()
    audit_modality("RGB", {"train": args.rgb_yolo / "train/images", "val": args.rgb_yolo / "val/images"}, args.output_dir / "rgb_duplicate_and_leakage_report.xlsx", args.threshold)
    audit_modality("Thermal", {"train": args.thermal / "train", "val": args.thermal / "val"}, args.output_dir / "thermal_duplicate_and_leakage_report.xlsx", args.threshold)


if __name__ == "__main__":
    main()