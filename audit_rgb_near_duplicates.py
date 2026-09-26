from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import imagehash
from openpyxl import Workbook
from PIL import Image


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
THRESHOLD = 4


class BKNode:
    def __init__(self, value: int, record: dict):
        self.value = value
        self.records = [record]
        self.children: dict[int, BKNode] = {}

    def add(self, value: int, record: dict) -> None:
        distance = (self.value ^ value).bit_count()
        child = self.children.get(distance)
        if child is None:
            self.children[distance] = BKNode(value, record)
        else:
            child.add(value, record)

    def search(self, value: int, radius: int, results: list[tuple[int, dict]]) -> None:
        distance = (self.value ^ value).bit_count()
        if distance <= radius:
            results.extend((distance, record) for record in self.records)
        for candidate_distance, child in self.children.items():
            if distance - radius <= candidate_distance <= distance + radius:
                child.search(value, radius, results)


def records(folder: Path, split: str) -> list[dict]:
    output = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as image:
            phash = imagehash.phash(image.convert("RGB"))
        output.append({
            "split": split,
            "path": str(path.resolve()),
            "filename": path.name,
            "sha256": digest,
            "phash": str(phash),
        })
    return output


def write_sheet(workbook: Workbook, name: str, rows: list[dict]) -> None:
    sheet = workbook.create_sheet(name)
    columns = sorted({key for row in rows for key in row}) if rows else ["result"]
    sheet.append(columns)
    if rows:
        for row in rows:
            sheet.append([row.get(column, "") for column in columns])
    else:
        sheet.append(["none"])
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions


root = Path("datasets/rgb_thermal/fasdd_yolo_real")
train = records(root / "train/images", "train")
val = records(root / "val/images", "val")
tree = None
for record in train:
    value = int(record["phash"], 16)
    if tree is None:
        tree = BKNode(value, record)
    else:
        tree.add(value, record)

cross = []
for record in val:
    matches: list[tuple[int, dict]] = []
    tree.search(int(record["phash"], 16), THRESHOLD, matches)
    for distance, candidate in matches:
        if candidate["sha256"] != record["sha256"]:
            cross.append({
                "image_1": candidate["path"],
                "image_2": record["path"],
                "phash_distance": distance,
                "type": "NEAR_CROSS_SPLIT",
                "review_status": "NEEDS_REVIEW",
            })

workbook = Workbook()
workbook.remove(workbook.active)
write_sheet(workbook, "RGB_Train_Val_Near", cross)
write_sheet(workbook, "RGB_Train_Val_Exact", [])
output = Path("artifacts/leakage_audit/rgb_near_duplicate_review.xlsx")
output.parent.mkdir(parents=True, exist_ok=True)
workbook.save(output)
Path("artifacts/leakage_audit/rgb_near_duplicate_summary.txt").write_text(
    f"train={len(train)}\nval={len(val)}\ncross_near_candidates={len(cross)}\nthreshold={THRESHOLD}\n",
    encoding="utf-8",
)
print(f"Wrote {output} with {len(cross)} cross-split near-duplicate candidates")