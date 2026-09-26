from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


source = Path("project_data/artifacts/leakage_audit/rgb_exact_group_audit.csv")
destination = Path("project_data/artifacts/leakage_audit/rgb_exact_cross_split_validation_quarantine.csv")

with source.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
for row in rows:
    by_hash[row["sha256"]].append(row)

cross_split_groups = [items for items in by_hash.values() if len({item["split"] for item in items}) > 1]
quarantine = [item for items in cross_split_groups for item in items if item["split"] == "val"]

with destination.open("w", newline="", encoding="utf-8") as handle:
    fieldnames = list(quarantine[0]) if quarantine else ["path"]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(quarantine)

print({
    "records": len(rows),
    "cross_hash_groups": len(cross_split_groups),
    "validation_quarantine": len(quarantine),
    "report": str(destination),
})