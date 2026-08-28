from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path


def latest_results(root: Path) -> Path | None:
    candidates = list(root.glob("runs/**/results.csv"))
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return [{key.strip(): value.strip() for key, value in row.items() if key} for row in csv.DictReader(file)]


def get_epochs(run_dir: Path) -> int:
    args = run_dir / "args.yaml"
    if not args.exists():
        return 40
    for line in args.read_text(encoding="utf-8").splitlines():
        if line.startswith("epochs:"):
            return int(line.split(":", 1)[1].strip())
    return 40


def metric(row: dict[str, str], name: str) -> float:
    return float(row.get(name, "0") or 0)


def projection(rows: list[dict[str, str]], total_epochs: int) -> str:
    if len(rows) < 5:
        return "Prediction: waiting for 5 completed epochs; early values are too unstable for a useful forecast."
    recent = rows[-5:]
    remaining = max(0, total_epochs - int(float(recent[-1]["epoch"])))
    values = {"precision": "metrics/precision(B)", "recall": "metrics/recall(B)", "mAP@50": "metrics/mAP50(B)"}
    estimates = []
    for label, column in values.items():
        slope = (metric(recent[-1], column) - metric(recent[0], column)) / 4
        estimate = min(1.0, max(0.0, metric(recent[-1], column) + slope * remaining * 0.35))
        estimates.append(f"{label} {estimate:.1%}")
    return "Trend estimate only (not a test result): " + ", ".join(estimates)


def render(root: Path) -> None:
    results = latest_results(root)
    if results is None:
        print("Waiting for a training results.csv file...")
        return
    rows = read_rows(results)
    if not rows:
        print(f"Waiting for first completed epoch: {results}")
        return
    row = rows[-1]
    epochs = get_epochs(results.parent)
    completed = int(float(row["epoch"]))
    print("Forest-fire training status")
    print(f"Run: {results.parent.relative_to(root)}")
    print(f"Completed: {completed}/{epochs} epochs ({completed / epochs:.0%})")
    print(f"Precision: {metric(row, 'metrics/precision(B)'):.1%} | Recall: {metric(row, 'metrics/recall(B)'):.1%}")
    print(f"mAP@50: {metric(row, 'metrics/mAP50(B)'):.1%} | mAP@50-95: {metric(row, 'metrics/mAP50-95(B)'):.1%}")
    print(projection(rows, epochs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Live terminal monitor for forest-fire training.")
    parser.add_argument("--refresh", type=float, default=10, help="Refresh interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Print one status update and exit.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        render(root)
        if args.once:
            return
        print(f"\nRefreshing every {args.refresh:g} seconds. Press Ctrl+C to stop monitoring.")
        time.sleep(args.refresh)


if __name__ == "__main__":
    main()
