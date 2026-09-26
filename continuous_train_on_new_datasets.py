#!/usr/bin/env python3
"""
Continuous Training Script for Forest Fire Detector
- Loads previously trained model weights
- Trains on new datasets in yet_to_train folder
- Displays real-time training status
- Moves datasets to trained_datasets after completion
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO
import sys
import json
from datetime import datetime


def print_status(message: str) -> None:
    """Print status message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()


def move_datasets(source_dir: Path, dest_dir: Path) -> None:
    """Move datasets from source to destination."""
    print_status(f"\n{'='*60}")
    print_status("MOVING DATASETS TO TRAINED_DATASETS")
    print_status(f"{'='*60}")
    
    if not source_dir.exists():
        print_status(f"⚠️  Source directory not found: {source_dir}")
        return
    
    if not dest_dir.exists():
        print_status(f"Creating destination directory: {dest_dir}")
        dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Items to move from yet_to_train (except data.yaml which will be handled)
    items_to_move = [
        'train',
        'test',
        'valid',
        'DFS-FIRE-SMOKE-Dataset-main',
        'FLAME 3 CV Dataset (Sycan Marsh)',
        'README.dataset.txt',
        'README.roboflow.txt'
    ]
    
    moved_items = []
    for item_name in items_to_move:
        source_item = source_dir / item_name
        if source_item.exists():
            dest_item = dest_dir / item_name
            # If destination exists, add a timestamp suffix
            if dest_item.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dest_item = dest_dir / f"{item_name}_{timestamp}"
            
            try:
                print_status(f"Moving: {item_name} → {dest_item.name}")
                if source_item.is_dir():
                    shutil.copytree(source_item, dest_item, dirs_exist_ok=True)
                    shutil.rmtree(source_item)
                else:
                    shutil.copy2(source_item, dest_item)
                    source_item.unlink()
                moved_items.append(item_name)
                print_status(f"✓ Successfully moved: {item_name}")
            except Exception as e:
                print_status(f"✗ Error moving {item_name}: {str(e)}")
    
    print_status(f"\n✓ Moved {len(moved_items)} items to trained_datasets")
    print_status(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous training on new datasets with status display."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("project_data/original_datasets/yet_to_train/data.yaml"),
        help="Path to data.yaml for yet_to_train dataset"
    )
    parser.add_argument(
        "--previous-weights",
        type=Path,
        default=Path("models/rgb_fire_smoke_best.pt"),
        help="Path to previously trained weights for continuous training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Number of training epochs (default: 60)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Device to use (0 for GPU, cpu for CPU)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of workers"
    )
    parser.add_argument(
        "--optimizer",
        default="AdamW",
        help="Optimizer: AdamW (default), SGD, Adam, etc."
    )
    parser.add_argument(
        "--no-move",
        action="store_true",
        help="Don't move datasets after training"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    data_yaml = args.data.resolve()
    weights_path = args.previous_weights.resolve()
    project_dir = Path(__file__).resolve().parent / "project_data" / "runs" / "forest_fire"
    
    # Validate paths
    print_status(f"\n{'='*60}")
    print_status("FOREST FIRE DETECTOR - CONTINUOUS TRAINING")
    print_status(f"{'='*60}\n")
    
    if not data_yaml.exists():
        raise SystemExit(f"✗ Data YAML not found: {data_yaml}")
    print_status(f"✓ Data YAML found: {data_yaml}")
    
    if not weights_path.exists():
        print_status(f"⚠️  Previous weights not found: {weights_path}")
        print_status(f"   Will start training with default YOLO11n weights")
        weights_to_load = "models/yolo11n.pt"
    else:
        print_status(f"✓ Previous weights found: {weights_path}")
        weights_to_load = str(weights_path)
    
    print_status(f"📊 Training Configuration:")
    print_status(f"   - Data: {data_yaml}")
    print_status(f"   - Weights: {weights_to_load}")
    print_status(f"   - Epochs: {args.epochs}")
    print_status(f"   - Batch Size: {args.batch}")
    print_status(f"   - Image Size: {args.imgsz}")
    print_status(f"   - Device: {args.device}")
    print_status(f"   - Optimizer: {args.optimizer}")
    print_status(f"\n{'='*60}\n")
    
    try:
        # Load model with previous weights
        print_status("Loading model...")
        model = YOLO(weights_to_load)
        
        # Start training with real-time monitoring
        print_status("Starting continuous training on new datasets...\n")
        print_status(f"{'='*60}")
        
        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(project_dir),
            name="fire_smoke",
            exist_ok=True,
            pretrained=True,
            patience=15,
            seed=42,
            optimizer=args.optimizer,
            cos_lr=True,
            lr0=0.001,
            lrf=0.01,
            warmup_epochs=3,
            warmup_bias_lr=0.1,
            weight_decay=0.0005,
            close_mosaic=10,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            verbose=True,
        )
        
        print_status(f"{'='*60}")
        print_status("✓ TRAINING COMPLETED SUCCESSFULLY!\n")
        
        # Display final metrics
        if hasattr(results, 'results_dict'):
            print_status("Final Training Metrics:")
            for key, value in results.results_dict.items():
                print_status(f"  {key}: {value}")
        
        # Move datasets if not disabled
        if not args.no_move:
            source_dir = Path("project_data/original_datasets/yet_to_train")
            dest_dir = Path("project_data/original_datasets/trained_datasets")
            move_datasets(source_dir, dest_dir)
        
        print_status(f"{'='*60}")
        print_status("All tasks completed!")
        print_status(f"Model saved at: {project_dir / 'fire_smoke'}")
        print_status(f"{'='*60}\n")
        
    except Exception as e:
        print_status(f"✗ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
