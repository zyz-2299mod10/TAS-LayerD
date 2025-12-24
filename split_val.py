#!/usr/bin/env python3
"""
Split training dataset into train and validation by keeping all layers from the same sample together.
"""

import os
import shutil
from pathlib import Path
import random
from collections import defaultdict

def get_sample_groups(image_dir):
    """Group images by sample ID (train_{idx}_{id})"""
    sample_groups = defaultdict(list)
    
    for img_file in sorted(os.listdir(image_dir)):
        if not img_file.endswith('.png'):
            continue
        
        # Parse filename: train_000000_589b107395a7a863ddcc47d8_00.png
        parts = img_file.rsplit('_', 1)  # Split from right: ['train_000000_589b107395a7a863ddcc47d8', '00.png']
        sample_key = parts[0]  # 'train_000000_589b107395a7a863ddcc47d8'
        
        sample_groups[sample_key].append(img_file)
    
    return sample_groups

def split_dataset(data_root, val_ratio=0.1, seed=42):
    """
    Split training data into train and validation sets by sample.
    
    Args:
        data_root: Root directory containing train/ folder
        val_ratio: Ratio of samples to use for validation (default: 0.1 = 10%)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    data_root = Path(data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "validation"
    
    # Check if train directory exists
    if not train_dir.exists():
        raise ValueError(f"Train directory not found: {train_dir}")
    
    # Get all samples grouped by sample ID
    im_dir = train_dir / "im"
    sample_groups = get_sample_groups(im_dir)
    
    print(f"Found {len(sample_groups)} unique samples")
    print(f"Total images: {sum(len(files) for files in sample_groups.values())}")
    
    # Split samples (not individual images)
    sample_keys = sorted(sample_groups.keys())
    random.shuffle(sample_keys)
    
    n_val = int(len(sample_keys) * val_ratio)
    val_samples = set(sample_keys[:n_val])
    
    print(f"\nSplitting:")
    print(f"  Training samples: {len(sample_keys) - n_val}")
    print(f"  Validation samples: {n_val}")
    
    # Create validation directory structure
    subdirs = ["im", "gt", "composite", "layers"]
    for subdir in subdirs:
        val_subdir = val_dir / subdir
        val_subdir.mkdir(parents=True, exist_ok=True)
        print(f"Created: {val_subdir}")
    
    # Move validation samples
    moved_count = 0
    for sample_key in val_samples:
        files = sample_groups[sample_key]
        
        for img_file in files:
            # Move from im/
            src_im = train_dir / "im" / img_file
            dst_im = val_dir / "im" / img_file
            if src_im.exists():
                shutil.move(str(src_im), str(dst_im))
                moved_count += 1
            
            # Move corresponding gt/
            src_gt = train_dir / "gt" / img_file
            dst_gt = val_dir / "gt" / img_file
            if src_gt.exists():
                shutil.move(str(src_gt), str(dst_gt))
        
        if moved_count % 100 == 0:
            print(f"Moved {moved_count} image pairs...")
    
    # Handle composite/ and layers/ (one per sample, not per layer)
    composite_dir = train_dir / "composite"
    layers_dir = train_dir / "layers"
    
    if composite_dir.exists():
        for sample_key in val_samples:
            composite_file = sample_key + ".png"
            src = composite_dir / composite_file
            dst = val_dir / "composite" / composite_file
            if src.exists():
                shutil.move(str(src), str(dst))
    
    if layers_dir.exists():
        for sample_key in val_samples:
            src_folder = layers_dir / sample_key
            dst_folder = val_dir / "layers" / sample_key
            if src_folder.exists():
                shutil.move(str(src_folder), str(dst_folder))
    
    # Final verification
    train_im_count = len(list((train_dir / "im").glob("*.png")))
    val_im_count = len(list((val_dir / "im").glob("*.png")))
    train_gt_count = len(list((train_dir / "gt").glob("*.png")))
    val_gt_count = len(list((val_dir / "gt").glob("*.png")))
    
    print(f"\n✓ Successfully split dataset!")
    print(f"Verification:")
    print(f"  Train: {train_im_count} images, {train_gt_count} gt masks")
    print(f"  Val:   {val_im_count} images, {val_gt_count} gt masks")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split training data into train/val sets")
    parser.add_argument("--data-root", type=str, default="../data/picCollege2",
                        help="Root directory containing train/ folder")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Ratio of data for validation (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    split_dataset(args.data_root, args.val_ratio, args.seed)
