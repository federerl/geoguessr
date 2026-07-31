#!/usr/bin/env python3
"""
Flatten the nested OSV-5M directory structure by moving all images
from subdirectories into a single flat directory.
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

def flatten_directory(source_dir, dest_dir):
    """
    Move all .jpg files from nested subdirectories to a flat destination directory.

    Args:
        source_dir: Directory with nested folders (e.g., osv-5m/images/train)
        dest_dir: Flat destination directory (e.g., osv-5m/images/train_flat)
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    # Create destination directory
    dest_path.mkdir(parents=True, exist_ok=True)

    print(f"Flattening {source_dir} -> {dest_dir}")

    # Find all subdirectories
    subdirs = sorted([d for d in source_path.iterdir() if d.is_dir()])
    print(f"Found {len(subdirs)} subdirectories")

    total_images = 0
    moved_images = 0

    # Count total images first
    print("Counting images...")
    for subdir in tqdm(subdirs, desc="Counting"):
        jpg_files = list(subdir.glob("*.jpg"))
        total_images += len(jpg_files)

    print(f"Total images to move: {total_images:,}")

    # Move all images
    print("Moving images...")
    with tqdm(total=total_images, desc="Moving images") as pbar:
        for subdir in subdirs:
            jpg_files = list(subdir.glob("*.jpg"))

            for img_file in jpg_files:
                dest_file = dest_path / img_file.name

                # Move the file
                shutil.move(str(img_file), str(dest_file))
                moved_images += 1
                pbar.update(1)

    print(f"\nSuccessfully moved {moved_images:,} images")

    # Remove empty subdirectories
    print("Removing empty subdirectories...")
    for subdir in subdirs:
        if subdir.is_dir() and not any(subdir.iterdir()):
            subdir.rmdir()
            print(f"  Removed {subdir.name}")

    print(f"\nFlattening complete!")
    print(f"All images are now in: {dest_dir}")


def main():
    # Resolve the base directory
    base_dir = Path("/home/carpenhm/CSSE416/Project/28").resolve()

    print("=" * 60)
    print("OSV-5M Directory Flattening Script")
    print("=" * 60)
    print(f"Base directory: {base_dir}")
    print()

    # Define source and destination directories
    train_source = base_dir / "osv-5m/images/train"
    train_dest = base_dir / "osv-5m/images/train_flat"

    test_source = base_dir / "osv-5m/images/test"
    test_dest = base_dir / "osv-5m/images/test_flat"

    # Check if source directories exist
    if not train_source.exists():
        print(f"ERROR: Train directory not found: {train_source}")
        return

    if not test_source.exists():
        print(f"ERROR: Test directory not found: {test_source}")
        return

    # Flatten train directory
    print("\n" + "=" * 60)
    print("FLATTENING TRAIN DIRECTORY")
    print("=" * 60)
    flatten_directory(train_source, train_dest)

    # Flatten test directory
    print("\n" + "=" * 60)
    print("FLATTENING TEST DIRECTORY")
    print("=" * 60)
    flatten_directory(test_source, test_dest)

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print(f"Flattened train images: {train_dest}")
    print(f"Flattened test images: {test_dest}")
    print()
    print("You can now update your notebook to use these flat directories.")


if __name__ == "__main__":
    main()
