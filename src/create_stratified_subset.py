#!/usr/bin/env python3
"""
Create stratified subsets of the OSV-5M dataset while maintaining continent and country distribution.

This script creates:
- 150,000 image subset from train
- 10,000 image subset from test
- Maintains proportional distribution of countries/regions
"""

import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse


def get_continent_mapping():
    """Map countries to continents for better distribution tracking."""
    # This is a simplified mapping - you may want to use the region column instead
    return {
        'US': 'North America', 'CA': 'North America', 'MX': 'North America',
        'BR': 'South America', 'AR': 'South America', 'CL': 'South America', 'CO': 'South America',
        'GB': 'Europe', 'FR': 'Europe', 'DE': 'Europe', 'IT': 'Europe', 'ES': 'Europe',
        'PL': 'Europe', 'RU': 'Europe', 'UA': 'Europe', 'RO': 'Europe', 'NL': 'Europe',
        'BE': 'Europe', 'SE': 'Europe', 'AT': 'Europe', 'CH': 'Europe', 'DK': 'Europe',
        'FI': 'Europe', 'NO': 'Europe', 'IE': 'Europe', 'PT': 'Europe', 'CZ': 'Europe',
        'HU': 'Europe', 'SK': 'Europe', 'BG': 'Europe', 'HR': 'Europe', 'LT': 'Europe',
        'SI': 'Europe', 'LV': 'Europe', 'EE': 'Europe', 'LU': 'Europe', 'MT': 'Europe',
        'CN': 'Asia', 'IN': 'Asia', 'JP': 'Asia', 'KR': 'Asia', 'TH': 'Asia',
        'ID': 'Asia', 'MY': 'Asia', 'PH': 'Asia', 'VN': 'Asia', 'BD': 'Asia',
        'PK': 'Asia', 'TR': 'Asia', 'IL': 'Asia', 'SG': 'Asia', 'KZ': 'Asia',
        'ZA': 'Africa', 'EG': 'Africa', 'NG': 'Africa', 'KE': 'Africa', 'ET': 'Africa',
        'GH': 'Africa', 'TZ': 'Africa', 'UG': 'Africa', 'DZ': 'Africa', 'MA': 'Africa',
        'AU': 'Oceania', 'NZ': 'Oceania', 'PG': 'Oceania', 'FJ': 'Oceania',
    }


def stratified_sample_by_country(df, n_samples, random_state=42):
    """
    Perform stratified sampling based on country distribution.

    Args:
        df: DataFrame to sample from
        n_samples: Number of samples to draw
        random_state: Random seed for reproducibility

    Returns:
        Sampled DataFrame
    """
    # Calculate country distribution
    country_counts = df['country'].value_counts()
    total = len(df)

    # Calculate samples per country proportionally
    samples_per_country = {}
    remaining_samples = n_samples

    for country, count in country_counts.items():
        proportion = count / total
        target_samples = int(n_samples * proportion)
        # Ensure at least 1 sample from each country if possible
        target_samples = max(1, target_samples) if count > 0 else 0
        # Don't sample more than available
        target_samples = min(target_samples, count)
        samples_per_country[country] = target_samples

    # Adjust if we're over/under the target due to rounding
    assigned_samples = sum(samples_per_country.values())
    if assigned_samples != n_samples:
        # Find countries with room to adjust
        diff = n_samples - assigned_samples
        countries_sorted = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)

        if diff > 0:
            # Need more samples - add to largest countries
            for country, count in countries_sorted:
                if diff == 0:
                    break
                if samples_per_country[country] < count:
                    add = min(diff, count - samples_per_country[country])
                    samples_per_country[country] += add
                    diff -= add
        else:
            # Need fewer samples - remove from largest countries
            for country, count in countries_sorted:
                if diff == 0:
                    break
                if samples_per_country[country] > 1:
                    remove = min(abs(diff), samples_per_country[country] - 1)
                    samples_per_country[country] -= remove
                    diff += remove

    # Sample from each country
    sampled_dfs = []
    for country, n in samples_per_country.items():
        if n > 0:
            country_df = df[df['country'] == country]
            sampled = country_df.sample(n=n, random_state=random_state)
            sampled_dfs.append(sampled)

    result = pd.concat(sampled_dfs, ignore_index=True)
    return result.sample(frac=1, random_state=random_state).reset_index(drop=True)


def copy_images(df, source_dir, dest_dir):
    """
    Copy images from source to destination directory.

    Args:
        df: DataFrame containing image IDs
        source_dir: Source directory path
        dest_dir: Destination directory path
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0

    print(f"Copying images from {source_dir} to {dest_dir}...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Copying images"):
        image_id = str(row['id'])
        # Try .jpg extension
        source_file = source_path / f"{image_id}.jpg"

        if source_file.exists():
            dest_file = dest_path / f"{image_id}.jpg"
            shutil.copy2(source_file, dest_file)
            copied += 1
        else:
            missing += 1
            if missing <= 10:  # Only print first 10 missing files
                print(f"Warning: Image not found: {source_file}")

    print(f"Copied: {copied}, Missing: {missing}")
    return copied, missing


def print_distribution(df, name):
    """Print distribution statistics for a dataset."""
    print(f"\n{'='*60}")
    print(f"{name} Distribution")
    print(f"{'='*60}")
    print(f"Total samples: {len(df)}")
    print(f"\nTop 10 countries by count:")
    country_dist = df['country'].value_counts().head(10)
    for country, count in country_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {country}: {count:>8} ({percentage:>5.2f}%)")

    # Print region distribution if available
    if 'region' in df.columns:
        print(f"\nTop 10 regions by count:")
        region_dist = df['region'].value_counts().head(10)
        for region, count in region_dist.items():
            percentage = (count / len(df)) * 100
            print(f"  {region}: {count:>8} ({percentage:>5.2f}%)")


def compare_distributions(original_df, subset_df, name):
    """Compare distributions between original and subset."""
    print(f"\n{'='*60}")
    print(f"{name} - Distribution Comparison")
    print(f"{'='*60}")

    orig_country = original_df['country'].value_counts(normalize=True).head(10)
    subset_country = subset_df['country'].value_counts(normalize=True)

    print(f"\n{'Country':<15} {'Original %':<12} {'Subset %':<12} {'Difference':<12}")
    print("-" * 60)
    for country in orig_country.index:
        orig_pct = orig_country[country] * 100
        subset_pct = subset_country.get(country, 0) * 100
        diff = subset_pct - orig_pct
        print(f"{country:<15} {orig_pct:>10.2f}% {subset_pct:>10.2f}% {diff:>+10.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Create stratified subsets of OSV-5M dataset'
    )
    parser.add_argument(
        '--train-csv',
        default='osv-5m/train.csv',
        help='Path to train CSV file'
    )
    parser.add_argument(
        '--test-csv',
        default='osv-5m/test.csv',
        help='Path to test CSV file'
    )
    parser.add_argument(
        '--train-images',
        default='osv-5m/images/train_flat',
        help='Path to train images directory'
    )
    parser.add_argument(
        '--test-images',
        default='osv-5m/images/test_flat',
        help='Path to test images directory'
    )
    parser.add_argument(
        '--output-dir',
        default='osv-5m_subset',
        help='Output directory for subsets'
    )
    parser.add_argument(
        '--train-samples',
        type=int,
        default=150000,
        help='Number of train samples to extract'
    )
    parser.add_argument(
        '--test-samples',
        type=int,
        default=10000,
        help='Number of test samples to extract'
    )
    parser.add_argument(
        '--no-copy-images',
        action='store_true',
        help='Skip copying images (only create CSVs)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    print("Loading train CSV...")
    train_df = pd.read_csv(args.train_csv)
    print(f"Loaded {len(train_df)} train samples")

    print("\nLoading test CSV...")
    test_df = pd.read_csv(args.test_csv)
    print(f"Loaded {len(test_df)} test samples")

    # Print original distributions
    print_distribution(train_df, "Original Train Dataset")
    print_distribution(test_df, "Original Test Dataset")

    # Create stratified samples
    print(f"\n{'='*60}")
    print("Creating stratified samples...")
    print(f"{'='*60}")

    print(f"\nSampling {args.train_samples} from train set...")
    train_subset = stratified_sample_by_country(
        train_df,
        args.train_samples,
        random_state=args.random_seed
    )

    print(f"\nSampling {args.test_samples} from test set...")
    test_subset = stratified_sample_by_country(
        test_df,
        args.test_samples,
        random_state=args.random_seed
    )

    # Print subset distributions
    print_distribution(train_subset, "Train Subset")
    print_distribution(test_subset, "Test Subset")

    # Compare distributions
    compare_distributions(train_df, train_subset, "Train")
    compare_distributions(test_df, test_subset, "Test")

    # Create output directories
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save CSV files
    train_csv_path = output_path / "train-150k.csv"
    test_csv_path = output_path / "test-10k.csv"

    print(f"\n{'='*60}")
    print("Saving CSV files...")
    print(f"{'='*60}")
    print(f"Train CSV: {train_csv_path}")
    print(f"Test CSV: {test_csv_path}")

    train_subset.to_csv(train_csv_path, index=False)
    test_subset.to_csv(test_csv_path, index=False)

    # Copy images
    if not args.no_copy_images:
        print(f"\n{'='*60}")
        print("Copying images...")
        print(f"{'='*60}")

        train_img_dest = output_path / "train-150k"
        test_img_dest = output_path / "test-10k"

        train_copied, train_missing = copy_images(
            train_subset,
            args.train_images,
            train_img_dest
        )

        test_copied, test_missing = copy_images(
            test_subset,
            args.test_images,
            test_img_dest
        )

        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"Train: {train_copied} images copied, {train_missing} missing")
        print(f"Test: {test_copied} images copied, {test_missing} missing")
    else:
        print("\nSkipping image copy (--no-copy-images flag set)")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}")
    print(f"Train subset: {len(train_subset)} samples")
    print(f"Test subset: {len(test_subset)} samples")


if __name__ == "__main__":
    main()
