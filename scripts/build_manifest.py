#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from explainai_thesis.manifest import build_png_mask_manifest, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a simple image/mask classification manifest.")
    parser.add_argument("dataset_root", help="Downloaded dataset root to scan.")
    parser.add_argument("--output", default="data/manifest.csv", help="CSV manifest output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    rows = build_png_mask_manifest(dataset_root)
    write_manifest(rows, Path(args.output))

    labels = Counter(int(row["label"]) for row in rows)
    with_masks = sum(1 for row in rows if row["mask_path"])
    print(f"Manifest written to {args.output}")
    print(f"Rows: {len(rows)}")
    print(f"Rows with masks: {with_masks}")
    print(f"Labels: {dict(labels)}")


if __name__ == "__main__":
    main()

