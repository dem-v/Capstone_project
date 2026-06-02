#!/usr/bin/env python3
"""Build the head-CT hemorrhage manifest from the PhysioNet ct-ich v1.3.1 data.

Reads `hemorrhage_diagnosis_raw_ct.csv` and the `ct_scans/` + `masks/`
NIfTI directories, enumerates per-slice rows, and writes
`data/ct_hemorrhage_manifest.csv` with a schema that mirrors the CXR
manifest (so cross-modality scripts can share loaders) plus a CT-specific
`slice_index` and `subtype` column.

Schema: filename,split,label,image_path,mask_path,slice_index,modality,subtype
  - label: binary hemorrhage flag (1 = any subtype, i.e. NOT `normal`).
  - slice_index: 0-based axial index into the NIfTI volume (SliceNumber - 1).
  - split: patient-level (whole patients go to one side) to avoid leakage.
  - subtype: the hemorrhage subtype for positive slices ("" for normal).
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


DEFAULT_DATA_ROOT = "data_local/physionet.org/files/ct-ich/1.3.1"
SUBTYPE_COLUMNS = (
    "Intraventricular",
    "Intraparenchymal",
    "Subarachnoid",
    "Epidural",
    "Subdural",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", default="data/ct_hemorrhage_manifest.csv")
    parser.add_argument(
        "--max-positive",
        type=int,
        default=0,
        help="Cap on hemorrhage-positive slices (0 = all).",
    )
    parser.add_argument(
        "--include-normal",
        type=int,
        default=0,
        help="Number of normal (no-hemorrhage) slices to also include (0 = none).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.3,
        help="Patient-level fraction assigned to the test split.",
    )
    parser.add_argument("--seed", type=int, default=20260515)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    label_csv = data_root / "hemorrhage_diagnosis_raw_ct.csv"
    ct_dir = data_root / "ct_scans"
    mask_dir = data_root / "masks"
    if not label_csv.exists():
        raise FileNotFoundError(label_csv)

    # utf-8-sig strips the BOM on the header.
    with label_csv.open(newline="", encoding="utf-8-sig") as handle:
        label_rows = list(csv.DictReader(handle))

    # Patient-level split for leakage safety.
    patients = sorted({int(row["PatientNumber"]) for row in label_rows})
    present = [p for p in patients if (ct_dir / f"{p:03d}.nii").exists()]
    rng = random.Random(args.seed)
    shuffled = present[:]
    rng.shuffle(shuffled)
    n_test = int(round(len(shuffled) * args.test_fraction))
    test_patients = set(shuffled[:n_test])

    positive: list[dict[str, str | int]] = []
    normal: list[dict[str, str | int]] = []
    for row in label_rows:
        patient = int(row["PatientNumber"])
        if patient not in present:
            continue
        slice_number = int(row["SliceNumber"])
        is_hemorrhage = str(row["No_Hemorrhage"]).strip() == "0"
        subtype = ""
        if is_hemorrhage:
            for column in SUBTYPE_COLUMNS:
                if str(row.get(column, "0")).strip() == "1":
                    subtype = column.lower()
                    break
        entry = {
            "filename": f"patient_{patient:03d}_slice_{slice_number:03d}.png",
            "split": "test" if patient in test_patients else "train",
            "label": 1 if is_hemorrhage else 0,
            "image_path": str(ct_dir / f"{patient:03d}.nii"),
            "mask_path": str(mask_dir / f"{patient:03d}.nii"),
            "slice_index": slice_number - 1,
            "modality": "ct",
            "subtype": subtype,
        }
        (positive if is_hemorrhage else normal).append(entry)

    if args.max_positive > 0:
        positive = positive[: args.max_positive]
    # include_normal is an exact count (0 = none); positive uses 0 = all.
    normal = normal[: max(0, args.include_normal)]
    rows = positive + normal

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename", "split", "label", "image_path", "mask_path",
        "slice_index", "modality", "subtype",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_test_rows = sum(1 for r in rows if r["split"] == "test")
    print(f"CT manifest written to: {output_path}")
    print(f"Patients present: {len(present)} (test patients: {len(test_patients)})")
    print(f"Rows: {len(rows)} (positive={len(positive)}, normal={len(normal)}; test={n_test_rows})")


if __name__ == "__main__":
    main()
