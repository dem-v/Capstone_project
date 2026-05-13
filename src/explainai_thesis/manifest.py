from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
BASE_FIELDNAMES = ["image_path", "mask_path", "label"]


def looks_like_mask(path: Path) -> bool:
    lowered = "/".join(part.lower() for part in path.parts)
    return "mask" in lowered or "label" in lowered or "segmentation" in lowered


def normalized_stem(path: Path) -> str:
    stem = path.stem
    for suffix in ("_mask", "-mask", "_label", "-label", "_seg", "-seg"):
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def mask_has_foreground(mask_path: Path) -> int:
    with Image.open(mask_path) as image:
        gray = image.convert("L")
        extrema = gray.getextrema()
    return int(extrema[1] > 0)


def infer_label_from_name(path: Path) -> int | None:
    name = path.name.lower()
    positive_markers = ("_1_", "-1-", "_positive", "-positive", "_pneumo", "-pneumo")
    negative_markers = ("_0_", "-0-", "_negative", "-negative", "_normal", "-normal")
    if any(marker in name for marker in positive_markers):
        return 1
    if any(marker in name for marker in negative_markers):
        return 0
    return None


def build_png_mask_manifest(root: Path) -> list[dict[str, str | int]]:
    files = [path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    mask_files = [path for path in files if looks_like_mask(path)]
    image_files = [path for path in files if not looks_like_mask(path)]

    masks_by_stem: dict[str, Path] = {}
    for mask_path in mask_files:
        masks_by_stem.setdefault(normalized_stem(mask_path), mask_path)

    rows: list[dict[str, str | int]] = []
    for image_path in image_files:
        stem = normalized_stem(image_path)
        mask_path = masks_by_stem.get(stem)
        if mask_path is not None:
            label = mask_has_foreground(mask_path)
            mask_value = str(mask_path)
        else:
            inferred = infer_label_from_name(image_path)
            if inferred is None:
                continue
            label = inferred
            mask_value = ""

        rows.append(
            {
                "image_path": str(image_path),
                "mask_path": mask_value,
                "label": label,
            }
        )

    return rows


def build_siim_pneumothorax_manifest(root: Path) -> list[dict[str, str | int]]:
    dataset_root = root / "siim-acr-pneumothorax"
    if not dataset_root.exists():
        dataset_root = root

    image_dir = dataset_root / "png_images"
    mask_dir = dataset_root / "png_masks"
    csv_by_split = {
        "train": dataset_root / "stage_1_train_images.csv",
        "test": dataset_root / "stage_1_test_images.csv",
    }

    missing_paths = [
        path
        for path in [image_dir, mask_dir, *csv_by_split.values()]
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "SIIM pneumothorax dataset layout is incomplete: "
            + ", ".join(str(path) for path in missing_paths)
        )

    rows: list[dict[str, str | int]] = []
    for split, csv_path in csv_by_split.items():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for source_row in reader:
                filename = source_row["new_filename"]
                rows.append(
                    {
                        "image_path": str(image_dir / filename),
                        "mask_path": str(mask_dir / filename),
                        "label": int(source_row["has_pneumo"]),
                        "split": split,
                        "image_id": source_row["ImageId"],
                        "filename": filename,
                    }
                )

    return rows


def build_manifest(root: Path) -> list[dict[str, str | int]]:
    dataset_root = root / "siim-acr-pneumothorax"
    if not dataset_root.exists():
        dataset_root = root

    if (
        (dataset_root / "stage_1_train_images.csv").exists()
        and (dataset_root / "stage_1_test_images.csv").exists()
    ):
        return build_siim_pneumothorax_manifest(root)

    return build_png_mask_manifest(root)


def write_manifest(rows: list[dict[str, str | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extra_fieldnames = sorted(
        {key for row in rows for key in row} - set(BASE_FIELDNAMES)
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*BASE_FIELDNAMES, *extra_fieldnames])
        writer.writeheader()
        writer.writerows(rows)
