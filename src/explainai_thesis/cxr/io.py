from __future__ import annotations

import csv
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchxrayvision as xrv


def read_positive_masked_rows(
    manifest_path: Path,
    split: str,
    limit: int,
    *,
    random_sample: bool = False,
    seed: int = 20260515,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["label"]) != 1:
                continue
            if split != "any" and row.get("split") != split:
                continue
            if not row.get("mask_path"):
                continue
            rows.append(row)
            if not random_sample and len(rows) >= limit:
                break
    if random_sample:
        rng = random.Random(seed)
        rng.shuffle(rows)
    return rows


def load_xray_image(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("L").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image)
    normalized = xrv.datasets.normalize(array, 255)
    return torch.from_numpy(normalized).unsqueeze(0).float()


def load_binary_mask(path: Path, image_size: int) -> torch.Tensor:
    mask = Image.open(path).convert("L").resize((image_size, image_size), Image.NEAREST)
    return torch.from_numpy(np.asarray(mask) > 0)


def safe_case_name(sample_idx: int, row: dict[str, str]) -> str:
    return f"case_{sample_idx:03d}_{safe_source_stem(row)}"


def safe_source_stem(row: dict[str, str]) -> str:
    stem = Path(row.get("filename") or row["image_path"]).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return safe_stem or "xray"


def read_manifest_rows(manifest_path: Path, split: str) -> list[dict[str, str]]:
    """Read every row from a CXR manifest, optionally filtered by split.

    Used by the classifier-outcome scan which needs both positive and
    negative labels. `split="any"` returns the full manifest.
    """
    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if split != "any" and row.get("split") != split:
                continue
            rows.append(row)
    return rows


def parse_threshold_fractions(raw: str) -> list[float]:
    """Parse a comma-separated string of fractions in (0, 1].

    Stricter than `parse_optional_fractions`: at least one value is
    required and zero is rejected. Used by the classifier-outcome
    threshold-sweep CLI where a non-empty list of fractions is mandatory.
    """
    fractions = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not fractions:
        raise ValueError("At least one fraction is required.")
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("All fractions must be in (0, 1].")
    return fractions


def read_calibrated_fractions(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}

    fractions: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            method = row.get("method")
            selected_fraction = row.get("selected_fraction")
            if method and selected_fraction:
                fractions[method] = float(selected_fraction)
    return fractions


def parse_optional_fractions(raw: str) -> list[float]:
    fractions = [float(value.strip()) for value in raw.split(",") if value.strip()]
    for fraction in fractions:
        if not 0 <= fraction <= 1:
            raise ValueError("Faithfulness fractions must be in [0, 1].")
    return fractions