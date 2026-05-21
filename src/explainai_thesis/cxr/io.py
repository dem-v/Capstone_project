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