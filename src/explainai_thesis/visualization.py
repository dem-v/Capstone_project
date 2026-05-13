from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .metrics import normalize_map


def _to_uint8(values: torch.Tensor) -> np.ndarray:
    return (normalize_map(values).numpy() * 255).astype(np.uint8)


def save_overlay(
    image: torch.Tensor,
    heatmap: torch.Tensor,
    mask: torch.Tensor,
    output_path: str | Path,
    alpha: float = 0.45,
) -> None:
    """Save grayscale image with red heatmap and green mask contour-like overlay."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = image.detach().cpu()
    if base.ndim == 3:
        base = base[0]

    gray = _to_uint8(base)
    heat = _to_uint8(heatmap.detach().cpu())
    true = mask.detach().cpu().bool().numpy()

    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    red = np.zeros_like(rgb)
    red[..., 0] = heat
    rgb = (1 - alpha) * rgb + alpha * red

    # Fill mask in green, lightly, to keep the attribution visible.
    rgb[true, 1] = np.maximum(rgb[true, 1], 220)
    rgb[true, 0] *= 0.65
    rgb[true, 2] *= 0.65

    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(output_path)

