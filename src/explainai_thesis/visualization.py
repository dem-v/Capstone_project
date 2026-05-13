from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .metrics import normalize_map


def _to_uint8(values: torch.Tensor) -> np.ndarray:
    return (normalize_map(values).numpy() * 255).astype(np.uint8)


def _mask_contour(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = mask.copy()
    for y_offset in range(3):
        for x_offset in range(3):
            eroded &= padded[y_offset : y_offset + mask.shape[0], x_offset : x_offset + mask.shape[1]]
    return mask & ~eroded


def save_overlay(
    image: torch.Tensor,
    heatmap: torch.Tensor,
    mask: torch.Tensor,
    output_path: str | Path,
    alpha: float = 0.45,
    contour_alpha: float = 0.55,
    heatmap_color: str = "red",
    negative_heatmap: torch.Tensor | None = None,
) -> None:
    """Save grayscale image with colored heatmap and green mask contour overlay."""
    if heatmap_color not in {"red", "blue"}:
        raise ValueError("heatmap_color must be 'red' or 'blue'.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = image.detach().cpu()
    if base.ndim == 3:
        base = base[0]

    gray = _to_uint8(base)
    heat = _to_uint8(heatmap.detach().cpu())
    true = mask.detach().cpu().bool().numpy()

    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    if negative_heatmap is not None:
        negative_heat = _to_uint8(negative_heatmap.detach().cpu())
        negative_colored = np.zeros_like(rgb)
        negative_colored[..., 2] = negative_heat
        rgb = (1 - alpha) * rgb + alpha * negative_colored

    colored = np.zeros_like(rgb)
    color_channel = 0 if heatmap_color == "red" else 2
    colored[..., color_channel] = heat
    rgb = (1 - alpha) * rgb + alpha * colored

    contour = _mask_contour(true)
    green = np.zeros_like(rgb)
    green[..., 1] = 255
    rgb[contour] = (1 - contour_alpha) * rgb[contour] + contour_alpha * green[contour]

    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(output_path)

