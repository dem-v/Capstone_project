from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .metrics import normalize_map


NEUTRAL_IMPACT_COLOR = np.array([180, 0, 255], dtype=np.float32)
# Diverging palette for signed scalar attribution maps (Phase 1.2-dispatch).
# Orange = positive side (positive - negative > 0); Teal = negative side
# (positive - negative < 0). Per AGENTS.md "Color Semantics", these are
# intentionally distinct from red/blue so a reader does not confuse a signed
# tug-of-war map with a pure positive- or negative-evidence map.
SIGNED_POSITIVE_COLOR = np.array([255, 140, 0], dtype=np.float32)   # orange
SIGNED_NEGATIVE_COLOR = np.array([0, 160, 160], dtype=np.float32)   # teal


def _to_uint8(values: torch.Tensor) -> np.ndarray:
    return (normalize_map(values).numpy() * 255).astype(np.uint8)


def _mask_contour(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = mask.copy()
    for y_offset in range(3):
        for x_offset in range(3):
            eroded &= padded[y_offset: y_offset +
                             mask.shape[0], x_offset: x_offset + mask.shape[1]]
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
    neutral_heatmap: torch.Tensor | None = None,
) -> None:
    """Save grayscale image with colored heatmap and green mask contour overlay."""
    if heatmap_color not in {"red", "blue", "neutral"}:
        raise ValueError("heatmap_color must be 'red', 'blue', or 'neutral'.")

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

    if neutral_heatmap is not None:
        neutral_heat = _to_uint8(neutral_heatmap.detach().cpu())
        neutral_colored = (
            neutral_heat[..., None].astype(np.float32) / 255.0
        ) * NEUTRAL_IMPACT_COLOR
        rgb = (1 - alpha) * rgb + alpha * neutral_colored

    if heatmap_color == "neutral":
        colored = (heat[..., None].astype(np.float32) / 255.0) * NEUTRAL_IMPACT_COLOR
    else:
        colored = np.zeros_like(rgb)
        color_channel = 0 if heatmap_color == "red" else 2
        colored[..., color_channel] = heat
    rgb = (1 - alpha) * rgb + alpha * colored

    contour = _mask_contour(true)
    green = np.zeros_like(rgb)
    green[..., 1] = 255
    rgb[contour] = (1 - contour_alpha) * rgb[contour] + \
        contour_alpha * green[contour]

    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(output_path)


def save_binary_selection(
    image: torch.Tensor,
    selected_mask: torch.Tensor,
    true_mask: torch.Tensor,
    output_path: str | Path,
    *,
    negative_style: bool = False,
    neutral_style: bool = False,
    negative_selected_mask: torch.Tensor | None = None,
    neutral_selected_mask: torch.Tensor | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = image.detach().cpu()
    if base.ndim == 3:
        base = base[0]
    gray = _to_uint8(base)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    pred = selected_mask.detach().cpu().bool().numpy()
    true = true_mask.detach().cpu().bool().numpy()
    tp = pred & true
    fp = pred & ~true
    fn = ~pred & true

    if neutral_selected_mask is not None:
        neutral_pred = neutral_selected_mask.detach().cpu().bool().numpy()
        rgb[neutral_pred] = 0.50 * rgb[neutral_pred] + 0.50 * NEUTRAL_IMPACT_COLOR

    if negative_selected_mask is not None:
        negative_pred = negative_selected_mask.detach().cpu().bool().numpy()
        negative_inside = negative_pred & true
        negative_outside = negative_pred & ~true
        rgb[negative_outside] = 0.50 * rgb[negative_outside] + \
            0.50 * np.array([0, 0, 255], dtype=np.float32)
        rgb[negative_inside] = 0.35 * rgb[negative_inside] + \
            0.65 * np.array([0, 255, 255], dtype=np.float32)

    if negative_style:
        selected_outside = np.array([0, 0, 255], dtype=np.float32)
        selected_inside = np.array([0, 255, 255], dtype=np.float32)
    elif neutral_style:
        selected_outside = NEUTRAL_IMPACT_COLOR
        selected_inside = NEUTRAL_IMPACT_COLOR
    else:
        selected_outside = np.array([255, 0, 0], dtype=np.float32)
        selected_inside = np.array([255, 255, 0], dtype=np.float32)

    rgb[fp] = 0.50 * rgb[fp] + 0.50 * selected_outside
    rgb[tp] = 0.35 * rgb[tp] + 0.65 * selected_inside
    rgb[fn] = 0.50 * rgb[fn] + 0.50 * np.array([0, 255, 0], dtype=np.float32)

    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(output_path)


def overlay_color_for_method(method_name: str) -> str:
    if method_name.endswith("_negative"):
        return "blue"
    if method_name.endswith("_magnitude"):
        return "neutral"
    return "red"


def signed_diverging_overlay(
    image: torch.Tensor,
    signed_map: torch.Tensor,
    mask: torch.Tensor,
    output_path: str | Path,
    alpha: float = 0.45,
    contour_alpha: float = 0.55,
) -> None:
    """Render a signed scalar map with orange/teal diverging palette.

    ``signed_map`` is expected to be a ``SignedAttribution.signed`` view
    (i.e. roughly ``[-1, 1]``-valued, sign-preserving). Positive pixels
    are tinted orange (positive evidence side of the tug-of-war),
    negative pixels are tinted teal. Magnitude controls per-pixel alpha
    by ``|value| / max(|value|)``, so weakly-signed pixels barely tint
    the base image and the strongest pixel reaches full ``alpha``.

    The ground-truth mask contour is overlaid in green, identical to
    :func:`save_overlay`, so a reader can compare lesion location across
    the four-view grid without re-orienting.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = image.detach().cpu()
    if base.ndim == 3:
        base = base[0]
    gray = _to_uint8(base)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    signed = signed_map.detach().cpu().float().numpy()
    abs_max = float(np.max(np.abs(signed)))
    if abs_max > 0:
        scale = np.abs(signed) / abs_max  # [0, 1] alpha weights
    else:
        scale = np.zeros_like(signed)

    positive_mask = signed > 0
    negative_mask = signed < 0
    weights = (scale * alpha)[..., None]
    pos_tint = positive_mask[..., None] * SIGNED_POSITIVE_COLOR[None, None, :]
    neg_tint = negative_mask[..., None] * SIGNED_NEGATIVE_COLOR[None, None, :]
    rgb = (1 - weights) * rgb + weights * (pos_tint + neg_tint)

    true = mask.detach().cpu().bool().numpy()
    contour = _mask_contour(true)
    green = np.zeros_like(rgb)
    green[..., 1] = 255
    rgb[contour] = (1 - contour_alpha) * rgb[contour] + \
        contour_alpha * green[contour]

    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(output_path)
