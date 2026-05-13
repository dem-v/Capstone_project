from __future__ import annotations

import torch


def normalize_map(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize a heatmap or image to [0, 1]."""
    values = values.detach().float()
    v_min = values.min()
    v_max = values.max()
    return (values - v_min) / (v_max - v_min + eps)


def threshold_top_fraction(heatmap: torch.Tensor, fraction: float = 0.15) -> torch.Tensor:
    """Keep the hottest fraction of pixels as a binary explanation mask."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")

    heatmap = normalize_map(heatmap)
    flat = heatmap.flatten()
    k = max(1, int(flat.numel() * fraction))
    threshold = torch.topk(flat, k).values.min()
    return heatmap >= threshold


def dice_score(pred_mask: torch.Tensor, true_mask: torch.Tensor, eps: float = 1e-8) -> float:
    pred = pred_mask.bool()
    true = true_mask.bool()
    intersection = (pred & true).sum().float()
    denominator = pred.sum().float() + true.sum().float()
    if denominator.item() == 0:
        return 1.0
    return ((2 * intersection + eps) / (denominator + eps)).item()


def iou_score(pred_mask: torch.Tensor, true_mask: torch.Tensor, eps: float = 1e-8) -> float:
    pred = pred_mask.bool()
    true = true_mask.bool()
    intersection = (pred & true).sum().float()
    union = (pred | true).sum().float()
    if union.item() == 0:
        return 1.0
    return ((intersection + eps) / (union + eps)).item()


def pointing_game_hit(heatmap: torch.Tensor, true_mask: torch.Tensor) -> float:
    """Return 1 if the maximum-attribution pixel lies inside the lesion mask."""
    heatmap = normalize_map(heatmap)
    flat_idx = int(torch.argmax(heatmap.flatten()).item())
    y = flat_idx // heatmap.shape[-1]
    x = flat_idx % heatmap.shape[-1]
    return float(bool(true_mask.bool()[y, x]))


def precision_at_fraction(
    heatmap: torch.Tensor,
    true_mask: torch.Tensor,
    fraction: float = 0.15,
    eps: float = 1e-8,
) -> float:
    pred = threshold_top_fraction(heatmap, fraction=fraction)
    true = true_mask.bool()
    selected = pred.sum().float()
    if selected.item() == 0:
        return 0.0
    true_positive = (pred & true).sum().float()
    return ((true_positive + eps) / (selected + eps)).item()


def localization_metrics(
    heatmap: torch.Tensor,
    true_mask: torch.Tensor,
    fraction: float = 0.15,
) -> dict[str, float]:
    pred_mask = threshold_top_fraction(heatmap, fraction=fraction)
    return {
        "iou": iou_score(pred_mask, true_mask),
        "dice": dice_score(pred_mask, true_mask),
        "pointing_hit": pointing_game_hit(heatmap, true_mask),
        "precision_at_fraction": precision_at_fraction(heatmap, true_mask, fraction=fraction),
    }

