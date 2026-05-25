from __future__ import annotations

import torch


def normalize_map(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize a heatmap or image to [0, 1]."""
    values = values.detach().float()
    v_min = values.min()
    v_max = values.max()
    return (values - v_min) / (v_max - v_min + eps)


def normalize_signed_map(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale a signed map to ``[-1, 1]`` while preserving the sign.

    Used as the canonical normalization for ``SignedAttribution.raw``. The
    output divides by ``max(|values|)`` so that the most extreme positive or
    negative pixel becomes ``+1`` or ``-1`` and the zero crossing is
    preserved (unlike ``normalize_map`` which is a min-max rescale that
    destroys the sign). ``eps`` guards the all-zero edge case.
    """
    values = values.detach().float()
    scale = values.abs().max()
    return values / (scale + eps)


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
    if intersection.item() == 0:
        return 0.0
    return ((2 * intersection + eps) / (denominator + eps)).item()


def iou_score(pred_mask: torch.Tensor, true_mask: torch.Tensor, eps: float = 1e-8) -> float:
    pred = pred_mask.bool()
    true = true_mask.bool()
    intersection = (pred & true).sum().float()
    union = (pred | true).sum().float()
    if union.item() == 0:
        return 1.0
    if intersection.item() == 0:
        return 0.0
    return ((intersection + eps) / (union + eps)).item()


def pointing_game_hit(heatmap: torch.Tensor, true_mask: torch.Tensor) -> float:
    """Return 1 if the maximum-attribution pixel lies inside the lesion mask."""
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
    return precision_for_mask(pred, true_mask, eps=eps)


def precision_for_mask(
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Precision of a precomputed binary explanation mask."""
    pred = pred_mask.bool()
    true = true_mask.bool()
    selected = pred.sum().float()
    if selected.item() == 0:
        return 0.0
    true_positive = (pred & true).sum().float()
    if true_positive.item() == 0:
        return 0.0
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
        "precision_at_fraction": precision_for_mask(pred_mask, true_mask),
    }


def negative_evidence_metrics(
    heatmap: torch.Tensor,
    true_mask: torch.Tensor,
    fraction: float,
) -> dict[str, float]:
    """Mask-overlap diagnostic for the negative view of a SignedAttribution.

    `negative_mask_overlap_fraction`: of the top-fraction-selected pixels,
    how many fall *inside* the lesion mask. Healthy negative evidence
    should *avoid* the lesion, so a low value is good.
    `negative_mask_avoidance_fraction = 1 - overlap`; reported as the
    complement for thesis-readable framing.
    """
    selected = threshold_top_fraction(heatmap, fraction=fraction)
    selected_count = selected.sum().float()
    if selected_count.item() == 0:
        return {
            "negative_mask_overlap_fraction": 0.0,
            "negative_mask_avoidance_fraction": 0.0,
        }
    overlap = (selected & true_mask.bool()).sum().float() / selected_count
    return {
        "negative_mask_overlap_fraction": overlap.item(),
        "negative_mask_avoidance_fraction": (1.0 - overlap).item(),
    }


def selection_counts(
    heatmap: torch.Tensor,
    true_mask: torch.Tensor,
    fraction: float,
) -> dict[str, int]:
    """Per-case pixel counts for the top-fraction selection vs. the lesion mask."""

    selected = threshold_top_fraction(heatmap, fraction=fraction).bool()
    true = true_mask.bool()
    return {
        "selected_pixel_count": int(selected.sum().item()),
        "mask_pixel_count": int(true.sum().item()),
        "intersection_pixel_count": int((selected & true).sum().item()),
        "union_pixel_count": int((selected | true).sum().item()),
    }
