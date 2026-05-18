"""Phase 1.3 — metric unit tests.

Covers the edge cases listed in `docs/refactor_plan.md` (Phase 1.3):

- Perfect overlap → IoU = Dice = 1.0.
- Disjoint masks → IoU = 0.0 and Dice = 0.0.
- Pointing-game hit returns 1.0 when the argmax pixel lies in the lesion and
  0.0 otherwise.
- All-zero heatmap edge case is well-defined (no NaN/exceptions; pointing
  game returns a deterministic 0/1; top-fraction returns a non-empty mask).
- ``threshold_top_fraction(fraction=1.0)`` returns an all-True mask.
"""

from __future__ import annotations

import pytest
import torch

from explainai_thesis.metrics import (
    dice_score,
    iou_score,
    localization_metrics,
    pointing_game_hit,
    precision_at_fraction,
    threshold_top_fraction,
)


@pytest.fixture
def lesion_mask() -> torch.Tensor:
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[2:5, 2:5] = True  # 3x3 lesion in the upper-left quadrant
    return mask


def test_perfect_overlap_iou_dice_equal_one(lesion_mask: torch.Tensor) -> None:
    pred = lesion_mask.clone()
    assert iou_score(pred, lesion_mask) == pytest.approx(1.0, abs=1e-6)
    assert dice_score(pred, lesion_mask) == pytest.approx(1.0, abs=1e-6)


def test_disjoint_masks_iou_dice_equal_zero(lesion_mask: torch.Tensor) -> None:
    pred = torch.zeros_like(lesion_mask)
    pred[6:8, 6:8] = True  # disjoint from the 2:5,2:5 lesion
    assert iou_score(pred, lesion_mask) == 0.0
    assert dice_score(pred, lesion_mask) == 0.0


def test_pointing_game_hit_inside_and_outside(lesion_mask: torch.Tensor) -> None:
    inside = torch.zeros(8, 8)
    inside[3, 3] = 1.0  # argmax inside the lesion
    assert pointing_game_hit(inside, lesion_mask) == 1.0

    outside = torch.zeros(8, 8)
    outside[0, 0] = 1.0  # argmax outside the lesion
    assert pointing_game_hit(outside, lesion_mask) == 0.0


def test_all_zero_heatmap_is_defined(lesion_mask: torch.Tensor) -> None:
    """All-zero heatmap should not raise, NaN, or return ill-defined values.

    Documents the current behavior: ``normalize_map`` with the ``eps`` guard
    yields a flat zero map, ``argmax`` returns index 0, and the top-fraction
    mask is non-empty (at least one pixel selected by construction).
    """
    flat = torch.zeros(8, 8)
    # pointing_game_hit on flat input: argmax → (0, 0). Mask is False at (0,0).
    hit = pointing_game_hit(flat, lesion_mask)
    assert hit in (0.0, 1.0)
    assert hit == 0.0  # for this lesion the (0,0) corner is outside

    # threshold_top_fraction must produce a non-empty selection.
    selected = threshold_top_fraction(flat, fraction=0.15)
    assert selected.sum().item() >= 1

    # precision_at_fraction on a flat heatmap must not raise.
    p = precision_at_fraction(flat, lesion_mask, fraction=0.15)
    assert 0.0 <= p <= 1.0


def test_threshold_top_fraction_full_returns_all_true() -> None:
    heatmap = torch.rand(8, 8)
    mask = threshold_top_fraction(heatmap, fraction=1.0)
    assert mask.dtype == torch.bool
    assert mask.all().item()


def test_threshold_top_fraction_rejects_out_of_range() -> None:
    heatmap = torch.rand(8, 8)
    with pytest.raises(ValueError):
        threshold_top_fraction(heatmap, fraction=0.0)
    with pytest.raises(ValueError):
        threshold_top_fraction(heatmap, fraction=1.5)


def test_localization_metrics_returns_all_keys(lesion_mask: torch.Tensor) -> None:
    heatmap = torch.zeros(8, 8)
    heatmap[3, 3] = 1.0
    out = localization_metrics(heatmap, lesion_mask, fraction=0.15)
    assert set(out.keys()) == {"iou", "dice", "pointing_hit", "precision_at_fraction"}
    for key, value in out.items():
        assert 0.0 <= value <= 1.0, f"{key} out of [0,1]: {value}"
