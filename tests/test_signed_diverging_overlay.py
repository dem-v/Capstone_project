"""Unit tests for the Phase 1.2-dispatch signed diverging overlay.

The CXR smoke script's end-to-end behavior (5 signed cores, v2 dispatch,
agreement.csv, metrics.csv schema bump) is exercised at integration scale
by the calibration + smoke runs themselves. These cheap unit tests guard
the two newly-introduced public APIs that the smoke script depends on:

- :func:`explainai_thesis.visualization.signed_diverging_overlay`
- :data:`explainai_thesis.visualization.SIGNED_POSITIVE_COLOR` / `SIGNED_NEGATIVE_COLOR`

and the v2 signed-attribution + agreement contract that the smoke script
hard-codes (cosine ``+1`` for identical maps, ``-1`` for negated maps).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from explainai_thesis.visualization import (
    SIGNED_NEGATIVE_COLOR,
    SIGNED_POSITIVE_COLOR,
    signed_diverging_overlay,
)
from explainai_thesis.xai import SignedAttribution, agreement_score


def _flat_gray_image(size: int = 32, value: float = 0.0) -> torch.Tensor:
    """One-channel image at a fixed gray level. Keeps tinting math obvious."""
    return torch.full((1, size, size), value, dtype=torch.float32)


def test_signed_diverging_overlay_emits_orange_for_positive(tmp_path: Path) -> None:
    """A pure-positive signed map must tint the image toward orange, not teal."""
    image = _flat_gray_image()
    mask = torch.zeros((32, 32), dtype=torch.bool)
    signed = torch.zeros((32, 32))
    signed[:16, :] = 1.0   # top half: positive evidence
    out = tmp_path / "overlay.png"
    signed_diverging_overlay(image, signed, mask, out, alpha=1.0, contour_alpha=0.0)
    assert out.is_file()
    rgb = np.asarray(Image.open(out))
    # Top half should be very close to SIGNED_POSITIVE_COLOR (orange).
    top_mean = rgb[:16, :, :].mean(axis=(0, 1))
    assert np.allclose(top_mean, SIGNED_POSITIVE_COLOR, atol=2.0), (
        f"top-half mean {top_mean} not near orange {SIGNED_POSITIVE_COLOR}"
    )
    # Bottom half is untinted (signed==0): stays at base gray (~0).
    bottom_mean = rgb[16:, :, :].mean()
    assert bottom_mean < 5.0, f"bottom should be untouched gray, got {bottom_mean}"


def test_signed_diverging_overlay_emits_teal_for_negative(tmp_path: Path) -> None:
    """A pure-negative signed map must tint toward teal, not orange."""
    image = _flat_gray_image()
    mask = torch.zeros((32, 32), dtype=torch.bool)
    signed = torch.zeros((32, 32))
    signed[:16, :] = -1.0
    out = tmp_path / "overlay.png"
    signed_diverging_overlay(image, signed, mask, out, alpha=1.0, contour_alpha=0.0)
    rgb = np.asarray(Image.open(out))
    top_mean = rgb[:16, :, :].mean(axis=(0, 1))
    assert np.allclose(top_mean, SIGNED_NEGATIVE_COLOR, atol=2.0), (
        f"top-half mean {top_mean} not near teal {SIGNED_NEGATIVE_COLOR}"
    )


def test_signed_diverging_overlay_handles_zero_map(tmp_path: Path) -> None:
    """All-zero signed map must not crash and must leave the image untouched."""
    image = _flat_gray_image(value=0.5)
    mask = torch.zeros((32, 32), dtype=torch.bool)
    signed = torch.zeros((32, 32))
    out = tmp_path / "overlay.png"
    signed_diverging_overlay(image, signed, mask, out, alpha=1.0, contour_alpha=0.0)
    rgb = np.asarray(Image.open(out))
    # Base gray at value=0.5 normalizes to 255 (single-value tensor -> max).
    # The key invariant is uniformity, not the exact byte value.
    assert rgb.std() < 1.0, "zero signed map must not introduce color variance"


def test_agreement_score_identity_and_negation() -> None:
    """Cosine similarity contract that the smoke script relies on."""
    raw = torch.randn(16, 16)
    attr = SignedAttribution(raw=raw)
    same = SignedAttribution(raw=raw.clone())
    flipped = SignedAttribution(raw=-raw)
    assert agreement_score(attr, same) == 1.0 or abs(
        agreement_score(attr, same) - 1.0) < 1e-5
    assert abs(agreement_score(attr, flipped) - (-1.0)) < 1e-5
