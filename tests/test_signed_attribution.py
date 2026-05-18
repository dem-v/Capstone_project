"""Phase 1.2-core tests: SignedAttribution decomposition and method cores.

Covers the four-view contract, the consensus signed/magnitude split, the
cross-method agreement score, and the regression guard that the new
single-compute Grad-CAM signed core, projected back through the
deprecated ``polarity=`` wrapper, still produces the pre-1.2
``[0, 1]`` heatmap shape and value range.

These tests intentionally use the same tiny CNN + briefly-trained-on-
synthetic-lesion fixture as ``tests/test_gradcam_polarity.py`` to keep
the suite fast on CPU and avoid pulling torchxrayvision into unit
tests.
"""
from __future__ import annotations

import torch
from torch import nn

from explainai_thesis.metrics import normalize_signed_map
from explainai_thesis.xai import (
    GradCAM,
    SignedAttribution,
    agreement_score,
    consensus_signed,
    gradient_shap_signed,
    integrated_gradients_signed,
    occlusion_sensitivity_signed,
)


# --------------------------------------------------------------------------- #
# Tiny model + synthetic lesion fixtures (duplicated locally on purpose so
# this test module is self-contained and doesn't reach into another test
# file's private helpers).
# --------------------------------------------------------------------------- #


class _TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=False)
        self.target_conv = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.target_conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _synthetic_positive_case(
    image_size: int = 32,
    lesion_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = torch.full((1, 1, image_size, image_size), 0.1)
    mask = torch.zeros((image_size, image_size), dtype=torch.bool)
    top, left = image_size // 4, image_size // 4
    image[..., top:top + lesion_size, left:left + lesion_size] = 0.9
    mask[top:top + lesion_size, left:left + lesion_size] = True
    return image, mask


def _train_tiny_cnn(model: _TinyCNN, steps: int = 60) -> _TinyCNN:
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(steps):
        pos_image, _ = _synthetic_positive_case()
        neg_image = torch.full_like(pos_image, 0.1)
        batch = torch.cat([pos_image, neg_image], dim=0)
        target = torch.tensor([1, 0])
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(batch), target)
        loss.backward()
        opt.step()
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# normalize_signed_map
# --------------------------------------------------------------------------- #


def test_normalize_signed_map_scales_to_unit_max_abs() -> None:
    """Confirms the canonical ``raw`` normalization preserves zero and
    maps the most extreme value to ±1.
    """
    values = torch.tensor([-4.0, -2.0, 0.0, 1.0, 3.0])
    normalized = normalize_signed_map(values)
    assert torch.isclose(normalized.abs().max(), torch.tensor(1.0))
    assert torch.isclose(normalized[2], torch.tensor(0.0))
    assert normalized[0].item() < 0
    assert normalized[-1].item() > 0


def test_normalize_signed_map_all_zero_does_not_blow_up() -> None:
    """All-zero edge case must not divide by zero; output is all zero."""
    zeros = torch.zeros(5)
    out = normalize_signed_map(zeros)
    assert torch.equal(out, torch.zeros_like(out))


# --------------------------------------------------------------------------- #
# SignedAttribution dataclass — view algebra
# --------------------------------------------------------------------------- #


def test_signed_attribution_view_decomposition() -> None:
    """``positive - negative == signed`` and ``positive + negative == magnitude``
    by construction. This is the headline contract the rest of the
    pipeline relies on for cheap view derivation.
    """
    torch.manual_seed(0)
    raw = torch.randn(8, 8)
    raw = normalize_signed_map(raw)
    attribution = SignedAttribution(raw=raw)

    assert torch.allclose(attribution.positive - attribution.negative, attribution.signed)
    assert torch.allclose(attribution.positive + attribution.negative, attribution.magnitude)


def test_signed_attribution_unknown_view_raises() -> None:
    raw = torch.zeros(4, 4)
    attribution = SignedAttribution(raw=raw)
    try:
        attribution.view("absolute")
    except ValueError as exc:
        assert "absolute" in str(exc)
    else:
        raise AssertionError("expected ValueError on unknown view name")


# --------------------------------------------------------------------------- #
# Each *_signed method returns a SignedAttribution whose views aren't
# numerically equal to each other in the general case.
# --------------------------------------------------------------------------- #


def _trained_model() -> _TinyCNN:
    torch.manual_seed(0)
    return _train_tiny_cnn(_TinyCNN())


def test_gradcam_signed_views_differ() -> None:
    model = _trained_model()
    image, _ = _synthetic_positive_case()
    cam = GradCAM(model, model.target_conv)
    try:
        attribution = cam.signed(image, class_idx=1)
    finally:
        cam.remove_hooks()
    assert not torch.allclose(attribution.positive, attribution.magnitude, atol=1e-6)
    assert not torch.allclose(attribution.signed, attribution.magnitude, atol=1e-6)


def test_integrated_gradients_signed_views_differ() -> None:
    model = _trained_model()
    image, _ = _synthetic_positive_case()
    attribution = integrated_gradients_signed(model, image, class_idx=1, steps=8)
    assert not torch.allclose(attribution.signed, attribution.magnitude, atol=1e-6)


def test_gradient_shap_signed_views_differ() -> None:
    model = _trained_model()
    image, _ = _synthetic_positive_case()
    attribution = gradient_shap_signed(model, image, class_idx=1, samples=4)
    assert not torch.allclose(attribution.signed, attribution.magnitude, atol=1e-6)


def test_occlusion_sensitivity_signed_view_contract_holds() -> None:
    """Occlusion sensitivity on the trained tiny-lesion fixture happens
    to produce an all-non-negative delta map (every patch occlusion
    lowers the class-1 score), so ``signed == magnitude`` for this
    specific case. The contract we *do* enforce is the four-view
    algebra and non-negativity of the polarity views.

    A "signed != magnitude" assertion is enforced for Grad-CAM, IG,
    and GradientSHAP above where the trained classifier does emit
    bidirectional gradients on this fixture; occlusion would need a
    different fixture (multi-class or mixed-feature image) to
    naturally produce negative deltas, which is out of scope for
    a Phase 1.2-core CPU unit test.
    """
    model = _trained_model()
    image, _ = _synthetic_positive_case()
    attribution = occlusion_sensitivity_signed(
        model, image, class_idx=1, patch_size=8, stride=4,
    )
    assert torch.allclose(
        attribution.positive - attribution.negative, attribution.signed
    )
    assert torch.allclose(
        attribution.positive + attribution.negative, attribution.magnitude
    )
    assert float(attribution.positive.min()) >= 0.0
    assert float(attribution.negative.min()) >= 0.0
    assert float(attribution.magnitude.min()) >= 0.0


# --------------------------------------------------------------------------- #
# Single-compute regression guard: the deprecated `polarity=` wrapper must
# still produce a `[0, 1]`-ranged tensor with the same spatial shape as
# the new signed core. This is the test that catches "I broke the
# legacy wrapper" regressions during the smoke-script port.
# --------------------------------------------------------------------------- #


def test_legacy_wrapper_returns_unit_range_and_matches_signed_core_shape() -> None:
    model = _trained_model()
    image, _ = _synthetic_positive_case()
    cam = GradCAM(model, model.target_conv)
    try:
        attribution = cam.signed(image, class_idx=1)
        legacy_positive = cam(image, class_idx=1, polarity="positive")
        legacy_negative = cam(image, class_idx=1, polarity="negative")
    finally:
        cam.remove_hooks()

    for legacy in (legacy_positive, legacy_negative):
        assert legacy.shape == attribution.positive.shape
        assert float(legacy.min()) >= 0.0 - 1e-6
        assert float(legacy.max()) <= 1.0 + 1e-6


# --------------------------------------------------------------------------- #
# consensus_signed — preserves direction
# --------------------------------------------------------------------------- #


def test_consensus_signed_contains_both_polarities_when_inputs_disagree() -> None:
    """If one input is uniformly positive and another is uniformly
    negative on opposite halves, the signed consensus must contain
    both signs (no full cancellation).
    """
    a = torch.zeros(4, 4)
    a[:, :2] = 1.0
    b = torch.zeros(4, 4)
    b[:, 2:] = -1.0
    consensus = consensus_signed([SignedAttribution(raw=a), SignedAttribution(raw=b)])
    assert (consensus.raw > 0).any()
    assert (consensus.raw < 0).any()


def test_consensus_signed_rejects_empty_list() -> None:
    try:
        consensus_signed([])
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty consensus input")


# --------------------------------------------------------------------------- #
# agreement_score
# --------------------------------------------------------------------------- #


def test_agreement_score_self_is_one() -> None:
    torch.manual_seed(0)
    raw = normalize_signed_map(torch.randn(8, 8))
    attribution = SignedAttribution(raw=raw)
    assert agreement_score(attribution, attribution) > 0.999


def test_agreement_score_negated_is_minus_one() -> None:
    """Two maps with opposite signs everywhere must score ``-1`` —
    this is the "methods disagree on direction" sentinel.
    """
    torch.manual_seed(0)
    raw = normalize_signed_map(torch.randn(8, 8))
    a = SignedAttribution(raw=raw)
    b = SignedAttribution(raw=-raw)
    assert agreement_score(a, b) < -0.999


def test_agreement_score_shape_mismatch_raises() -> None:
    a = SignedAttribution(raw=torch.zeros(4, 4))
    b = SignedAttribution(raw=torch.zeros(8, 8))
    try:
        agreement_score(a, b)
    except ValueError as exc:
        assert "shape" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError on shape mismatch")
