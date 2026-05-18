"""Regression tests for Phase 1.1: Grad-CAM++ polarity double-flip fix.

Bug (pre-fix): the ``grad_cam_plus_plus`` branch in
``src/explainai_thesis/xai.py`` sign-flipped ``self.gradients`` when
``polarity == "negative"`` *and* then re-applied ``F.relu(-cam)`` further
down, which is a double sign flip. The negative-polarity Grad-CAM++ output
therefore degenerated into its positive-polarity output.

Fix: polarity is applied exactly once, post-weight, via the
``F.relu(cam)`` / ``F.relu(-cam)`` block — same as the standard
``grad_cam`` branch.

These tests use a tiny CNN + synthetic lesion image so they run in
fractions of a second on CPU and do not depend on torchxrayvision.
"""
from __future__ import annotations

import torch
from torch import nn

from explainai_thesis.xai import GradCAM


class _TinyCNN(nn.Module):
    """Two-conv classifier with a named final conv as the Grad-CAM target.

    The convolutional stack is intentionally shallow so the saliency map
    inherits the spatial structure of the input — that makes the
    "lesion inside / outside the positive map" assertion in
    ``test_gradcam_negative_does_not_peak_inside_lesion`` meaningful.
    """

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
    """Return (image, lesion_mask). The image has a bright square in the
    upper-left quadrant; the mask marks the same square.
    """
    image = torch.full((1, 1, image_size, image_size), 0.1)
    mask = torch.zeros((image_size, image_size), dtype=torch.bool)
    top, left = image_size // 4, image_size // 4
    image[..., top:top + lesion_size, left:left + lesion_size] = 0.9
    mask[top:top + lesion_size, left:left + lesion_size] = True
    return image, mask


def _gradcampp_map(model: _TinyCNN, image: torch.Tensor, polarity: str) -> torch.Tensor:
    cam = GradCAM(model, model.target_conv)
    try:
        return cam(image, class_idx=1, polarity=polarity, variant="grad_cam_plus_plus")
    finally:
        cam.remove_hooks()


def _train_tiny_cnn(model: _TinyCNN, steps: int = 60) -> _TinyCNN:
    """Briefly train the tiny CNN so class 1 actually corresponds to
    "bright lesion present". Without this the gradient w.r.t. class 1 is
    random and the negative-polarity Grad-CAM++ map has no principled
    spatial preference, defeating the point of the regression assertion.
    """
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


def test_gradcam_plus_plus_negative_differs_from_positive() -> None:
    """Negative-polarity Grad-CAM++ must not equal positive-polarity output.

    With the pre-fix double sign flip the two maps were numerically equal;
    after the fix they must differ on at least one pixel.
    """
    torch.manual_seed(0)
    model = _train_tiny_cnn(_TinyCNN())
    image, _ = _synthetic_positive_case()

    positive = _gradcampp_map(model, image.clone(), "positive")
    negative = _gradcampp_map(model, image.clone(), "negative")

    assert positive.shape == negative.shape
    assert not torch.allclose(positive, negative, atol=1e-6), (
        "grad_cam_plus_plus negative map equals the positive map — the "
        "polarity double-flip regression is back."
    )


def test_gradcam_plus_plus_negative_does_not_peak_inside_lesion() -> None:
    """Across a small synthetic split, the negative-polarity argmax should
    fall *outside* the lesion mask more often than inside.

    Rationale: the negative-polarity map highlights regions whose
    activations push the score *down*; on synthetic positive cases the
    lesion drives the score *up*, so the negative-map peak should be off
    the lesion the majority of the time. This is the qualitative
    behavior the fix is meant to restore — pre-fix the negative map was
    identical to the positive map and therefore always peaked inside.

    The assertion is probabilistic (> 0.5 of cases peak outside the
    lesion), matching the plan's wording.
    """
    torch.manual_seed(0)
    cases = 5
    outside_hits = 0

    for seed in range(cases):
        torch.manual_seed(seed)
        model = _train_tiny_cnn(_TinyCNN())
        image, mask = _synthetic_positive_case()
        negative = _gradcampp_map(model, image, "negative")

        flat_idx = int(torch.argmax(negative))
        h = negative.shape[-1]
        row, col = divmod(flat_idx, h)
        if not bool(mask[row, col]):
            outside_hits += 1

    assert outside_hits / cases > 0.5, (
        f"negative-polarity Grad-CAM++ peaked inside the lesion in "
        f"{cases - outside_hits}/{cases} synthetic cases — expected the "
        f"majority of peaks to fall outside the lesion."
    )
