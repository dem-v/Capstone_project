"""Phase 1.4 — faithfulness sanity test.

Assertion: insertion AUC of ``grad_cam`` (positive view) on a synthetic
positive case beats the insertion AUC of a random heatmap, averaged across
a small fixed set of seeds. If this regresses, downstream thesis claims
about ``Grad-CAM`` being a meaningful explanation tool on this codebase
are unsafe.

The test is intentionally self-contained:

- Uses the tiny CNN + bright-square synthetic case from
  ``tests/test_gradcam_polarity.py`` (replicated here so the two test
  files do not couple, matching the pattern already in the repo).
- Implements a minimal insertion curve directly against the model probe so
  the sanity assertion stays independent from the production curve helper.

Insertion semantics (matching ``AGENTS.md`` Faithfulness Evaluation
Rules): start from a baseline image (here: zeros), restore the top-K
pixels by attribution at increasing K, and read the class-1 softmax
probability after each restore step. A good explanation should restore
probability quickly → higher AUC.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from explainai_thesis.xai import GradCAM


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
) -> torch.Tensor:
    image = torch.full((1, 1, image_size, image_size), 0.1)
    top, left = image_size // 4, image_size // 4
    image[..., top:top + lesion_size, left:left + lesion_size] = 0.9
    return image


def _train_tiny_cnn(model: _TinyCNN, steps: int = 60) -> _TinyCNN:
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(steps):
        pos_image = _synthetic_positive_case()
        neg_image = torch.full_like(pos_image, 0.1)
        batch = torch.cat([pos_image, neg_image], dim=0)
        target = torch.tensor([1, 0])
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(batch), target)
        loss.backward()
        opt.step()
    model.eval()
    return model


def _insertion_auc(
    model: nn.Module,
    image: torch.Tensor,
    heatmap: torch.Tensor,
    class_idx: int = 1,
    steps: int = 8,
) -> float:
    """Insertion curve AUC.

    Start from a zero baseline; at each step restore the top fraction of
    pixels (ranked by ``heatmap``), measure the class-``class_idx``
    softmax probability, and return the mean across step probabilities
    as the AUC proxy (trapezoid not needed for a strict ordinal sanity
    check on a uniform fraction grid).
    """
    flat_scores = heatmap.flatten()
    order = torch.argsort(flat_scores, descending=True)
    baseline = torch.zeros_like(image)
    h, w = image.shape[-2:]
    total = h * w

    fractions = torch.linspace(1.0 / steps, 1.0, steps)
    probs: list[float] = []
    with torch.no_grad():
        for frac in fractions:
            k = max(1, int(total * float(frac)))
            mask = torch.zeros(total, dtype=torch.bool)
            mask[order[:k]] = True
            mask = mask.view(1, 1, h, w)
            restored = torch.where(mask, image, baseline)
            logits = model(restored)
            probs.append(float(F.softmax(logits, dim=1)[0, class_idx]))
    return float(sum(probs) / len(probs))


def _gradcam_positive_map(model: _TinyCNN, image: torch.Tensor) -> torch.Tensor:
    cam = GradCAM(model, model.target_conv)
    try:
        return cam(image, class_idx=1, polarity="positive", variant="grad_cam")
    finally:
        cam.remove_hooks()


def test_gradcam_insertion_auc_beats_random() -> None:
    """Averaged across a small fixed set of seeds, Grad-CAM's insertion
    AUC must exceed a random heatmap's insertion AUC.

    Uses 5 seeds so a single unlucky init cannot tank the assertion. The
    margin requirement is small (Grad-CAM strictly > random) — anything
    closer would indicate the saliency signal has collapsed.
    """
    seeds = list(range(5))
    gradcam_aucs: list[float] = []
    random_aucs: list[float] = []

    for seed in seeds:
        torch.manual_seed(seed)
        model = _train_tiny_cnn(_TinyCNN())
        image = _synthetic_positive_case()

        gradcam_map = _gradcam_positive_map(model, image)
        # Random heatmap drawn under the same seed budget so the test is
        # fully deterministic.
        random_map = torch.rand_like(gradcam_map)

        gradcam_aucs.append(_insertion_auc(model, image, gradcam_map))
        random_aucs.append(_insertion_auc(model, image, random_map))

    mean_gradcam = sum(gradcam_aucs) / len(gradcam_aucs)
    mean_random = sum(random_aucs) / len(random_aucs)

    assert mean_gradcam > mean_random, (
        f"Grad-CAM insertion AUC ({mean_gradcam:.4f}) did not beat the "
        f"random heatmap AUC ({mean_random:.4f}) across seeds {seeds}. "
        f"Per-seed Grad-CAM: {gradcam_aucs}; per-seed random: {random_aucs}."
    )
