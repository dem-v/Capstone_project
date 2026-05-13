from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .metrics import normalize_map


class GradCAM:
    """Minimal Grad-CAM implementation for binary classification smoke tests."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, _module: nn.Module, _inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(
        self,
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor],
        grad_output: tuple[torch.Tensor],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()

    def __call__(
        self,
        image: torch.Tensor,
        class_idx: int = 1,
        polarity: str = "positive",
        variant: str = "grad_cam",
    ) -> torch.Tensor:
        if polarity not in {"positive", "negative"}:
            raise ValueError("polarity must be 'positive' or 'negative'.")
        if variant not in {"grad_cam", "grad_cam_plus_plus"}:
            raise ValueError(
                "variant must be 'grad_cam' or 'grad_cam_plus_plus'.")

        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        score = logits[:, class_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations/gradients.")

        if variant == "grad_cam_plus_plus":
            gradients = self.gradients if polarity == "positive" else -self.gradients
            gradients_power_2 = gradients.pow(2)
            gradients_power_3 = gradients_power_2 * gradients
            denominator = 2 * gradients_power_2 + (
                self.activations * gradients_power_3
            ).sum(dim=(2, 3), keepdim=True)
            denominator = torch.where(
                denominator != 0,
                denominator,
                torch.ones_like(denominator),
            )
            alpha = gradients_power_2 / denominator
            weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        else:
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        if polarity == "positive":
            cam = F.relu(cam)
        else:
            cam = F.relu(-cam)
        cam = F.interpolate(
            cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return normalize_map(cam[0, 0].cpu())


def integrated_gradients(
    model: nn.Module,
    image: torch.Tensor,
    class_idx: int = 1,
    steps: int = 32,
    baseline: torch.Tensor | None = None,
    polarity: str = "magnitude",
) -> torch.Tensor:
    """Small Integrated Gradients implementation for single-image smoke tests."""
    if polarity not in {"magnitude", "positive", "negative"}:
        raise ValueError(
            "polarity must be 'magnitude', 'positive', or 'negative'.")

    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(image)

    scaled_images = [baseline + (float(i) / steps) * (image - baseline)
                     for i in range(1, steps + 1)]
    total_gradients = torch.zeros_like(image)

    for scaled in scaled_images:
        scaled = scaled.detach().requires_grad_(True)
        logits = model(scaled)
        score = logits[:, class_idx].sum()
        gradients = torch.autograd.grad(score, scaled)[0]
        total_gradients += gradients.detach()

    avg_gradients = total_gradients / steps
    attribution = (image - baseline) * avg_gradients
    if polarity == "positive":
        attribution = F.relu(attribution)
    elif polarity == "negative":
        attribution = F.relu(-attribution)
    else:
        attribution = attribution.abs()
    heatmap = attribution.sum(dim=1)[0]
    return normalize_map(heatmap.cpu())


def consensus_heatmap(heatmaps: list[torch.Tensor]) -> torch.Tensor:
    """Average normalized maps as a simple explanation-improvement baseline."""
    if not heatmaps:
        raise ValueError("At least one heatmap is required.")
    stacked = torch.stack([normalize_map(h) for h in heatmaps], dim=0)
    return normalize_map(stacked.mean(dim=0))
