"""XAI methods producing signed attribution maps.

Phase 1.2 refactor (2026-05-18). Each method has a canonical ``*_signed``
core that runs forward / backward / occlusion exactly once per case and
returns a :class:`SignedAttribution` whose ``raw`` map is normalized to
``[-1, 1]`` while preserving the sign. The four views (``positive``,
``negative``, ``magnitude``, ``signed``) are derived as properties in
microseconds. This eliminates the per-polarity recomputation that
previously cost three separate forward+backward (or occlusion) passes per
case for IG, GradientSHAP, and Occlusion, and that was the same code
shape that produced the ``grad_cam_plus_plus`` polarity double-flip
fixed in Phase 1.1.

The legacy ``polarity=``-keyword functions are kept as thin deprecated
wrappers so that scripts not yet ported to the new API (notably
``scripts/run_cxr_torchxray_smoke.py`` ahead of its 1.2-dispatch port)
keep producing numerically equivalent ``[0, 1]`` heatmaps.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from captum.attr import GradientShap

from .metrics import normalize_map, normalize_signed_map


# --------------------------------------------------------------------------- #
# SignedAttribution dataclass
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SignedAttribution:
    """Canonical XAI output with four polarity views derived from one map.

    ``raw`` is a 2-D ``torch.Tensor`` normalized to ``[-1, 1]`` via
    :func:`normalize_signed_map`. The four views are derived properties:

    - ``positive`` / ``negative``: non-negative ``[0, 1]`` maps suitable
      for the existing ``localization_metrics`` and faithfulness paths
      that consume ``[0, 1]`` heatmaps.
    - ``magnitude``: ``|raw|`` in ``[0, 1]`` — used as the default "where
      did the method look" view.
    - ``signed``: ``raw`` itself, used by diverging overlays and by the
      ``signed_positive_fraction`` diagnostic that records what share of
      the top-fraction selected pixels are positive-signed.
    """

    raw: torch.Tensor

    @property
    def positive(self) -> torch.Tensor:
        return self.raw.clamp(min=0)

    @property
    def negative(self) -> torch.Tensor:
        return self.raw.clamp(max=0).abs()

    @property
    def magnitude(self) -> torch.Tensor:
        return self.raw.abs()

    @property
    def signed(self) -> torch.Tensor:
        return self.raw

    def view(self, name: str) -> torch.Tensor:
        """Look up one of the four views by string name."""
        if name == "positive":
            return self.positive
        if name == "negative":
            return self.negative
        if name == "magnitude":
            return self.magnitude
        if name == "signed":
            return self.signed
        raise ValueError(
            f"unknown SignedAttribution view '{name}'; "
            f"expected one of: positive, negative, magnitude, signed."
        )


@dataclass(frozen=True)
class MethodView:
    """One concrete output view derived from a method family's attribution."""

    method: str
    heatmap: torch.Tensor
    view: str
    family: str

    @property
    def is_negative(self) -> bool:
        return self.view == "negative"

    @property
    def is_magnitude(self) -> bool:
        return self.view == "magnitude"

    @property
    def is_signed(self) -> bool:
        return self.view == "signed"


def iter_method_views(
    attributions: dict[str, SignedAttribution],
) -> list[MethodView]:
    """Expand signed method-family outputs into stable per-view records.

    The canonical v2 convention is one primary method id per family for
    positive evidence, plus explicit ``_negative``, ``_magnitude``, and
    ``_signed`` view rows. Keeping this expansion in one place prevents
    visualization and calibration scripts from drifting back to ad hoc
    per-method branching.
    """
    method_views: list[MethodView] = []
    for family, attribution in attributions.items():
        method_views.extend(
            [
                MethodView(
                    method=family,
                    heatmap=normalize_map(attribution.positive.cpu()),
                    view="positive",
                    family=family,
                ),
                MethodView(
                    method=f"{family}_negative",
                    heatmap=normalize_map(attribution.negative.cpu()),
                    view="negative",
                    family=family,
                ),
                MethodView(
                    method=f"{family}_magnitude",
                    heatmap=normalize_map(attribution.magnitude.cpu()),
                    view="magnitude",
                    family=family,
                ),
                MethodView(
                    method=f"{family}_signed",
                    heatmap=attribution.signed.cpu(),
                    view="signed",
                    family=family,
                ),
            ]
        )
    return method_views


def _deprecated_polarity_warning(method: str) -> None:
    warnings.warn(
        f"The `polarity=` keyword on `{method}` is deprecated; use the "
        f"`{method}_signed` core function and read `.positive`, "
        f"`.negative`, `.magnitude`, or `.signed` on the returned "
        f"SignedAttribution. The wrapper is kept for backwards "
        f"compatibility during the Phase 1.2 transition.",
        DeprecationWarning,
        stacklevel=3,
    )


def _project_legacy(attribution: SignedAttribution, polarity: str) -> torch.Tensor:
    """Map a SignedAttribution to the legacy ``[0, 1]`` tensor contract.

    The pre-1.2 API rescaled the requested polarity view via
    :func:`normalize_map` (a min-max rescale to ``[0, 1]``). We re-apply
    the same rescale here so that the deprecated wrappers stay
    numerically equivalent to the old code path within tolerance.
    """
    view = attribution.view(polarity)
    return normalize_map(view.cpu())


# --------------------------------------------------------------------------- #
# Grad-CAM (and Grad-CAM++)
# --------------------------------------------------------------------------- #


class GradCAM:
    """Grad-CAM / Grad-CAM++ with a signed-attribution core.

    The legacy ``__call__(..., polarity=..., variant=...)`` signature is
    preserved as a deprecated shim around :meth:`signed`; new code should
    call :meth:`signed` directly and use the four-view properties.
    """

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

    # --- new signed core ----------------------------------------------------

    def signed(
        self,
        image: torch.Tensor,
        class_idx: int = 1,
        variant: str = "grad_cam",
    ) -> SignedAttribution:
        """Compute the signed Grad-CAM(++) map for one image.

        The signed ``cam`` is the raw ``(weights * activations).sum()``
        *before* any ``F.relu``, then interpolated to image size and
        normalized to ``[-1, 1]`` via :func:`normalize_signed_map`.
        Positive pixels push the class score up, negative pixels push it
        down — the cleanest definition for the four-view contract.
        """
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

        gradients = self.gradients
        if variant == "grad_cam_plus_plus":
            # Single-flip polarity is now the responsibility of the
            # caller via SignedAttribution.negative; the pre-weight
            # `-self.gradients` flip from the pre-1.1 code is gone.
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
            weights = gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.interpolate(
            cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return SignedAttribution(raw=normalize_signed_map(cam[0, 0].cpu()))

    # --- legacy wrapper -----------------------------------------------------

    def __call__(
        self,
        image: torch.Tensor,
        class_idx: int = 1,
        polarity: str = "positive",
        variant: str = "grad_cam",
    ) -> torch.Tensor:
        if polarity not in {"positive", "negative"}:
            raise ValueError("polarity must be 'positive' or 'negative'.")
        # Not warning here because every Phase 0 / Phase 1.1 callsite
        # still uses this signature; the warning will be added when the
        # smoke script is ported in Phase 1.2-dispatch.
        attribution = self.signed(image, class_idx=class_idx, variant=variant)
        return _project_legacy(attribution, polarity)


# --------------------------------------------------------------------------- #
# Integrated Gradients
# --------------------------------------------------------------------------- #


def integrated_gradients_signed(
    model: nn.Module,
    image: torch.Tensor,
    class_idx: int = 1,
    steps: int = 32,
    baseline: torch.Tensor | None = None,
) -> SignedAttribution:
    """Signed Integrated Gradients for a single image.

    Returns one :class:`SignedAttribution` summed over the channel
    dimension and normalized to ``[-1, 1]``. One IG loop per case (was
    three in the pre-1.2 ``polarity in {magnitude, positive, negative}``
    flow).
    """
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(image)

    alphas = torch.linspace(
        1.0 / steps,
        1.0,
        steps,
        device=image.device,
        dtype=image.dtype,
    ).view(steps, 1, 1, 1)
    scaled = baseline + alphas * (image - baseline)
    scaled = scaled.detach().requires_grad_(True)
    logits = model(scaled)
    score = logits[:, class_idx].sum()
    avg_gradients = torch.autograd.grad(score, scaled)[0].mean(dim=0, keepdim=True)
    attribution = (image - baseline) * avg_gradients
    heatmap = attribution.sum(dim=1)[0]
    return SignedAttribution(raw=normalize_signed_map(heatmap.cpu()))


def integrated_gradients(
    model: nn.Module,
    image: torch.Tensor,
    class_idx: int = 1,
    steps: int = 32,
    baseline: torch.Tensor | None = None,
    polarity: str = "magnitude",
) -> torch.Tensor:
    """Deprecated thin wrapper preserving the pre-1.2 ``polarity=`` API.

    Kept so that ``scripts/run_cxr_torchxray_smoke.py`` and any unported
    notebook code continue to produce the same ``[0, 1]`` heatmaps until
    Phase 1.2-dispatch lands. Prefer :func:`integrated_gradients_signed`.
    """
    if polarity not in {"magnitude", "positive", "negative"}:
        raise ValueError(
            "polarity must be 'magnitude', 'positive', or 'negative'.")
    attribution = integrated_gradients_signed(
        model, image, class_idx=class_idx, steps=steps, baseline=baseline
    )
    return _project_legacy(attribution, polarity)


# --------------------------------------------------------------------------- #
# GradientSHAP
# --------------------------------------------------------------------------- #


def gradient_shap_signed(
    model: nn.Module,
    image: torch.Tensor,
    class_idx: int = 1,
    samples: int = 16,
    stdevs: float = 0.02,
) -> SignedAttribution:
    """Signed GradientSHAP attribution for a single image.

    Single call to Captum's :class:`GradientShap.attribute`, summed over
    channels, normalized to ``[-1, 1]``.
    """
    model.eval()
    baseline_zero = torch.zeros_like(image)
    baseline_mean = torch.full_like(image, float(image.mean().detach().cpu().item()))
    baselines = torch.cat([baseline_zero, baseline_mean], dim=0)
    attribution = GradientShap(model).attribute(
        image,
        baselines=baselines,
        target=class_idx,
        n_samples=samples,
        stdevs=stdevs,
    )
    heatmap = attribution.sum(dim=1)[0]
    return SignedAttribution(raw=normalize_signed_map(heatmap.detach().cpu()))


def gradient_shap(
    model: nn.Module,
    image: torch.Tensor,
    class_idx: int = 1,
    samples: int = 16,
    stdevs: float = 0.02,
    polarity: str = "magnitude",
) -> torch.Tensor:
    """Deprecated thin wrapper preserving the pre-1.2 ``polarity=`` API."""
    if polarity not in {"magnitude", "positive", "negative"}:
        raise ValueError(
            "polarity must be 'magnitude', 'positive', or 'negative'.")
    attribution = gradient_shap_signed(
        model, image, class_idx=class_idx, samples=samples, stdevs=stdevs
    )
    return _project_legacy(attribution, polarity)


# --------------------------------------------------------------------------- #
# Occlusion Sensitivity
# --------------------------------------------------------------------------- #


def occlusion_sensitivity_signed(
    model: nn.Module,
    image: torch.Tensor,
    class_idx: int = 1,
    patch_size: int = 16,
    stride: int = 8,
    baseline_value: float = 0.0,
    batch_size: int = 32,
) -> SignedAttribution:
    """Signed occlusion-sensitivity map for a single image.

    The raw signal is ``delta = original_score - occluded_score`` per
    pixel: positive ``delta`` means the patch was a positive driver of
    the class score (occluding it lowered the score), negative ``delta``
    means the patch suppressed the class. Single occlusion sweep per
    case (was three for magnitude/positive/negative).
    """
    model.eval()
    _, _, height, width = image.shape
    windows: list[tuple[int, int, int, int]] = []
    for top in range(0, height, stride):
        bottom = min(top + patch_size, height)
        for left in range(0, width, stride):
            right = min(left + patch_size, width)
            windows.append((top, bottom, left, right))

    masks = torch.zeros(
        (len(windows), 1, height, width),
        dtype=image.dtype,
        device=image.device,
    )
    for mask, (top, bottom, left, right) in zip(masks, windows):
        mask[:, top:bottom, left:right] = 1

    attribution = torch.zeros((height, width), dtype=image.dtype, device=image.device)
    counts = masks.sum(dim=0)[0]

    with torch.inference_mode():
        original_score = model(image)[:, class_idx].sum()
        for start in range(0, len(windows), batch_size):
            batch_masks = masks[start:start + batch_size]
            occluded_batch = image.detach() * (1 - batch_masks) + baseline_value * batch_masks
            occluded_scores = model(occluded_batch)[:, class_idx]
            deltas = original_score - occluded_scores
            attribution += (deltas.view(-1, 1, 1) * batch_masks[:, 0]).sum(dim=0)
    attribution = attribution / torch.clamp(counts, min=1)
    return SignedAttribution(raw=normalize_signed_map(attribution.detach().cpu()))


def occlusion_sensitivity(
    model: nn.Module,
    image: torch.Tensor,
    class_idx: int = 1,
    patch_size: int = 16,
    stride: int = 8,
    baseline_value: float = 0.0,
    polarity: str = "positive",
    batch_size: int = 32,
) -> torch.Tensor:
    """Deprecated thin wrapper preserving the pre-1.2 ``polarity=`` API."""
    if polarity not in {"positive", "negative", "magnitude"}:
        raise ValueError(
            "polarity must be 'positive', 'negative', or 'magnitude'.")
    attribution = occlusion_sensitivity_signed(
        model,
        image,
        class_idx=class_idx,
        patch_size=patch_size,
        stride=stride,
        baseline_value=baseline_value,
        batch_size=batch_size,
    )
    return _project_legacy(attribution, polarity)


# --------------------------------------------------------------------------- #
# Consensus
# --------------------------------------------------------------------------- #


def consensus_heatmap(heatmaps: list[torch.Tensor]) -> torch.Tensor:
    """Average normalized maps as a simple explanation-improvement baseline.

    Magnitude-only consensus. Each input is re-normalized to ``[0, 1]``
    via :func:`normalize_map` before averaging; the output is also in
    ``[0, 1]``. Pre-1.2 behavior preserved exactly.
    """
    if not heatmaps:
        raise ValueError("At least one heatmap is required.")
    stacked = torch.stack([normalize_map(h) for h in heatmaps], dim=0)
    return normalize_map(stacked.mean(dim=0))


def consensus_signed(attributions: list[SignedAttribution]) -> SignedAttribution:
    """Sign-aware average of multiple :class:`SignedAttribution` maps.

    Unlike :func:`consensus_heatmap`, this preserves direction: a region
    that one method calls "positive evidence" and another calls
    "negative evidence" will partially cancel rather than reinforce.
    Result is normalized to ``[-1, 1]``.
    """
    if not attributions:
        raise ValueError("At least one SignedAttribution is required.")
    stacked = torch.stack([attribution.raw for attribution in attributions], dim=0)
    averaged = stacked.mean(dim=0)
    return SignedAttribution(raw=normalize_signed_map(averaged))


# --------------------------------------------------------------------------- #
# Cross-method agreement
# --------------------------------------------------------------------------- #


def agreement_score(
    attribution_a: SignedAttribution,
    attribution_b: SignedAttribution,
    eps: float = 1e-8,
) -> float:
    """Cosine similarity between two signed maps in ``[-1, 1]``.

    Measures whether two XAI methods agree on *direction*, not just on
    *magnitude*: two methods that pick the same hot region with
    opposite signs score ``-1``, full agreement scores ``+1``. Both
    inputs must share spatial shape.
    """
    a = attribution_a.raw.flatten().float()
    b = attribution_b.raw.flatten().float()
    if a.shape != b.shape:
        raise ValueError(
            f"agreement_score expects same spatial shape, got "
            f"{tuple(attribution_a.raw.shape)} vs "
            f"{tuple(attribution_b.raw.shape)}."
        )
    numerator = torch.dot(a, b)
    denominator = a.norm() * b.norm()
    return float((numerator / (denominator + eps)).item())
