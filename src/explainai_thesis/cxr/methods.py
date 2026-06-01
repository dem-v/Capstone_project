"""MethodSpec registry for CXR signed-attribution dispatch.

Replaces the inline per-method blocks in CXR scripts with a single
extensible registry. Each spec maps a stable method name (the family id
used in `iter_method_views(...)`) to a callable that returns a
`SignedAttribution`. New methods (Eigen-CAM, Score-CAM) plug in here
without touching call sites.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from ..xai import (
    GradCAM,
    SignedAttribution,
    consensus_signed,
    eigen_cam_signed,
    gradient_shap_signed,
    integrated_gradients_signed,
    occlusion_sensitivity_signed,
    score_cam_signed,
)


@dataclass
class MethodContext:
    """All per-case inputs a MethodSpec compute callable may consume."""

    model: torch.nn.Module
    model_input: torch.Tensor
    class_idx: int
    gradcam: GradCAM
    ig_steps: int = 16
    gradshap_samples: int = 8
    gradshap_stdevs: float = 0.09
    gradshap_internal_batch_size: int | None = None
    occlusion_patch_size: int = 32
    occlusion_stride: int = 16
    score_cam_channels_cap: int = 256


@dataclass(frozen=True)
class MethodSpec:
    """A named signed-attribution producer."""

    name: str
    compute: Callable[[MethodContext], SignedAttribution]


def _grad_cam(ctx: MethodContext) -> SignedAttribution:
    return ctx.gradcam.signed(ctx.model_input, class_idx=ctx.class_idx)


def _grad_cam_pp(ctx: MethodContext) -> SignedAttribution:
    return ctx.gradcam.signed(
        ctx.model_input,
        class_idx=ctx.class_idx,
        variant="grad_cam_plus_plus",
    )


def _ig(ctx: MethodContext) -> SignedAttribution:
    return integrated_gradients_signed(
        ctx.model,
        ctx.model_input,
        class_idx=ctx.class_idx,
        steps=ctx.ig_steps,
    )


def _gradshap(ctx: MethodContext) -> SignedAttribution:
    return gradient_shap_signed(
        ctx.model,
        ctx.model_input,
        class_idx=ctx.class_idx,
        samples=ctx.gradshap_samples,
        stdevs=ctx.gradshap_stdevs,
        internal_batch_size=ctx.gradshap_internal_batch_size,
    )


def _occlusion(ctx: MethodContext) -> SignedAttribution:
    return occlusion_sensitivity_signed(
        ctx.model,
        ctx.model_input,
        class_idx=ctx.class_idx,
        patch_size=ctx.occlusion_patch_size,
        stride=ctx.occlusion_stride,
    )


def _eigen_cam(ctx: MethodContext) -> SignedAttribution:
    return eigen_cam_signed(
        ctx.model,
        ctx.model_input,
        ctx.gradcam.target_layer,
        class_idx=ctx.class_idx,
    )


def _score_cam(ctx: MethodContext) -> SignedAttribution:
    return score_cam_signed(
        ctx.model,
        ctx.model_input,
        ctx.gradcam.target_layer,
        class_idx=ctx.class_idx,
        channels_cap=ctx.score_cam_channels_cap,
    )


DEFAULT_METHOD_SPECS: tuple[MethodSpec, ...] = (
    MethodSpec("grad_cam", _grad_cam),
    MethodSpec("grad_cam_plus_plus", _grad_cam_pp),
    MethodSpec("integrated_gradients", _ig),
    MethodSpec("gradient_shap", _gradshap),
    MethodSpec("occlusion", _occlusion),
    MethodSpec("eigen_cam", _eigen_cam),
    MethodSpec("score_cam", _score_cam),
)


CONSENSUS_CONSTITUENTS: tuple[str, ...] = (
    "grad_cam",
    "integrated_gradients",
    "gradient_shap",
    "occlusion",
)


def compute_signed_attributions(
    ctx: MethodContext,
    *,
    specs: tuple[MethodSpec, ...] = DEFAULT_METHOD_SPECS,
    include_consensus: bool = True,
    consensus_constituents: tuple[str, ...] = CONSENSUS_CONSTITUENTS,
) -> dict[str, SignedAttribution]:
    """Run every spec in order and return a name → SignedAttribution map.

    The order in `specs` is the order callbacks fire — kept stable so
    that any reproducibility-sensitive (PRNG-touching) method composes
    deterministically across runs.
    """

    result: dict[str, SignedAttribution] = {
        spec.name: spec.compute(ctx) for spec in specs
    }
    if include_consensus:
        constituents = [
            result[name] for name in consensus_constituents if name in result
        ]
        if constituents:
            result["consensus"] = consensus_signed(constituents)
    return result
