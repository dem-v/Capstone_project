"""Reusable CT attribution method dispatch.

The Phase 5.4 CT pilot deliberately uses only input-space methods whose
implementations transfer unchanged across modalities.  This module keeps the
method set, consensus semantics, and runtime settings in one reusable place so
CLI scripts do not grow their own ad hoc dispatch logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import nn

from explainai_thesis.xai import (
    SignedAttribution,
    consensus_signed,
    gradient_shap_signed,
    integrated_gradients_signed,
    occlusion_sensitivity_signed,
)


CT_INPUT_SPACE_METHODS = ("integrated_gradients", "gradient_shap", "occlusion")
CT_CONSENSUS_NAME = "consensus_input3"
CT_VISUAL_METHODS = (*CT_INPUT_SPACE_METHODS, CT_CONSENSUS_NAME)


@dataclass(frozen=True)
class CTAttributionSettings:
    """Runtime settings for CT input-space attribution methods."""

    ig_steps: int = 16
    gradshap_samples: int = 32
    gradshap_stdevs: float = 0.02
    occlusion_patch_size: int = 32
    occlusion_stride: int = 16


def normalize_ct_method_names(methods: Sequence[str]) -> tuple[str, ...]:
    """Validate and de-duplicate CT method names while preserving order."""

    normalized: list[str] = []
    allowed = set(CT_VISUAL_METHODS)
    for raw_method in methods:
        method = raw_method.strip()
        if not method:
            continue
        if method not in allowed:
            raise ValueError(
                f"Unsupported CT XAI method {method!r}. Expected one of: "
                f"{', '.join(CT_VISUAL_METHODS)}."
            )
        if method not in normalized:
            normalized.append(method)
    if not normalized:
        raise ValueError("At least one CT XAI method is required.")
    return tuple(normalized)


def required_input_methods(methods: Sequence[str]) -> tuple[str, ...]:
    """Return the input-space methods that must be computed.

    ``consensus_input3`` depends on all three transferable CT methods, even if
    the individual overlays are not requested for output.
    """

    requested = normalize_ct_method_names(methods)
    if CT_CONSENSUS_NAME in requested:
        return CT_INPUT_SPACE_METHODS
    return tuple(method for method in requested if method in CT_INPUT_SPACE_METHODS)


def compute_ct_signed_attributions(
    model: nn.Module,
    image: torch.Tensor,
    *,
    class_idx: int,
    methods: Sequence[str] = CT_VISUAL_METHODS,
    settings: CTAttributionSettings | None = None,
) -> dict[str, SignedAttribution]:
    """Compute requested CT signed attributions.

    The returned dictionary contains exactly the requested method names.  When
    ``consensus_input3`` is requested, its three constituents are computed as
    dependencies but omitted unless they were explicitly requested too.
    """

    requested = normalize_ct_method_names(methods)
    settings = settings or CTAttributionSettings()
    computed: dict[str, SignedAttribution] = {}

    for method in required_input_methods(requested):
        if method == "integrated_gradients":
            computed[method] = integrated_gradients_signed(
                model,
                image,
                class_idx=class_idx,
                steps=settings.ig_steps,
            )
        elif method == "gradient_shap":
            computed[method] = gradient_shap_signed(
                model,
                image,
                class_idx=class_idx,
                samples=settings.gradshap_samples,
                stdevs=settings.gradshap_stdevs,
            )
        elif method == "occlusion":
            computed[method] = occlusion_sensitivity_signed(
                model,
                image,
                class_idx=class_idx,
                patch_size=settings.occlusion_patch_size,
                stride=settings.occlusion_stride,
            )

    if CT_CONSENSUS_NAME in requested:
        computed[CT_CONSENSUS_NAME] = consensus_signed(
            [computed[method] for method in CT_INPUT_SPACE_METHODS]
        )

    return {method: computed[method] for method in requested}