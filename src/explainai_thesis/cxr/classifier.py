"""Classifier-loading seam for CXR pneumothorax experiments.

Phase 1.7 Stage A prerequisite: a single `load_classifier(name, device)`
entry point that returns the four artifacts every CXR script needs:

    (model, target_layer, class_idx, preprocess_fn)

This isolates the model-loading decision (which TorchXRayVision weights, or
an out-of-family external model) from the rest of the pipeline. Existing
scripts can route through this seam without changing their CLI flag names
or defaults; the diagnostic A/B sweep in Phase 1.7 Stage A then loops over
weight names and uses the same downstream code unchanged.

Design notes
------------
- The preprocess function maps a HxW uint8 numpy array (0..255 grayscale)
  to a `[1, H, W]` float tensor in the model's expected input space. For
  TorchXRayVision DenseNets this is `xrv.datasets.normalize(array, 255)`
  followed by `from_numpy(...).unsqueeze(0).float()` — exactly what the
  smoke and calibration scripts do inline today.
- The target layer is the convolutional block consumed by Grad-CAM. For
  TorchXRayVision DenseNet-121 weights this is `model.features.denseblock4`,
  which is what every current CXR script uses.
- `class_idx` is resolved from `model.pathologies` for TorchXRayVision
  weights. The seam raises a clear error if the requested pathology is not
  present in the model's head — this catches "wrong model family" mistakes
  early instead of silently scoring the wrong class.
- The current implementation only covers the in-family TorchXRayVision
  DenseNet-121 weights named in `docs/refactor_plan.md` Phase 1.7. An
  out-of-family external model is added at Phase 1.7 implementation time
  through the same dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn


# Preprocess signature: HxW uint8 numpy array -> [1, H, W] float tensor on CPU.
# Callers move the tensor to the model's device themselves (consistent with
# the pre-seam scripts).
PreprocessFn = Callable[[np.ndarray], torch.Tensor]


@dataclass(frozen=True)
class ClassifierBundle:
    """Everything the XAI pipeline needs to run against a classifier."""

    model: nn.Module
    target_layer: nn.Module
    class_idx: int
    preprocess: PreprocessFn


# Phase 1.7 Stage A candidate weights (in-family TorchXRayVision DenseNet-121).
# Listed explicitly so a typo in `--weights` fails fast at the seam rather
# than at TorchXRayVision's checkpoint downloader.
TORCHXRAYVISION_DENSENET_WEIGHTS: tuple[str, ...] = (
    "densenet121-res224-all",      # current baseline / control
    "densenet121-res224-chex",
    "densenet121-res224-mimic_ch",
    "densenet121-res224-mimic_nb",
    "densenet121-res224-rsna",
    "densenet121-res224-nih",
    "densenet121-res224-pc",
)

# TorchXRayVision ResNet-50 classifier weights. Architecturally different from
# the DenseNet-121 family (different stem, residual blocks, 512x512 native input
# instead of 224x224). Same `pathologies` head, so the seam contract is
# identical aside from the larger native input size — callers that pass
# `--image-size 224` will get a downscaled forward pass, which is consistent
# with TorchXRayVision's own resizer behaviour but should be reported.
TORCHXRAYVISION_RESNET_WEIGHTS: tuple[str, ...] = (
    "resnet50-res512-all",
)

# TorchXRayVision ResNet autoencoder. This is NOT a pathology classifier:
# `ResNetAE` has no `pathologies` attribute and no class head; it produces
# image reconstructions / latent vectors. We expose it through the same seam
# so future feature-extractor / reconstruction-baseline experiments can load
# it uniformly, but `load_classifier(..., pathology=...)` will refuse the call
# unless `pathology` is explicitly set to `None`, because there is no class
# score for the current pneumothorax XAI pipeline to attribute against.
TORCHXRAYVISION_AUTOENCODER_WEIGHTS: tuple[str, ...] = (
    "resnetae-101-elastic",
)


def _torchxrayvision_preprocess_factory() -> PreprocessFn:
    # Imported lazily so unit tests that stub `load_classifier` don't pay
    # the TorchXRayVision import cost.
    import torchxrayvision as xrv

    def preprocess(array: np.ndarray) -> torch.Tensor:
        normalized = xrv.datasets.normalize(array, 255)
        return torch.from_numpy(normalized).unsqueeze(0).float()

    return preprocess


def _pathology_index(model: nn.Module, pathology: str) -> int:
    pathologies = list(getattr(model, "pathologies", []))
    try:
        return pathologies.index(pathology)
    except ValueError as exc:
        raise ValueError(
            f"{pathology!r} is not available in model pathologies: {pathologies}"
        ) from exc


def load_classifier(
    name: str,
    device: torch.device | str = "cpu",
    pathology: Optional[str] = "Pneumothorax",
) -> ClassifierBundle:
    """Load a CXR classifier by name and return the artifacts the pipeline needs.

    Parameters
    ----------
    name
        Either a TorchXRayVision DenseNet weight identifier (see
        `TORCHXRAYVISION_DENSENET_WEIGHTS`) or a future external-model key.
    device
        Torch device the model is moved to. Caller still owns moving inputs.
    pathology
        Name of the target head. Defaults to `"Pneumothorax"`, which is what
        every current CXR script uses.

    Returns
    -------
    ClassifierBundle
        Frozen dataclass with `(model, target_layer, class_idx, preprocess)`.

    Notes
    -----
    Existing scripts access fields by attribute (`bundle.model`, etc.); a
    tuple form is intentionally not provided to keep the call sites
    self-documenting.
    """

    if name in TORCHXRAYVISION_DENSENET_WEIGHTS:
        import torchxrayvision as xrv

        model = xrv.models.DenseNet(weights=name).to(device)
        model.eval()
        target_layer = model.features.denseblock4
        if pathology is None:
            raise ValueError(
                f"{name!r} is a pathology classifier; `pathology` must be set "
                "(default 'Pneumothorax'). Pass `pathology=None` only for "
                "autoencoder weights."
            )
        class_idx = _pathology_index(model, pathology)
        return ClassifierBundle(
            model=model,
            target_layer=target_layer,
            class_idx=class_idx,
            preprocess=_torchxrayvision_preprocess_factory(),
        )

    if name in TORCHXRAYVISION_RESNET_WEIGHTS:
        import torchxrayvision as xrv

        # The TorchXRayVision ResNet wrapper stores the torchvision ResNet on
        # `model.model`; Grad-CAM needs the final residual stage, i.e.
        # `model.model.layer4` (verified at implementation time on
        # `resnet50-res512-all`). Native input is 512x512 — callers that pass
        # a different `--image-size` get bilinear resizing inside the model.
        model = xrv.models.ResNet(weights=name).to(device)
        model.eval()
        target_layer = model.model.layer4
        if pathology is None:
            raise ValueError(
                f"{name!r} is a pathology classifier; `pathology` must be set "
                "(default 'Pneumothorax'). Pass `pathology=None` only for "
                "autoencoder weights."
            )
        class_idx = _pathology_index(model, pathology)
        return ClassifierBundle(
            model=model,
            target_layer=target_layer,
            class_idx=class_idx,
            preprocess=_torchxrayvision_preprocess_factory(),
        )

    if name in TORCHXRAYVISION_AUTOENCODER_WEIGHTS:
        # ResNetAE has no classification head and no `pathologies`. The seam
        # still loads it uniformly (same `preprocess`, same device move) so
        # downstream feature-extractor / reconstruction baselines can reuse
        # the bundle, but `class_idx` is set to a sentinel `-1` and the
        # current Pneumothorax XAI scripts will refuse to run against it.
        # The `pathology` arg must be explicitly `None` to acknowledge that
        # no class score will be attributed.
        if pathology is not None:
            raise ValueError(
                f"{name!r} is a ResNet autoencoder, not a pathology "
                "classifier — it has no class head. Call "
                "`load_classifier(name, ..., pathology=None)` to load it for "
                "latent/reconstruction use; it is not compatible with the "
                "current Pneumothorax classification + XAI pipeline."
            )
        import torchxrayvision as xrv

        # Weight name `resnetae-101-elastic` maps to TorchXRayVision's
        # documented `weights='101-elastic'` for `xrv.autoencoders.ResNetAE`.
        # The longer form is kept for the seam allow-list so a single
        # `--weights` value uniquely identifies the model class.
        ae_weights = name.split("-", 1)[1]  # "resnetae-101-elastic" -> "101-elastic"
        model = xrv.autoencoders.ResNetAE(weights=ae_weights).to(device)
        model.eval()
        # Deepest encoder stage — closest analogue to a Grad-CAM target layer
        # if a downstream consumer wants to attribute latent activations.
        target_layer = model.layer4
        return ClassifierBundle(
            model=model,
            target_layer=target_layer,
            class_idx=-1,
            preprocess=_torchxrayvision_preprocess_factory(),
        )

    known = (
        TORCHXRAYVISION_DENSENET_WEIGHTS
        + TORCHXRAYVISION_RESNET_WEIGHTS
        + TORCHXRAYVISION_AUTOENCODER_WEIGHTS
    )
    raise ValueError(
        f"Unknown classifier name: {name!r}. "
        f"Known TorchXRayVision weights: {known}. "
        "External-model identifiers are added at Phase 1.7 Stage A "
        "implementation time."
    )
