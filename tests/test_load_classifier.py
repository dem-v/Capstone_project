"""Tests for the Phase 1.7 `load_classifier(name)` seam.

Two layers of coverage:

1. Fast unit tests (always run): a synthetic stub plugged in via
   `monkeypatch` exercises the `_pathology_index` resolution, the
   `ClassifierBundle` shape contract, and the unknown-name `ValueError`.
   These run in milliseconds and do not download any weights.

2. `@pytest.mark.slow` real-weights tests (opt-in): instantiate at least
   two real TorchXRayVision DenseNet-121 checkpoints and assert that the
   same input produces *different* outputs across them. This is the
   "are we really loading different weights, not silently caching the
   same checkpoint?" parity check from `docs/refactor_plan.md` Phase 1.7.

   These are skipped by default to keep the suite under 5 minutes on CPU
   without a network. Enable with `pytest -m slow`.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch
from torch import nn

from explainai_thesis.cxr import classifier as classifier_module
from explainai_thesis.cxr.classifier import (
    ClassifierBundle,
    TORCHXRAYVISION_AUTOENCODER_WEIGHTS,
    TORCHXRAYVISION_DENSENET_WEIGHTS,
    TORCHXRAYVISION_RESNET_WEIGHTS,
    load_classifier,
)


class _StubDenseNet(nn.Module):
    """Minimal stand-in for `xrv.models.DenseNet` used in fast unit tests.

    Mirrors the surface area `load_classifier` touches: a `pathologies`
    attribute, a `features.denseblock4` submodule, `.to(device)`, and
    `.eval()`. No actual classification capability is required.
    """

    def __init__(self, weights: str) -> None:
        super().__init__()
        self.weights_name = weights
        self.pathologies = [
            "Atelectasis",
            "Consolidation",
            "Pneumothorax",
            "Edema",
        ]
        self.features = nn.Sequential()
        self.features.add_module("denseblock4", nn.Conv2d(1, 1, kernel_size=1))


class _StubResNet(nn.Module):
    """Minimal stand-in for `xrv.models.ResNet`.

    The real wrapper stores the torchvision ResNet under `self.model`, with
    `layer4` as the final residual stage — exactly what `load_classifier`
    grabs for Grad-CAM. Pneumothorax sits at a different index (1) than in
    the DenseNet stub so an accidental class-index reuse would surface.
    """

    def __init__(self, weights: str) -> None:
        super().__init__()
        self.weights_name = weights
        self.pathologies = ["Atelectasis", "Pneumothorax", "Edema"]
        inner = nn.Module()
        inner.layer4 = nn.Conv2d(1, 1, kernel_size=1)
        self.model = inner


class _StubResNetAE(nn.Module):
    """Minimal stand-in for `xrv.autoencoders.ResNetAE`.

    Crucially has NO `pathologies` attribute — mirrors the real autoencoder,
    which has no classification head. `layer4` is the deepest encoder stage.
    """

    def __init__(self, weights: str) -> None:
        super().__init__()
        self.weights_name = weights
        self.layer4 = nn.Conv2d(1, 1, kernel_size=1)


def _install_torchxrayvision_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace `torchxrayvision` with a stub for fast unit testing.

    The seam imports `torchxrayvision` lazily inside `load_classifier`
    (and inside the preprocess factory), so we can install a stub module
    in `sys.modules` before the call and the production code path picks
    it up transparently.
    """

    stub = types.ModuleType("torchxrayvision")
    stub.models = types.SimpleNamespace(
        DenseNet=_StubDenseNet, ResNet=_StubResNet
    )
    stub.autoencoders = types.SimpleNamespace(ResNetAE=_StubResNetAE)
    stub.datasets = types.SimpleNamespace(
        normalize=lambda array, max_value: (
            array.astype(np.float32) / float(max_value) * 2048.0 - 1024.0
        )
    )
    monkeypatch.setitem(sys.modules, "torchxrayvision", stub)


def test_load_classifier_returns_bundle_with_expected_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_torchxrayvision_stub(monkeypatch)

    bundle = load_classifier("densenet121-res224-all", device="cpu")

    assert isinstance(bundle, ClassifierBundle)
    assert isinstance(bundle.model, nn.Module)
    assert isinstance(bundle.target_layer, nn.Module)
    # Pneumothorax sits at index 2 in the stub pathology list — the seam
    # must resolve to that exact index, not e.g. 0 by default.
    assert bundle.class_idx == 2
    assert callable(bundle.preprocess)


def test_load_classifier_preprocess_shape_and_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_torchxrayvision_stub(monkeypatch)

    bundle = load_classifier("densenet121-res224-all", device="cpu")

    array = np.full((224, 224), fill_value=128, dtype=np.uint8)
    tensor = bundle.preprocess(array)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 224, 224)
    assert tensor.dtype == torch.float32


def test_load_classifier_rejects_unknown_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_torchxrayvision_stub(monkeypatch)

    with pytest.raises(ValueError, match="Unknown classifier name"):
        load_classifier("definitely-not-a-real-weight-name", device="cpu")


def test_load_classifier_raises_when_pathology_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_torchxrayvision_stub(monkeypatch)

    with pytest.raises(ValueError, match="not available in model pathologies"):
        load_classifier(
            "densenet121-res224-all",
            device="cpu",
            pathology="NotARealPathology",
        )


def test_known_weights_list_contains_baseline_and_a_b_candidates() -> None:
    # Guardrail: the list of in-family candidates documented in the
    # refactor plan must stay in sync with the seam's allow-list, so a
    # typo or accidental rename surfaces here instead of mid-A/B run.
    expected_subset = {
        "densenet121-res224-all",
        "densenet121-res224-chex",
        "densenet121-res224-mimic_ch",
        "densenet121-res224-mimic_nb",
        "densenet121-res224-rsna",
    }
    assert expected_subset.issubset(set(TORCHXRAYVISION_DENSENET_WEIGHTS))


def test_load_classifier_resnet_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    # ResNet path: same `ClassifierBundle` contract as the DenseNet branch,
    # but `target_layer` comes from the inner torchvision ResNet's `layer4`
    # rather than DenseNet's `features.denseblock4`, and the pathology index
    # is resolved against the ResNet's own `pathologies` list.
    _install_torchxrayvision_stub(monkeypatch)

    bundle = load_classifier("resnet50-res512-all", device="cpu")

    assert isinstance(bundle, ClassifierBundle)
    # Pneumothorax sits at index 1 in the ResNet stub — distinct from the
    # DenseNet stub's index 2, so an accidental hard-coded index would fail.
    assert bundle.class_idx == 1
    assert isinstance(bundle.target_layer, nn.Conv2d)


def test_load_classifier_resnet_ae_requires_pathology_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ResNetAE has no classification head; the seam must refuse to silently
    # produce a bundle with a bogus `class_idx`. Default `pathology` arg
    # path is the failure mode we want to catch loudly.
    _install_torchxrayvision_stub(monkeypatch)

    with pytest.raises(ValueError, match="autoencoder, not a pathology"):
        load_classifier("resnetae-101-elastic", device="cpu")


def test_load_classifier_resnet_ae_with_pathology_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Opt-in autoencoder path: caller acknowledges no class head by passing
    # `pathology=None`. Bundle still loads (so future latent/reconstruction
    # experiments share the seam), with sentinel `class_idx=-1`.
    _install_torchxrayvision_stub(monkeypatch)

    bundle = load_classifier(
        "resnetae-101-elastic", device="cpu", pathology=None
    )

    assert isinstance(bundle, ClassifierBundle)
    assert bundle.class_idx == -1
    assert isinstance(bundle.target_layer, nn.Conv2d)


def test_load_classifier_classifier_with_pathology_none_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Inverse guardrail: a true pathology classifier must not be loaded with
    # `pathology=None`. Otherwise downstream scripts would silently run on
    # an undefined class head.
    _install_torchxrayvision_stub(monkeypatch)

    with pytest.raises(ValueError, match="`pathology` must be set"):
        load_classifier(
            "densenet121-res224-all", device="cpu", pathology=None
        )
    with pytest.raises(ValueError, match="`pathology` must be set"):
        load_classifier(
            "resnet50-res512-all", device="cpu", pathology=None
        )


def test_known_weights_lists_contain_resnet_and_autoencoder() -> None:
    assert "resnet50-res512-all" in TORCHXRAYVISION_RESNET_WEIGHTS
    assert "resnetae-101-elastic" in TORCHXRAYVISION_AUTOENCODER_WEIGHTS


@pytest.mark.slow
def test_real_torchxrayvision_weights_load_and_differ() -> None:
    """Two real TorchXRayVision checkpoints must produce different outputs.

    Sanity-check that the seam genuinely swaps weights, not silently caches
    the same file. Skipped by default — opt in with `pytest -m slow`. Needs
    network access on first run for the checkpoint downloads.
    """

    pytest.importorskip("torchxrayvision")
    # Avoid the stub installed by earlier tests in the same process.
    import importlib

    importlib.reload(classifier_module)

    bundle_all = classifier_module.load_classifier(
        "densenet121-res224-all", device="cpu"
    )
    bundle_chex = classifier_module.load_classifier(
        "densenet121-res224-chex", device="cpu"
    )

    torch.manual_seed(0)
    dummy = torch.randn(1, 1, 224, 224)

    with torch.no_grad():
        out_all = bundle_all.model(dummy)
        out_chex = bundle_chex.model(dummy)

    # The Pneumothorax-head score must differ between weights; identical
    # scores would imply the same checkpoint was loaded twice.
    score_all = out_all[0, bundle_all.class_idx].item()
    score_chex = out_chex[0, bundle_chex.class_idx].item()
    assert score_all != pytest.approx(score_chex, abs=1e-6)
