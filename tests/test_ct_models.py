from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from explainai_thesis.ct.models import (
    HemorrhageBinaryHead,
    resolve_label_indices,
)


# id2label of the actual DifeiT checkpoint (gate-verified 2026-06-02).
DIFEIT_ID2LABEL = {
    0: "epidural",
    1: "intraparenchymal",
    2: "intraventricular",
    3: "normal",
    4: "subarachnoid",
    5: "subdural",
}


class _StubBackbone(nn.Module):
    """Returns fixed logits regardless of input, mimicking the HF model's
    `.logits` output object."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self._logits = logits

    def forward(self, pixel_values: torch.Tensor):  # noqa: ARG002
        batch = pixel_values.shape[0]
        out = self._logits.unsqueeze(0).expand(batch, -1)
        return type("Out", (), {"logits": out})()


def test_resolve_label_indices_difeit() -> None:
    normal_index, subtype_indices = resolve_label_indices(DIFEIT_ID2LABEL)
    assert normal_index == 3
    assert subtype_indices == [0, 1, 2, 4, 5]


def test_resolve_label_indices_rejects_no_normal() -> None:
    with pytest.raises(ValueError, match="No 'normal' class"):
        resolve_label_indices({0: "epidural", 1: "subdural"})


def test_resolve_label_indices_rejects_no_subtypes() -> None:
    with pytest.raises(ValueError, match="No hemorrhage-subtype"):
        resolve_label_indices({0: "normal"})


def test_binary_head_matches_one_minus_p_normal() -> None:
    # The core identity: sigmoid(logsumexp(subtypes) - normal) == 1 - softmax(normal).
    rng = np.random.default_rng(0)
    for _ in range(20):
        logits = torch.tensor(rng.normal(size=6), dtype=torch.float32)
        head = HemorrhageBinaryHead(_StubBackbone(logits), normal_index=3, subtype_indices=[0, 1, 2, 4, 5])
        out = head(torch.zeros(1, 3, 224, 224))
        assert out.shape == (1, 1)
        hemorrhage_via_head = float(torch.sigmoid(out[0, 0]))
        one_minus_p_normal = float(1.0 - torch.softmax(logits, dim=-1)[3])
        assert hemorrhage_via_head == pytest.approx(one_minus_p_normal, abs=1e-6)


def test_binary_head_is_differentiable() -> None:
    # IG/GradientSHAP need gradients to flow back to the input.
    logits = torch.tensor([0.5, 1.0, -0.3, 0.2, 0.1, -1.0])

    class _GradBackbone(nn.Module):
        def forward(self, pixel_values):
            scaled = logits * pixel_values.mean()
            return type("Out", (), {"logits": scaled.unsqueeze(0)})()

    head = HemorrhageBinaryHead(_GradBackbone(), normal_index=3, subtype_indices=[0, 1, 2, 4, 5])
    x = torch.full((1, 3, 8, 8), 0.7, requires_grad=True)
    out = head(x)
    out[0, 0].backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_binary_head_handles_batch() -> None:
    logits = torch.tensor([0.5, 1.0, -0.3, 0.2, 0.1, -1.0])
    head = HemorrhageBinaryHead(_StubBackbone(logits), normal_index=3, subtype_indices=[0, 1, 2, 4, 5])
    out = head(torch.zeros(4, 3, 224, 224))
    assert out.shape == (4, 1)
    assert torch.allclose(out, out[0])  # stub: identical across the batch
