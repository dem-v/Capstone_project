"""CT hemorrhage classifier loader for the Phase 5.4 pilot.

Wraps the DifeiT RSNA-IHD ViT (single-label softmax over
{epidural, intraparenchymal, intraventricular, normal, subarachnoid,
subdural}) and exposes a single **binary hemorrhage logit** at
`output[:, 0]`, defined as

    hemorrhage_logit = logsumexp(subtype_logits) - logit_normal

so that `sigmoid(output[:, 0]) == 1 - P(normal)` exactly (a softmax
identity, verified numerically on real data: 0.991754 both ways). This is
the binary attribution target chosen for the pilot — the CT analogue of
the CXR binary `Pneumothorax` head.

With this wrapper the input-space XAI methods whose `xai.py`
implementations are byte-identical across modalities — Integrated
Gradients, GradientSHAP, Occlusion — target `class_idx = 0` unchanged, and
the faithfulness sigmoid yields `1 - P(normal)`. CAM-family methods
(Grad-CAM/++, Eigen-CAM, Score-CAM) are intentionally NOT supported here:
they require 4-D conv activations and a ViT would need a separate
token→grid reimplementation, which would make the cross-modality
comparison uncontrolled. See `docs/refactor_plan.md` § "Phase 5.4".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn

from .io import BRAIN_WINDOW_WIDTH, preprocess_ct_slice


DIFEIT_MODEL_ID = "DifeiT/rsna-intracranial-hemorrhage-detection"

# Preprocess signature: a 2-D HU slice (float numpy) -> [1, 3, H, W] tensor
# in the model's normalized input space (CPU). Callers move to device.
CTPreprocessFn = Callable[[np.ndarray], torch.Tensor]


@dataclass(frozen=True)
class CTClassifierBundle:
    """Everything the input-space XAI pipeline needs for the CT pilot.

    Deliberately omits `target_layer`: the supported methods (IG,
    GradientSHAP, Occlusion) are input-space and need no conv target.
    `class_idx` is always 0 — the wrapper exposes exactly one logit.
    """

    model: nn.Module
    class_idx: int
    preprocess: CTPreprocessFn
    id2label: dict[int, str]


class HemorrhageBinaryHead(nn.Module):
    """Reduce a multi-class hemorrhage classifier to a single binary logit.

    `forward(pixel_values)` returns shape `(B, 1)` where column 0 is
    `logsumexp(subtype_logits) - logit_normal`. Fully differentiable, so
    the input-gradient methods (IG, GradientSHAP) backprop through it; the
    occlusion method only needs the forward score.
    """

    def __init__(
        self,
        backbone: nn.Module,
        normal_index: int,
        subtype_indices: list[int],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.normal_index = int(normal_index)
        self.register_buffer(
            "subtype_indices",
            torch.tensor(sorted(subtype_indices), dtype=torch.long),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(pixel_values=pixel_values).logits
        normal = logits[:, self.normal_index]
        subtype = logits.index_select(1, self.subtype_indices)
        hemorrhage_logit = torch.logsumexp(subtype, dim=1) - normal
        return hemorrhage_logit.unsqueeze(1)


def resolve_label_indices(id2label: dict[int, str]) -> tuple[int, list[int]]:
    """Return (normal_index, subtype_indices) from a hub id2label mapping.

    Robust to label ordering: locates the `normal` head by name; every
    other head is treated as a hemorrhage subtype.
    """
    normal_index: int | None = None
    subtype_indices: list[int] = []
    for idx, label in id2label.items():
        if str(label).strip().lower() == "normal":
            normal_index = int(idx)
        else:
            subtype_indices.append(int(idx))
    if normal_index is None:
        raise ValueError(f"No 'normal' class found in id2label: {id2label}")
    if not subtype_indices:
        raise ValueError(f"No hemorrhage-subtype classes found in id2label: {id2label}")
    return normal_index, sorted(subtype_indices)


def _difeit_preprocess_factory(processor, window_width: float) -> CTPreprocessFn:
    # Route through the same processor used in the verified gate so the
    # preprocessing is byte-identical to the 0.9918 verification: window
    # (locked WL=40/WW=80) -> uint8 -> processor resize+rescale+normalize.
    def preprocess(slice_hu: np.ndarray) -> torch.Tensor:
        display = preprocess_ct_slice(slice_hu, size=224, window_width=window_width)
        as_uint8 = (np.clip(display, 0.0, 1.0) * 255).astype(np.uint8)
        return processor(images=as_uint8, return_tensors="pt")["pixel_values"]

    return preprocess


def load_ct_classifier(
    name: str = DIFEIT_MODEL_ID,
    device: torch.device | str = "cpu",
    *,
    window_width: float = BRAIN_WINDOW_WIDTH,
) -> CTClassifierBundle:
    """Load the CT hemorrhage classifier and return the pilot bundle.

    Only `DifeiT/rsna-intracranial-hemorrhage-detection` is supported
    initially (the gate-verified primary). Backup checkpoints are wired in
    here at integration time if the primary regresses.
    """
    if name != DIFEIT_MODEL_ID:
        raise ValueError(
            f"Unsupported CT classifier {name!r}. Only {DIFEIT_MODEL_ID!r} is "
            "wired in for the Phase 5.4 pilot."
        )
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    backbone = AutoModelForImageClassification.from_pretrained(name).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(name)
    id2label = {int(k): str(v) for k, v in backbone.config.id2label.items()}
    normal_index, subtype_indices = resolve_label_indices(id2label)
    model = HemorrhageBinaryHead(backbone, normal_index, subtype_indices).to(device).eval()
    return CTClassifierBundle(
        model=model,
        class_idx=0,
        preprocess=_difeit_preprocess_factory(processor, window_width),
        id2label=id2label,
    )
