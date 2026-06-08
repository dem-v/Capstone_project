#!/usr/bin/env python3
"""Phase 5.4 hour-1 gate, steps 4-5: real-slice end-to-end verification.

Loads one hemorrhage-positive and one normal slice from the PhysioNet
ct-ich NIfTI volumes, windows them through the locked `ct/io.py` brain
window, runs them through the DifeiT ViT, and reports the binary
hemorrhage target (1 - P(normal)) plus the mask alignment. Confirms the
model produces meaningful, hemorrhage-discriminating signal on real data
before any Branch-A module is built. GPU-free.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from explainai_thesis.ct.io import (
    extract_slice,
    load_nifti_volume,
    preprocess_ct_slice,
)

MODEL_ID = "DifeiT/rsna-intracranial-hemorrhage-detection"
DATA = Path("data_local/physionet.org/files/ct-ich/1.3.1")
NORMAL_IDX = 3  # id2label index of the `normal` (no-hemorrhage) class.


def run_slice(model, processor, id2label, volume, mask_vol, slice_number, window_width):
    # CSV SliceNumber is 1-indexed; volume slices are on axis=2.
    idx = int(slice_number) - 1
    slice_hu = extract_slice(volume, idx, axis=2)
    processed = preprocess_ct_slice(slice_hu, size=224, window_width=window_width)
    pil_uint8 = (np.clip(processed, 0.0, 1.0) * 255).astype(np.uint8)
    inputs = processor(images=pil_uint8, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    p_normal = float(probs[NORMAL_IDX])
    hemorrhage = 1.0 - p_normal
    top = int(torch.argmax(probs))
    mask_slice = extract_slice(mask_vol, idx, axis=2)
    nonzero = int((mask_slice > 0).sum())
    print(
        f"  slice#{slice_number} (idx {idx}, WW={window_width}): "
        f"1-P(normal)={hemorrhage:.4f} | P(normal)={p_normal:.4f} | "
        f"argmax={id2label[top]!r} | mask_nonzero_px={nonzero} "
        f"| mask_shape={mask_slice.shape}"
    )
    return hemorrhage, nonzero


def main() -> None:
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    model = AutoModelForImageClassification.from_pretrained(MODEL_ID).eval()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    id2label = model.config.id2label

    patient = 49
    volume = load_nifti_volume(DATA / "ct_scans" / f"{patient:03d}.nii")
    mask_vol = load_nifti_volume(DATA / "masks" / f"{patient:03d}.nii")
    print(f"patient {patient}: volume shape {volume.shape}, mask shape {mask_vol.shape}")
    print(f"HU range: [{volume.min():.0f}, {volume.max():.0f}]  (raw Hounsfield preserved)")

    print("POSITIVE slice (Epidural hemorrhage, slice 14):")
    pos80, pos_mask = run_slice(model, processor, id2label, volume, mask_vol, 14, 80.0)
    pos120, _ = run_slice(model, processor, id2label, volume, mask_vol, 14, 120.0)

    print("NORMAL slice (slice 1):")
    neg80, neg_mask = run_slice(model, processor, id2label, volume, mask_vol, 1, 80.0)

    print("\n=== gate verdict (steps 4-5) ===")
    checks = {
        "hemorrhage_prob_higher_on_positive (WW=80)": pos80 > neg80,
        "positive_mask_has_nonzero_pixels": pos_mask > 0,
        "normal_mask_empty": neg_mask == 0,
        "hu_preserved (range exceeds [0,1])": float(volume.max()) > 1.5,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"  (windowing sensitivity: WW=80 -> {pos80:.4f}, WW=120 -> {pos120:.4f})")
    print(f"\nALL PASS: {all(checks.values())}")


if __name__ == "__main__":
    main()
