#!/usr/bin/env python3
"""Phase 5.4 hour-1 gate probe (model track, steps 1-2).

GPU-free. Loads the DifeiT RSNA-IHD ViT, inspects its label heads to locate
the binary `any` hemorrhage target, runs one CPU forward pass on a random
224x224x3 image to confirm finite probabilities, reads the image
processor's normalization (which fixes the deferred `brain_window_center`
baseline constant), and reports the license. Prints a Branch-A/Branch-B
verdict for the model track only (the mask track is checked separately).
"""
from __future__ import annotations

import json

import numpy as np
import torch

MODEL_ID = "DifeiT/rsna-intracranial-hemorrhage-detection"


def main() -> None:
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    print(f"[1] Loading {MODEL_ID} ...")
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    cfg = model.config
    print(f"    architecture: {cfg.architectures}")
    print(f"    num_labels:   {cfg.num_labels}")
    print(f"    problem_type: {getattr(cfg, 'problem_type', None)}")
    print(f"    id2label:     {cfg.id2label}")

    # Locate the binary `any` head (case-insensitive).
    any_idx = None
    for idx, label in cfg.id2label.items():
        if str(label).strip().lower() == "any":
            any_idx = int(idx)
    print(f"    'any' head index: {any_idx}")

    # Forward pass on a random 224x224x3 image through the real processor.
    rng = np.random.default_rng(0)
    fake = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    inputs = processor(images=fake, return_tensors="pt")
    pixel = inputs["pixel_values"]
    print(f"[2] processor output pixel_values shape: {tuple(pixel.shape)}, "
          f"range [{float(pixel.min()):.3f}, {float(pixel.max()):.3f}]")
    with torch.inference_mode():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits)[0]  # multi-label -> per-head sigmoid
    finite = bool(torch.isfinite(probs).all())
    print(f"    logits shape: {tuple(logits.shape)}; all finite: {finite}")
    print(f"    sigmoid probs: {[round(float(p), 4) for p in probs]}")
    if any_idx is not None:
        print(f"    p(any) on random input: {float(probs[any_idx]):.4f}")

    # brain_window_center constant for the deferred faithfulness branch:
    # display midpoint 0.5 mapped through the processor normalization.
    mean = getattr(processor, "image_mean", None)
    std = getattr(processor, "image_std", None)
    size = getattr(processor, "size", None)
    print(f"[norm] image_mean={mean} image_std={std} size={size}")
    if mean and std:
        bwc = [round((0.5 - m) / s, 6) for m, s in zip(mean, std)]
        print(f"    brain_window_center_normalized (per channel) = {bwc}")

    # License (best-effort, from the hub model card metadata).
    license_tag = None
    try:
        from huggingface_hub import model_info

        info = model_info(MODEL_ID)
        license_tag = (info.card_data or {}).get("license") if info.card_data else None
    except Exception as exc:  # noqa: BLE001 - best effort
        print(f"    (license lookup failed: {exc})")
    print(f"[license] {license_tag}")

    # Verdict (model track only).
    model_pass = (any_idx is not None) and finite
    print(json.dumps({
        "model_track_pass": model_pass,
        "any_head_index": any_idx,
        "num_labels": cfg.num_labels,
        "license": license_tag,
        "finite_probabilities": finite,
    }, indent=2))


if __name__ == "__main__":
    main()
