#!/usr/bin/env python3
"""Throwaway structural probe: does a HuggingFace ViT CXR classifier with a
native NIH ChestX-ray14 head (taheera/vit-in1k-chestxray14) plug into the
existing input-space XAI pipeline the same way the DifeiT CT model did?

This is NOT a thesis experiment. It only verifies:
  1. the model loads via timm and exposes a Pneumothorax logit,
  2. the Pneumothorax label index is read/verified from config.json,
  3. `occlusion_sensitivity_signed` (the real pipeline function) attributes
     against that logit and returns a sane 224x224 heatmap.

Output is isolated under outputs/iter_60_taheera_vit_probe/. No thesis
artifact, manifest, or locked config is modified.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from explainai_thesis.xai import normalize_map, occlusion_sensitivity_signed

MODEL_ID = "taheera/vit-in1k-chestxray14"
TARGET_LABEL = "Pneumothorax"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/cxr_pneumothorax_manifest.csv")
    p.add_argument("--output-dir", default="outputs/iter_60_taheera_vit_probe")
    p.add_argument("--max-cases", type=int, default=3)
    p.add_argument("--occlusion-patch-size", type=int, default=32)
    p.add_argument("--occlusion-stride", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def resolve_target_index(repo: str) -> tuple[int, list[str]]:
    cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
    labels = cfg.get("labels") or cfg.get("label_names")
    if not labels or TARGET_LABEL not in labels:
        raise RuntimeError(f"{TARGET_LABEL!r} not in config labels: {labels}")
    return labels.index(TARGET_LABEL), labels


def find_manifest(path_arg: str) -> Path:
    p = Path(path_arg)
    if p.is_file():
        return p
    matches = sorted(Path("data").glob("*manifest*.csv"))
    if not matches:
        raise FileNotFoundError("No CXR manifest found under data/.")
    return matches[0]


def positive_rows(manifest: Path, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("label")) == "1":
                rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def save_overlay(input_chw: torch.Tensor, heatmap_2d: np.ndarray, out_path: Path) -> None:
    # Denormalize the mean=std=0.5 input back to [0,1] grayscale for display.
    disp = (input_chw[0].cpu() * 0.5 + 0.5).clamp(0, 1).mean(0).numpy()
    rgb = np.stack([disp, disp, disp], axis=-1)
    rgb[..., 0] = np.clip(rgb[..., 0] + 0.6 * heatmap_2d, 0, 1)  # red = positive evidence
    Image.fromarray((rgb * 255).astype(np.uint8)).save(out_path)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    target_idx, labels = resolve_target_index(MODEL_ID)
    print(f"[probe] {MODEL_ID}: {len(labels)} labels; {TARGET_LABEL} -> index {target_idx}")

    model = timm.create_model(f"hf_hub:{MODEL_ID}", pretrained=True).eval().to(device)
    assert model.num_classes == len(labels), (model.num_classes, len(labels))
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg, is_training=False)
    print(f"[probe] data config: {cfg}")

    manifest = find_manifest(args.manifest)
    rows = positive_rows(manifest, args.max_cases)
    if not rows:
        raise RuntimeError(f"No positive (label==1) rows in {manifest}.")

    results: list[dict[str, object]] = []
    for i, row in enumerate(rows):
        img = Image.open(row["image_path"]).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
        prob = torch.sigmoid(logits[0, target_idx]).item()

        attr = occlusion_sensitivity_signed(
            model, x, class_idx=target_idx,
            patch_size=args.occlusion_patch_size, stride=args.occlusion_stride,
        )
        pos = normalize_map(attr.positive.cpu()).numpy()
        overlay_path = out / f"case_{i:02d}_{Path(row['filename']).stem}_pneumothorax_occlusion.png"
        save_overlay(x, pos, overlay_path)

        rec = {
            "case": i, "filename": row["filename"], "label": row["label"],
            "pneumothorax_prob": round(prob, 4),
            "logits_shape": tuple(logits.shape),
            "heatmap_shape": tuple(pos.shape),
            "heatmap_min": round(float(pos.min()), 4),
            "heatmap_max": round(float(pos.max()), 4),
            "peak_yx": tuple(int(v) for v in np.unravel_index(int(pos.argmax()), pos.shape)),
            "overlay": overlay_path.name,
        }
        results.append(rec)
        print(f"[probe] case {i}: {row['filename']} | P(pneumothorax)={prob:.4f} | "
              f"heatmap {pos.shape} range[{pos.min():.3f},{pos.max():.3f}] peak={rec['peak_yx']}")

    with (out / "probe_results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print(f"\n[probe] STRUCTURE OK: ViT CXR-14 logit -> occlusion heatmap pipeline works.")
    print(f"[probe] outputs in {out}/ (overlays + probe_results.csv). Not a thesis artifact.")


if __name__ == "__main__":
    main()
