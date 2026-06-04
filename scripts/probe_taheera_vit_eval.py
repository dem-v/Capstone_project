#!/usr/bin/env python3
"""Exploratory probe: evaluate the taheera/vit-in1k-chestxray14 ViT CXR
classifier on SIIM pneumothorax positives with all three input-space XAI
methods (Integrated Gradients, GradientSHAP, Occlusion), producing per-case
image panels and localization metrics.

NOT a thesis experiment — it is an isolated, uncalibrated read on how a
drop-in transformer CXR classifier behaves in the existing pipeline. Masks
are aligned to the model's Resize->CenterCrop geometry so IoU / Dice /
pointing-hit / precision are computed against the same 224x224 frame the
heatmaps live in. Output is isolated under outputs/iter_60_taheera_vit_eval/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from PIL import Image

from explainai_thesis.metrics import localization_metrics
from explainai_thesis.xai import (
    gradient_shap_signed,
    integrated_gradients_signed,
    normalize_map,
    occlusion_sensitivity_signed,
)

MODEL_ID = "taheera/vit-in1k-chestxray14"
TARGET_LABEL = "Pneumothorax"
METRICS = ("iou", "dice", "pointing_hit", "precision_at_fraction")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/cxr_pneumothorax_manifest.csv")
    p.add_argument("--output-dir", default="outputs/iter_60_taheera_vit_eval")
    p.add_argument("--max-cases", type=int, default=10)
    p.add_argument("--fraction", type=float, default=0.10)
    p.add_argument("--ig-steps", type=int, default=32)
    p.add_argument("--gradshap-samples", type=int, default=16)
    p.add_argument("--gradshap-stdevs", type=float, default=0.02)
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


def build_mask_transform(input_size: int, crop_pct: float) -> T.Compose:
    """Mirror timm's eval geometry (Resize shorter edge -> CenterCrop) with
    NEAREST interpolation so the mask aligns with the model input frame."""
    resize = int(math.floor(input_size / crop_pct))
    return T.Compose([
        T.Resize(resize, interpolation=T.InterpolationMode.NEAREST),
        T.CenterCrop(input_size),
    ])


def overlay_axis(ax, gray: np.ndarray, heat: np.ndarray | None, mask: np.ndarray, title: str) -> None:
    ax.imshow(gray, cmap="gray")
    if heat is not None:
        ax.imshow(heat, cmap="jet", alpha=0.45)
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors="lime", linewidths=1.0)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    target_idx, labels = resolve_target_index(MODEL_ID)
    print(f"[eval] {MODEL_ID}: {len(labels)} labels; {TARGET_LABEL} -> index {target_idx}")

    model = timm.create_model(f"hf_hub:{MODEL_ID}", pretrained=True).eval().to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    img_tf = timm.data.create_transform(**cfg, is_training=False)
    input_size = cfg["input_size"][-1]
    mask_tf = build_mask_transform(input_size, cfg.get("crop_pct", 0.9))

    manifest = find_manifest(args.manifest)
    per_case: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    used = 0
    with manifest.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if used >= args.max_cases:
                break
            if str(row.get("label")) != "1" or not row.get("mask_path"):
                continue
            mask_pil = Image.open(row["mask_path"]).convert("L")
            mask = torch.from_numpy(np.asarray(mask_tf(mask_pil)) > 0)
            if not bool(mask.any()):  # lesion cropped out by center-crop; skip
                continue

            img = Image.open(row["image_path"]).convert("RGB")
            x = img_tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.sigmoid(model(x)[0, target_idx]).item()

            attrs = {
                "integrated_gradients": integrated_gradients_signed(
                    model, x, class_idx=target_idx, steps=args.ig_steps),
                "gradient_shap": gradient_shap_signed(
                    model, x, class_idx=target_idx,
                    samples=args.gradshap_samples, stdevs=args.gradshap_stdevs),
                "occlusion": occlusion_sensitivity_signed(
                    model, x, class_idx=target_idx,
                    patch_size=args.occlusion_patch_size, stride=args.occlusion_stride),
            }
            heatmaps = {m: normalize_map(a.positive.cpu()) for m, a in attrs.items()}
            metrics = {
                m: localization_metrics(h, mask, fraction=args.fraction)
                for m, h in heatmaps.items()
            }
            for m, md in metrics.items():
                metric_rows.append({
                    "case": used, "filename": row["filename"], "method": m,
                    "pneumothorax_prob": round(prob, 4), "fraction": args.fraction,
                    **{k: round(float(v), 6) for k, v in md.items()},
                })

            # Per-case panel: source | mask | IG | GradientSHAP | Occlusion.
            gray = (x[0].cpu() * 0.5 + 0.5).clamp(0, 1).mean(0).numpy()
            mask_np = mask.numpy()
            fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), constrained_layout=True)
            overlay_axis(axes[0], gray, None, np.zeros_like(mask_np), f"{row['filename']}\nP(ptx)={prob:.3f}")
            overlay_axis(axes[1], gray, None, mask_np, "ground-truth mask")
            for ax, m in zip(axes[2:], ("integrated_gradients", "gradient_shap", "occlusion")):
                md = metrics[m]
                overlay_axis(ax, gray, heatmaps[m].numpy(), mask_np,
                             f"{m}\nDice={md['dice']:.3f} hit={int(md['pointing_hit'])}")
            fig.savefig(out / f"case_{used:02d}_{Path(row['filename']).stem}_panel.png", dpi=130)
            plt.close(fig)

            per_case.append({"filename": row["filename"], "prob": prob})
            print(f"[eval] case {used}: {row['filename']} P(ptx)={prob:.3f} "
                  + " ".join(f"{m[:4]}:D={metrics[m]['dice']:.3f}/hit={int(metrics[m]['pointing_hit'])}"
                             for m in attrs))
            used += 1

    if not metric_rows:
        raise RuntimeError("No usable positive masked cases found.")

    # Aggregate per method.
    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in metric_rows:
        for k in METRICS:
            agg[str(r["method"])][k].append(float(r[k]))
    summary_rows = []
    for m in ("integrated_gradients", "gradient_shap", "occlusion"):
        means = {k: round(float(np.mean(agg[m][k])), 4) for k in METRICS}
        summary_rows.append({"method": m, "n_cases": used, **means})

    with (out / "metrics_per_case.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(metric_rows[0].keys()))
        w.writeheader(); w.writerows(metric_rows)
    with (out / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    probs = [c["prob"] for c in per_case]
    print(f"\n[eval] {used} positive masked cases | mean P(pneumothorax)={np.mean(probs):.3f} "
          f"| predicted-positive@0.5: {sum(p >= 0.5 for p in probs)}/{used}")
    print("[eval] aggregate localization means (uncalibrated, fraction="
          f"{args.fraction}):")
    for s in summary_rows:
        print(f"   {s['method']:22s} IoU={s['iou']:.4f} Dice={s['dice']:.4f} "
              f"pointing={s['pointing_hit']:.4f} precision={s['precision_at_fraction']:.4f}")
    print(f"[eval] panels + CSVs in {out}/ (isolated, not a thesis artifact).")


if __name__ == "__main__":
    main()
