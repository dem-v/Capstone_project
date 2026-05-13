#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from explainai_thesis.xai import GradCAM, consensus_heatmap, integrated_gradients
from explainai_thesis.visualization import save_overlay
from explainai_thesis.metrics import localization_metrics, normalize_map, threshold_top_fraction
from PIL import Image, ImageDraw
import torchxrayvision as xrv
import torch
import numpy as np


NEUTRAL_IMPACT_COLOR = np.array([180, 0, 255], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize top-fraction threshold selection for one CXR XAI case."
    )
    parser.add_argument(
        "--manifest", default="data/cxr_pneumothorax_manifest.csv")
    parser.add_argument(
        "--output-dir", default="outputs/iter_02_threshold_selection_single_image")
    parser.add_argument("--weights", default="densenet121-res224-all")
    parser.add_argument("--split", default="train",
                        choices=["train", "test", "any"])
    parser.add_argument("--case-index", type=int, default=0,
                        help="Zero-based index among positive masked cases.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument(
        "--fractions",
        default="0.05,0.10,0.15,0.20,0.25,0.30",
        help="Comma-separated top-fractions to visualize.",
    )
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(choice)


def read_positive_rows(manifest_path: Path, split: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["label"]) != 1:
                continue
            if split != "any" and row.get("split") != split:
                continue
            if not row.get("mask_path"):
                continue
            rows.append(row)
    return rows


def load_image(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("L").resize(
        (image_size, image_size), Image.BILINEAR)
    array = np.asarray(image)
    normalized = xrv.datasets.normalize(array, 255)
    return torch.from_numpy(normalized).unsqueeze(0).float()


def load_mask(path: Path, image_size: int) -> torch.Tensor:
    mask = Image.open(path).convert("L").resize(
        (image_size, image_size), Image.NEAREST)
    return torch.from_numpy(np.asarray(mask) > 0)


def pathology_index(model: torch.nn.Module, pathology: str) -> int:
    pathologies = list(model.pathologies)
    try:
        return pathologies.index(pathology)
    except ValueError as exc:
        raise ValueError(
            f"{pathology!r} is not available in model pathologies: {pathologies}") from exc


def parse_fractions(raw: str) -> list[float]:
    fractions = [float(value.strip())
                 for value in raw.split(",") if value.strip()]
    if not fractions:
        raise ValueError("At least one fraction is required.")
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("All fractions must be in (0, 1].")
    return fractions


def write_rows(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_binary_selection(
    image: torch.Tensor,
    selected_mask: torch.Tensor,
    true_mask: torch.Tensor,
    output_path: Path,
    *,
    negative_style: bool = False,
    neutral_style: bool = False,
    negative_selected_mask: torch.Tensor | None = None,
) -> None:
    base = image.detach().cpu()
    if base.ndim == 3:
        base = base[0]
    gray = (normalize_map(base).numpy() * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    pred = selected_mask.detach().cpu().bool().numpy()
    true = true_mask.detach().cpu().bool().numpy()
    tp = pred & true
    fp = pred & ~true
    fn = ~pred & true

    if negative_selected_mask is not None:
        negative_pred = negative_selected_mask.detach().cpu().bool().numpy()
        negative_inside = negative_pred & true
        negative_outside = negative_pred & ~true
        rgb[negative_outside] = 0.50 * rgb[negative_outside] + \
            0.50 * np.array([0, 0, 255], dtype=np.float32)
        rgb[negative_inside] = 0.35 * rgb[negative_inside] + \
            0.65 * np.array([0, 255, 255], dtype=np.float32)

    if negative_style:
        selected_outside = np.array([0, 0, 255], dtype=np.float32)
        selected_inside = np.array([0, 255, 255], dtype=np.float32)
    elif neutral_style:
        selected_outside = NEUTRAL_IMPACT_COLOR
        selected_inside = NEUTRAL_IMPACT_COLOR
    else:
        selected_outside = np.array([255, 0, 0], dtype=np.float32)
        selected_inside = np.array([255, 255, 0], dtype=np.float32)

    rgb[fp] = 0.50 * rgb[fp] + 0.50 * selected_outside
    rgb[tp] = 0.35 * rgb[tp] + 0.65 * selected_inside
    rgb[fn] = 0.50 * rgb[fn] + 0.50 * np.array([0, 255, 0], dtype=np.float32)

    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(output_path)


def is_negative_method(method_name: str) -> bool:
    return method_name.endswith("_negative")


def overlay_color_for_method(method_name: str) -> str:
    if method_name == "integrated_gradients":
        return "neutral"
    if is_negative_method(method_name):
        return "blue"
    return "red"


def make_contact_sheet(image_paths: list[Path], captions: list[str], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not images:
        return
    width, height = images[0].size
    caption_height = 28
    sheet = Image.new("RGB", (width * len(images),
                      height + caption_height), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, image in enumerate(images):
        x = idx * width
        sheet.paste(image, (x, 0))
        draw.text((x + 6, height + 7), captions[idx], fill=(0, 0, 0))
    sheet.save(output_path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fractions = parse_fractions(args.fractions)
    rows = read_positive_rows(Path(args.manifest), args.split)
    if not rows:
        raise RuntimeError(
            f"No positive rows with masks found in {args.manifest} for split={args.split}.")
    if not 0 <= args.case_index < len(rows):
        raise ValueError(
            f"case-index must be in [0, {len(rows) - 1}] for split={args.split}.")

    row = rows[args.case_index]
    image = load_image(Path(row["image_path"]), args.image_size)
    mask = load_mask(Path(row["mask_path"]), args.image_size)

    device = resolve_device(args.device)
    model = xrv.models.DenseNet(weights=args.weights).to(device)
    model.eval()
    class_idx = pathology_index(model, "Pneumothorax")
    gradcam = GradCAM(model, model.features.denseblock4)
    model_input = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(model_input)
        score = float(output[0, class_idx].detach().cpu().item())
        probability = float(torch.sigmoid(
            output[0, class_idx]).detach().cpu().item())

    cam_map = gradcam(model_input, class_idx=class_idx)
    negative_cam_map = gradcam(
        model_input, class_idx=class_idx, polarity="negative")
    ig_map = integrated_gradients(
        model, model_input, class_idx=class_idx, steps=args.ig_steps)
    ig_positive_map = integrated_gradients(
        model, model_input, class_idx=class_idx, steps=args.ig_steps, polarity="positive")
    ig_negative_map = integrated_gradients(
        model, model_input, class_idx=class_idx, steps=args.ig_steps, polarity="negative")
    gradcam.remove_hooks()

    methods = {
        "grad_cam": cam_map,
        "grad_cam_negative": negative_cam_map,
        "integrated_gradients": ig_map,
        "integrated_gradients_positive": ig_positive_map,
        "integrated_gradients_negative": ig_negative_map,
        "integrated_gradients_signed": ig_positive_map,
        "consensus": consensus_heatmap([cam_map, ig_map]),
    }

    metadata_rows = [
        {
            "case_index": args.case_index,
            "filename": row.get("filename", Path(row["image_path"]).name),
            "split": row.get("split", ""),
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "xrv_pneumothorax_score": round(score, 6),
            "xrv_pneumothorax_sigmoid": round(probability, 6),
        }
    ]
    write_rows(output_dir / "case_metadata.csv", metadata_rows)

    metric_rows: list[dict[str, str | int | float]] = []
    for method_name, heatmap in methods.items():
        method_dir = output_dir / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        save_overlay(
            image,
            heatmap,
            mask,
            method_dir / "continuous_heatmap.png",
            heatmap_color=overlay_color_for_method(method_name),
            negative_heatmap=(
                negative_cam_map if method_name == "consensus"
                else ig_negative_map if method_name == "integrated_gradients_signed"
                else None
            ),
        )

        binary_paths: list[Path] = []
        binary_captions: list[str] = []
        for fraction in fractions:
            selected = threshold_top_fraction(heatmap, fraction=fraction)
            metrics = localization_metrics(heatmap, mask, fraction=fraction)
            metric_rows.append(
                {
                    "method": method_name,
                    "top_fraction": round(fraction, 6),
                    **{key: round(value, 6) for key, value in metrics.items()},
                }
            )
            image_path = method_dir / \
                f"selected_top_{int(round(fraction * 100)):02d}pct.png"
            save_binary_selection(
                image,
                selected,
                mask,
                image_path,
                negative_style=is_negative_method(method_name),
                neutral_style=method_name == "integrated_gradients",
                negative_selected_mask=threshold_top_fraction(
                    negative_cam_map, fraction=fraction)
                if method_name == "consensus"
                else threshold_top_fraction(ig_negative_map, fraction=fraction)
                if method_name == "integrated_gradients_signed"
                else None,
            )
            binary_paths.append(image_path)
            binary_captions.append(
                f"top {fraction:.0%} | Dice {metrics['dice']:.3f} | IoU {metrics['iou']:.3f}"
            )
        make_contact_sheet(binary_paths, binary_captions,
                           method_dir / "threshold_sweep_panel.png")

    write_rows(output_dir / "threshold_metrics.csv", metric_rows)
    print(f"Single-image threshold visualization complete on {device}.")
    print(f"Output directory: {output_dir}")
    print(f"Case: {row.get('filename', Path(row['image_path']).name)}")


if __name__ == "__main__":
    main()
