#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


from explainai_thesis.xai import (
    GradCAM,
    SignedAttribution,
    consensus_signed,
    gradient_shap_signed,
    iter_method_views,
    integrated_gradients_signed,
    occlusion_sensitivity_signed,
)
from explainai_thesis.visualization import save_overlay, signed_diverging_overlay
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
    parser.add_argument(
        "--case-filename",
        default="",
        help="Optional exact manifest filename; overrides --case-index when provided.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--gradshap-samples", type=int, default=8)
    parser.add_argument("--gradshap-stdevs", type=float, default=0.02)
    parser.add_argument("--occlusion-patch-size", type=int, default=32)
    parser.add_argument("--occlusion-stride", type=int, default=16)
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


def read_positive_rows(
    manifest_path: Path,
    split: str,
    *,
    positive_only: bool = True,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if positive_only and int(row["label"]) != 1:
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


def safe_source_stem(row: dict[str, str]) -> str:
    stem = Path(row.get("filename") or row["image_path"]).stem
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)


def safe_case_name(case_index: int, row: dict[str, str]) -> str:
    return f"case_{case_index:03d}_{safe_source_stem(row)}"


def save_binary_selection(
    image: torch.Tensor,
    selected_mask: torch.Tensor,
    true_mask: torch.Tensor,
    output_path: Path,
    *,
    negative_style: bool = False,
    neutral_style: bool = False,
    negative_selected_mask: torch.Tensor | None = None,
    neutral_selected_mask: torch.Tensor | None = None,
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

    if neutral_selected_mask is not None:
        neutral_pred = neutral_selected_mask.detach().cpu().bool().numpy()
        rgb[neutral_pred] = 0.50 * rgb[neutral_pred] + 0.50 * NEUTRAL_IMPACT_COLOR

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
    if is_negative_method(method_name):
        return "blue"
    if method_name.endswith("_magnitude"):
        return "neutral"
    return "red"


def selected_pixel_counts(
    selected_mask: torch.Tensor,
    true_mask: torch.Tensor,
) -> dict[str, int]:
    selected = selected_mask.bool()
    true = true_mask.bool()
    intersection = selected & true
    union = selected | true
    return {
        "selected_pixel_count": int(selected.sum().item()),
        "mask_pixel_count": int(true.sum().item()),
        "intersection_pixel_count": int(intersection.sum().item()),
        "union_pixel_count": int(union.sum().item()),
    }


def negative_evidence_metrics(
    heatmap: torch.Tensor,
    true_mask: torch.Tensor,
    fraction: float,
) -> dict[str, float]:
    selected = threshold_top_fraction(heatmap, fraction=fraction)
    true = true_mask.bool()
    selected_count = selected.sum().float()
    if selected_count.item() == 0:
        return {
            "negative_mask_overlap_fraction": 0.0,
            "negative_mask_avoidance_fraction": 0.0,
        }
    overlap = (selected & true).sum().float() / selected_count
    return {
        "negative_mask_overlap_fraction": overlap.item(),
        "negative_mask_avoidance_fraction": (1.0 - overlap).item(),
    }


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
    rows = read_positive_rows(
        Path(args.manifest),
        args.split,
        positive_only=not bool(args.case_filename),
    )
    if not rows:
        raise RuntimeError(
            f"No positive rows with masks found in {args.manifest} for split={args.split}.")
    if args.case_filename:
        matches = [
            (idx, row) for idx, row in enumerate(rows)
            if row.get("filename", Path(row["image_path"]).name) == args.case_filename
        ]
        if not matches:
            raise ValueError(
                f"case-filename={args.case_filename!r} was not found among positive masked cases for split={args.split}."
            )
        case_index, row = matches[0]
    else:
        if not 0 <= args.case_index < len(rows):
            raise ValueError(
                f"case-index must be in [0, {len(rows) - 1}] for split={args.split}.")
        case_index = args.case_index
        row = rows[case_index]
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

    cam_attr = gradcam.signed(model_input, class_idx=class_idx)
    cam_plus_plus_attr = gradcam.signed(
        model_input, class_idx=class_idx, variant="grad_cam_plus_plus")
    ig_attr = integrated_gradients_signed(
        model, model_input, class_idx=class_idx, steps=args.ig_steps)
    gradient_shap_attr = gradient_shap_signed(
        model,
        model_input,
        class_idx=class_idx,
        samples=args.gradshap_samples,
        stdevs=args.gradshap_stdevs,
    )
    occlusion_attr = occlusion_sensitivity_signed(
        model,
        model_input,
        class_idx=class_idx,
        patch_size=args.occlusion_patch_size,
        stride=args.occlusion_stride,
    )
    consensus_attr = consensus_signed([cam_attr, ig_attr, gradient_shap_attr, occlusion_attr])
    gradcam.remove_hooks()

    signed_attributions: dict[str, SignedAttribution] = {
        "grad_cam": cam_attr,
        "grad_cam_plus_plus": cam_plus_plus_attr,
        "integrated_gradients": ig_attr,
        "gradient_shap": gradient_shap_attr,
        "occlusion": occlusion_attr,
        "consensus": consensus_attr,
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
    source_stem = safe_source_stem(row)
    case_dir = output_dir / safe_case_name(args.case_index, row)
    case_dir.mkdir(parents=True, exist_ok=True)
    for method_view in iter_method_views(signed_attributions):
        method_name = method_view.method
        heatmap = method_view.heatmap
        view_kind = method_view.view
        family = method_view.family
        overlay_path = case_dir / f"{source_stem}_{method_name}_continuous_heatmap.png"
        if view_kind == "signed":
            signed_diverging_overlay(image, heatmap, mask, overlay_path)
        else:
            save_overlay(
                image,
                heatmap,
                mask,
                overlay_path,
                heatmap_color=overlay_color_for_method(method_name),
            )

        binary_paths: list[Path] = []
        binary_captions: list[str] = []
        for fraction in fractions:
            metrics_input = (
                signed_attributions[family].magnitude
                if view_kind == "signed"
                else heatmap
            )
            selected = threshold_top_fraction(metrics_input, fraction=fraction)
            metrics = localization_metrics(metrics_input, mask, fraction=fraction)
            negative_metrics: dict[str, str | float] = {
                "negative_mask_overlap_fraction": "",
                "negative_mask_avoidance_fraction": "",
            }
            if view_kind == "negative":
                negative_metrics = {
                    key: round(value, 6)
                    for key, value in negative_evidence_metrics(
                        heatmap, mask, fraction
                    ).items()
                }
            metric_rows.append(
                {
                    "filename": row.get("filename", Path(row["image_path"]).name),
                    "source_stem": source_stem,
                    "image_path": row["image_path"],
                    "mask_path": row["mask_path"],
                    "method": method_name,
                    "view": view_kind,
                    "family": family,
                    "metric_component": view_kind,
                    "top_fraction": round(fraction, 6),
                    "top_fraction_percent": int(round(fraction * 100)),
                    **selected_pixel_counts(selected, mask),
                    **{key: round(value, 6) for key, value in metrics.items()},
                    **negative_metrics,
                }
            )
            image_path = case_dir / f"{source_stem}_{method_name}_selected_top_{int(round(fraction * 100)):02d}pct.png"
            save_binary_selection(
                image,
                selected,
                mask,
                image_path,
                negative_style=view_kind == "negative",
                neutral_style=view_kind == "magnitude",
            )
            binary_paths.append(image_path)
            binary_captions.append(
                f"top {fraction:.0%} | Dice {metrics['dice']:.3f} | IoU {metrics['iou']:.3f}"
            )
        make_contact_sheet(binary_paths, binary_captions,
                           case_dir / f"{source_stem}_{method_name}_threshold_sweep_panel.png")

    write_rows(output_dir / "threshold_metrics.csv", metric_rows)
    print(f"Single-image threshold visualization complete on {device}.")
    print(f"Output directory: {output_dir}")
    print(f"Case: {row.get('filename', Path(row['image_path']).name)}")


if __name__ == "__main__":
    main()
