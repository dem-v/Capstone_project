#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from explainai_thesis.xai import GradCAM, consensus_heatmap, integrated_gradients
from explainai_thesis.visualization import save_overlay
from explainai_thesis.metrics import (
    localization_metrics,
    normalize_map,
    threshold_top_fraction,
)
from PIL import Image
import torchxrayvision as xrv
import torch
import numpy as np


NEUTRAL_IMPACT_COLOR = np.array([180, 0, 255], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tiny real-data TorchXRayVision explainability pass on SIIM pneumothorax PNGs."
    )
    parser.add_argument(
        "--manifest",
        default="data/cxr_pneumothorax_manifest.csv",
        help="Input manifest CSV.",
    )
    parser.add_argument(
        "--output-dir", default="outputs/cxr_torchxray_smoke", help="Output directory."
    )
    parser.add_argument(
        "--weights",
        default="densenet121-res224-all",
        help="TorchXRayVision DenseNet weights.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test", "any"],
        help="Manifest split to sample.",
    )
    parser.add_argument(
        "--max-positive",
        type=int,
        default=6,
        help="Number of positive cases to evaluate.",
    )
    parser.add_argument("--image-size", type=int,
                        default=224, help="Model input size.")
    parser.add_argument(
        "--ig-steps", type=int, default=16, help="Integrated Gradients steps."
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.15,
        help="Heatmap fraction used for binary metrics.",
    )
    parser.add_argument(
        "--calibrated-fractions",
        default=None,
        help="Optional calibration CSV with method,selected_fraction columns.",
    )
    parser.add_argument(
        "--max-overlays",
        type=int,
        default=12,
        help="Maximum number of cases for which overlay PNGs are exported.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device.",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(choice)


def read_positive_rows(
    manifest_path: Path, split: str, limit: int
) -> list[dict[str, str]]:
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
            if len(rows) >= limit:
                break
    return rows


def load_image(path: Path, image_size: int) -> torch.Tensor:
    image = (
        Image.open(path).convert("L").resize(
            (image_size, image_size), Image.BILINEAR)
    )
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
            f"{pathology!r} is not available in model pathologies: {pathologies}"
        ) from exc


def write_metric_summary(
    metric_rows: list[dict[str, str | int | float]], output_path: Path
) -> None:
    grouped: dict[str, list[dict[str, str | int | float]]] = defaultdict(list)
    for row in metric_rows:
        grouped[str(row["method"])].append(row)

    metric_names = ["iou", "dice", "pointing_hit", "precision_at_fraction"]
    optional_metric_names = [
        "negative_mask_overlap_fraction",
        "negative_mask_avoidance_fraction",
    ]
    fieldnames = ["method", "n"]
    for metric_name in metric_names + optional_metric_names:
        fieldnames.extend([f"{metric_name}_mean", f"{metric_name}_std"])

    summary_rows: list[dict[str, str | int | float]] = []
    for method_name, rows in sorted(grouped.items()):
        summary: dict[str, str | int | float] = {
            "method": method_name,
            "n": len(rows),
        }
        for metric_name in metric_names:
            values = np.asarray([float(row[metric_name])
                                for row in rows], dtype=float)
            summary[f"{metric_name}_mean"] = round(float(values.mean()), 6)
            summary[f"{metric_name}_std"] = round(float(values.std(ddof=0)), 6)
        for metric_name in optional_metric_names:
            values = [row.get(metric_name) for row in rows]
            numeric_values = np.asarray(
                [float(value) for value in values if value != ""],
                dtype=float,
            )
            if numeric_values.size:
                summary[f"{metric_name}_mean"] = round(
                    float(numeric_values.mean()), 6)
                summary[f"{metric_name}_std"] = round(
                    float(numeric_values.std(ddof=0)), 6
                )
            else:
                summary[f"{metric_name}_mean"] = ""
                summary[f"{metric_name}_std"] = ""
        summary_rows.append(summary)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def read_calibrated_fractions(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}

    fractions: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            method = row.get("method")
            selected_fraction = row.get("selected_fraction")
            if method and selected_fraction:
                fractions[method] = float(selected_fraction)
    return fractions


def save_selected_threshold_image(
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
        negative_tp = negative_pred & true
        negative_fp = negative_pred & ~true
        rgb[negative_fp] = 0.50 * rgb[negative_fp] + 0.50 * np.array(
            [0, 0, 255], dtype=np.float32
        )
        rgb[negative_tp] = 0.35 * rgb[negative_tp] + 0.65 * np.array(
            [0, 255, 255], dtype=np.float32
        )

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


def is_negative_method(method_name: str) -> bool:
    return method_name.endswith("_negative")


def safe_case_name(sample_idx: int, row: dict[str, str]) -> str:
    stem = Path(row.get("filename") or row["image_path"]).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    if not safe_stem:
        safe_stem = "xray"
    return f"case_{sample_idx:03d}_{safe_stem}"


def overlay_color_for_method(method_name: str) -> str:
    if method_name == "integrated_gradients":
        return "neutral"
    if is_negative_method(method_name):
        return "blue"
    return "red"


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    calibrated_fractions = read_calibrated_fractions(
        Path(args.calibrated_fractions) if args.calibrated_fractions else None
    )
    rows = read_positive_rows(
        manifest_path, split=args.split, limit=args.max_positive)
    if not rows:
        raise RuntimeError(
            f"No positive rows with masks found in {manifest_path} for split={args.split}."
        )

    model = xrv.models.DenseNet(weights=args.weights).to(device)
    model.eval()
    class_idx = pathology_index(model, "Pneumothorax")
    gradcam = GradCAM(model, model.features.denseblock4)

    metric_rows: list[dict[str, str | int | float]] = []
    for sample_idx, row in enumerate(rows):
        image = load_image(Path(row["image_path"]), args.image_size)
        mask = load_mask(Path(row["mask_path"]), args.image_size)
        model_input = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(model_input)
            score = float(output[0, class_idx].detach().cpu().item())
            probability = float(
                torch.sigmoid(output[0, class_idx]).detach().cpu().item()
            )

        cam_map = gradcam(model_input, class_idx=class_idx)
        negative_cam_map = gradcam(
            model_input, class_idx=class_idx, polarity="negative"
        )
        ig_map = integrated_gradients(
            model, model_input, class_idx=class_idx, steps=args.ig_steps
        )
        ig_positive_map = integrated_gradients(
            model,
            model_input,
            class_idx=class_idx,
            steps=args.ig_steps,
            polarity="positive",
        )
        ig_negative_map = integrated_gradients(
            model,
            model_input,
            class_idx=class_idx,
            steps=args.ig_steps,
            polarity="negative",
        )
        consensus = consensus_heatmap([cam_map, ig_map])

        methods = {
            "grad_cam": cam_map,
            "grad_cam_negative": negative_cam_map,
            "integrated_gradients": ig_map,
            "integrated_gradients_positive": ig_positive_map,
            "integrated_gradients_negative": ig_negative_map,
            "integrated_gradients_signed": ig_positive_map,
            "consensus": consensus,
        }

        for method_name, heatmap in methods.items():
            top_fraction = calibrated_fractions.get(
                method_name, args.top_fraction)
            metrics = localization_metrics(
                heatmap, mask, fraction=top_fraction)
            negative_metrics = {
                "negative_mask_overlap_fraction": "",
                "negative_mask_avoidance_fraction": "",
            }
            if is_negative_method(method_name):
                negative_metrics = {
                    key: round(value, 6)
                    for key, value in negative_evidence_metrics(
                        heatmap,
                        mask,
                        top_fraction,
                    ).items()
                }
            elif method_name == "consensus":
                negative_fraction = calibrated_fractions.get(
                    "grad_cam_negative", args.top_fraction
                )
                negative_metrics = {
                    key: round(value, 6)
                    for key, value in negative_evidence_metrics(
                        negative_cam_map,
                        mask,
                        negative_fraction,
                    ).items()
                }
            elif method_name == "integrated_gradients_signed":
                negative_fraction = calibrated_fractions.get(
                    "integrated_gradients_negative", args.top_fraction
                )
                negative_metrics = {
                    key: round(value, 6)
                    for key, value in negative_evidence_metrics(
                        ig_negative_map,
                        mask,
                        negative_fraction,
                    ).items()
                }
            metric_rows.append(
                {
                    "sample_id": sample_idx,
                    "filename": row.get("filename", Path(row["image_path"]).name),
                    "split": row.get("split", ""),
                    "xrv_pneumothorax_score": round(score, 6),
                    "xrv_pneumothorax_sigmoid": round(probability, 6),
                    "method": method_name,
                    "top_fraction": round(top_fraction, 6),
                    **{key: round(value, 6) for key, value in metrics.items()},
                    **negative_metrics,
                }
            )
            if sample_idx < args.max_overlays:
                case_dir = output_dir / safe_case_name(sample_idx, row)
                case_dir.mkdir(parents=True, exist_ok=True)
                save_overlay(
                    image,
                    heatmap,
                    mask,
                    case_dir / f"{method_name}.png",
                    heatmap_color=overlay_color_for_method(method_name),
                    negative_heatmap=(
                        negative_cam_map if method_name == "consensus"
                        else ig_negative_map if method_name == "integrated_gradients_signed"
                        else None
                    ),
                    neutral_heatmap=ig_map if method_name == "consensus" else None,
                )
                selected_mask = threshold_top_fraction(
                    heatmap, fraction=top_fraction)
                negative_selected_mask = (
                    threshold_top_fraction(
                        ig_negative_map,
                        fraction=calibrated_fractions.get(
                            "integrated_gradients_negative", args.top_fraction
                        ),
                    )
                    if method_name == "integrated_gradients_signed"
                    else None
                )
                neutral_selected_mask = (
                    threshold_top_fraction(
                        ig_map,
                        fraction=calibrated_fractions.get(
                            "integrated_gradients", args.top_fraction
                        ),
                    )
                    if method_name == "consensus"
                    else None
                )
                save_selected_threshold_image(
                    image,
                    selected_mask,
                    mask,
                    case_dir / f"{method_name}_selected.png",
                    negative_style=is_negative_method(method_name),
                    neutral_style=method_name == "integrated_gradients",
                    negative_selected_mask=negative_selected_mask,
                    neutral_selected_mask=neutral_selected_mask,
                )

    gradcam.remove_hooks()

    metrics_path = output_dir / "metrics.csv"
    fieldnames = [
        "sample_id",
        "filename",
        "split",
        "xrv_pneumothorax_score",
        "xrv_pneumothorax_sigmoid",
        "method",
        "top_fraction",
        "iou",
        "dice",
        "pointing_hit",
        "precision_at_fraction",
        "negative_mask_overlap_fraction",
        "negative_mask_avoidance_fraction",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    summary_path = output_dir / "metrics_summary.csv"
    write_metric_summary(metric_rows, summary_path)

    print(f"TorchXRayVision CXR smoke test complete on {device}.")
    print(f"Weights: {args.weights}")
    print(f"Positive cases evaluated: {len(rows)}")
    print(f"Metrics written to: {metrics_path}")
    print(f"Metric summary written to: {summary_path}")
    print(f"Overlay case folders written to: {output_dir}")


if __name__ == "__main__":
    main()
