#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from explainai_thesis.xai import (
    GradCAM,
    SignedAttribution,
    agreement_score,
    consensus_signed,
    gradient_shap_signed,
    integrated_gradients_signed,
    occlusion_sensitivity_signed,
)
from explainai_thesis.visualization import save_overlay, signed_diverging_overlay
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
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Randomly sample positive cases after filtering instead of taking the first rows.",
    )
    parser.add_argument(
        "--case-filename",
        default="",
        help="Optional exact manifest filename to evaluate as a one-case targeted diagnostic.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260515,
        help="Seed used with --random-sample for reproducible case selection.",
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
        "--faithfulness-fractions",
        default="",
        help="Optional comma-separated fractions for deletion/insertion faithfulness curves.",
    )
    parser.add_argument(
        "--faithfulness-baseline",
        default="zero_tensor",
        choices=["zero_tensor", "black", "white", "case_mean"],
        help=(
            "Baseline used for deletion/insertion faithfulness. "
            "zero_tensor preserves the historical behavior; black/white/case_mean "
            "use normalized image-space baselines."
        ),
    )
    parser.add_argument(
        "--gradshap-samples",
        type=int,
        default=8,
        help="GradientSHAP samples per image. Increase for final runs.",
    )
    parser.add_argument(
        "--gradshap-stdevs",
        type=float,
        default=0.02,
        help="GradientSHAP noise standard deviation.",
    )
    parser.add_argument(
        "--occlusion-patch-size",
        type=int,
        default=32,
        help="Occlusion Sensitivity square patch size in resized image pixels.",
    )
    parser.add_argument(
        "--occlusion-stride",
        type=int,
        default=16,
        help="Occlusion Sensitivity stride in resized image pixels.",
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
    manifest_path: Path, split: str, limit: int, *, random_sample: bool = False, seed: int = 20260515
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
            if not random_sample and len(rows) >= limit:
                break
    if random_sample:
        rng = random.Random(seed)
        rng.shuffle(rows)
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


def parse_optional_fractions(raw: str) -> list[float]:
    fractions = [float(value.strip()) for value in raw.split(",") if value.strip()]
    for fraction in fractions:
        if not 0 <= fraction <= 1:
            raise ValueError("Faithfulness fractions must be in [0, 1].")
    return fractions


def model_probability(model: torch.nn.Module, image: torch.Tensor, class_idx: int) -> float:
    with torch.no_grad():
        output = model(image)
        return float(torch.sigmoid(output[0, class_idx]).detach().cpu().item())


def faithfulness_baseline_tensor(model_input: torch.Tensor, baseline: str) -> torch.Tensor:
    if baseline == "zero_tensor":
        return torch.zeros_like(model_input)
    if baseline == "black":
        return torch.full_like(model_input, -1024.0)
    if baseline == "white":
        return torch.full_like(model_input, 1024.0)
    if baseline == "case_mean":
        return torch.full_like(model_input, float(model_input.mean().item()))
    raise ValueError(f"Unsupported faithfulness baseline: {baseline}")


def faithfulness_curve_rows(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    heatmap: torch.Tensor,
    class_idx: int,
    fractions: list[float],
    baseline: torch.Tensor,
) -> list[dict[str, float]]:
    if not fractions:
        return []
    flat_order = torch.argsort(heatmap.flatten().to(model_input.device), descending=True)
    original_flat = model_input.detach().clone().flatten()
    baseline_flat = baseline.detach().clone().flatten().to(model_input.device)
    rows: list[dict[str, float]] = []
    total_pixels = flat_order.numel()
    for fraction in fractions:
        keep_count = int(round(total_pixels * fraction))
        insertion_flat = baseline_flat.clone()
        deletion_flat = original_flat.clone()
        if keep_count > 0:
            selected = flat_order[:keep_count]
            insertion_flat[selected] = original_flat[selected]
            deletion_flat[selected] = baseline_flat[selected]
        insertion = insertion_flat.view_as(model_input)
        deletion = deletion_flat.view_as(model_input)
        rows.append(
            {
                "fraction": round(fraction, 6),
                "insertion_probability": round(
                    model_probability(model, insertion, class_idx), 6
                ),
                "deletion_probability": round(
                    model_probability(model, deletion, class_idx), 6
                ),
            }
        )
    return rows


def curve_auc(rows: list[dict[str, str | int | float]], value_key: str) -> float:
    points = sorted((float(row["fraction"]), float(row[value_key])) for row in rows)
    if len(points) < 2:
        return 0.0
    auc = 0.0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        auc += (x1 - x0) * (y0 + y1) / 2.0
    return auc


def write_faithfulness_summary(
    faithfulness_rows: list[dict[str, str | int | float]], output_path: Path
) -> None:
    grouped: dict[tuple[int, str], list[dict[str, str | int | float]]] = defaultdict(list)
    for row in faithfulness_rows:
        grouped[(int(row["sample_id"]), str(row["method"]))].append(row)

    per_case: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (_sample_id, method), rows in grouped.items():
        insertion_auc = curve_auc(rows, "insertion_probability")
        deletion_auc = curve_auc(rows, "deletion_probability")
        per_case[method].append(
            {
                "insertion_auc": insertion_auc,
                "deletion_auc": deletion_auc,
                "deletion_drop_auc": 1.0 - deletion_auc,
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "case_count",
                "insertion_auc_mean",
                "deletion_auc_mean",
                "deletion_drop_auc_mean",
            ],
        )
        writer.writeheader()
        for method, values in sorted(per_case.items()):
            writer.writerow(
                {
                    "method": method,
                    "case_count": len(values),
                    "insertion_auc_mean": round(
                        float(np.mean([item["insertion_auc"] for item in values])), 6
                    ),
                    "deletion_auc_mean": round(
                        float(np.mean([item["deletion_auc"] for item in values])), 6
                    ),
                    "deletion_drop_auc_mean": round(
                        float(np.mean([item["deletion_drop_auc"] for item in values])), 6
                    ),
                }
            )


def faithfulness_method_family(method: str) -> str:
    if method.startswith("grad_cam") or method == "consensus":
        return "cam_family"
    if method.startswith("integrated_gradients"):
        return "integrated_gradients_family"
    if method.startswith("gradient_shap"):
        return "gradient_shap_family"
    if method.startswith("occlusion"):
        return "occlusion_family"
    return "other"


def plot_faithfulness_curves(
    faithfulness_rows: list[dict[str, str | int | float]],
    output_path: Path,
    title: str,
    *,
    zoom_y: bool = False,
    y_limits: tuple[float, float] | None = None,
) -> None:
    if not faithfulness_rows:
        return
    grouped: dict[str, list[dict[str, str | int | float]]] = defaultdict(list)
    for row in faithfulness_rows:
        grouped[str(row["method"])].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    plotted_values: list[float] = []
    for method, rows in sorted(grouped.items()):
        by_fraction: dict[float, list[dict[str, str | int | float]]] = defaultdict(list)
        for row in rows:
            by_fraction[float(row["fraction"])].append(row)
        fractions = sorted(by_fraction)
        insertion_values = [
            float(
                np.mean(
                    [float(row["insertion_probability"]) for row in by_fraction[fraction]]
                )
            )
            for fraction in fractions
        ]
        deletion_values = [
            float(
                np.mean(
                    [float(row["deletion_probability"]) for row in by_fraction[fraction]]
                )
            )
            for fraction in fractions
        ]
        plotted_values.extend(insertion_values)
        plotted_values.extend(deletion_values)
        axes[0].plot(
            fractions,
            insertion_values,
            marker="o",
            linewidth=1.5,
            label=method,
        )
        axes[1].plot(
            fractions,
            deletion_values,
            marker="o",
            linewidth=1.5,
            label=method,
        )

    axes[0].set_title("Insertion")
    axes[0].set_xlabel("Fraction of top-attributed pixels restored")
    axes[0].set_ylabel("Pneumothorax probability")
    axes[1].set_title("Deletion")
    axes[1].set_xlabel("Fraction of top-attributed pixels removed")
    if y_limits is not None:
        y_min, y_max = y_limits
    elif zoom_y and plotted_values:
        y_min = max(0.0, min(plotted_values) - 0.03)
        y_max = min(1.0, max(plotted_values) + 0.03)
    else:
        y_min = 0.0
        y_max = 1.0
    for axis in axes:
        axis.set_ylim(y_min, y_max)
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_faithfulness_plots(
    faithfulness_rows: list[dict[str, str | int | float]], output_dir: Path, title: str
) -> None:
    plotted_values = [
        float(row[key])
        for row in faithfulness_rows
        for key in ["insertion_probability", "deletion_probability"]
    ]
    shared_zoom_limits = None
    if plotted_values:
        shared_zoom_limits = (
            max(0.0, min(plotted_values) - 0.03),
            min(1.0, max(plotted_values) + 0.03),
        )
    plot_faithfulness_curves(
        faithfulness_rows,
        output_dir / "faithfulness_curves.png",
        title,
    )
    plot_faithfulness_curves(
        faithfulness_rows,
        output_dir / "faithfulness_curves_zoomed.png",
        f"{title} (zoomed y-axis)",
        y_limits=shared_zoom_limits,
    )
    families: dict[str, list[dict[str, str | int | float]]] = defaultdict(list)
    for row in faithfulness_rows:
        families[faithfulness_method_family(str(row["method"]))].append(row)
    for family, rows in families.items():
        if rows:
            plot_faithfulness_curves(
                rows,
                output_dir / f"faithfulness_curves_{family}.png",
                f"{title}: {family.replace('_', ' ')} (shared zoom scale)",
                y_limits=shared_zoom_limits,
            )


def plot_faithfulness_summary(summary_path: Path, output_path: Path) -> None:
    if not summary_path.exists():
        return
    rows: list[dict[str, str]] = []
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
    if not rows:
        return
    methods = [row["method"] for row in rows]
    x = np.arange(len(methods))
    width = 0.38
    insertion = [float(row["insertion_auc_mean"]) for row in rows]
    deletion_drop = [float(row["deletion_drop_auc_mean"]) for row in rows]
    fig, axis = plt.subplots(figsize=(max(10, len(methods) * 0.8), 5))
    axis.bar(x - width / 2, insertion, width, label="Insertion AUC")
    axis.bar(x + width / 2, deletion_drop, width, label="Deletion-drop AUC")
    axis.set_ylabel("AUC")
    axis.set_title("Faithfulness AUC summary")
    axis.set_xticks(x)
    axis.set_xticklabels(methods, rotation=45, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


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


def safe_source_stem(row: dict[str, str]) -> str:
    stem = Path(row.get("filename") or row["image_path"]).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return safe_stem or "xray"


def overlay_color_for_method(method_name: str) -> str:
    if method_name in {"integrated_gradients", "gradient_shap", "occlusion"}:
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
    faithfulness_fractions = parse_optional_fractions(args.faithfulness_fractions)
    calibrated_fractions = read_calibrated_fractions(
        Path(args.calibrated_fractions) if args.calibrated_fractions else None
    )
    rows = read_positive_rows(
        manifest_path,
        split=args.split,
        limit=args.max_positive,
        random_sample=args.random_sample,
        seed=args.seed,
    )
    if args.case_filename:
        rows = [
            row for row in rows
            if row.get("filename", Path(row["image_path"]).name) == args.case_filename
        ]
    rows = rows[: args.max_positive]
    if not rows:
        raise RuntimeError(
            f"No positive rows with masks found in {manifest_path} for split={args.split}."
        )

    model = xrv.models.DenseNet(weights=args.weights).to(device)
    model.eval()
    class_idx = pathology_index(model, "Pneumothorax")
    gradcam = GradCAM(model, model.features.denseblock4)

    metric_rows: list[dict[str, str | int | float]] = []
    faithfulness_rows: list[dict[str, str | int | float]] = []
    agreement_rows: list[dict[str, str | int | float]] = []
    for sample_idx, row in enumerate(rows):
        image = load_image(Path(row["image_path"]), args.image_size)
        mask = load_mask(Path(row["mask_path"]), args.image_size)
        model_input = image.unsqueeze(0).to(device)
        faithfulness_baseline = faithfulness_baseline_tensor(
            model_input, args.faithfulness_baseline
        )
        case_faithfulness_rows: list[dict[str, str | int | float]] = []

        with torch.no_grad():
            output = model(model_input)
            score = float(output[0, class_idx].detach().cpu().item())
            probability = float(
                torch.sigmoid(output[0, class_idx]).detach().cpu().item()
            )

        # Phase 1.2-dispatch: 5 signed cores per case (one forward/backward
        # or one occlusion sweep each) replace the pre-1.2 16-call polarity
        # fan-out. The four views (positive/negative/magnitude/signed) are
        # derived in microseconds from each SignedAttribution.
        gradcam_attr = gradcam.signed(model_input, class_idx=class_idx)
        gradcam_pp_attr = gradcam.signed(
            model_input, class_idx=class_idx, variant="grad_cam_plus_plus"
        )
        ig_attr = integrated_gradients_signed(
            model, model_input, class_idx=class_idx, steps=args.ig_steps
        )
        gradshap_attr = gradient_shap_signed(
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
        consensus_attr = consensus_signed(
            [gradcam_attr, ig_attr, gradshap_attr, occlusion_attr]
        )

        # SignedAttribution registry — the source of truth for the v2 dispatch.
        # Each entry is (method-family id, SignedAttribution). The smoke loop
        # below expands each family into per-view rows.
        signed_attributions: dict[str, SignedAttribution] = {
            "grad_cam": gradcam_attr,
            "grad_cam_plus_plus": gradcam_pp_attr,
            "integrated_gradients": ig_attr,
            "gradient_shap": gradshap_attr,
            "occlusion": occlusion_attr,
            "consensus": consensus_attr,
        }

        # methods: v2 method id -> (heatmap tensor, view kind, family id).
        # `view kind` ∈ {"positive", "negative", "magnitude", "signed"} and
        # drives both the metrics row and the overlay color choice.
        methods: dict[str, tuple[torch.Tensor, str, str]] = {}
        for family, attr in signed_attributions.items():
            # Positive view — canonical localization heatmap, [0, 1].
            methods[family] = (
                normalize_map(attr.positive), "positive", family,
            )
            # Negative view — suppressive evidence, [0, 1].
            methods[f"{family}_negative"] = (
                normalize_map(attr.negative), "negative", family,
            )
            # Magnitude view — only meaningful for the gradient-flavor methods
            # and Grad-CAM(++). Consensus and IG/GradientSHAP all expose it.
            methods[f"{family}_magnitude"] = (
                normalize_map(attr.magnitude), "magnitude", family,
            )
            # Signed view — tug-of-war diverging map. Rendered with the
            # orange/teal palette by `signed_diverging_overlay`. Used in
            # metrics.csv but with `signed_positive_fraction` instead of the
            # standard top-fraction selection (see below).
            methods[f"{family}_signed"] = (
                attr.signed, "signed", family,
            )

        # Back-compat tensors for downstream consumers that haven't been
        # ported yet (e.g. the negative_evidence_metrics block below uses
        # the family.negative view as the "negative evidence" reference).
        cam_map = methods["grad_cam"][0]
        negative_cam_map = methods["grad_cam_negative"][0]
        ig_map = methods["integrated_gradients"][0]
        ig_negative_map = methods["integrated_gradients_negative"][0]
        gradient_shap_map = methods["gradient_shap"][0]
        gradient_shap_negative_map = methods["gradient_shap_negative"][0]
        occlusion_map = methods["occlusion"][0]

        # Cross-method agreement (cosine similarity between signed maps) —
        # one row per unordered pair, per case. Per AGENTS.md, reported when
        # more than one signed-capable method is run on the same case.
        agreement_families = ["grad_cam", "grad_cam_plus_plus",
                              "integrated_gradients", "gradient_shap",
                              "occlusion"]
        for i, family_a in enumerate(agreement_families):
            for family_b in agreement_families[i + 1:]:
                agreement_rows.append({
                    "sample_id": sample_idx,
                    "filename": row.get("filename", Path(row["image_path"]).name),
                    "split": row.get("split", ""),
                    "method_a": family_a,
                    "method_b": family_b,
                    "agreement_score": round(
                        agreement_score(
                            signed_attributions[family_a],
                            signed_attributions[family_b],
                        ),
                        6,
                    ),
                })

        for method_name, (heatmap, view_kind, family) in methods.items():
            top_fraction = calibrated_fractions.get(
                method_name, args.top_fraction)
            # localization_metrics expects a [0, 1] map. For the signed view
            # we feed the magnitude as a stand-in so the standard IoU/Dice/
            # pointing_hit/precision columns are still defined; signed-
            # specific behavior is captured by `signed_positive_fraction`.
            metrics_input = signed_attributions[family].magnitude \
                if view_kind == "signed" else heatmap
            metrics = localization_metrics(
                metrics_input, mask, fraction=top_fraction)
            for faithfulness_row in faithfulness_curve_rows(
                model,
                model_input,
                metrics_input,
                class_idx,
                faithfulness_fractions,
                faithfulness_baseline,
            ):
                enriched_faithfulness_row = {
                    "sample_id": sample_idx,
                    "filename": row.get("filename", Path(row["image_path"]).name),
                    "split": row.get("split", ""),
                    "method": method_name,
                    "baseline": args.faithfulness_baseline,
                    **faithfulness_row,
                }
                faithfulness_rows.append(enriched_faithfulness_row)
                case_faithfulness_rows.append(enriched_faithfulness_row)

            # negative-evidence diagnostics: only meaningful for the
            # negative view (`*_negative`) — the signed view is reported
            # separately via `signed_positive_fraction`.
            negative_metrics = {
                "negative_mask_overlap_fraction": "",
                "negative_mask_avoidance_fraction": "",
            }
            if view_kind == "negative":
                negative_metrics = {
                    key: round(value, 6)
                    for key, value in negative_evidence_metrics(
                        heatmap, mask, top_fraction,
                    ).items()
                }

            # signed_positive_fraction: of the |signed| top-fraction
            # selected pixels, what fraction came from the positive side
            # (`signed > 0`)? Defined only on `*_signed` rows; blank
            # elsewhere to keep the column unambiguous.
            if view_kind == "signed":
                signed_tensor = signed_attributions[family].signed
                abs_map = signed_tensor.abs()
                selected = threshold_top_fraction(abs_map, fraction=top_fraction)
                selected_count = float(selected.sum().item())
                if selected_count > 0:
                    positive_count = float(
                        (selected & (signed_tensor > 0)).sum().item()
                    )
                    signed_positive_fraction: str | float = round(
                        positive_count / selected_count, 6
                    )
                else:
                    signed_positive_fraction = 0.0
            else:
                signed_positive_fraction = ""

            metric_rows.append(
                {
                    "sample_id": sample_idx,
                    "filename": row.get("filename", Path(row["image_path"]).name),
                    "split": row.get("split", ""),
                    "xrv_pneumothorax_score": round(score, 6),
                    "xrv_pneumothorax_sigmoid": round(probability, 6),
                    "method": method_name,
                    "view": view_kind,
                    "family": family,
                    "top_fraction": round(top_fraction, 6),
                    **{key: round(value, 6) for key, value in metrics.items()},
                    **negative_metrics,
                    "signed_positive_fraction": signed_positive_fraction,
                }
            )

            if sample_idx < args.max_overlays:
                case_dir = output_dir / safe_case_name(sample_idx, row)
                source_stem = safe_source_stem(row)
                case_dir.mkdir(parents=True, exist_ok=True)
                overlay_path = case_dir / f"{source_stem}_{method_name}.png"
                if view_kind == "signed":
                    # Orange/teal diverging palette per AGENTS.md.
                    signed_diverging_overlay(
                        image,
                        signed_attributions[family].signed,
                        mask,
                        overlay_path,
                    )
                else:
                    color = {
                        "positive": "red",
                        "negative": "blue",
                        "magnitude": "neutral",
                    }[view_kind]
                    save_overlay(
                        image,
                        heatmap,
                        mask,
                        overlay_path,
                        heatmap_color=color,
                    )
                # Selected-threshold image only for positive/negative/magnitude
                # views; the signed view is communicated by the overlay itself.
                if view_kind != "signed":
                    selected_mask = threshold_top_fraction(
                        heatmap, fraction=top_fraction)
                    save_selected_threshold_image(
                        image,
                        selected_mask,
                        mask,
                        case_dir / f"{source_stem}_{method_name}_selected.png",
                        negative_style=(view_kind == "negative"),
                        neutral_style=(view_kind == "magnitude"),
                    )

        if faithfulness_fractions and sample_idx < args.max_overlays:
            case_dir = output_dir / safe_case_name(sample_idx, row)
            source_stem = safe_source_stem(row)
            case_dir.mkdir(parents=True, exist_ok=True)
            plot_faithfulness_curves(
                case_faithfulness_rows,
                case_dir / f"{source_stem}_faithfulness_curves.png",
                f"Faithfulness curves: {row.get('filename', Path(row['image_path']).name)}",
            )

    gradcam.remove_hooks()

    metrics_path = output_dir / "metrics.csv"
    # Phase 1.2-dispatch schema bump: added `view` and `family` (v2 dispatch
    # provenance) plus `signed_positive_fraction` (signed-view-only column).
    # The Phase 0 golden-output snapshot is on `scripts/run_smoke_test.py`
    # (synthetic), not this CXR script, so this schema change does not
    # regress the frozen public contract guarded by `test_golden_outputs.py`.
    fieldnames = [
        "sample_id",
        "filename",
        "split",
        "xrv_pneumothorax_score",
        "xrv_pneumothorax_sigmoid",
        "method",
        "view",
        "family",
        "top_fraction",
        "iou",
        "dice",
        "pointing_hit",
        "precision_at_fraction",
        "negative_mask_overlap_fraction",
        "negative_mask_avoidance_fraction",
        "signed_positive_fraction",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    # Cross-method agreement: cosine similarity between signed maps,
    # per unordered method pair, per case. Empty file emitted on
    # single-method runs so downstream tooling can rely on its presence.
    agreement_path = output_dir / "agreement.csv"
    with agreement_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "filename",
                "split",
                "method_a",
                "method_b",
                "agreement_score",
            ],
        )
        writer.writeheader()
        writer.writerows(agreement_rows)

    if faithfulness_fractions:
        faithfulness_path = output_dir / "faithfulness_curves.csv"
        with faithfulness_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_id",
                    "filename",
                    "split",
                    "method",
                    "baseline",
                    "fraction",
                    "insertion_probability",
                    "deletion_probability",
                ],
            )
            writer.writeheader()
            writer.writerows(faithfulness_rows)
        faithfulness_summary_path = output_dir / "faithfulness_summary.csv"
        write_faithfulness_summary(faithfulness_rows, faithfulness_summary_path)
        write_faithfulness_plots(
            faithfulness_rows,
            output_dir,
            "Aggregate faithfulness curves",
        )
        plot_faithfulness_summary(
            faithfulness_summary_path,
            output_dir / "faithfulness_summary.png",
        )
        plot_faithfulness_summary(
            faithfulness_summary_path,
            output_dir / "faithfulness_auc_bars.png",
        )

    summary_path = output_dir / "metrics_summary.csv"
    write_metric_summary(metric_rows, summary_path)

    print(f"TorchXRayVision CXR smoke test complete on {device}.")
    print(f"Weights: {args.weights}")
    print(f"Positive cases evaluated: {len(rows)}")
    print(f"Metrics written to: {metrics_path}")
    print(f"Metric summary written to: {summary_path}")
    if faithfulness_fractions:
        print(f"Faithfulness curves written to: {output_dir / 'faithfulness_curves.csv'}")
        print(f"Faithfulness summary written to: {output_dir / 'faithfulness_summary.csv'}")
        print(f"Faithfulness plot written to: {output_dir / 'faithfulness_curves.png'}")
        print(f"Faithfulness AUC bar plot written to: {output_dir / 'faithfulness_auc_bars.png'}")
    print(f"Overlay case folders written to: {output_dir}")


if __name__ == "__main__":
    main()
