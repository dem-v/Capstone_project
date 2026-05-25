#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from explainai_thesis.cli.common import resolve_device
from explainai_thesis.cxr.classifier import load_classifier
from explainai_thesis.run_metadata import write_run_metadata
from explainai_thesis.faithfulness import (
    curve_auc,
    faithfulness_baseline_tensor,
    faithfulness_curve_rows,
)
from explainai_thesis.metrics import (
    localization_metrics,
    negative_evidence_metrics,
    threshold_top_fraction,
)
from explainai_thesis.visualization import (
    overlay_color_for_method,
    save_binary_selection,
    save_overlay,
    signed_diverging_overlay,
)
from explainai_thesis.xai import (
    GradCAM,
    SignedAttribution,
    consensus_signed,
    gradient_shap_signed,
    integrated_gradients_signed,
    iter_method_views,
    occlusion_sensitivity_signed,
)


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
    parser.add_argument(
        "--gradshap-internal-batch-size",
        type=int,
        default=8,
        help=(
            "Captum internal batch size for GradientSHAP noisy samples. "
            "Lower values reduce CUDA memory use for 512px ResNet diagnostics."
        ),
    )
    parser.add_argument("--occlusion-patch-size", type=int, default=32)
    parser.add_argument("--occlusion-stride", type=int, default=16)
    parser.add_argument(
        "--fractions",
        default="0.05,0.10,0.15,0.20,0.25,0.30",
        help="Comma-separated top-fractions to visualize.",
    )
    parser.add_argument(
        "--stop-fractions-at-coverage",
        type=float,
        default=0.95,
        help=(
            "Stop rendering larger top-fractions for a method view once the "
            "actual selected mask covers at least this fraction of image pixels. "
            "This avoids redundant near-full-image threshold panels when heatmap "
            "ties fill more than the requested fraction. Use 1.0 to continue "
            "until the whole image is selected."
        ),
    )
    parser.add_argument(
        "--pixel-attribution-mask-smoothing",
        type=int,
        default=9,
        help=(
            "Odd average-pooling kernel used only for IG/GradientSHAP overlay and "
            "top-fraction mask readability. Set 1 to disable."
        ),
    )
    parser.add_argument(
        "--faithfulness-fractions",
        default="",
        help="Optional comma-separated fractions for deletion/insertion faithfulness curves.",
    )
    parser.add_argument(
        "--faithfulness-baseline",
        default="black",
        choices=["zero_tensor", "black", "white", "case_mean"],
        help="Baseline used for optional deletion/insertion faithfulness curves.",
    )
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


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


def load_image(path: Path, image_size: int, preprocess) -> torch.Tensor:
    image = Image.open(path).convert("L").resize(
        (image_size, image_size), Image.BILINEAR)
    array = np.asarray(image)
    return preprocess(array)


def load_mask(path: Path, image_size: int) -> torch.Tensor:
    mask = Image.open(path).convert("L").resize(
        (image_size, image_size), Image.NEAREST)
    return torch.from_numpy(np.asarray(mask) > 0)


def parse_fractions(raw: str) -> list[float]:
    fractions = [float(value.strip())
                 for value in raw.split(",") if value.strip()]
    if not fractions:
        raise ValueError("At least one fraction is required.")
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("All fractions must be in (0, 1].")
    return fractions


def parse_optional_fractions(raw: str) -> list[float]:
    if not raw.strip():
        return []
    return parse_fractions(raw)


def write_rows(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_faithfulness_rows(
    rows: list[dict[str, str | int | float]],
) -> list[dict[str, str | int | float]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str | int | float]]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["view"]), str(row["family"]))
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, str | int | float]] = []
    for (method, view, family), group_rows in sorted(grouped.items()):
        output.append(
            {
                "method": method,
                "view": view,
                "family": family,
                "faithfulness_insertion_auc": round(
                    curve_auc(group_rows, "insertion_probability"), 6
                ),
                "faithfulness_deletion_auc": round(
                    curve_auc(group_rows, "deletion_probability"), 6
                ),
                "faithfulness_deletion_drop": round(
                    float(group_rows[0]["original_probability"])
                    - min(float(row["deletion_probability"]) for row in group_rows),
                    6,
                ),
                "faithfulness_insertion_gain": round(
                    max(float(row["insertion_probability"]) for row in group_rows)
                    - float(group_rows[0]["baseline_probability"]),
                    6,
                ),
                "n_curve_points": len(group_rows),
            }
        )
    return output


def safe_source_stem(row: dict[str, str]) -> str:
    stem = Path(row.get("filename") or row["image_path"]).stem
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)


def safe_case_name(case_index: int, row: dict[str, str]) -> str:
    return f"case_{case_index:03d}_{safe_source_stem(row)}"


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


def selected_image_coverage(selected_mask: torch.Tensor) -> float:
    return float(selected_mask.float().mean().item())


def readable_heatmap_for_method(
    heatmap: torch.Tensor,
    family: str,
    smoothing_kernel: int,
) -> torch.Tensor:
    if family not in {"integrated_gradients", "gradient_shap"}:
        return heatmap
    if smoothing_kernel <= 1:
        return heatmap
    if smoothing_kernel % 2 == 0:
        raise ValueError("--pixel-attribution-mask-smoothing must be odd or 1.")

    heatmap_2d = heatmap.detach().float()
    padding = smoothing_kernel // 2
    smoothed = F.avg_pool2d(
        heatmap_2d.unsqueeze(0).unsqueeze(0),
        kernel_size=smoothing_kernel,
        stride=1,
        padding=padding,
    )[0, 0]
    return smoothed.to(device=heatmap.device, dtype=heatmap.dtype)


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


def log_progress(message: str, start_time: float) -> None:
    elapsed_minutes = (time.perf_counter() - start_time) / 60.0
    print(f"[{elapsed_minutes:6.1f} min] {message}", flush=True)


def main() -> None:
    run_start = time.perf_counter()
    args = parse_args()
    if not 0 < args.stop_fractions_at_coverage <= 1:
        raise ValueError("--stop-fractions-at-coverage must be in (0, 1].")
    if args.pixel_attribution_mask_smoothing < 1 or args.pixel_attribution_mask_smoothing % 2 == 0:
        raise ValueError("--pixel-attribution-mask-smoothing must be a positive odd integer.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_progress("started threshold visualization", run_start)

    fractions = parse_fractions(args.fractions)
    faithfulness_fractions = parse_optional_fractions(args.faithfulness_fractions)
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
    case_filename = row.get("filename", Path(row["image_path"]).name)
    log_progress(f"selected case {case_filename}", run_start)
    device = resolve_device(args.device)
    log_progress(f"loading classifier {args.weights} on {device}", run_start)
    classifier = load_classifier(args.weights, device=device, pathology="Pneumothorax")
    model = classifier.model
    class_idx = classifier.class_idx
    gradcam = GradCAM(model, classifier.target_layer)

    image = load_image(Path(row["image_path"]), args.image_size, classifier.preprocess)
    mask = load_mask(Path(row["mask_path"]), args.image_size)
    model_input = image.unsqueeze(0).to(device)
    log_progress("loaded image/mask and prepared model input", run_start)

    with torch.no_grad():
        output = model(model_input)
        score = float(output[0, class_idx].detach().cpu().item())
        probability = float(torch.sigmoid(
            output[0, class_idx]).detach().cpu().item())
    log_progress(f"model score ready: probability={probability:.4f}", run_start)

    log_progress("computing Grad-CAM", run_start)
    cam_attr = gradcam.signed(model_input, class_idx=class_idx)
    log_progress("computing Grad-CAM++", run_start)
    cam_plus_plus_attr = gradcam.signed(
        model_input, class_idx=class_idx, variant="grad_cam_plus_plus")
    log_progress(f"computing Integrated Gradients ({args.ig_steps} steps)", run_start)
    ig_attr = integrated_gradients_signed(
        model, model_input, class_idx=class_idx, steps=args.ig_steps)
    log_progress(
        f"computing GradientSHAP ({args.gradshap_samples} samples, internal batch {args.gradshap_internal_batch_size})",
        run_start,
    )
    gradient_shap_attr = gradient_shap_signed(
        model,
        model_input,
        class_idx=class_idx,
        samples=args.gradshap_samples,
        stdevs=args.gradshap_stdevs,
        internal_batch_size=args.gradshap_internal_batch_size,
    )
    log_progress(
        f"computing Occlusion (patch {args.occlusion_patch_size}, stride {args.occlusion_stride})",
        run_start,
    )
    occlusion_attr = occlusion_sensitivity_signed(
        model,
        model_input,
        class_idx=class_idx,
        patch_size=args.occlusion_patch_size,
        stride=args.occlusion_stride,
    )
    log_progress("computing consensus attribution", run_start)
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
            "weights": args.weights,
            "image_size": args.image_size,
            "xrv_pneumothorax_score": round(score, 6),
            "xrv_pneumothorax_sigmoid": round(probability, 6),
        }
    ]
    write_rows(output_dir / "case_metadata.csv", metadata_rows)

    metric_rows: list[dict[str, str | int | float]] = []
    faithfulness_rows: list[dict[str, str | int | float]] = []
    faithfulness_baseline = faithfulness_baseline_tensor(
        model_input, args.faithfulness_baseline
    ) if faithfulness_fractions else None
    baseline_probability = None
    if faithfulness_baseline is not None:
        log_progress(
            f"preparing faithfulness baseline={args.faithfulness_baseline} with {len(faithfulness_fractions)} fractions",
            run_start,
        )
        with torch.no_grad():
            baseline_output = model(faithfulness_baseline)
            baseline_probability = float(
                torch.sigmoid(baseline_output[0, class_idx]).detach().cpu().item()
            )
    source_stem = safe_source_stem(row)
    case_dir = output_dir / safe_case_name(args.case_index, row)
    case_dir.mkdir(parents=True, exist_ok=True)
    method_views = list(iter_method_views(signed_attributions))
    log_progress(f"rendering {len(method_views)} method/view blocks", run_start)
    for view_index, method_view in enumerate(method_views, start=1):
        method_name = method_view.method
        heatmap = method_view.heatmap
        view_kind = method_view.view
        family = method_view.family
        display_heatmap = readable_heatmap_for_method(
            heatmap, family, args.pixel_attribution_mask_smoothing
        )
        log_progress(f"[{view_index}/{len(method_views)}] {method_name}: start", run_start)
        if faithfulness_fractions and faithfulness_baseline is not None:
            log_progress(f"[{view_index}/{len(method_views)}] {method_name}: faithfulness curves", run_start)
            faithfulness_input = (
                signed_attributions[family].magnitude
                if view_kind == "signed"
                else display_heatmap
            )
            if view_kind == "signed":
                faithfulness_input = readable_heatmap_for_method(
                    faithfulness_input, family, args.pixel_attribution_mask_smoothing
                )
            for faithfulness_row in faithfulness_curve_rows(
                model,
                model_input,
                faithfulness_input,
                class_idx,
                faithfulness_fractions,
                faithfulness_baseline,
            ):
                faithfulness_rows.append(
                    {
                        "filename": row.get("filename", Path(row["image_path"]).name),
                        "source_stem": source_stem,
                        "image_path": row["image_path"],
                        "method": method_name,
                        "view": view_kind,
                        "family": family,
                        "baseline": args.faithfulness_baseline,
                        "original_probability": round(probability, 6),
                        "baseline_probability": round(float(baseline_probability), 6),
                        **faithfulness_row,
                    }
                )
        overlay_path = case_dir / f"{source_stem}_{method_name}_continuous_heatmap.png"
        if view_kind == "signed":
            signed_diverging_overlay(image, display_heatmap, mask, overlay_path)
        else:
            save_overlay(
                image,
                display_heatmap,
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
                else display_heatmap
            )
            if view_kind == "signed":
                metrics_input = readable_heatmap_for_method(
                    metrics_input, family, args.pixel_attribution_mask_smoothing
                )
            selected = threshold_top_fraction(metrics_input, fraction=fraction)
            metrics = localization_metrics(metrics_input, mask, fraction=fraction)
            selected_coverage = selected_image_coverage(selected)
            negative_metrics: dict[str, str | float] = {
                "negative_mask_overlap_fraction": "",
                "negative_mask_avoidance_fraction": "",
            }
            if view_kind == "negative":
                negative_metrics = {
                    key: round(value, 6)
                    for key, value in negative_evidence_metrics(
                        metrics_input, mask, fraction
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
                    "pixel_attribution_mask_smoothing": (
                        args.pixel_attribution_mask_smoothing
                        if family in {"integrated_gradients", "gradient_shap"}
                        else 1
                    ),
                    "top_fraction": round(fraction, 6),
                    "top_fraction_percent": int(round(fraction * 100)),
                    "selected_image_coverage": round(selected_coverage, 6),
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
            if selected_coverage >= args.stop_fractions_at_coverage:
                break
        make_contact_sheet(binary_paths, binary_captions,
                           case_dir / f"{source_stem}_{method_name}_threshold_sweep_panel.png")
        log_progress(f"[{view_index}/{len(method_views)}] {method_name}: rendered", run_start)

    log_progress("writing CSV outputs", run_start)
    write_rows(output_dir / "threshold_metrics.csv", metric_rows)
    if faithfulness_rows:
        write_rows(output_dir / "faithfulness_curves.csv", faithfulness_rows)
        write_rows(
            output_dir / "faithfulness_summary.csv",
            summarize_faithfulness_rows(faithfulness_rows),
        )
    run_meta_path = write_run_metadata(
        output_dir,
        args,
        weights=args.weights,
        split=args.split,
        case=row.get("filename", Path(row["image_path"]).name),
    )

    log_progress("completed threshold visualization", run_start)
    print(f"Single-image threshold visualization complete on {device}.")
    print(f"Output directory: {output_dir}")
    print(f"Case: {row.get('filename', Path(row['image_path']).name)}")
    print(f"Run metadata written to: {run_meta_path}")


if __name__ == "__main__":
    main()
