#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
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
from explainai_thesis.cxr.classifier import load_classifier
from explainai_thesis.metrics import (
    localization_metrics,
    normalize_map,
    threshold_top_fraction,
)
from explainai_thesis.run_metadata import write_run_metadata
from PIL import Image
import torchxrayvision as xrv
import torch
import numpy as np

from explainai_thesis.cli.common import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate top-fraction thresholds for CXR XAI heatmaps on positive masked cases."
    )
    parser.add_argument(
        "--manifest", default="data/cxr_pneumothorax_manifest.csv")
    parser.add_argument(
        "--output-dir", default="outputs/cxr_xai_threshold_calibration")
    parser.add_argument(
        "--weights",
        default="densenet121-res224-all",
        help=(
            "TorchXRayVision classifier weights name (DenseNet-121 224x224 or "
            "ResNet-50 512x512). For ResNet pass `--image-size 512`."
        ),
    )
    parser.add_argument("--split", default="train",
                        choices=["train", "test", "any"])
    parser.add_argument("--max-positive", type=int, default=200)
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Randomly sample positive masked cases instead of taking the first rows in manifest order.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260515,
        help="Random seed used with --random-sample.",
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
        help="Comma-separated top-fractions to sweep.",
    )
    parser.add_argument(
        "--selection-metric",
        default="dice",
        choices=[
            "dice",
            "iou",
            "precision_at_fraction",
            "pointing_hit",
            "negative_mask_avoidance_fraction",
            "negative_mask_overlap_fraction",
        ],
    )
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--classifier-threshold",
        type=float,
        default=0.62,
        help=(
            "Sigmoid cutoff for deriving the model's predicted class on each "
            "case. Used only by the `signed_prediction_alignment` (SPA) column "
            "on `*_signed` rows; does NOT affect calibrated top-fraction "
            "selection. Default 0.62 matches the train-calibrated TorchXRayVision "
            "cutoff documented in AGENTS.md."
        ),
    )
    return parser.parse_args()


def read_positive_rows(
    manifest_path: Path,
    split: str,
    limit: int,
    random_sample: bool,
    seed: int,
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
        rows = rows[:limit]
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


def negative_evidence_metrics(heatmap: torch.Tensor, true_mask: torch.Tensor, fraction: float) -> dict[str, float]:
    selected = threshold_top_fraction(heatmap, fraction=fraction)
    selected_count = selected.sum().float()
    if selected_count.item() == 0:
        return {
            "negative_mask_overlap_fraction": 0.0,
            "negative_mask_avoidance_fraction": 0.0,
        }
    overlap = (selected & true_mask.bool()).sum().float() / selected_count
    return {
        "negative_mask_overlap_fraction": overlap.item(),
        "negative_mask_avoidance_fraction": (1.0 - overlap).item(),
    }




def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fractions = parse_fractions(args.fractions)
    rows = read_positive_rows(
        Path(args.manifest),
        args.split,
        args.max_positive,
        args.random_sample,
        args.seed,
    )
    if not rows:
        raise RuntimeError(
            f"No positive rows with masks found in {args.manifest} for split={args.split}.")

    device = resolve_device(args.device)
    # Phase 1.7 seam: same constructor / target layer / pathology index as the
    # pre-seam inline path; only the entry point changes. See
    # `src/explainai_thesis/cxr/classifier.py` for the full contract.
    bundle = load_classifier(args.weights, device=device, pathology="Pneumothorax")
    model = bundle.model
    class_idx = bundle.class_idx
    gradcam = GradCAM(model, bundle.target_layer)

    metric_rows: list[dict[str, str | int | float]] = []
    for sample_idx, row in enumerate(rows):
        image = load_image(Path(row["image_path"]), args.image_size)
        mask = load_mask(Path(row["mask_path"]), args.image_size)
        model_input = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(model_input)
            score = float(output[0, class_idx].detach().cpu().item())
            probability = float(torch.sigmoid(
                output[0, class_idx]).detach().cpu().item())

        # Phase 1.2.5 (v2): 5 signed cores per case replace the pre-1.2 16-call
        # polarity fan-out. The four views (positive/negative/magnitude/signed)
        # are derived in microseconds from each SignedAttribution.
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

        signed_attributions: dict[str, SignedAttribution] = {
            "grad_cam": gradcam_attr,
            "grad_cam_plus_plus": gradcam_pp_attr,
            "integrated_gradients": ig_attr,
            "gradient_shap": gradshap_attr,
            "occlusion": occlusion_attr,
            "consensus": consensus_attr,
        }

        for method_view in iter_method_views(signed_attributions):
            method_name = method_view.method
            heatmap = method_view.heatmap
            view_kind = method_view.view
            family = method_view.family
            for fraction in fractions:
                # For the signed view, feed magnitude into localization_metrics
                # so the standard IoU/Dice/pointing_hit/precision columns are
                # still defined; signed-specific behavior is captured by
                # `signed_positive_fraction`.
                metrics_input = (
                    signed_attributions[family].magnitude
                    if view_kind == "signed"
                    else heatmap
                )
                metrics = localization_metrics(
                    metrics_input, mask, fraction=fraction)
                negative_metrics = {
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
                if view_kind == "signed":
                    signed_tensor = signed_attributions[family].signed
                    abs_map = signed_tensor.abs()
                    selected = threshold_top_fraction(abs_map, fraction=fraction)
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
                    # SPA: fraction of top-|signed| pixels whose sign matches
                    # the model's predicted class. On `model predicts positive`
                    # this equals SPF; on `model predicts negative` it equals
                    # 1 - SPF. Comparable across TP/TN at high confidence and
                    # diagnostic on FP/FN. Calibration set is positives-only,
                    # so on this script SPA is dominated by the SPF branch;
                    # the FN branch (sigmoid < threshold on a label=1 case)
                    # is where the 1-SPF flip kicks in.
                    if isinstance(signed_positive_fraction, float):
                        if probability >= args.classifier_threshold:
                            signed_prediction_alignment: str | float = \
                                signed_positive_fraction
                        else:
                            signed_prediction_alignment = round(
                                1.0 - signed_positive_fraction, 6
                            )
                    else:
                        signed_prediction_alignment = ""
                else:
                    signed_positive_fraction = ""
                    signed_prediction_alignment = ""
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
                        "top_fraction": round(fraction, 6),
                        **{key: round(value, 6) for key, value in metrics.items()},
                        **negative_metrics,
                        "signed_positive_fraction": signed_positive_fraction,
                        "signed_prediction_alignment": signed_prediction_alignment,
                    }
                )

    gradcam.remove_hooks()

    grouped: dict[tuple[str, float],
                  list[dict[str, str | int | float]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(str(row["method"]), float(row["top_fraction"]))].append(row)

    summary_rows: list[dict[str, str | int | float]] = []
    for (method_name, fraction), rows_for_fraction in sorted(grouped.items()):
        summary: dict[str, str | int | float] = {
            "method": method_name,
            "top_fraction": round(fraction, 6),
            "n": len(rows_for_fraction),
        }
        for metric_name in [
            "iou",
            "dice",
            "pointing_hit",
            "precision_at_fraction",
            "negative_mask_overlap_fraction",
            "negative_mask_avoidance_fraction",
            # Report-only signed-view diagnostics. Aggregated mean/std over
            # `*_signed` rows only (other views leave these blank, and the
            # filter on `row[metric_name] != ""` excludes them naturally).
            # These are NOT added to the selection-metric list — they must
            # not influence calibrated top-fraction choice.
            "signed_positive_fraction",
            "signed_prediction_alignment",
        ]:
            values = np.asarray(
                [float(row[metric_name]) for row in rows_for_fraction if row[metric_name] != ""],
                dtype=float,
            )
            if values.size:
                summary[f"{metric_name}_mean"] = round(float(values.mean()), 6)
                summary[f"{metric_name}_std"] = round(float(values.std(ddof=0)), 6)
            else:
                summary[f"{metric_name}_mean"] = ""
                summary[f"{metric_name}_std"] = ""
        summary_rows.append(summary)

    selected_rows: list[dict[str, str | int | float]] = []
    for method_name in sorted({str(row["method"]) for row in summary_rows}):
        candidates = [
            row for row in summary_rows if row["method"] == method_name]
        valid_candidates = [
            row for row in candidates if row.get(f"{args.selection_metric}_mean") != ""
        ]
        if not valid_candidates:
            continue
        selected = max(valid_candidates, key=lambda row: float(
            row[f"{args.selection_metric}_mean"]))
        selected_rows.append(
            {
                "method": method_name,
                "selected_fraction": selected["top_fraction"],
                "selection_metric": args.selection_metric,
                "selection_metric_mean": selected[f"{args.selection_metric}_mean"],
                "n": selected["n"],
            }
        )

    selected_by_metric_rows: list[dict[str, str | int | float]] = []
    for metric_name in [
        "iou",
        "dice",
        "precision_at_fraction",
        "pointing_hit",
        "negative_mask_avoidance_fraction",
        "negative_mask_overlap_fraction",
    ]:
        for method_name in sorted({str(row["method"]) for row in summary_rows}):
            candidates = [
                row
                for row in summary_rows
                if row["method"] == method_name and row.get(f"{metric_name}_mean") != ""
            ]
            if not candidates:
                continue
            selected = max(candidates, key=lambda row: float(row[f"{metric_name}_mean"]))
            selected_by_metric_rows.append(
                {
                    "method": method_name,
                    "selected_fraction": selected["top_fraction"],
                    "selection_metric": metric_name,
                    "selection_metric_mean": selected[f"{metric_name}_mean"],
                    "n": selected["n"],
                }
            )

    write_rows(output_dir / "calibration_metrics.csv", metric_rows)
    write_rows(output_dir / "calibration_summary.csv", summary_rows)
    write_rows(output_dir / "selected_fractions.csv", selected_rows)
    write_rows(output_dir / "selected_fractions_by_metric.csv", selected_by_metric_rows)
    # Phase 1.2.5: v2-canonical alias. Downstream scripts (smoke,
    # improvement experiment) consume calibrated top-fractions per
    # (method, selection_metric) via --calibrated-fractions and rely on
    # this filename to disambiguate v2 calibration from v1 outputs.
    write_rows(output_dir / "calibrated_thresholds_v2.csv", selected_by_metric_rows)

    run_meta_path = write_run_metadata(
        output_dir,
        args,
        classifier_threshold=args.classifier_threshold,
        faithfulness_baseline=None,
        weights=args.weights,
        split=args.split,
    )

    print(f"CXR XAI threshold calibration complete on {device}.")
    print(f"Positive calibration cases: {len(rows)}")
    print(f"Selection metric: {args.selection_metric}")
    print(
        f"Selected fractions written to: {output_dir / 'selected_fractions.csv'}")
    print(
        f"v2 calibrated thresholds: {output_dir / 'calibrated_thresholds_v2.csv'}")
    print(f"Run metadata written to: {run_meta_path}")


if __name__ == "__main__":
    main()
