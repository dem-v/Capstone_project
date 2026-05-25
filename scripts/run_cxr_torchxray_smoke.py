#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from explainai_thesis.xai import (
    GradCAM,
    SignedAttribution,
    agreement_score,
    iter_method_views,
)
from explainai_thesis.cli.common import resolve_device
from explainai_thesis.cxr.classifier import load_classifier
from explainai_thesis.cxr.methods import (
    MethodContext,
    compute_signed_attributions,
)
from explainai_thesis.cxr.io import (
    load_binary_mask,
    load_xray_image,
    parse_optional_fractions,
    read_calibrated_fractions,
    read_positive_masked_rows,
    safe_case_name,
    safe_source_stem,
)
from explainai_thesis.faithfulness import (
    curve_auc,
    faithfulness_baseline_tensor,
    faithfulness_curve_rows,
    faithfulness_method_family,
    model_probability,
    plot_faithfulness_curves,
    plot_faithfulness_summary,
    write_faithfulness_plots,
    write_faithfulness_summary,
)
from explainai_thesis.visualization import save_overlay, signed_diverging_overlay
from explainai_thesis.metrics import (
    localization_metrics,
    negative_evidence_metrics,
    normalize_map,
    threshold_top_fraction,
)
from explainai_thesis.io import (
    AGREEMENT_FIELDS,
    FAITHFULNESS_CURVE_FIELDS,
    METRICS_FIELDS,
)
from explainai_thesis.run_metadata import write_run_metadata
from PIL import Image
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
        help=(
            "TorchXRayVision classifier weights name. Routed through "
            "`load_classifier()`. Supported families: DenseNet-121 "
            "(`densenet121-res224-*`, native 224x224), ResNet-50 "
            "(`resnet50-res512-all`, native 512x512 -- pass `--image-size 512`)."
        ),
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
        default="black",
        choices=["zero_tensor", "black", "white", "case_mean"],
        help=(
            "Baseline used for deletion/insertion faithfulness. "
            "Default 'black' per AGENTS.md (Faithfulness Evaluation Rules): "
            "'zero_tensor' is historical / not recommended because it is not a "
            "true black image in the normalized TorchXRayVision input space and "
            "can still score ~60pct pneumothorax. 'black'/'white'/'case_mean' use "
            "normalized image-space baselines."
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
    parser.add_argument(
        "--classifier-threshold",
        type=float,
        default=0.62,
        help=(
            "Sigmoid cutoff for deriving the model's predicted class on each "
            "case. Used only by the `signed_prediction_alignment` (SPA) column "
            "on `*_signed` rows of metrics.csv; does NOT affect overlays, "
            "calibrated top-fractions, or faithfulness. Default 0.62 matches "
            "the train-calibrated TorchXRayVision cutoff in AGENTS.md."
        ),
    )
    return parser.parse_args()


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
        # Report-only signed-view diagnostics. Populated only on `*_signed`
        # rows; the `value != ""` filter below excludes other views naturally.
        "signed_positive_fraction",
        "signed_prediction_alignment",
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


def is_negative_method(method_name: str) -> bool:
    return method_name.endswith("_negative")


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
    rows = read_positive_masked_rows(
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

    # Phase 1.7 seam: route classifier loading through `load_classifier(name)`.
    # Behaviour for the default `--weights densenet121-res224-all` is identical
    # to the pre-seam path (same `xrv.models.DenseNet` constructor, same
    # `.to(device).eval()`, same `model.features.denseblock4` Grad-CAM layer,
    # same Pneumothorax pathology index). Diagnostic A/B runs simply pass a
    # different `--weights` value; no other code in this script changes.
    bundle = load_classifier(args.weights, device=device, pathology="Pneumothorax")
    model = bundle.model
    class_idx = bundle.class_idx
    gradcam = GradCAM(model, bundle.target_layer)

    metric_rows: list[dict[str, str | int | float]] = []
    faithfulness_rows: list[dict[str, str | int | float]] = []
    agreement_rows: list[dict[str, str | int | float]] = []
    for sample_idx, row in enumerate(rows):
        image = load_xray_image(Path(row["image_path"]), args.image_size)
        mask = load_binary_mask(Path(row["mask_path"]), args.image_size)
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

        # Phase 1.2-dispatch + Phase 2 MethodSpec registry: 5 signed cores
        # per case (one forward/backward or one occlusion sweep each)
        # replace the pre-1.2 16-call polarity fan-out. The four views
        # (positive/negative/magnitude/signed) are derived in microseconds
        # from each SignedAttribution by `iter_method_views`. The registry
        # in `explainai_thesis.cxr.methods` is the single place to add new
        # signed-attribution methods (Eigen-CAM, Score-CAM).
        method_ctx = MethodContext(
            model=model,
            model_input=model_input,
            class_idx=class_idx,
            gradcam=gradcam,
            ig_steps=args.ig_steps,
            gradshap_samples=args.gradshap_samples,
            gradshap_stdevs=args.gradshap_stdevs,
            occlusion_patch_size=args.occlusion_patch_size,
            occlusion_stride=args.occlusion_stride,
        )
        signed_attributions: dict[str, SignedAttribution] = (
            compute_signed_attributions(method_ctx)
        )

        # Canonical v2 method-view expansion. `MethodView.view` ∈
        # {"positive", "negative", "magnitude", "signed"} and drives both
        # metrics provenance and overlay color choice.
        method_views = iter_method_views(signed_attributions)

        # Back-compat tensors for downstream consumers that haven't been
        # ported yet (e.g. the negative_evidence_metrics block below uses
        # the family.negative view as the "negative evidence" reference).
        method_heatmaps = {method_view.method: method_view.heatmap for method_view in method_views}
        cam_map = method_heatmaps["grad_cam"]
        negative_cam_map = method_heatmaps["grad_cam_negative"]
        ig_map = method_heatmaps["integrated_gradients"]
        ig_negative_map = method_heatmaps["integrated_gradients_negative"]
        gradient_shap_map = method_heatmaps["gradient_shap"]
        gradient_shap_negative_map = method_heatmaps["gradient_shap_negative"]
        occlusion_map = method_heatmaps["occlusion"]

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

        for method_view in method_views:
            method_name = method_view.method
            heatmap = method_view.heatmap
            view_kind = method_view.view
            family = method_view.family
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
                # SPA: model-relative coherence diagnostic. On a positive
                # prediction it equals SPF; on a negative prediction it
                # equals 1 - SPF. Lets TN cases (model says "no", evidence
                # should be blue-side) be summarized on the same scale as
                # TP cases. Report-only — not used for top-fraction
                # selection or overlay rendering.
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
                    "top_fraction": round(top_fraction, 6),
                    **{key: round(value, 6) for key, value in metrics.items()},
                    **negative_metrics,
                    "signed_positive_fraction": signed_positive_fraction,
                    "signed_prediction_alignment": signed_prediction_alignment,
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
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRICS_FIELDS))
        writer.writeheader()
        writer.writerows(metric_rows)

    # Cross-method agreement: cosine similarity between signed maps,
    # per unordered method pair, per case. Empty file emitted on
    # single-method runs so downstream tooling can rely on its presence.
    agreement_path = output_dir / "agreement.csv"
    with agreement_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AGREEMENT_FIELDS))
        writer.writeheader()
        writer.writerows(agreement_rows)

    if faithfulness_fractions:
        faithfulness_path = output_dir / "faithfulness_curves.csv"
        with faithfulness_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(FAITHFULNESS_CURVE_FIELDS)
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

    run_meta_path = write_run_metadata(
        output_dir,
        args,
        classifier_threshold=args.classifier_threshold,
        faithfulness_baseline=args.faithfulness_baseline,
        weights=args.weights,
        split=args.split,
    )

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
    print(f"Run metadata written to: {run_meta_path}")


if __name__ == "__main__":
    main()
