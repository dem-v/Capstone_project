from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from explainai_thesis.cli.common import resolve_device
from explainai_thesis.cli.progress import RollingLogDisplay
from explainai_thesis.cxr.classifier import load_classifier
from explainai_thesis.cxr.io import (
    load_binary_mask,
    load_xray_image,
    read_positive_masked_rows,
)
from explainai_thesis.cxr.methods import MethodContext, compute_signed_attributions
from explainai_thesis.metrics import localization_metrics
from explainai_thesis.run_metadata import write_run_metadata
from explainai_thesis.stats import wilcoxon_paired
from explainai_thesis.xai import GradCAM, iter_method_views


METRICS = ("iou", "dice", "pointing_hit", "precision_at_fraction")

IMPROVEMENT_FIELDS = [
    "sample_id",
    "filename",
    "split",
    "weights",
    "method",
    "view",
    "family",
    "top_fraction",
    "iou",
    "dice",
    "pointing_hit",
    "precision_at_fraction",
]

PAIRED_FIELDS = [
    "metric",
    "reference",
    "compared",
    "n_pairs",
    "median_diff",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "wilcoxon_stat",
    "p_raw",
    "p_holm_adjusted",
    "p_holm_threshold",
    "holm_significant_bool",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the held-out consensus-vs-individual CXR XAI improvement experiment."
    )
    parser.add_argument("--manifest", default="data/cxr_pneumothorax_manifest.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-csv", required=True)
    parser.add_argument(
        "--weights",
        default="densenet121-res224-all",
        help="TorchXRayVision classifier weights name; use --image-size 512 for resnet50-res512-all.",
    )
    parser.add_argument("--split", default="test", choices=["test", "train", "any"])
    parser.add_argument("--max-positive", type=int, default=0, help="0 means all positive masked rows.")
    parser.add_argument("--random-sample", action="store_true")
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--gradshap-samples", type=int, default=8)
    parser.add_argument("--gradshap-stdevs", type=float, default=0.02)
    parser.add_argument("--occlusion-patch-size", type=int, default=32)
    parser.add_argument("--occlusion-stride", type=int, default=16)
    parser.add_argument(
        "--score-cam-channels-cap",
        type=int,
        default=256,
        help="Maximum Score-CAM activation channels; use 0 to evaluate all channels.",
    )
    parser.add_argument(
        "--selection-metric",
        default="dice",
        choices=METRICS,
        help="Calibration metric used to choose the frozen top-fraction per positive-view method.",
    )
    parser.add_argument("--reference-method", default="consensus")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--default-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--allow-stale-calibration",
        action="store_true",
        help="Allow rerunning into an output folder whose existing CSV is newer than the calibration CSV.",
    )
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, str | int | float | bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_calibrated_fractions_by_metric(
    path: Path,
    selection_metric: str,
) -> dict[str, float]:
    fractions: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("selection_metric") != selection_metric:
                continue
            method = row.get("method")
            selected_fraction = row.get("selected_fraction")
            if method and selected_fraction:
                fractions[method] = float(selected_fraction)
    if not fractions:
        raise RuntimeError(
            f"No calibrated fractions found in {path} for selection_metric={selection_metric!r}."
        )
    return fractions


def guard_existing_outputs(
    output_dir: Path,
    calibration_csv: Path,
    *,
    allow_stale_calibration: bool,
) -> None:
    existing_metrics = output_dir / "improvement_experiment.csv"
    if not existing_metrics.exists() or allow_stale_calibration:
        return
    if calibration_csv.stat().st_mtime > existing_metrics.stat().st_mtime:
        raise RuntimeError(
            "Calibration CSV is newer than existing improvement_experiment.csv. "
            "Use a fresh output directory or pass --allow-stale-calibration to overwrite knowingly."
        )


def paired_rows(
    metric_rows: list[dict[str, str | int | float | bool]],
    *,
    reference_method: str,
    alpha: float,
    bootstrap_resamples: int,
    seed: int,
) -> list[dict[str, str | int | float | bool]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in metric_rows:
        method = str(row["method"])
        for metric in METRICS:
            grouped[metric][method].append(float(row[metric]))

    rows: list[dict[str, str | int | float | bool]] = []
    for metric in METRICS:
        if reference_method not in grouped[metric]:
            raise RuntimeError(f"Reference method {reference_method!r} is missing for metric {metric!r}.")
        reference = np.asarray(grouped[metric][reference_method], dtype=float)
        alternatives = {
            method: np.asarray(values, dtype=float)
            for method, values in grouped[metric].items()
            if method != reference_method
        }
        stats_rows = wilcoxon_paired(
            reference,
            alternatives,
            alpha=alpha,
            n_resamples=bootstrap_resamples,
            seed=seed,
        )
        for method, values in stats_rows.items():
            rows.append(
                {
                    "metric": metric,
                    "reference": reference_method,
                    "compared": method,
                    **values,
                }
            )
    return rows


def write_summary(output_path: Path, paired_metric_rows: list[dict[str, str | int | float | bool]]) -> None:
    lines = [
        "### Improvement experiment summary",
        "",
        "- Reference method: `" + str(paired_metric_rows[0]["reference"]) + "`" if paired_metric_rows else "- No paired rows produced.",
        "- Positive-view localization metrics only: `IoU`, `Dice`, `pointing_hit`, `precision_at_fraction`.",
        "- Statistical test: paired Wilcoxon signed-rank, two-sided, with Holm-Bonferroni correction.",
        "",
    ]
    for metric in METRICS:
        metric_rows = [row for row in paired_metric_rows if row["metric"] == metric]
        significant = [row for row in metric_rows if row["holm_significant_bool"]]
        if significant:
            winners = ", ".join(
                f"`{row['compared']}` (median Δ={float(row['median_diff']):.4f})"
                for row in significant
            )
            lines.append(f"- `{metric}`: reference differs significantly from {winners}.")
        else:
            lines.append(f"- `{metric}`: no Holm-significant reference-vs-individual difference.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_boxplots(metric_rows: list[dict[str, str | int | float | bool]], output_path: Path) -> None:
    methods = sorted({str(row["method"]) for row in metric_rows})
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for axis, metric in zip(axes.ravel(), METRICS):
        data = [
            [float(row[metric]) for row in metric_rows if row["method"] == method]
            for method in methods
        ]
        axis.boxplot(data, labels=methods, showfliers=False)
        axis.set_title(metric)
        axis.set_ylim(0.0, 1.0)
        axis.tick_params(axis="x", rotation=45)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_paired_differences(
    paired_metric_rows: list[dict[str, str | int | float | bool]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for axis, metric in zip(axes.ravel(), METRICS):
        rows = [row for row in paired_metric_rows if row["metric"] == metric]
        labels = [str(row["compared"]) for row in rows]
        medians = np.asarray([float(row["median_diff"]) for row in rows], dtype=float)
        lows = np.asarray([float(row["bootstrap_ci_low"]) for row in rows], dtype=float)
        highs = np.asarray([float(row["bootstrap_ci_high"]) for row in rows], dtype=float)
        lower_err = np.maximum(0.0, medians - lows)
        upper_err = np.maximum(0.0, highs - medians)
        axis.bar(labels, medians, yerr=np.vstack([lower_err, upper_err]), capsize=3)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(metric)
        axis.tick_params(axis="x", rotation=45)
        axis.set_ylabel("reference − compared")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    calibration_csv = Path(args.calibration_csv)
    if not calibration_csv.exists():
        raise FileNotFoundError(calibration_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    guard_existing_outputs(
        output_dir,
        calibration_csv,
        allow_stale_calibration=args.allow_stale_calibration,
    )

    run_start = time.perf_counter()
    progress = RollingLogDisplay(line_count=15)
    progress.log("started CXR XAI improvement experiment", run_start)

    device = resolve_device(args.device)
    row_limit = args.max_positive if args.max_positive > 0 else 10**9
    rows = read_positive_masked_rows(
        manifest_path,
        args.split,
        row_limit,
        random_sample=args.random_sample,
        seed=args.seed,
    )
    if args.max_positive > 0:
        rows = rows[: args.max_positive]
    if not rows:
        raise RuntimeError(
            f"No positive rows with masks found in {manifest_path} for split={args.split}."
        )

    calibrated_fractions = read_calibrated_fractions_by_metric(
        calibration_csv,
        args.selection_metric,
    )

    progress.log(f"loading classifier {args.weights} on {device}", run_start)
    bundle = load_classifier(args.weights, device=device, pathology="Pneumothorax")
    model = bundle.model
    class_idx = bundle.class_idx
    gradcam = GradCAM(model, bundle.target_layer)
    progress.log(
        f"classifier ready | cases={len(rows)} selection_metric={args.selection_metric} "
        f"ig_steps={args.ig_steps} gradshap_samples={args.gradshap_samples} "
        f"score_cam_channels_cap={args.score_cam_channels_cap}",
        run_start,
    )

    metric_rows: list[dict[str, str | int | float | bool]] = []
    for sample_idx, row in enumerate(rows):
        case_filename = row.get("filename", Path(row["image_path"]).name)
        progress.log(
            f"case {sample_idx + 1}/{len(rows)} | start | {case_filename}",
            run_start,
        )
        image = load_xray_image(Path(row["image_path"]), args.image_size)
        mask = load_binary_mask(Path(row["mask_path"]), args.image_size)
        model_input = image.unsqueeze(0).to(device)

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
            score_cam_channels_cap=args.score_cam_channels_cap,
        )
        progress.log(
            f"  case {sample_idx + 1}: computing signed attributions (7 methods + consensus)",
            run_start,
        )
        signed_attributions = compute_signed_attributions(method_ctx)
        for method_view in iter_method_views(signed_attributions):
            if method_view.view != "positive":
                continue
            method_name = method_view.method
            top_fraction = calibrated_fractions.get(method_name, args.default_fraction)
            metrics = localization_metrics(method_view.heatmap, mask, fraction=top_fraction)
            metric_rows.append(
                {
                    "sample_id": sample_idx,
                    "filename": row.get("filename", Path(row["image_path"]).name),
                    "split": row.get("split", ""),
                    "weights": args.weights,
                    "method": method_name,
                    "view": method_view.view,
                    "family": method_view.family,
                    "top_fraction": round(top_fraction, 6),
                    **{key: round(value, 6) for key, value in metrics.items()},
                }
            )
        progress.log(
            f"case {sample_idx + 1}/{len(rows)} | done | {case_filename}",
            run_start,
        )

    gradcam.remove_hooks()
    progress.log("all cases processed; running paired Wilcoxon + Holm-Bonferroni", run_start)

    if not any(row["method"] == args.reference_method for row in metric_rows):
        raise RuntimeError(f"Reference method {args.reference_method!r} was not produced.")

    paired_metric_rows = paired_rows(
        metric_rows,
        reference_method=args.reference_method,
        alpha=args.alpha,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )

    progress.log("writing CSV/summary/plot outputs", run_start)
    write_rows(output_dir / "improvement_experiment.csv", metric_rows)
    write_rows(output_dir / "improvement_experiment_paired.csv", paired_metric_rows)
    write_summary(output_dir / "improvement_experiment_summary.md", paired_metric_rows)
    plot_boxplots(metric_rows, output_dir / "improvement_experiment_boxplots.png")
    plot_paired_differences(paired_metric_rows, output_dir / "improvement_experiment_paired_diff.png")
    run_meta_path = write_run_metadata(
        output_dir,
        args,
        weights=args.weights,
        split=args.split,
        calibration_csv=str(calibration_csv),
        selection_metric=args.selection_metric,
        reference_method=args.reference_method,
    )

    progress.log("improvement experiment complete", run_start)
    progress.finish()
    print(f"Improvement experiment complete on {device}.")
    print(f"Positive held-out cases: {len(rows)}")
    print(f"Per-case metrics: {output_dir / 'improvement_experiment.csv'}")
    print(f"Paired statistics: {output_dir / 'improvement_experiment_paired.csv'}")
    print(f"Run metadata written to: {run_meta_path}")


if __name__ == "__main__":
    main()