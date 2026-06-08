#!/usr/bin/env python3
"""Head-CT held-out consensus-vs-individual improvement experiment (Phase 5.4).

CT analogue of `run_improvement_experiment.py`, scoped to the three
input-space methods that transfer with byte-identical `xai.py` code
(Integrated Gradients, GradientSHAP, Occlusion) plus the exploratory
3-method `consensus_input3`. Mirrors the CXR protocol so the two
modalities are comparable:

  1. Calibrate a per-method top-fraction on the TRAIN split (pick the
     fraction maximizing the selection metric's mean).
  2. Freeze those fractions and evaluate held-out TEST positives.
  3. Compare the consensus reference against each individual method with
     paired Wilcoxon signed-rank + Holm-Bonferroni and a 10k bootstrap CI
     on the paired median difference.

The CT consensus is the 3-method input-space aggregate, NOT the frozen
CXR 4-method consensus (Grad-CAM cannot transfer to the ViT); this is
stated wherever the result is reported.
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from explainai_thesis.cli.common import add_device_arg, resolve_device
from explainai_thesis.cli.progress import RollingLogDisplay
from explainai_thesis.ct.io import extract_slice, load_nifti_volume
from explainai_thesis.ct.models import load_ct_classifier
from explainai_thesis.metrics import localization_metrics
from explainai_thesis.run_metadata import write_run_metadata
from explainai_thesis.stats import wilcoxon_paired
from explainai_thesis.xai import (
    SignedAttribution,
    consensus_signed,
    gradient_shap_signed,
    integrated_gradients_signed,
    iter_method_views,
    occlusion_sensitivity_signed,
)


METRICS = ("iou", "dice", "pointing_hit", "precision_at_fraction")
CONSENSUS_NAME = "consensus_input3"
METHOD_ORDER = ("integrated_gradients", "gradient_shap", "occlusion", CONSENSUS_NAME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/ct_hemorrhage_manifest.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--max-positive-cal", type=int, default=0, help="0 = all train positives.")
    parser.add_argument("--max-positive-eval", type=int, default=0, help="0 = all test positives.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--fractions", default="0.05,0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--selection-metric", default="dice", choices=METRICS)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--gradshap-samples", type=int, default=8)
    parser.add_argument("--gradshap-stdevs", type=float, default=0.02)
    parser.add_argument("--occlusion-patch-size", type=int, default=32)
    parser.add_argument("--occlusion-stride", type=int, default=16)
    parser.add_argument("--reference-method", default=CONSENSUS_NAME)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--window-width", type=float, default=80.0)
    add_device_arg(parser, help=None)
    return parser.parse_args()


def read_positive_masked_ct_rows(manifest_path: Path, split: str, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["label"]) != 1 or not row.get("mask_path"):
                continue
            if split != "any" and row.get("split") != split:
                continue
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def load_mask_slice(mask_path: Path, slice_index: int, image_size: int) -> torch.Tensor:
    mask_2d = extract_slice(load_nifti_volume(mask_path), slice_index, axis=2) > 0
    resized = Image.fromarray((mask_2d * 255).astype(np.uint8)).resize(
        (image_size, image_size), Image.NEAREST
    )
    return torch.from_numpy(np.asarray(resized) > 0)


def parse_fractions(raw: str) -> list[float]:
    fractions = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if not fractions:
        raise ValueError("At least one fraction is required.")
    return fractions


def positive_heatmaps(bundle, model_input: torch.Tensor, args) -> dict[str, torch.Tensor]:
    """Compute the four methods' attributions once and return the
    positive-view heatmap for each (keyed by method/family name)."""
    ig = integrated_gradients_signed(
        bundle.model, model_input, class_idx=bundle.class_idx, steps=args.ig_steps
    )
    gshap = gradient_shap_signed(
        bundle.model, model_input, class_idx=bundle.class_idx,
        samples=args.gradshap_samples, stdevs=args.gradshap_stdevs,
    )
    occ = occlusion_sensitivity_signed(
        bundle.model, model_input, class_idx=bundle.class_idx,
        patch_size=args.occlusion_patch_size, stride=args.occlusion_stride,
    )
    attrs: dict[str, SignedAttribution] = {
        "integrated_gradients": ig,
        "gradient_shap": gshap,
        "occlusion": occ,
        CONSENSUS_NAME: consensus_signed([ig, gshap, occ]),
    }
    return {
        mv.method: mv.heatmap
        for mv in iter_method_views(attrs)
        if mv.view == "positive"
    }


def calibrate_fractions(
    bundle, rows, fractions, args, device, progress, run_start
) -> dict[str, float]:
    """Per-method top-fraction that maximizes the selection metric's mean
    on the calibration split."""
    sums: dict[tuple[str, float], float] = defaultdict(float)
    counts: dict[tuple[str, float], int] = defaultdict(int)
    for sample_idx, row in enumerate(rows):
        progress.log(f"calibrate {sample_idx + 1}/{len(rows)} | {row['filename']}", run_start)
        slice_index = int(row["slice_index"])
        slice_hu = extract_slice(load_nifti_volume(Path(row["image_path"])), slice_index, axis=2)
        model_input = bundle.preprocess(slice_hu).to(device)
        mask = load_mask_slice(Path(row["mask_path"]), slice_index, args.image_size)
        heatmaps = positive_heatmaps(bundle, model_input, args)
        for method, heatmap in heatmaps.items():
            for fraction in fractions:
                value = localization_metrics(heatmap, mask, fraction=fraction)[args.selection_metric]
                sums[(method, fraction)] += float(value)
                counts[(method, fraction)] += 1
    selected: dict[str, float] = {}
    for method in METHOD_ORDER:
        best_fraction, best_mean = None, -1.0
        for fraction in fractions:
            key = (method, fraction)
            if counts[key] == 0:
                continue
            mean = sums[key] / counts[key]
            if mean > best_mean:
                best_mean, best_fraction = mean, fraction
        if best_fraction is not None:
            selected[method] = best_fraction
    return selected


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_start = time.perf_counter()
    progress = RollingLogDisplay(line_count=15)
    progress.log("started head-CT improvement experiment", run_start)

    device = resolve_device(args.device)
    fractions = parse_fractions(args.fractions)
    manifest_path = Path(args.manifest)
    cal_rows = read_positive_masked_ct_rows(manifest_path, args.calibration_split, args.max_positive_cal)
    eval_rows = read_positive_masked_ct_rows(manifest_path, args.eval_split, args.max_positive_eval)
    if not cal_rows or not eval_rows:
        raise RuntimeError(
            f"Need both splits populated: cal={len(cal_rows)} eval={len(eval_rows)}."
        )

    progress.log(f"loading CT classifier on {device}", run_start)
    bundle = load_ct_classifier(device=device, window_width=args.window_width)
    progress.log(
        f"calibrating on {len(cal_rows)} {args.calibration_split} positives "
        f"({args.selection_metric}, fractions={fractions})",
        run_start,
    )
    calibrated = calibrate_fractions(bundle, cal_rows, fractions, args, device, progress, run_start)
    progress.log(f"calibrated fractions: {calibrated}", run_start)

    metric_rows: list[dict[str, str | int | float]] = []
    for sample_idx, row in enumerate(eval_rows):
        progress.log(f"eval {sample_idx + 1}/{len(eval_rows)} | {row['filename']}", run_start)
        slice_index = int(row["slice_index"])
        slice_hu = extract_slice(load_nifti_volume(Path(row["image_path"])), slice_index, axis=2)
        model_input = bundle.preprocess(slice_hu).to(device)
        mask = load_mask_slice(Path(row["mask_path"]), slice_index, args.image_size)
        heatmaps = positive_heatmaps(bundle, model_input, args)
        for method, heatmap in heatmaps.items():
            top_fraction = calibrated.get(method, 0.2)
            metrics = localization_metrics(heatmap, mask, fraction=top_fraction)
            metric_rows.append({
                "sample_id": sample_idx,
                "filename": row["filename"],
                "split": row.get("split", ""),
                "subtype": row.get("subtype", ""),
                "method": method,
                "top_fraction": round(top_fraction, 6),
                **{key: round(value, 6) for key, value in metrics.items()},
            })

    if not any(r["method"] == args.reference_method for r in metric_rows):
        raise RuntimeError(f"Reference method {args.reference_method!r} not produced.")

    # Paired consensus-vs-individual statistics per metric.
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in metric_rows:
        for metric in METRICS:
            grouped[metric][str(row["method"])].append(float(row[metric]))
    paired_rows: list[dict[str, str | int | float | bool]] = []
    for metric in METRICS:
        reference = np.asarray(grouped[metric][args.reference_method], dtype=float)
        alternatives = {
            m: np.asarray(v, dtype=float)
            for m, v in grouped[metric].items() if m != args.reference_method
        }
        for method, values in wilcoxon_paired(
            reference, alternatives, alpha=args.alpha,
            n_resamples=args.bootstrap_resamples, seed=args.seed,
        ).items():
            paired_rows.append({
                "metric": metric, "reference": args.reference_method,
                "compared": method, **values,
            })

    def write_rows(path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    progress.log("writing outputs", run_start)
    write_rows(output_dir / "ct_improvement_experiment.csv", metric_rows)
    write_rows(output_dir / "ct_improvement_experiment_paired.csv", paired_rows)
    write_rows(
        output_dir / "ct_calibrated_fractions.csv",
        [{"method": m, "selected_fraction": f, "selection_metric": args.selection_metric}
         for m, f in calibrated.items()],
    )
    run_meta_path = write_run_metadata(
        output_dir, args,
        model_id="DifeiT/rsna-intracranial-hemorrhage-detection",
        attribution_target="1 - P(normal)",
        reference_method=args.reference_method,
        selection_metric=args.selection_metric,
        modality="ct",
    )

    progress.log("CT improvement experiment complete", run_start)
    progress.finish()
    print(f"Head-CT improvement experiment complete on {device}.")
    print(f"Calibration positives: {len(cal_rows)} ({args.calibration_split})")
    print(f"Held-out eval positives: {len(eval_rows)} ({args.eval_split})")
    print(f"Calibrated fractions: {calibrated}")
    print(f"Per-case metrics: {output_dir / 'ct_improvement_experiment.csv'}")
    print(f"Paired statistics: {output_dir / 'ct_improvement_experiment_paired.csv'}")
    print(f"Run metadata: {run_meta_path}")


if __name__ == "__main__":
    main()
