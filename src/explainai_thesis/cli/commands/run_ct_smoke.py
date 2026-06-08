#!/usr/bin/env python3
"""Head-CT hemorrhage XAI smoke (Phase 5.4, Branch A).

CT analogue of `run_cxr_torchxray_smoke.py`, deliberately scoped to the
methods whose `xai.py` implementations are byte-identical across
modalities: Integrated Gradients, GradientSHAP, Occlusion. This keeps the
cross-modality transfer comparison controlled. A clearly-labeled
exploratory 3-method consensus (`consensus_input3`) is included as a
secondary finding — it is NOT the frozen CXR 4-method consensus (which
includes Grad-CAM and cannot transfer to a ViT without an uncontrolled
token->grid reimplementation).

The model is the DifeiT ViT wrapped to expose 1 - P(normal) at
`class_idx = 0` (see `ct/models.py`). Faithfulness baseline defaults to
`zero_tensor`, which in this model's [-1, 1] input space equals the
brain-window-center fill (HU=40 midpoint) — the principled CT neutral.
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from PIL import Image

from explainai_thesis.cli.common import add_device_arg, add_split_arg, resolve_device
from explainai_thesis.cli.progress import RollingLogDisplay
from explainai_thesis.ct.io import extract_slice, load_nifti_volume
from explainai_thesis.ct.models import load_ct_classifier
from explainai_thesis.faithfulness import (
    faithfulness_baseline_tensor,
    faithfulness_curve_rows,
    model_probability,
    write_faithfulness_plots,
    write_faithfulness_summary,
)
from explainai_thesis.metrics import localization_metrics
from explainai_thesis.run_metadata import write_run_metadata
from explainai_thesis.xai import (
    SignedAttribution,
    agreement_score,
    consensus_signed,
    gradient_shap_signed,
    integrated_gradients_signed,
    iter_method_views,
    occlusion_sensitivity_signed,
)


# The three input-space methods that transfer with identical code.
TRANSFER_METHODS = ("integrated_gradients", "gradient_shap", "occlusion")
CONSENSUS_NAME = "consensus_input3"
METRIC_KEYS = ("iou", "dice", "pointing_hit", "precision_at_fraction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Head-CT hemorrhage XAI smoke (input-space methods).")
    parser.add_argument("--manifest", default="data/ct_hemorrhage_manifest.csv")
    parser.add_argument("--output-dir", required=True)
    add_split_arg(parser, choices=("test", "train", "any"), help=None)
    parser.add_argument("--max-positive", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--gradshap-samples", type=int, default=8)
    parser.add_argument("--gradshap-stdevs", type=float, default=0.02)
    parser.add_argument("--occlusion-patch-size", type=int, default=32)
    parser.add_argument("--occlusion-stride", type=int, default=16)
    parser.add_argument("--faithfulness-fractions", default="")
    parser.add_argument(
        "--faithfulness-baseline",
        default="zero_tensor",
        choices=["zero_tensor", "black", "white", "case_mean"],
        help="Default zero_tensor == brain-window-center (HU=40) in this ViT's "
        "[-1,1] input space. 'black' (-1024) is the out-of-range air stress baseline.",
    )
    add_device_arg(parser, help=None)
    parser.add_argument("--window-width", type=float, default=80.0)
    return parser.parse_args()


def read_positive_masked_ct_rows(manifest_path: Path, split: str, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["label"]) != 1:
                continue
            if split != "any" and row.get("split") != split:
                continue
            if not row.get("mask_path"):
                continue
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def load_mask_slice(mask_path: Path, slice_index: int, image_size: int) -> torch.Tensor:
    mask_vol = load_nifti_volume(mask_path)
    mask_2d = extract_slice(mask_vol, slice_index, axis=2) > 0
    resized = Image.fromarray((mask_2d * 255).astype(np.uint8)).resize(
        (image_size, image_size), Image.NEAREST
    )
    return torch.from_numpy(np.asarray(resized) > 0)


def parse_fractions(raw: str) -> list[float]:
    return [float(v.strip()) for v in raw.split(",") if v.strip()]


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_start = time.perf_counter()
    progress = RollingLogDisplay(line_count=15)
    progress.log("started head-CT hemorrhage XAI smoke", run_start)

    device = resolve_device(args.device)
    faithfulness_fractions = parse_fractions(args.faithfulness_fractions)
    rows = read_positive_masked_ct_rows(manifest_path, args.split, args.max_positive)
    if not rows:
        raise RuntimeError(f"No positive masked CT rows in {manifest_path} for split={args.split}.")

    progress.log(f"loading CT classifier (DifeiT ViT) on {device}", run_start)
    bundle = load_ct_classifier(device=device, window_width=args.window_width)
    model = bundle.model
    class_idx = bundle.class_idx
    progress.log(
        f"classifier ready | cases={len(rows)} top_fraction={args.top_fraction} "
        f"ig_steps={args.ig_steps} baseline={args.faithfulness_baseline}",
        run_start,
    )

    metric_rows: list[dict[str, str | int | float]] = []
    faithfulness_rows: list[dict[str, str | int | float]] = []
    agreement_rows: list[dict[str, str | int | float]] = []

    for sample_idx, row in enumerate(rows):
        slice_index = int(row["slice_index"])
        progress.log(
            f"case {sample_idx + 1}/{len(rows)} | start | {row['filename']}",
            run_start,
        )
        volume = load_nifti_volume(Path(row["image_path"]))
        slice_hu = extract_slice(volume, slice_index, axis=2)
        model_input = bundle.preprocess(slice_hu).to(device)
        mask = load_mask_slice(Path(row["mask_path"]), slice_index, args.image_size)
        faithfulness_baseline = faithfulness_baseline_tensor(model_input, args.faithfulness_baseline)

        with torch.inference_mode():
            hemorrhage_prob = float(torch.sigmoid(model(model_input)[0, class_idx]))

        progress.log(f"  case {sample_idx + 1}: IG / GradientSHAP / Occlusion", run_start)
        ig = integrated_gradients_signed(model, model_input, class_idx=class_idx, steps=args.ig_steps)
        gshap = gradient_shap_signed(
            model, model_input, class_idx=class_idx,
            samples=args.gradshap_samples, stdevs=args.gradshap_stdevs,
        )
        occ = occlusion_sensitivity_signed(
            model, model_input, class_idx=class_idx,
            patch_size=args.occlusion_patch_size, stride=args.occlusion_stride,
        )
        # Exploratory CT-specific consensus over the three input-space methods.
        # NOT the frozen CXR 4-method consensus (no Grad-CAM on a ViT).
        consensus = consensus_signed([ig, gshap, occ])
        signed_attributions: dict[str, SignedAttribution] = {
            "integrated_gradients": ig,
            "gradient_shap": gshap,
            "occlusion": occ,
            CONSENSUS_NAME: consensus,
        }

        # Cross-method agreement on signed maps (one row per unordered pair).
        families = list(signed_attributions)
        for i, family_a in enumerate(families):
            for family_b in families[i + 1:]:
                agreement_rows.append({
                    "sample_id": sample_idx,
                    "filename": row["filename"],
                    "method_a": family_a,
                    "method_b": family_b,
                    "agreement_score": round(
                        agreement_score(signed_attributions[family_a], signed_attributions[family_b]), 6
                    ),
                })

        for method_view in iter_method_views(signed_attributions):
            if method_view.view != "positive":
                continue
            metrics = localization_metrics(method_view.heatmap, mask, fraction=args.top_fraction)
            metric_rows.append({
                "sample_id": sample_idx,
                "filename": row["filename"],
                "split": row.get("split", ""),
                "subtype": row.get("subtype", ""),
                "slice_index": slice_index,
                "hemorrhage_prob": round(hemorrhage_prob, 6),
                "method": method_view.method,
                "view": method_view.view,
                "family": method_view.family,
                "top_fraction": round(args.top_fraction, 6),
                **{key: round(value, 6) for key, value in metrics.items()},
            })
            for faithfulness_row in faithfulness_curve_rows(
                model, model_input, method_view.heatmap, class_idx,
                faithfulness_fractions, faithfulness_baseline,
            ):
                faithfulness_rows.append({
                    "sample_id": sample_idx,
                    "filename": row["filename"],
                    "method": method_view.method,
                    "baseline": args.faithfulness_baseline,
                    **faithfulness_row,
                })
        progress.log(f"case {sample_idx + 1}/{len(rows)} | done | {row['filename']}", run_start)

    progress.log("writing CSV outputs", run_start)
    metrics_fields = [
        "sample_id", "filename", "split", "subtype", "slice_index",
        "hemorrhage_prob", "method", "view", "family", "top_fraction", *METRIC_KEYS,
    ]
    with (output_dir / "ct_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metrics_fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    with (output_dir / "ct_agreement.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "filename", "method_a", "method_b", "agreement_score"]
        )
        writer.writeheader()
        writer.writerows(agreement_rows)

    # Per-method localization summary.
    grouped: dict[str, list[dict[str, str | int | float]]] = defaultdict(list)
    for metric_row in metric_rows:
        grouped[str(metric_row["method"])].append(metric_row)
    summary_fields = ["method", "n"] + [f"{key}_mean" for key in METRIC_KEYS]
    with (output_dir / "ct_metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for method_name, method_metric_rows in sorted(grouped.items()):
            summary = {"method": method_name, "n": len(method_metric_rows)}
            for key in METRIC_KEYS:
                values = np.asarray([float(r[key]) for r in method_metric_rows], dtype=float)
                summary[f"{key}_mean"] = round(float(values.mean()), 6)
            writer.writerow(summary)

    if faithfulness_fractions:
        with (output_dir / "ct_faithfulness_curves.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["sample_id", "filename", "method", "baseline", "fraction",
                      "insertion_probability", "deletion_probability"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(faithfulness_rows)
        write_faithfulness_summary(faithfulness_rows, output_dir / "ct_faithfulness_summary.csv")
        write_faithfulness_plots(faithfulness_rows, output_dir, "Head-CT hemorrhage faithfulness")

    run_meta_path = write_run_metadata(
        output_dir, args,
        model_id="DifeiT/rsna-intracranial-hemorrhage-detection",
        attribution_target="1 - P(normal)",
        faithfulness_baseline=args.faithfulness_baseline,
        split=args.split,
        modality="ct",
    )

    progress.log("CT smoke complete", run_start)
    progress.finish()
    print(f"Head-CT hemorrhage XAI smoke complete on {device}.")
    print(f"Positive masked cases: {len(rows)}")
    print(f"Metrics: {output_dir / 'ct_metrics.csv'}")
    print(f"Summary: {output_dir / 'ct_metrics_summary.csv'}")
    print(f"Run metadata: {run_meta_path}")


if __name__ == "__main__":
    main()
