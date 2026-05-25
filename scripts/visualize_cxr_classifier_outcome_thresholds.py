#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
import time
from pathlib import Path


from explainai_thesis.cxr.classifier import load_classifier
from explainai_thesis.cxr.io import (
    parse_threshold_fractions as parse_fractions,
    read_manifest_rows as read_rows,
)
from explainai_thesis.cxr.outcome import (
    case_dir_name,
    classifier_outcome,
    completed_source_keys,
    read_existing_rows,
    target_case_count,
    write_progress_checkpoint,
    write_rows,
)
from explainai_thesis.cli.progress import (
    LiveProgress,
    estimate_eta,
    format_duration,
    progress_stats_line,
    timestamp,
)
from explainai_thesis.xai import (
    GradCAM,
    SignedAttribution,
    consensus_signed,
    gradient_shap_signed,
    iter_method_views,
    integrated_gradients_signed,
    occlusion_sensitivity_signed,
)
from explainai_thesis.visualization import (
    is_negative_method,
    make_contact_sheet,
    overlay_color_for_method,
    readable_heatmap_for_method,
    save_binary_selection,
    save_overlay,
    signed_diverging_overlay,
)
from explainai_thesis.metrics import (
    localization_metrics,
    negative_evidence_metrics,
    selection_counts,
    threshold_top_fraction,
)
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

from explainai_thesis.cli.common import resolve_device
from explainai_thesis.run_metadata import write_run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize XAI threshold selections for classifier TP/FP/TN/FN CXR cases."
    )
    parser.add_argument(
        "--manifest", default="data/cxr_pneumothorax_manifest.csv")
    parser.add_argument(
        "--output-dir", default="outputs/iter_05_classifier_outcome_thresholds_test100")
    parser.add_argument("--weights", default="densenet121-res224-all")
    parser.add_argument("--split", default="test",
                        choices=["train", "test", "any"])
    parser.add_argument("--max-cases", type=int, default=100)
    parser.add_argument("--max-per-outcome", type=int, default=0,
                        help="Stop after this many cases per TP/FP/TN/FN outcome; 0 disables the cap.")
    parser.add_argument("--random-sample", action="store_true",
                        help="Shuffle candidate manifest rows reproducibly before evaluation.")
    parser.add_argument("--seed", type=int, default=20260517,
                        help="Random seed used when --random-sample is enabled.")
    parser.add_argument("--threshold", type=float, default=0.62)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--gradshap-samples", type=int, default=8)
    parser.add_argument("--occlusion-patch-size", type=int, default=56)
    parser.add_argument("--occlusion-stride", type=int, default=56)
    parser.add_argument(
        "--fractions",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50",
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
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--progress-every", type=int, default=25,
                        help="Print progress after this many candidate rows; 0 disables periodic candidate progress logs.")
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="Rewrite partial CSVs and progress.json after this many selected cases; 0 disables checkpointing until the end.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing cases.csv and threshold_metrics.csv in --output-dir, skipping already completed source images.")
    return parser.parse_args()


def load_image(path: Path, image_size: int, preprocess) -> torch.Tensor:
    image = Image.open(path).convert("L").resize(
        (image_size, image_size), Image.BILINEAR)
    array = np.asarray(image)
    return preprocess(array)


def load_mask(row: dict[str, str], image_size: int) -> torch.Tensor:
    mask_path = row.get("mask_path", "")
    if mask_path:
        path = Path(mask_path)
        if path.exists():
            mask = Image.open(path).convert("L").resize(
                (image_size, image_size), Image.NEAREST)
            return torch.from_numpy(np.asarray(mask) > 0)
    return torch.zeros((image_size, image_size), dtype=torch.bool)


def selected_image_coverage(selected_mask: torch.Tensor) -> float:
    return float(selected_mask.float().mean().item())


def safe_source_stem(row: dict[str, str]) -> str:
    stem = Path(row.get("filename") or row["image_path"]).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    if not safe_stem:
        safe_stem = "xray"
    return safe_stem


def main() -> None:
    args = parse_args()
    if not 0 < args.stop_fractions_at_coverage <= 1:
        raise ValueError("--stop-fractions-at-coverage must be in (0, 1].")
    if args.pixel_attribution_mask_smoothing < 1 or args.pixel_attribution_mask_smoothing % 2 == 0:
        raise ValueError("--pixel-attribution-mask-smoothing must be a positive odd integer.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fractions = parse_fractions(args.fractions)
    rows = read_rows(Path(args.manifest), args.split)
    if not rows:
        raise RuntimeError(
            f"No rows found in {args.manifest} for split={args.split}.")
    if args.random_sample:
        rows = rows.copy()
        random.Random(args.seed).shuffle(rows)
    if args.max_cases > 0:
        rows = rows[:args.max_cases]

    device = resolve_device(args.device)
    classifier = load_classifier(args.weights, device=device, pathology="Pneumothorax")
    model = classifier.model
    class_idx = classifier.class_idx
    gradcam = GradCAM(model, classifier.target_layer)

    case_rows: list[dict[str, str | int | float]] = []
    metric_rows: list[dict[str, str | int | float]] = []
    outcome_counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    completed_keys: set[str] = set()
    if args.resume:
        case_rows = read_existing_rows(output_dir / "cases.csv")
        metric_rows = read_existing_rows(output_dir / "threshold_metrics.csv")
        completed_keys = completed_source_keys(case_rows)
        outcome_counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        for existing_case in case_rows:
            existing_outcome = str(existing_case.get("classifier_outcome", ""))
            if existing_outcome in outcome_counts:
                outcome_counts[existing_outcome] += 1
    candidate_total = len(rows)
    target_total = target_case_count(args.max_per_outcome, candidate_total)
    start_time = time.monotonic()
    last_completed_selected = len(case_rows)
    progress = LiveProgress()

    progress.update(
        progress_stats_line(
            candidate_number=0,
            candidate_total=candidate_total,
            selected_total=len(case_rows),
            target_total=target_total,
            outcome_counts=outcome_counts,
            elapsed=0,
            eta=None,
        ),
        f"Starting | resume={int(args.resume)} | existing_cases={len(case_rows)} | max_per_outcome={args.max_per_outcome} | output={output_dir}",
    )

    for candidate_idx, row in enumerate(rows):
        candidate_number = candidate_idx + 1
        elapsed = time.monotonic() - start_time
        selected_total = len(case_rows)
        if args.progress_every > 0 and candidate_number % args.progress_every == 0:
            eta = estimate_eta(last_completed_selected, target_total, elapsed)
            progress.update(
                progress_stats_line(
                    candidate_number=candidate_number,
                    candidate_total=candidate_total,
                    selected_total=selected_total,
                    target_total=target_total,
                    outcome_counts=outcome_counts,
                    elapsed=elapsed,
                    eta=eta,
                ),
                "Scanning candidates",
            )
        label = int(row["label"])
        row_filename = row.get("filename", Path(row["image_path"]).name)
        if args.resume and (row["image_path"] in completed_keys or row_filename in completed_keys):
            continue
        image = load_image(Path(row["image_path"]), args.image_size, classifier.preprocess)
        mask = load_mask(row, args.image_size)
        model_input = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(model_input)
            score = float(output[0, class_idx].detach().cpu().item())
            probability = float(torch.sigmoid(
                output[0, class_idx]).detach().cpu().item())

        outcome = classifier_outcome(label, probability, args.threshold)
        if args.max_per_outcome > 0 and outcome_counts[outcome] >= args.max_per_outcome:
            if all(count >= args.max_per_outcome for count in outcome_counts.values()):
                break
            continue
        outcome_counts[outcome] += 1
        sample_idx = len(case_rows)
        selected_total = sample_idx + 1
        source_stem = safe_source_stem(row)
        case_start = time.monotonic()
        eta = estimate_eta(last_completed_selected, target_total, elapsed)
        case_detail = (
            f"Selected case {sample_idx + 1}/{target_total} | candidate={candidate_number}/{candidate_total} | "
            f"outcome={outcome} | file={row.get('filename', Path(row['image_path']).name)} | "
            f"probability={probability:.6f}"
        )
        progress.update(
            progress_stats_line(
                candidate_number=candidate_number,
                candidate_total=candidate_total,
                selected_total=selected_total,
                target_total=target_total,
                outcome_counts=outcome_counts,
                elapsed=elapsed,
                eta=eta,
            ),
            case_detail,
        )
        case_name = case_dir_name(sample_idx, outcome, source_stem)
        case_dir = output_dir / outcome / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        cam_attr = gradcam.signed(model_input, class_idx=class_idx)
        cam_plus_plus_attr = gradcam.signed(
            model_input, class_idx=class_idx, variant="grad_cam_plus_plus")
        progress.update(
            progress_stats_line(
                candidate_number=candidate_number,
                candidate_total=candidate_total,
                selected_total=selected_total,
                target_total=target_total,
                outcome_counts=outcome_counts,
                elapsed=time.monotonic() - start_time,
                eta=eta,
            ),
            f"{case_detail} | Grad-CAM / Grad-CAM++ done",
        )
        ig_attr = integrated_gradients_signed(
            model, model_input, class_idx=class_idx, steps=args.ig_steps)
        progress.update(
            progress_stats_line(
                candidate_number=candidate_number,
                candidate_total=candidate_total,
                selected_total=selected_total,
                target_total=target_total,
                outcome_counts=outcome_counts,
                elapsed=time.monotonic() - start_time,
                eta=eta,
            ),
            f"{case_detail} | Integrated Gradients done",
        )
        shap_attr = gradient_shap_signed(
            model, model_input, class_idx=class_idx, samples=args.gradshap_samples)
        progress.update(
            progress_stats_line(
                candidate_number=candidate_number,
                candidate_total=candidate_total,
                selected_total=selected_total,
                target_total=target_total,
                outcome_counts=outcome_counts,
                elapsed=time.monotonic() - start_time,
                eta=eta,
            ),
            f"{case_detail} | GradientSHAP done",
        )
        occlusion_attr = occlusion_sensitivity_signed(
            model,
            model_input,
            class_idx=class_idx,
            patch_size=args.occlusion_patch_size,
            stride=args.occlusion_stride,
        )
        progress.update(
            progress_stats_line(
                candidate_number=candidate_number,
                candidate_total=candidate_total,
                selected_total=selected_total,
                target_total=target_total,
                outcome_counts=outcome_counts,
                elapsed=time.monotonic() - start_time,
                eta=eta,
            ),
            f"{case_detail} | Occlusion done",
        )
        consensus_attr = consensus_signed([cam_attr, ig_attr, shap_attr, occlusion_attr])
        signed_attributions: dict[str, SignedAttribution] = {
            "grad_cam": cam_attr,
            "grad_cam_plus_plus": cam_plus_plus_attr,
            "integrated_gradients": ig_attr,
            "gradient_shap": shap_attr,
            "occlusion": occlusion_attr,
            "consensus": consensus_attr,
        }
        case_rows.append(
            {
                "sample_index": sample_idx,
                "candidate_index": candidate_idx,
                "filename": row.get("filename", Path(row["image_path"]).name),
                "split": row.get("split", ""),
                "label": label,
                "prediction": int(probability >= args.threshold),
                "classifier_outcome": outcome,
                "xrv_pneumothorax_score": round(score, 8),
                "xrv_pneumothorax_sigmoid": round(probability, 8),
                "classifier_threshold": args.threshold,
                "weights": args.weights,
                "image_size": args.image_size,
                "image_path": row["image_path"],
                "mask_path": row.get("mask_path", ""),
            }
        )

        for method_view in iter_method_views(signed_attributions):
            method_name = method_view.method
            heatmap = method_view.heatmap
            view_kind = method_view.view
            family = method_view.family
            display_heatmap = readable_heatmap_for_method(
                heatmap, family, args.pixel_attribution_mask_smoothing
            )
            overlay_path = case_dir / f"{source_stem}_{method_name}.png"
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
            panel_paths: list[Path] = []
            captions: list[str] = []
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
                selected_coverage = selected_image_coverage(selected)
                selection_path = (
                    case_dir
                    / f"{source_stem}_{method_name}_selected_top_{int(round(fraction * 100)):02d}.png"
                )
                save_binary_selection(
                    image,
                    selected,
                    mask,
                    selection_path,
                    negative_style=view_kind == "negative",
                    neutral_style=view_kind == "magnitude",
                )
                panel_paths.append(selection_path)
                captions.append(f"top {fraction:.0%}")
                positive_localization_applicable = label == 1
                metrics = localization_metrics(metrics_input, mask, fraction=fraction)
                if not positive_localization_applicable:
                    metrics = {
                        "iou": "",
                        "dice": "",
                        "pointing_hit": "",
                        "precision_at_fraction": "",
                    }
                counts = selection_counts(metrics_input, mask, fraction)
                negative_metrics = {
                    "negative_mask_overlap_fraction": "", "negative_mask_avoidance_fraction": ""}
                if view_kind == "negative":
                    negative_metrics = negative_evidence_metrics(
                        metrics_input, mask, fraction)
                metric_rows.append(
                    {
                        "sample_index": sample_idx,
                        "filename": row.get("filename", Path(row["image_path"]).name),
                        "source_stem": source_stem,
                        "image_path": row["image_path"],
                        "mask_path": row.get("mask_path", ""),
                        "label": label,
                        "prediction": int(probability >= args.threshold),
                        "classifier_outcome": outcome,
                        "weights": args.weights,
                        "image_size": args.image_size,
                        "method": method_name,
                        "view": view_kind,
                        "family": family,
                        "metric_component": view_kind,
                        "pixel_attribution_mask_smoothing": (
                            args.pixel_attribution_mask_smoothing
                            if family in {"integrated_gradients", "gradient_shap"}
                            else 1
                        ),
                        "top_fraction": fraction,
                        "top_fraction_percent": int(round(fraction * 100)),
                        "selected_image_coverage": round(selected_coverage, 6),
                        "positive_localization_applicable": int(positive_localization_applicable),
                        **counts,
                        **metrics,
                        **negative_metrics,
                    }
                )
                if selected_coverage >= args.stop_fractions_at_coverage:
                    break
            make_contact_sheet(
                panel_paths,
                captions,
                case_dir / f"{source_stem}_{method_name}_threshold_sweep_panel.png",
            )

        last_completed_selected = len(case_rows)
        elapsed = time.monotonic() - start_time
        case_elapsed = time.monotonic() - case_start
        eta = estimate_eta(last_completed_selected, target_total, elapsed)
        progress.update(
            progress_stats_line(
                candidate_number=candidate_number,
                candidate_total=candidate_total,
                selected_total=last_completed_selected,
                target_total=target_total,
                outcome_counts=outcome_counts,
                elapsed=elapsed,
                eta=eta,
            ),
            f"Completed case {last_completed_selected}/{target_total} | case_time={format_duration(case_elapsed)}",
        )
        if args.checkpoint_every > 0 and last_completed_selected % args.checkpoint_every == 0:
            write_rows(output_dir / "cases.csv", case_rows)
            write_rows(output_dir / "threshold_metrics.csv", metric_rows)
            write_progress_checkpoint(
                output_dir,
                candidate_index=candidate_number,
                candidate_total=candidate_total,
                selected_total=last_completed_selected,
                target_total=target_total,
                outcome_counts=outcome_counts,
                elapsed_seconds=elapsed,
                eta_seconds=eta,
                status="running",
            )
            progress.update(
                progress_stats_line(
                    candidate_number=candidate_number,
                    candidate_total=candidate_total,
                    selected_total=last_completed_selected,
                    target_total=target_total,
                    outcome_counts=outcome_counts,
                    elapsed=elapsed,
                    eta=eta,
                ),
                f"Checkpoint written | cases.csv + threshold_metrics.csv + progress.json",
            )

    gradcam.remove_hooks()
    write_rows(output_dir / "cases.csv", case_rows)
    write_rows(output_dir / "threshold_metrics.csv", metric_rows)
    write_rows(
        output_dir / "outcome_summary.csv",
        [
            {
                "threshold": args.threshold,
                "n": len(rows),
                "tp": outcome_counts["tp"],
                "fp": outcome_counts["fp"],
                "tn": outcome_counts["tn"],
                "fn": outcome_counts["fn"],
            }
        ],
    )
    elapsed = time.monotonic() - start_time
    write_progress_checkpoint(
        output_dir,
        candidate_index=candidate_number if rows else 0,
        candidate_total=candidate_total,
        selected_total=len(case_rows),
        target_total=target_total,
        outcome_counts=outcome_counts,
        elapsed_seconds=elapsed,
        eta_seconds=0,
        status="completed",
    )
    progress.finish()

    run_meta_path = write_run_metadata(
        output_dir,
        args,
        weights=args.weights,
        split=args.split,
        elapsed_seconds=elapsed,
        candidates_scanned=candidate_number if rows else 0,
        candidates_total=candidate_total,
        selected_total=len(case_rows),
        target_total=target_total,
        outcome_counts=outcome_counts,
    )

    print(f"[{timestamp()}] Saved classifier-outcome threshold visualizations to {output_dir}")
    print(f"[{timestamp()}] Completed in {format_duration(elapsed)}")
    print(f"[{timestamp()}] Candidates scanned: {candidate_number if rows else 0}/{candidate_total}")
    print(f"[{timestamp()}] Cases selected: {len(case_rows)}/{target_total}")
    print(f"[{timestamp()}] Outcome counts at threshold {args.threshold}: {outcome_counts}")
    print(f"[{timestamp()}] Rows written: cases={len(case_rows)}, threshold_metrics={len(metric_rows)}")
    print(f"[{timestamp()}] Run metadata written to: {run_meta_path}")


if __name__ == "__main__":
    main()
