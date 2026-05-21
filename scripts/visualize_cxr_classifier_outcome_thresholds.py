#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
import textwrap
import time
from datetime import datetime
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
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--progress-every", type=int, default=25,
                        help="Print progress after this many candidate rows; 0 disables periodic candidate progress logs.")
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="Rewrite partial CSVs and progress.json after this many selected cases; 0 disables checkpointing until the end.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing cases.csv and threshold_metrics.csv in --output-dir, skipping already completed source images.")
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(choice)


def read_rows(manifest_path: Path, split: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if split != "any" and row.get("split") != split:
                continue
            rows.append(row)
    return rows


def load_image(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("L").resize(
        (image_size, image_size), Image.BILINEAR)
    array = np.asarray(image)
    normalized = xrv.datasets.normalize(array, 255)
    return torch.from_numpy(normalized).unsqueeze(0).float()


def load_mask(row: dict[str, str], image_size: int) -> torch.Tensor:
    mask_path = row.get("mask_path", "")
    if mask_path:
        path = Path(mask_path)
        if path.exists():
            mask = Image.open(path).convert("L").resize(
                (image_size, image_size), Image.NEAREST)
            return torch.from_numpy(np.asarray(mask) > 0)
    return torch.zeros((image_size, image_size), dtype=torch.bool)


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


def read_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def completed_source_keys(case_rows: list[dict[str, str | int | float]]) -> set[str]:
    keys: set[str] = set()
    for row in case_rows:
        image_path = str(row.get("image_path", ""))
        filename = str(row.get("filename", ""))
        if image_path:
            keys.add(image_path)
        if filename:
            keys.add(filename)
    return keys


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def target_case_count(max_per_outcome: int, candidate_count: int) -> int:
    if max_per_outcome > 0:
        return max_per_outcome * 4
    return candidate_count


def estimate_eta(completed: int, total: int, elapsed: float) -> float | None:
    if completed <= 0 or total <= 0:
        return None
    remaining = max(0, total - completed)
    return elapsed / completed * remaining


def write_progress_checkpoint(
    output_dir: Path,
    *,
    candidate_index: int,
    candidate_total: int,
    selected_total: int,
    target_total: int,
    outcome_counts: dict[str, int],
    elapsed_seconds: float,
    eta_seconds: float | None,
    status: str,
) -> None:
    payload = {
        "status": status,
        "candidate_index": candidate_index,
        "candidate_total": candidate_total,
        "selected_total": selected_total,
        "target_total": target_total,
        "outcome_counts": outcome_counts,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "elapsed": format_duration(elapsed_seconds),
        "eta_seconds": round(eta_seconds, 3) if eta_seconds is not None else None,
        "eta": format_duration(eta_seconds),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with (output_dir / "progress.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


class LiveProgress:
    LINE_COUNT = 6

    def __init__(self) -> None:
        self._started = False
        self._live = sys.stdout.isatty()
        self._latest_lines: list[str] = []

    @staticmethod
    def _terminal_width() -> int:
        try:
            return max(40, shutil.get_terminal_size(fallback=(120, 20)).columns)
        except OSError:
            return 120

    def _fit_line(self, text: str) -> str:
        width = self._terminal_width()
        max_len = max(1, width - 1)
        clean = text.replace("\n", " ")
        if len(clean) > max_len:
            return clean[: max(1, max_len - 1)] + "…"
        return clean.ljust(max_len)

    def _compose_lines(self, stats: str, detail: str) -> list[str]:
        width = self._terminal_width()
        max_len = max(20, width - 1)
        parts = [part.strip() for part in f"{stats} | {detail}".split(" | ") if part.strip()]
        lines: list[str] = []
        current = ""
        for part in parts:
            candidate = part if not current else f"{current} | {part}"
            if len(candidate) <= max_len:
                current = candidate
                continue
            if current:
                lines.extend(textwrap.wrap(current, width=max_len) or [current])
            current = part
        if current:
            lines.extend(textwrap.wrap(current, width=max_len) or [current])
        if len(lines) > self.LINE_COUNT:
            overflow = " | ".join(lines[self.LINE_COUNT - 1:])
            lines = lines[: self.LINE_COUNT - 1] + [overflow]
        lines.extend([""] * (self.LINE_COUNT - len(lines)))
        return lines[: self.LINE_COUNT]

    def update(self, stats: str, detail: str) -> None:
        lines = self._compose_lines(stats, detail)
        self._latest_lines = lines
        if not self._live:
            return
        if self._started:
            sys.stdout.write(f"\x1b[{self.LINE_COUNT}F")
        else:
            self._started = True
        for line in lines:
            sys.stdout.write(f"\r\x1b[2K{self._fit_line(line)}\n")
        sys.stdout.flush()

    def finish(self) -> None:
        if self._live and self._started:
            sys.stdout.write("\n")
            sys.stdout.flush()
        elif not self._live and self._latest_lines:
            for line in self._latest_lines:
                if line:
                    print(line, flush=True)


def progress_stats_line(
    *,
    candidate_number: int,
    candidate_total: int,
    selected_total: int,
    target_total: int,
    outcome_counts: dict[str, int],
    elapsed: float,
    eta: float | None,
) -> str:
    return (
        f"[{timestamp()}] Candidate {candidate_number}/{candidate_total} | "
        f"kept={selected_total}/{target_total} | "
        f"TP={outcome_counts['tp']} FP={outcome_counts['fp']} "
        f"TN={outcome_counts['tn']} FN={outcome_counts['fn']} | "
        f"elapsed={format_duration(elapsed)} | ETA≈{format_duration(eta)}"
    )


def classifier_outcome(label: int, probability: float, threshold: float) -> str:
    prediction = int(probability >= threshold)
    if label == 1 and prediction == 1:
        return "tp"
    if label == 0 and prediction == 1:
        return "fp"
    if label == 0 and prediction == 0:
        return "tn"
    return "fn"


def safe_source_stem(row: dict[str, str]) -> str:
    stem = Path(row.get("filename") or row["image_path"]).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    if not safe_stem:
        safe_stem = "xray"
    return safe_stem


def safe_case_name(sample_idx: int, outcome: str, source_stem: str) -> str:
    return f"case_{sample_idx:03d}_{outcome}_{source_stem}"


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


def negative_evidence_metrics(heatmap: torch.Tensor, true_mask: torch.Tensor, fraction: float) -> dict[str, float]:
    selected = threshold_top_fraction(heatmap, fraction=fraction)
    selected_count = selected.sum().float()
    if selected_count.item() == 0:
        return {"negative_mask_overlap_fraction": 0.0, "negative_mask_avoidance_fraction": 0.0}
    overlap = (selected & true_mask.bool()).sum().float() / selected_count
    return {
        "negative_mask_overlap_fraction": overlap.item(),
        "negative_mask_avoidance_fraction": (1.0 - overlap).item(),
    }


def selection_counts(heatmap: torch.Tensor, true_mask: torch.Tensor, fraction: float) -> dict[str, int]:
    selected = threshold_top_fraction(heatmap, fraction=fraction).bool()
    true = true_mask.bool()
    return {
        "selected_pixel_count": int(selected.sum().item()),
        "mask_pixel_count": int(true.sum().item()),
        "intersection_pixel_count": int((selected & true).sum().item()),
        "union_pixel_count": int((selected | true).sum().item()),
    }


def is_negative_method(method_name: str) -> bool:
    return method_name.endswith("_negative")


def overlay_color_for_method(method_name: str) -> str:
    if is_negative_method(method_name):
        return "blue"
    if method_name.endswith("_magnitude"):
        return "neutral"
    return "red"


def main() -> None:
    args = parse_args()
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
    model = xrv.models.DenseNet(weights=args.weights).to(device)
    model.eval()
    class_idx = pathology_index(model, "Pneumothorax")
    gradcam = GradCAM(model, model.features.denseblock4)

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
        image = load_image(Path(row["image_path"]), args.image_size)
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
        case_name = safe_case_name(sample_idx, outcome, source_stem)
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
                "image_path": row["image_path"],
                "mask_path": row.get("mask_path", ""),
            }
        )

        for method_view in iter_method_views(signed_attributions):
            method_name = method_view.method
            heatmap = method_view.heatmap
            view_kind = method_view.view
            family = method_view.family
            overlay_path = case_dir / f"{source_stem}_{method_name}.png"
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
            panel_paths: list[Path] = []
            captions: list[str] = []
            for fraction in fractions:
                metrics_input = (
                    signed_attributions[family].magnitude
                    if view_kind == "signed"
                    else heatmap
                )
                selected = threshold_top_fraction(metrics_input, fraction=fraction)
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
                        heatmap, mask, fraction)
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
                        "method": method_name,
                        "view": view_kind,
                        "family": family,
                        "metric_component": view_kind,
                        "top_fraction": fraction,
                        "top_fraction_percent": int(round(fraction * 100)),
                        "positive_localization_applicable": int(positive_localization_applicable),
                        **counts,
                        **metrics,
                        **negative_metrics,
                    }
                )
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

    print(f"[{timestamp()}] Saved classifier-outcome threshold visualizations to {output_dir}")
    print(f"[{timestamp()}] Completed in {format_duration(elapsed)}")
    print(f"[{timestamp()}] Candidates scanned: {candidate_number if rows else 0}/{candidate_total}")
    print(f"[{timestamp()}] Cases selected: {len(case_rows)}/{target_total}")
    print(f"[{timestamp()}] Outcome counts at threshold {args.threshold}: {outcome_counts}")
    print(f"[{timestamp()}] Rows written: cases={len(case_rows)}, threshold_metrics={len(metric_rows)}")


if __name__ == "__main__":
    main()
