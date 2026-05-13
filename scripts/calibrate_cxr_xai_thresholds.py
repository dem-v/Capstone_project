#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image

from explainai_thesis.metrics import localization_metrics
from explainai_thesis.xai import GradCAM, consensus_heatmap, integrated_gradients


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate top-fraction thresholds for CXR XAI heatmaps on positive masked cases."
    )
    parser.add_argument("--manifest", default="data/cxr_pneumothorax_manifest.csv")
    parser.add_argument("--output-dir", default="outputs/cxr_xai_threshold_calibration")
    parser.add_argument("--weights", default="densenet121-res224-all")
    parser.add_argument("--split", default="train", choices=["train", "test", "any"])
    parser.add_argument("--max-positive", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument(
        "--fractions",
        default="0.05,0.10,0.15,0.20,0.25,0.30",
        help="Comma-separated top-fractions to sweep.",
    )
    parser.add_argument(
        "--selection-metric",
        default="dice",
        choices=["dice", "iou", "precision_at_fraction", "pointing_hit"],
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(choice)


def read_positive_rows(manifest_path: Path, split: str, limit: int) -> list[dict[str, str]]:
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
    image = Image.open(path).convert("L").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image)
    normalized = xrv.datasets.normalize(array, 255)
    return torch.from_numpy(normalized).unsqueeze(0).float()


def load_mask(path: Path, image_size: int) -> torch.Tensor:
    mask = Image.open(path).convert("L").resize((image_size, image_size), Image.NEAREST)
    return torch.from_numpy(np.asarray(mask) > 0)


def pathology_index(model: torch.nn.Module, pathology: str) -> int:
    pathologies = list(model.pathologies)
    try:
        return pathologies.index(pathology)
    except ValueError as exc:
        raise ValueError(f"{pathology!r} is not available in model pathologies: {pathologies}") from exc


def parse_fractions(raw: str) -> list[float]:
    fractions = [float(value.strip()) for value in raw.split(",") if value.strip()]
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fractions = parse_fractions(args.fractions)
    rows = read_positive_rows(Path(args.manifest), args.split, args.max_positive)
    if not rows:
        raise RuntimeError(f"No positive rows with masks found in {args.manifest} for split={args.split}.")

    device = resolve_device(args.device)
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
            probability = float(torch.sigmoid(output[0, class_idx]).detach().cpu().item())

        cam_map = gradcam(model_input, class_idx=class_idx)
        negative_cam_map = gradcam(model_input, class_idx=class_idx, polarity="negative")
        ig_map = integrated_gradients(model, model_input, class_idx=class_idx, steps=args.ig_steps)
        methods = {
            "grad_cam": cam_map,
            "grad_cam_negative": negative_cam_map,
            "integrated_gradients": ig_map,
            "consensus": consensus_heatmap([cam_map, ig_map]),
        }

        for method_name, heatmap in methods.items():
            for fraction in fractions:
                metrics = localization_metrics(heatmap, mask, fraction=fraction)
                metric_rows.append(
                    {
                        "sample_id": sample_idx,
                        "filename": row.get("filename", Path(row["image_path"]).name),
                        "split": row.get("split", ""),
                        "xrv_pneumothorax_score": round(score, 6),
                        "xrv_pneumothorax_sigmoid": round(probability, 6),
                        "method": method_name,
                        "top_fraction": round(fraction, 6),
                        **{key: round(value, 6) for key, value in metrics.items()},
                    }
                )

    gradcam.remove_hooks()

    grouped: dict[tuple[str, float], list[dict[str, str | int | float]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(str(row["method"]), float(row["top_fraction"]))].append(row)

    summary_rows: list[dict[str, str | int | float]] = []
    for (method_name, fraction), rows_for_fraction in sorted(grouped.items()):
        summary: dict[str, str | int | float] = {
            "method": method_name,
            "top_fraction": round(fraction, 6),
            "n": len(rows_for_fraction),
        }
        for metric_name in ["iou", "dice", "pointing_hit", "precision_at_fraction"]:
            values = np.asarray([float(row[metric_name]) for row in rows_for_fraction], dtype=float)
            summary[f"{metric_name}_mean"] = round(float(values.mean()), 6)
            summary[f"{metric_name}_std"] = round(float(values.std(ddof=0)), 6)
        summary_rows.append(summary)

    selected_rows: list[dict[str, str | int | float]] = []
    for method_name in sorted({str(row["method"]) for row in summary_rows}):
        candidates = [row for row in summary_rows if row["method"] == method_name]
        selected = max(candidates, key=lambda row: float(row[f"{args.selection_metric}_mean"]))
        selected_rows.append(
            {
                "method": method_name,
                "selected_fraction": selected["top_fraction"],
                "selection_metric": args.selection_metric,
                "selection_metric_mean": selected[f"{args.selection_metric}_mean"],
                "n": selected["n"],
            }
        )

    write_rows(output_dir / "calibration_metrics.csv", metric_rows)
    write_rows(output_dir / "calibration_summary.csv", summary_rows)
    write_rows(output_dir / "selected_fractions.csv", selected_rows)

    print(f"CXR XAI threshold calibration complete on {device}.")
    print(f"Positive calibration cases: {len(rows)}")
    print(f"Selection metric: {args.selection_metric}")
    print(f"Selected fractions written to: {output_dir / 'selected_fractions.csv'}")


if __name__ == "__main__":
    main()