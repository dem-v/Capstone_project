#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image

from explainai_thesis.metrics import localization_metrics
from explainai_thesis.visualization import save_overlay
from explainai_thesis.xai import GradCAM, consensus_heatmap, integrated_gradients

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


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
    parser.add_argument("--image-size", type=int, default=224, help="Model input size.")
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
    manifest_path: Path, split: str, limit: int
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
            if len(rows) >= limit:
                break
    return rows


def load_image(path: Path, image_size: int) -> torch.Tensor:
    image = (
        Image.open(path).convert("L").resize((image_size, image_size), Image.BILINEAR)
    )
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
        raise ValueError(
            f"{pathology!r} is not available in model pathologies: {pathologies}"
        ) from exc


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    rows = read_positive_rows(manifest_path, split=args.split, limit=args.max_positive)
    if not rows:
        raise RuntimeError(
            f"No positive rows with masks found in {manifest_path} for split={args.split}."
        )

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
            probability = float(
                torch.sigmoid(output[0, class_idx]).detach().cpu().item()
            )

        cam_map = gradcam(model_input, class_idx=class_idx)
        ig_map = integrated_gradients(
            model, model_input, class_idx=class_idx, steps=args.ig_steps
        )
        consensus = consensus_heatmap([cam_map, ig_map])

        methods = {
            "grad_cam": cam_map,
            "integrated_gradients": ig_map,
            "consensus": consensus,
        }

        for method_name, heatmap in methods.items():
            metrics = localization_metrics(heatmap, mask, fraction=args.top_fraction)
            metric_rows.append(
                {
                    "sample_id": sample_idx,
                    "filename": row.get("filename", Path(row["image_path"]).name),
                    "split": row.get("split", ""),
                    "xrv_pneumothorax_score": round(score, 6),
                    "xrv_pneumothorax_sigmoid": round(probability, 6),
                    "method": method_name,
                    **{key: round(value, 6) for key, value in metrics.items()},
                }
            )
            save_overlay(
                image,
                heatmap,
                mask,
                output_dir / f"sample_{sample_idx:02d}_{method_name}.png",
            )

    gradcam.remove_hooks()

    metrics_path = output_dir / "metrics.csv"
    fieldnames = [
        "sample_id",
        "filename",
        "split",
        "xrv_pneumothorax_score",
        "xrv_pneumothorax_sigmoid",
        "method",
        "iou",
        "dice",
        "pointing_hit",
        "precision_at_fraction",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    print(f"TorchXRayVision CXR smoke test complete on {device}.")
    print(f"Weights: {args.weights}")
    print(f"Positive cases evaluated: {len(rows)}")
    print(f"Metrics written to: {metrics_path}")
    print(f"Overlays written to: {output_dir}")


if __name__ == "__main__":
    main()
