#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image, ImageFilter



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose TorchXRayVision probabilities for blank and perturbed CXR baselines."
    )
    parser.add_argument("--manifest", default="data/cxr_pneumothorax_manifest.csv")
    parser.add_argument("--output-dir", default="outputs/iter_23_torchxray_baseline_diagnostic")
    parser.add_argument("--weights", default="densenet121-res224-all")
    parser.add_argument("--split", default="test", choices=["train", "test", "any"])
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--blur-radius", type=float, default=12.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(choice)


def read_rows(manifest_path: Path, split: str, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if split != "any" and row.get("split") != split:
                continue
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def normalize_array(array: np.ndarray) -> torch.Tensor:
    normalized = xrv.datasets.normalize(array, 255)
    return torch.from_numpy(normalized).unsqueeze(0).float()


def load_image(path: Path, image_size: int) -> Image.Image:
    return Image.open(path).convert("L").resize((image_size, image_size), Image.BILINEAR)


def image_tensor(image: Image.Image) -> torch.Tensor:
    return normalize_array(np.asarray(image))


def constant_tensor(image_size: int, pixel_value: int) -> torch.Tensor:
    array = np.full((image_size, image_size), pixel_value, dtype=np.uint8)
    return normalize_array(array)


def pathology_index(model: torch.nn.Module, pathology: str) -> int:
    pathologies = list(model.pathologies)
    try:
        return pathologies.index(pathology)
    except ValueError as exc:
        raise ValueError(
            f"{pathology!r} is not available in model pathologies: {pathologies}"
        ) from exc


def model_score(
    model: torch.nn.Module, tensor: torch.Tensor, class_idx: int, device: torch.device
) -> tuple[float, float]:
    with torch.no_grad():
        logit = model(tensor.unsqueeze(0).to(device))[0, class_idx]
        probability = torch.sigmoid(logit)
    return float(logit.detach().cpu().item()), float(probability.detach().cpu().item())


def write_summary(rows: list[dict[str, str | int | float]], output_path: Path) -> None:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(float(row["probability"]))

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "n",
                "probability_mean",
                "probability_std",
                "probability_min",
                "probability_max",
            ],
        )
        writer.writeheader()
        for variant, values in sorted(grouped.items()):
            array = np.asarray(values, dtype=float)
            writer.writerow(
                {
                    "variant": variant,
                    "n": int(array.size),
                    "probability_mean": round(float(array.mean()), 6),
                    "probability_std": round(float(array.std(ddof=0)), 6),
                    "probability_min": round(float(array.min()), 6),
                    "probability_max": round(float(array.max()), 6),
                }
            )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(Path(args.manifest), args.split, args.max_cases)
    if not rows:
        raise RuntimeError(f"No rows found in {args.manifest} for split={args.split}.")

    device = resolve_device(args.device)
    model = xrv.models.DenseNet(weights=args.weights).to(device)
    model.eval()
    class_idx = pathology_index(model, "Pneumothorax")

    normalized_zero = torch.zeros(1, args.image_size, args.image_size)
    black = constant_tensor(args.image_size, 0)
    mid_gray = constant_tensor(args.image_size, 128)
    white = constant_tensor(args.image_size, 255)

    diagnostic_rows: list[dict[str, str | int | float]] = []
    for sample_index, row in enumerate(rows):
        image = load_image(Path(row["image_path"]), args.image_size)
        original = image_tensor(image)
        blurred = image_tensor(image.filter(ImageFilter.GaussianBlur(radius=args.blur_radius)))
        image_mean_pixel = int(round(float(np.asarray(image).mean())))
        image_mean = constant_tensor(args.image_size, image_mean_pixel)

        variants = {
            "original_image": original,
            "current_faithfulness_zero_tensor": normalized_zero,
            "black_pixel_0_normalized": black,
            "mid_gray_pixel_128_normalized": mid_gray,
            "white_pixel_255_normalized": white,
            "blurred_original_normalized": blurred,
            "case_mean_pixel_normalized": image_mean,
        }
        for variant, tensor in variants.items():
            logit, probability = model_score(model, tensor, class_idx, device)
            diagnostic_rows.append(
                {
                    "sample_index": sample_index,
                    "filename": Path(row["image_path"]).name,
                    "image_path": row["image_path"],
                    "label": int(row["label"]),
                    "variant": variant,
                    "image_mean_pixel": image_mean_pixel,
                    "tensor_min": round(float(tensor.min().item()), 6),
                    "tensor_max": round(float(tensor.max().item()), 6),
                    "tensor_mean": round(float(tensor.mean().item()), 6),
                    "pneumothorax_logit": round(logit, 6),
                    "probability": round(probability, 6),
                }
            )

    details_path = output_dir / "baseline_diagnostics.csv"
    with details_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostic_rows[0].keys()))
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    summary_path = output_dir / "baseline_diagnostics_summary.csv"
    write_summary(diagnostic_rows, summary_path)

    print(f"Baseline diagnostics written to: {details_path}")
    print(f"Baseline summary written to: {summary_path}")


if __name__ == "__main__":
    main()