#!/usr/bin/env python3
from __future__ import annotations
from explainai_thesis.xai import GradCAM, consensus_heatmap, integrated_gradients
from explainai_thesis.visualization import save_overlay
from explainai_thesis.metrics import localization_metrics, normalize_map, threshold_top_fraction
from PIL import Image, ImageDraw
import torchxrayvision as xrv
import torch
import numpy as np

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


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
    parser.add_argument("--threshold", type=float, default=0.61)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument(
        "--fractions",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50",
        help="Comma-separated top-fractions to visualize.",
    )
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
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


def classifier_outcome(label: int, probability: float, threshold: float) -> str:
    prediction = int(probability >= threshold)
    if label == 1 and prediction == 1:
        return "tp"
    if label == 0 and prediction == 1:
        return "fp"
    if label == 0 and prediction == 0:
        return "tn"
    return "fn"


def save_binary_selection(
    image: torch.Tensor,
    selected_mask: torch.Tensor,
    true_mask: torch.Tensor,
    output_path: Path,
    *,
    negative_style: bool = False,
    negative_selected_mask: torch.Tensor | None = None,
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fractions = parse_fractions(args.fractions)
    rows = read_rows(Path(args.manifest), args.split, args.max_cases)
    if not rows:
        raise RuntimeError(
            f"No rows found in {args.manifest} for split={args.split}.")

    device = resolve_device(args.device)
    model = xrv.models.DenseNet(weights=args.weights).to(device)
    model.eval()
    class_idx = pathology_index(model, "Pneumothorax")
    gradcam = GradCAM(model, model.features.denseblock4)

    case_rows: list[dict[str, str | int | float]] = []
    metric_rows: list[dict[str, str | int | float]] = []
    outcome_counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    for sample_idx, row in enumerate(rows):
        label = int(row["label"])
        image = load_image(Path(row["image_path"]), args.image_size)
        mask = load_mask(row, args.image_size)
        model_input = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(model_input)
            score = float(output[0, class_idx].detach().cpu().item())
            probability = float(torch.sigmoid(
                output[0, class_idx]).detach().cpu().item())

        outcome = classifier_outcome(label, probability, args.threshold)
        outcome_counts[outcome] += 1
        case_name = f"case_{sample_idx:03d}_{outcome}_{row.get('filename', Path(row['image_path']).stem)}"
        case_dir = output_dir / outcome / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        cam_map = gradcam(model_input, class_idx=class_idx)
        negative_cam_map = gradcam(
            model_input, class_idx=class_idx, polarity="negative")
        ig_map = integrated_gradients(
            model, model_input, class_idx=class_idx, steps=args.ig_steps)
        consensus = consensus_heatmap([cam_map, ig_map])

        methods = {
            "grad_cam": cam_map,
            "grad_cam_negative": negative_cam_map,
            "integrated_gradients": ig_map,
            "consensus": consensus,
        }
        case_rows.append(
            {
                "sample_index": sample_idx,
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

        for method_name, heatmap in methods.items():
            method_dir = case_dir / method_name
            method_dir.mkdir(parents=True, exist_ok=True)
            save_overlay(
                image,
                heatmap,
                mask,
                method_dir / "continuous_heatmap.png",
                heatmap_color="blue" if method_name == "grad_cam_negative" else "red",
                negative_heatmap=negative_cam_map if method_name == "consensus" else None,
            )
            panel_paths: list[Path] = []
            captions: list[str] = []
            for fraction in fractions:
                selected = threshold_top_fraction(heatmap, fraction=fraction)
                selection_path = method_dir / \
                    f"selected_top_{int(round(fraction * 100)):02d}.png"
                save_binary_selection(
                    image,
                    selected,
                    mask,
                    selection_path,
                    negative_style=(method_name == "grad_cam_negative"),
                    negative_selected_mask=threshold_top_fraction(
                        negative_cam_map, fraction=fraction)
                    if method_name == "consensus"
                    else None,
                )
                panel_paths.append(selection_path)
                captions.append(f"top {fraction:.0%}")
                metrics = localization_metrics(
                    heatmap, mask, fraction=fraction)
                negative_metrics = {
                    "negative_mask_overlap_fraction": "", "negative_mask_avoidance_fraction": ""}
                if method_name == "grad_cam_negative":
                    negative_metrics = negative_evidence_metrics(
                        heatmap, mask, fraction)
                elif method_name == "consensus":
                    negative_metrics = negative_evidence_metrics(
                        negative_cam_map, mask, fraction)
                metric_rows.append(
                    {
                        "sample_index": sample_idx,
                        "filename": row.get("filename", Path(row["image_path"]).name),
                        "label": label,
                        "prediction": int(probability >= args.threshold),
                        "classifier_outcome": outcome,
                        "method": method_name,
                        "top_fraction": fraction,
                        **metrics,
                        **negative_metrics,
                    }
                )
            make_contact_sheet(panel_paths, captions,
                               method_dir / "threshold_sweep_panel.png")

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

    print(f"Saved classifier-outcome threshold visualizations to {output_dir}")
    print(f"Outcome counts at threshold {args.threshold}: {outcome_counts}")


if __name__ == "__main__":
    main()
