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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a TorchXRayVision pneumothorax classifier on the CXR manifest."
    )
    parser.add_argument(
        "--manifest", default="data/cxr_pneumothorax_manifest.csv")
    parser.add_argument(
        "--output-dir", default="outputs/cxr_torchxray_model_eval")
    parser.add_argument("--weights", default="densenet121-res224-all")
    parser.add_argument("--split", default="test",
                        choices=["train", "test", "any"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--high-sensitivity-min", type=float, default=0.95)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
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
    image = (
        Image.open(path).convert("L").resize(
            (image_size, image_size), Image.BILINEAR)
    )
    array = np.asarray(image)
    normalized = xrv.datasets.normalize(array, 255)
    return torch.from_numpy(normalized).unsqueeze(0).float()


def pathology_index(model: torch.nn.Module, pathology: str) -> int:
    pathologies = list(model.pathologies)
    try:
        return pathologies.index(pathology)
    except ValueError as exc:
        raise ValueError(
            f"{pathology!r} is not available in model pathologies: {pathologies}"
        ) from exc


def batched(iterable: list[dict[str, str]], batch_size: int):
    for start in range(0, len(iterable), batch_size):
        yield iterable[start: start + batch_size]


def metrics_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict[str, float | int]:
    y_pred = y_score >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "threshold": round(float(threshold), 6),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "sensitivity": round(float(sensitivity), 6),
        "specificity": round(float(specificity), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def selected_threshold_rows(
    threshold_rows: list[dict[str, float | int]], high_sensitivity_min: float
) -> list[dict[str, str | float | int]]:
    best_f1 = max(threshold_rows, key=lambda row: float(row["f1"]))
    best_youden = max(
        threshold_rows,
        key=lambda row: float(row["sensitivity"]) + float(row["specificity"]) - 1.0,
    )
    high_sensitivity_candidates = [
        row for row in threshold_rows
        if float(row["sensitivity"]) >= high_sensitivity_min
    ]
    if high_sensitivity_candidates:
        high_sensitivity = max(
            high_sensitivity_candidates,
            key=lambda row: (float(row["specificity"]), float(row["precision"]), float(row["f1"])),
        )
    else:
        high_sensitivity = max(threshold_rows, key=lambda row: float(row["sensitivity"]))

    selected = [
        ("best_f1", "Maximizes F1 on this calibration split.", best_f1),
        ("best_youden_j", "Maximizes sensitivity + specificity - 1 on this calibration split.", best_youden),
        (
            "high_sensitivity",
            f"Highest-specificity threshold with sensitivity >= {high_sensitivity_min:.2f}; falls back to maximum sensitivity if unavailable.",
            high_sensitivity,
        ),
    ]
    return [
        {
            "selection_name": name,
            "selection_note": note,
            "youden_j": round(float(row["sensitivity"]) + float(row["specificity"]) - 1.0, 6),
            **row,
        }
        for name, note, row in selected
    ]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(Path(args.manifest), args.split)
    if not rows:
        raise RuntimeError(
            f"No rows found in {args.manifest} for split={args.split}.")

    device = resolve_device(args.device)
    model = xrv.models.DenseNet(weights=args.weights).to(device)
    model.eval()
    class_idx = pathology_index(model, "Pneumothorax")

    prediction_rows: list[dict[str, str | int | float]] = []
    labels: list[int] = []
    scores: list[float] = []
    raw_scores: list[float] = []

    for batch_rows in batched(rows, args.batch_size):
        images = torch.stack(
            [
                load_image(Path(row["image_path"]), args.image_size)
                for row in batch_rows
            ],
            dim=0,
        ).to(device)
        with torch.no_grad():
            output = model(images)[:, class_idx].detach().cpu()
            probabilities = torch.sigmoid(output)

        for row, score, probability in zip(
            batch_rows, output.tolist(), probabilities.tolist(), strict=False
        ):
            label = int(row["label"])
            labels.append(label)
            raw_scores.append(float(score))
            scores.append(float(probability))
            prediction_rows.append(
                {
                    "filename": row.get("filename", Path(row["image_path"]).name),
                    "split": row.get("split", ""),
                    "label": label,
                    "xrv_pneumothorax_score": round(float(score), 8),
                    "xrv_pneumothorax_sigmoid": round(float(probability), 8),
                }
            )

    y_true = np.asarray(labels, dtype=int)
    y_score = np.asarray(scores, dtype=float)
    y_raw = np.asarray(raw_scores, dtype=float)

    thresholds = np.linspace(0.05, 0.95, 181)
    threshold_rows = [
        metrics_at_threshold(y_true, y_score, float(threshold))
        for threshold in thresholds
    ]
    best_f1_row = max(threshold_rows, key=lambda row: float(row["f1"]))
    selected_rows = selected_threshold_rows(threshold_rows, args.high_sensitivity_min)

    summary = {
        "weights": args.weights,
        "split": args.split,
        "n": int(y_true.size),
        "positives": int(y_true.sum()),
        "negatives": int((y_true == 0).sum()),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 6),
        "average_precision": round(float(average_precision_score(y_true, y_score)), 6),
        "score_mean_positive": round(float(y_score[y_true == 1].mean()), 6),
        "score_mean_negative": round(float(y_score[y_true == 0].mean()), 6),
        "raw_mean_positive": round(float(y_raw[y_true == 1].mean()), 6),
        "raw_mean_negative": round(float(y_raw[y_true == 0].mean()), 6),
        **{
            f"default_{key}": value
            for key, value in metrics_at_threshold(
                y_true, y_score, args.threshold
            ).items()
        },
        **{f"best_f1_{key}": value for key, value in best_f1_row.items()},
    }

    predictions_path = output_dir / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "split",
                "label",
                "xrv_pneumothorax_score",
                "xrv_pneumothorax_sigmoid",
            ],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    threshold_path = output_dir / "threshold_sweep.csv"
    with threshold_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(threshold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(threshold_rows)

    selected_thresholds_path = output_dir / "selected_thresholds.csv"
    with selected_thresholds_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selected_rows)

    summary_path = output_dir / "classification_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"TorchXRayVision model evaluation complete on {device}.")
    print(
        f"Rows evaluated: {summary['n']} ({summary['positives']} positive, {summary['negatives']} negative)"
    )
    print(f"ROC AUC: {summary['roc_auc']}")
    print(f"Average precision: {summary['average_precision']}")
    print(
        "Default threshold metrics: "
        f"acc={summary['default_accuracy']}, sens={summary['default_sensitivity']}, "
        f"spec={summary['default_specificity']}, f1={summary['default_f1']}"
    )
    print(
        "Best-F1 threshold metrics: "
        f"threshold={summary['best_f1_threshold']}, acc={summary['best_f1_accuracy']}, "
        f"sens={summary['best_f1_sensitivity']}, spec={summary['best_f1_specificity']}, "
        f"f1={summary['best_f1_f1']}"
    )
    print(f"Outputs written to: {output_dir}")
    print(f"Selected threshold candidates written to: {selected_thresholds_path}")


if __name__ == "__main__":
    main()
