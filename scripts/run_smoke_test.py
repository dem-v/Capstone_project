#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from explainai_thesis.metrics import localization_metrics
from explainai_thesis.models import TinyCnn
from explainai_thesis.synthetic import SyntheticLesionDataset
from explainai_thesis.visualization import save_overlay
from explainai_thesis.xai import GradCAM, consensus_heatmap, integrated_gradients


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a dependency-light XAI smoke test.")
    parser.add_argument("--output-dir", default="outputs/smoke_test", help="Directory for metrics and overlays.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs for the tiny synthetic model.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Execution device.")
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(choice)


def train_model(model: nn.Module, train_loader: DataLoader, device: torch.device, epochs: int) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    for _epoch in range(epochs):
        for images, labels, _masks in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels, _masks in loader:
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images).argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += int(labels.numel())
    return correct / max(total, 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    dataset = SyntheticLesionDataset(n_samples=320, image_size=96, positive_fraction=0.5, seed=args.seed)
    train_set, test_set = random_split(
        dataset,
        [240, 80],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    model = TinyCnn().to(device)
    train_model(model, train_loader, device, epochs=args.epochs)
    accuracy = evaluate_accuracy(model, test_loader, device)

    model.eval()
    gradcam = GradCAM(model, model.target_layer)
    rows: list[dict[str, float | str | int]] = []
    selected = 0

    for image, label, mask in test_set:
        if int(label.item()) != 1:
            continue

        model_input = image.unsqueeze(0).to(device)
        cam_map = gradcam(model_input, class_idx=1)
        ig_map = integrated_gradients(model, model_input, class_idx=1, steps=24)
        consensus = consensus_heatmap([cam_map, ig_map])

        methods = {
            "grad_cam": cam_map,
            "integrated_gradients": ig_map,
            "consensus": consensus,
        }

        for method_name, heatmap in methods.items():
            metrics = localization_metrics(heatmap, mask, fraction=0.15)
            rows.append(
                {
                    "sample_id": selected,
                    "method": method_name,
                    "test_accuracy": round(accuracy, 4),
                    **{key: round(value, 4) for key, value in metrics.items()},
                }
            )
            save_overlay(image, heatmap, mask, output_dir / f"sample_{selected:02d}_{method_name}.png")

        selected += 1
        if selected >= 6:
            break

    gradcam.remove_hooks()

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "method", "test_accuracy", "iou", "dice", "pointing_hit", "precision_at_fraction"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Smoke test complete on {device}.")
    print(f"Synthetic test accuracy: {accuracy:.3f}")
    print(f"Metrics written to: {metrics_path}")
    print(f"Overlays written to: {output_dir}")


if __name__ == "__main__":
    main()
