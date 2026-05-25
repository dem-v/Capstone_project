from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def model_probability(model: torch.nn.Module, image: torch.Tensor, class_idx: int) -> float:
    with torch.inference_mode():
        output = model(image)
        return float(torch.sigmoid(output[0, class_idx]).detach().cpu().item())


def faithfulness_baseline_tensor(model_input: torch.Tensor, baseline: str) -> torch.Tensor:
    if baseline == "zero_tensor":
        return torch.zeros_like(model_input)
    if baseline == "black":
        return torch.full_like(model_input, -1024.0)
    if baseline == "white":
        return torch.full_like(model_input, 1024.0)
    if baseline == "case_mean":
        return torch.full_like(model_input, float(model_input.mean().item()))
    raise ValueError(f"Unsupported faithfulness baseline: {baseline}")


def faithfulness_curve_rows(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    heatmap: torch.Tensor,
    class_idx: int,
    fractions: list[float],
    baseline: torch.Tensor,
) -> list[dict[str, float]]:
    if not fractions:
        return []
    flat_order = torch.argsort(heatmap.flatten().to(model_input.device), descending=True)
    original_flat = model_input.detach().clone().flatten()
    baseline_flat = baseline.detach().clone().flatten().to(model_input.device)
    rows: list[dict[str, float]] = []
    total_pixels = flat_order.numel()
    for fraction in fractions:
        keep_count = int(round(total_pixels * fraction))
        insertion_flat = baseline_flat.clone()
        deletion_flat = original_flat.clone()
        if keep_count > 0:
            selected = flat_order[:keep_count]
            insertion_flat[selected] = original_flat[selected]
            deletion_flat[selected] = baseline_flat[selected]
        insertion = insertion_flat.view_as(model_input)
        deletion = deletion_flat.view_as(model_input)
        rows.append(
            {
                "fraction": round(fraction, 6),
                "insertion_probability": round(
                    model_probability(model, insertion, class_idx), 6
                ),
                "deletion_probability": round(
                    model_probability(model, deletion, class_idx), 6
                ),
            }
        )
    return rows


def curve_auc(rows: list[dict[str, str | int | float]], value_key: str) -> float:
    points = sorted((float(row["fraction"]), float(row[value_key])) for row in rows)
    if len(points) < 2:
        return 0.0
    auc = 0.0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        auc += (x1 - x0) * (y0 + y1) / 2.0
    return auc


def faithfulness_method_family(method: str) -> str:
    if method.startswith("grad_cam") or method == "consensus":
        return "cam_family"
    if method.startswith("integrated_gradients"):
        return "integrated_gradients_family"
    if method.startswith("gradient_shap"):
        return "gradient_shap_family"
    if method.startswith("occlusion"):
        return "occlusion_family"
    return "other"


def write_faithfulness_summary(
    faithfulness_rows: list[dict[str, str | int | float]], output_path: Path
) -> None:
    grouped: dict[tuple[int, str], list[dict[str, str | int | float]]] = defaultdict(list)
    for row in faithfulness_rows:
        grouped[(int(row["sample_id"]), str(row["method"]))].append(row)

    per_case: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (_sample_id, method), rows in grouped.items():
        insertion_auc = curve_auc(rows, "insertion_probability")
        deletion_auc = curve_auc(rows, "deletion_probability")
        per_case[method].append(
            {
                "insertion_auc": insertion_auc,
                "deletion_auc": deletion_auc,
                "deletion_drop_auc": 1.0 - deletion_auc,
            }
        )

    from .io import FAITHFULNESS_SUMMARY_FIELDS

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FAITHFULNESS_SUMMARY_FIELDS))
        writer.writeheader()
        for method, values in sorted(per_case.items()):
            writer.writerow(
                {
                    "method": method,
                    "case_count": len(values),
                    "insertion_auc_mean": round(
                        float(np.mean([item["insertion_auc"] for item in values])), 6
                    ),
                    "deletion_auc_mean": round(
                        float(np.mean([item["deletion_auc"] for item in values])), 6
                    ),
                    "deletion_drop_auc_mean": round(
                        float(np.mean([item["deletion_drop_auc"] for item in values])), 6
                    ),
                }
            )


def plot_faithfulness_curves(
    faithfulness_rows: list[dict[str, str | int | float]],
    output_path: Path,
    title: str,
    *,
    zoom_y: bool = False,
    y_limits: tuple[float, float] | None = None,
) -> None:
    if not faithfulness_rows:
        return
    grouped: dict[str, list[dict[str, str | int | float]]] = defaultdict(list)
    for row in faithfulness_rows:
        grouped[str(row["method"])].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    plotted_values: list[float] = []
    for method, rows in sorted(grouped.items()):
        by_fraction: dict[float, list[dict[str, str | int | float]]] = defaultdict(list)
        for row in rows:
            by_fraction[float(row["fraction"])].append(row)
        fractions = sorted(by_fraction)
        insertion_values = [
            float(
                np.mean(
                    [float(row["insertion_probability"]) for row in by_fraction[fraction]]
                )
            )
            for fraction in fractions
        ]
        deletion_values = [
            float(
                np.mean(
                    [float(row["deletion_probability"]) for row in by_fraction[fraction]]
                )
            )
            for fraction in fractions
        ]
        plotted_values.extend(insertion_values)
        plotted_values.extend(deletion_values)
        axes[0].plot(
            fractions,
            insertion_values,
            marker="o",
            linewidth=1.5,
            label=method,
        )
        axes[1].plot(
            fractions,
            deletion_values,
            marker="o",
            linewidth=1.5,
            label=method,
        )

    axes[0].set_title("Insertion")
    axes[0].set_xlabel("Fraction of top-attributed pixels restored")
    axes[0].set_ylabel("Pneumothorax probability")
    axes[1].set_title("Deletion")
    axes[1].set_xlabel("Fraction of top-attributed pixels removed")
    if y_limits is not None:
        y_min, y_max = y_limits
    elif zoom_y and plotted_values:
        y_min = max(0.0, min(plotted_values) - 0.03)
        y_max = min(1.0, max(plotted_values) + 0.03)
    else:
        y_min = 0.0
        y_max = 1.0
    for axis in axes:
        axis.set_ylim(y_min, y_max)
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_faithfulness_plots(
    faithfulness_rows: list[dict[str, str | int | float]], output_dir: Path, title: str
) -> None:
    plotted_values = [
        float(row[key])
        for row in faithfulness_rows
        for key in ["insertion_probability", "deletion_probability"]
    ]
    shared_zoom_limits = None
    if plotted_values:
        shared_zoom_limits = (
            max(0.0, min(plotted_values) - 0.03),
            min(1.0, max(plotted_values) + 0.03),
        )
    plot_faithfulness_curves(
        faithfulness_rows,
        output_dir / "faithfulness_curves.png",
        title,
    )
    plot_faithfulness_curves(
        faithfulness_rows,
        output_dir / "faithfulness_curves_zoomed.png",
        f"{title} (zoomed y-axis)",
        y_limits=shared_zoom_limits,
    )
    families: dict[str, list[dict[str, str | int | float]]] = defaultdict(list)
    for row in faithfulness_rows:
        families[faithfulness_method_family(str(row["method"]))].append(row)
    for family, rows in families.items():
        if rows:
            plot_faithfulness_curves(
                rows,
                output_dir / f"faithfulness_curves_{family}.png",
                f"{title}: {family.replace('_', ' ')} (shared zoom scale)",
                y_limits=shared_zoom_limits,
            )


def plot_faithfulness_summary(summary_path: Path, output_path: Path) -> None:
    if not summary_path.exists():
        return
    rows: list[dict[str, str]] = []
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
    if not rows:
        return
    methods = [row["method"] for row in rows]
    x = np.arange(len(methods))
    width = 0.38
    insertion = [float(row["insertion_auc_mean"]) for row in rows]
    deletion_drop = [float(row["deletion_drop_auc_mean"]) for row in rows]
    fig, axis = plt.subplots(figsize=(max(10, len(methods) * 0.8), 5))
    axis.bar(x - width / 2, insertion, width, label="Insertion AUC")
    axis.bar(x + width / 2, deletion_drop, width, label="Deletion-drop AUC")
    axis.set_ylabel("AUC")
    axis.set_title("Faithfulness AUC summary")
    axis.set_xticks(x)
    axis.set_xticklabels(methods, rotation=45, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)