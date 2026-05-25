from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explainai_thesis.run_metadata import write_run_metadata


DEFAULT_METRICS = (
    "iou",
    "dice",
    "pointing_hit",
    "precision_at_fraction",
    "negative_mask_overlap_fraction",
    "negative_mask_avoidance_fraction",
    "signed_positive_fraction",
    "signed_prediction_alignment",
    "agreement_score",
    "insertion_auc",
    "deletion_auc",
    "faithfulness_insertion_auc",
    "faithfulness_deletion_auc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze metric-metric correlations from one or more XAI metrics CSV files."
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        action="append",
        required=True,
        help="Path to a metrics CSV. May be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/metric_correlations"),
        help="Directory where correlation matrices and heatmaps will be written.",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metric columns to analyze when present.",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=3,
        help="Minimum finite row pairs required for a correlation cell.",
    )
    parser.add_argument(
        "--filter-view",
        default="",
        help="Optional value for the v2 'view' column, e.g. positive, negative, magnitude, signed.",
    )
    parser.add_argument(
        "--filter-method",
        action="append",
        default=[],
        help="Optional method to keep. May be passed multiple times.",
    )
    return parser.parse_args()


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["source_csv"] = str(path)
                rows.append(row)
    return rows


def as_float(value: str | None) -> float:
    if value in (None, ""):
        return math.nan
    try:
        out = float(value)
    except ValueError:
        return math.nan
    return out if math.isfinite(out) else math.nan


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return math.nan
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom == 0.0:
        return math.nan
    return float(np.sum(x * y) / denom)


def correlation_matrix(data: dict[str, np.ndarray], method: str, min_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    names = list(data.keys())
    matrix = np.full((len(names), len(names)), np.nan, dtype=float)
    counts = np.zeros((len(names), len(names)), dtype=int)
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            mask = np.isfinite(data[left]) & np.isfinite(data[right])
            counts[i, j] = int(mask.sum())
            if counts[i, j] < min_pairs:
                continue
            x = data[left][mask]
            y = data[right][mask]
            if method == "spearman":
                x = rankdata(x)
                y = rankdata(y)
            matrix[i, j] = pearson(x, y)
    return matrix, counts


def write_matrix(path: Path, names: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", *names])
        for name, row in zip(names, matrix):
            writer.writerow([name, *["" if math.isnan(v) else f"{v:.6g}" for v in row]])


def write_counts(path: Path, names: list[str], counts: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", *names])
        for name, row in zip(names, counts):
            writer.writerow([name, *row.tolist()])


def plot_heatmap(path: Path, title: str, names: list[str], matrix: np.ndarray) -> None:
    fig_width = max(8.0, len(names) * 0.7)
    fig_height = max(6.0, len(names) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(names)), labels=names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(names)), labels=names)
    for i in range(len(names)):
        for j in range(len(names)):
            value = matrix[i, j]
            if math.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.metrics_csv)
    input_row_count = len(rows)
    if args.filter_view:
        rows = [row for row in rows if row.get("view") == args.filter_view]
    if args.filter_method:
        allowed = set(args.filter_method)
        rows = [row for row in rows if row.get("method") in allowed]

    requested = [m.strip() for m in args.metrics.split(",") if m.strip()]
    present = [m for m in requested if any(row.get(m) not in (None, "") for row in rows)]
    if len(present) < 2:
        raise SystemExit("Need at least two present numeric metric columns after filtering.")

    data = {metric: np.array([as_float(row.get(metric)) for row in rows], dtype=float) for metric in present}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    names = list(data.keys())
    for method in ("pearson", "spearman"):
        matrix, counts = correlation_matrix(data, method=method, min_pairs=args.min_pairs)
        write_matrix(args.output_dir / f"{method}_correlations.csv", names, matrix)
        write_counts(args.output_dir / f"{method}_pair_counts.csv", names, counts)
        plot_heatmap(args.output_dir / f"{method}_correlations.png", f"{method.title()} metric correlations", names, matrix)

    summary_path = args.output_dir / "correlation_input_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["input_rows", "kept_rows", "metric_columns"])
        writer.writerow([input_row_count, len(rows), ",".join(names)])

    run_meta_path = write_run_metadata(
        args.output_dir,
        args,
        input_rows=input_row_count,
        kept_rows=len(rows),
        metric_columns=names,
    )

    print(f"Analyzed {len(rows)} rows and {len(names)} metric columns.")
    print(f"Wrote correlation outputs to {args.output_dir}")
    print(f"Run metadata written to: {run_meta_path}")


if __name__ == "__main__":
    main()