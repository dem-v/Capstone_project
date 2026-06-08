#!/usr/bin/env python3
"""Render Phase 5.4 CT improvement-experiment figures from existing CSVs.

Reads the per-case and paired CSVs already produced by
`run_ct_improvement_experiment.py` and emits the same two figures the CXR
`run_improvement_experiment.py` produces (boxplots + paired-difference
bars), with identical style so Chapter 4.5 can show a CT figure next to
the CXR ones. This does NOT re-run the experiment; it only plots.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Same metric panel and order as the CT improvement experiment.
METRICS = ("iou", "dice", "pointing_hit", "precision_at_fraction")
METHOD_ORDER = ("integrated_gradients", "gradient_shap", "occlusion", "consensus_input3")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_boxplots(metric_rows: list[dict[str, str]], output_path: Path) -> None:
    present = {row["method"] for row in metric_rows}
    methods = [m for m in METHOD_ORDER if m in present]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for axis, metric in zip(axes.ravel(), METRICS):
        data = [
            [float(row[metric]) for row in metric_rows if row["method"] == method]
            for method in methods
        ]
        axis.boxplot(data, labels=methods, showfliers=False)
        axis.set_title(metric)
        axis.set_ylim(0.0, 1.0)
        axis.tick_params(axis="x", rotation=45)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_paired_differences(paired_rows: list[dict[str, str]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for axis, metric in zip(axes.ravel(), METRICS):
        rows = [row for row in paired_rows if row["metric"] == metric]
        labels = [str(row["compared"]) for row in rows]
        medians = np.asarray([float(row["median_diff"]) for row in rows], dtype=float)
        lows = np.asarray([float(row["bootstrap_ci_low"]) for row in rows], dtype=float)
        highs = np.asarray([float(row["bootstrap_ci_high"]) for row in rows], dtype=float)
        lower_err = np.maximum(0.0, medians - lows)
        upper_err = np.maximum(0.0, highs - medians)
        axis.bar(labels, medians, yerr=np.vstack([lower_err, upper_err]), capsize=3)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(metric)
        axis.tick_params(axis="x", rotation=45)
        axis.set_ylabel("reference − compared")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/iter_54_ct_improvement_test")
    parser.add_argument("--metrics-csv", default="ct_improvement_experiment.csv")
    parser.add_argument("--paired-csv", default="ct_improvement_experiment_paired.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    metric_rows = read_rows(output_dir / args.metrics_csv)
    paired_rows = read_rows(output_dir / args.paired_csv)

    boxplot_path = output_dir / "ct_improvement_experiment_boxplots.png"
    paired_path = output_dir / "ct_improvement_experiment_paired_diff.png"
    plot_boxplots(metric_rows, boxplot_path)
    plot_paired_differences(paired_rows, paired_path)

    print(f"Boxplots: {boxplot_path}")
    print(f"Paired differences: {paired_path}")


if __name__ == "__main__":
    main()
