#!/usr/bin/env python3
"""Generate the final thesis Chapter 4 charts from frozen review-count CSVs.

Reads the canonical balanced 40-case ResNet-50 review aggregates and renders
the two thesis charts referenced in `thesis/thesis_skeleton.md`:

  - chart_4_2_review_scores.png   : Table 4.4 (localization + usefulness counts)
  - chart_4_3_failure_taxonomy.png: Table 4.5 (failure categories) + review flags

Source counts (canonical, frozen review = iter_48):
  - outputs/iter_48_resnet_review_analysis_balanced40_smoothed_faithfulness/review_score_counts.csv
  - outputs/iter_48_resnet_review_analysis_balanced40_smoothed_faithfulness/review_flag_counts.csv

Charts only; no experiment is re-run and no source CSV is modified.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REVIEW = Path(
    "outputs/iter_48_resnet_review_analysis_balanced40_smoothed_faithfulness"
)
SCORE_COUNTS = REVIEW / "review_score_counts.csv"
FLAG_COUNTS = REVIEW / "review_flag_counts.csv"
OUT = Path("outputs/iter_61_thesis_charts")

N_CASES = 40

# Stable category orders matching Tables 4.4 and 4.5 in thesis_skeleton.md.
LOC_ORDER = ["correct", "partial", "incorrect"]
USE_ORDER = ["useful", "potentially_useful", "misleading", "not_useful"]
FAIL_ORDER = [
    "correct",
    "partial",
    "non_pathological_high_contrast",
    "clinically_misleading",
    "devices_text_artifacts",
]
FLAG_ORDER = [
    "flag_devices_or_tubes",
    "flag_indirect_evidence",
    "flag_method_disagreement",
    "flag_subcutaneous_emphysema",
    "flag_weak_pixel_attribution",
    "flag_mask_quality_issue",
]


def read_score_counts() -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    with open(SCORE_COUNTS, newline="") as fh:
        for row in csv.DictReader(fh):
            counts[row["field"]][row["value"]] = int(row["n"])
    return counts


def read_flag_counts() -> dict[str, int]:
    present: dict[str, int] = {}
    with open(FLAG_COUNTS, newline="") as fh:
        for row in csv.DictReader(fh):
            present[row["flag"]] = int(row["present"])
    return present


def _annotate(ax, bars, vals) -> None:
    for bar, val in zip(bars, vals):
        ax.annotate(
            str(val),
            (bar.get_x() + bar.get_width() / 2, val),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=9,
        )


def chart_4_2(score_counts: dict[str, dict[str, int]]) -> Path:
    loc = [score_counts["localization_score"].get(k, 0) for k in LOC_ORDER]
    use = [score_counts["usefulness_score"].get(k, 0) for k in USE_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, (title, order, vals, color) in zip(
        axes,
        [
            ("Localization score", LOC_ORDER, loc, "#4c72b0"),
            ("Usefulness score", USE_ORDER, use, "#55a868"),
        ],
    ):
        bars = ax.bar([o.replace("_", "\n") for o in order], vals, color=color)
        ax.set_title(title)
        ax.set_ylabel(f"cases / {N_CASES}")
        ax.set_ylim(0, 16)
        ax.tick_params(axis="x", labelsize=8)
        _annotate(ax, bars, vals)
    fig.suptitle(
        "Chart 4.2 — Balanced 40-case ResNet-50 radiologist review score distribution "
        "(10 per tp/fp/tn/fn outcome)"
    )
    path = OUT / "chart_4_2_review_scores.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def chart_4_3(
    score_counts: dict[str, dict[str, int]], flag_counts: dict[str, int]
) -> Path:
    fail = [score_counts["failure_category"].get(k, 0) for k in FAIL_ORDER]
    flags = [flag_counts.get(k, 0) for k in FLAG_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    bars = axes[0].bar(
        [o.replace("_", "\n") for o in FAIL_ORDER], fail, color="#c44e52"
    )
    axes[0].set_title("Failure taxonomy")
    axes[0].set_ylabel(f"cases / {N_CASES}")
    axes[0].set_ylim(0, 16)
    axes[0].tick_params(axis="x", labelsize=8)
    _annotate(axes[0], bars, fail)

    flag_labels = [f.replace("flag_", "").replace("_", "\n") for f in FLAG_ORDER]
    bars = axes[1].bar(flag_labels, flags, color="#8172b3")
    axes[1].set_title("Qualitative review flags (present)")
    axes[1].set_ylabel(f"cases / {N_CASES}")
    axes[1].set_ylim(0, 22)
    axes[1].tick_params(axis="x", labelsize=8)
    _annotate(axes[1], bars, flags)

    fig.suptitle(
        "Chart 4.3 — Balanced 40-case review failure taxonomy and qualitative flags"
    )
    path = OUT / "chart_4_3_failure_taxonomy.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    score_counts = read_score_counts()
    flag_counts = read_flag_counts()
    paths = [
        chart_4_2(score_counts),
        chart_4_3(score_counts, flag_counts),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
