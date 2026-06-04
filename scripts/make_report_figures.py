#!/usr/bin/env python3
"""Generate mentor-facing summary charts for the Week 4 / Week 5 reports.

Reads committed experiment CSVs and renders clean summary figures into
`outputs/iter_59_report_figures/`:

  - stage_a_model_comparison.png  (Week 4): Stage A localization sweep
  - review_distributions.png      (Week 4): balanced 40-case review counts
  - cxr_classifier_comparison.png (Week 5): chex vs ResNet-50 classifier metrics

Charts only; no experiment is re-run.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("outputs/iter_59_report_figures")
OUT.mkdir(parents=True, exist_ok=True)


def stage_a_chart() -> None:
    rows = list(csv.DictReader(open("outputs/iter_33_stage_a_diagnostic_ab/weights_ab_summary.csv")))
    rows.sort(key=lambda r: float(r["mean_dice"]), reverse=True)
    models = [r["model"].replace("densenet121-res224-", "dn-").replace("resnet50-res512-", "rn50-") for r in rows]
    dice = [float(r["mean_dice"]) for r in rows]
    iou = [float(r["mean_iou"]) for r in rows]
    prec = [float(r["mean_precision_at_fraction"]) for r in rows]
    x = np.arange(len(models))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.bar(x - w, dice, w, label="mean Dice")
    ax.bar(x, iou, w, label="mean IoU")
    ax.bar(x + w, prec, w, label="mean precision@fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("localization metric (positive masked cases)")
    ax.set_title("Stage A localization sweep — 7 TorchXRayVision checkpoints (n=180 cases each)")
    ax.legend()
    # Annotate the two selected baselines.
    for i, r in enumerate(rows):
        if r["model"] in ("resnet50-res512-all", "densenet121-res224-chex"):
            ax.annotate("selected", (i, float(r["mean_dice"])), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8, fontweight="bold")
    fig.savefig(OUT / "stage_a_model_comparison.png", dpi=160)
    plt.close(fig)


def review_distributions_chart() -> None:
    rows = list(csv.DictReader(open(
        "outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/review/scores.csv")))

    def counts(col, order):
        from collections import Counter
        c = Counter(r[col] for r in rows)
        return [c.get(k, 0) for k in order]

    loc_order = ["correct", "partial", "incorrect"]
    use_order = ["useful", "potentially_useful", "misleading", "not_useful"]
    fail_order = ["correct", "partial", "non_pathological_high_contrast",
                  "clinically_misleading", "devices_text_artifacts"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, (title, order, col, color) in zip(axes, [
        ("Localization score", loc_order, "localization_score", "#4c72b0"),
        ("Usefulness score", use_order, "usefulness_score", "#55a868"),
        ("Failure taxonomy", fail_order, "failure_category", "#c44e52"),
    ]):
        vals = counts(col, order)
        bars = ax.bar([o.replace("_", "\n") for o in order], vals, color=color)
        ax.set_title(title)
        ax.set_ylabel("cases / 40")
        ax.set_ylim(0, 16)
        ax.tick_params(axis="x", labelsize=8)
        for b, v in zip(bars, vals):
            ax.annotate(str(v), (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=9)
    fig.suptitle("Balanced 40-case ResNet-50 radiologist review (10 per tp/fp/tn/fn outcome)")
    fig.savefig(OUT / "review_distributions.png", dpi=160)
    plt.close(fig)


def classifier_comparison_chart() -> None:
    def metrics(path):
        r = next(csv.DictReader(open(path)))
        return {
            "AUC": float(r["roc_auc"]), "AP": float(r["average_precision"]),
            "Accuracy": float(r["default_accuracy"]), "Sensitivity": float(r["default_sensitivity"]),
            "Specificity": float(r["default_specificity"]), "F1": float(r["default_f1"]),
        }

    chex = metrics("outputs/iter_56_chex_classifier_eval_test/classification_metrics.csv")
    resnet = metrics("outputs/iter_55_resnet_classifier_eval_test/classification_metrics.csv")
    labels = list(chex.keys())
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    b1 = ax.bar(x - w / 2, [chex[k] for k in labels], w, label="densenet121-res224-chex")
    b2 = ax.bar(x + w / 2, [resnet[k] for k in labels], w, label="resnet50-res512-all")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score (SIIM test split, n=1372)")
    ax.set_title("CXR classifier performance — selected DenseNet (CheX) vs ResNet-50")
    ax.legend()
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    fig.savefig(OUT / "cxr_classifier_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    stage_a_chart()
    review_distributions_chart()
    classifier_comparison_chart()
    for p in sorted(OUT.glob("*.png")):
        print(p)


if __name__ == "__main__":
    main()
