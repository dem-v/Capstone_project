#!/usr/bin/env python3
"""Render an improvement-experiment paired-statistics CSV as a thesis-ready
Markdown table.

Reads the `improvement_experiment_paired.csv` produced by
`scripts/run_improvement_experiment.py` and emits one table per localization
metric, comparing the consensus reference against each individual method:
median paired difference (reference - compared), 95% bootstrap CI, raw and
Holm-Bonferroni-adjusted Wilcoxon p-values, and the FWER-controlled
significance verdict. Reusable across models (DenseNet, ResNet) so the
Chapter 4 results tables stay byte-for-byte reproducible from the CSVs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


# Canonical metric order (mirrors run_improvement_experiment.METRICS).
METRICS = ("iou", "dice", "pointing_hit", "precision_at_fraction")

METRIC_LABELS = {
    "iou": "IoU",
    "dice": "Dice",
    "pointing_hit": "Pointing-game hit rate",
    "precision_at_fraction": "Precision@fraction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-csv", required=True)
    parser.add_argument(
        "--output",
        default=None,
        help="Markdown output path. Defaults to improvement_experiment_table.md "
        "next to the paired CSV.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional table heading; defaults to a weights-derived title.",
    )
    return parser.parse_args()


def fmt(value: str, places: int = 4) -> str:
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return str(value)


def read_run_meta(paired_csv: Path) -> dict:
    meta_path = paired_csv.parent / "run_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def render(rows: list[dict[str, str]], *, title: str, reference: str) -> str:
    n_pairs = rows[0]["n_pairs"] if rows else "?"
    lines = [
        f"## {title}",
        "",
        f"Reference method: `{reference}`. Paired across n = {n_pairs} held-out "
        "positive cases. Test: two-sided paired Wilcoxon signed-rank; "
        "Holm-Bonferroni FWER control at α = 0.05; 95 % bootstrap CI "
        "(10 000 resamples) on the paired median difference (reference − compared).",
        "",
    ]
    by_metric: dict[str, list[dict[str, str]]] = {metric: [] for metric in METRICS}
    for row in rows:
        by_metric.setdefault(row["metric"], []).append(row)

    for metric in METRICS:
        metric_rows = by_metric.get(metric, [])
        if not metric_rows:
            continue
        lines.append(f"### {METRIC_LABELS.get(metric, metric)}")
        lines.append("")
        lines.append(
            "| Compared method | Median Δ (ref − cmp) | 95 % CI | p (raw) | "
            "p (Holm-adj.) | Significant |"
        )
        lines.append("|---|---|---|---|---|---|")
        for row in sorted(metric_rows, key=lambda r: str(r["compared"])):
            significant = str(row.get("holm_significant_bool", "")).strip().lower() == "true"
            ci = f"[{fmt(row['bootstrap_ci_low'])}, {fmt(row['bootstrap_ci_high'])}]"
            lines.append(
                f"| `{row['compared']}` | {fmt(row['median_diff'])} | {ci} | "
                f"{fmt(row['p_raw'])} | {fmt(row['p_holm_adjusted'])} | "
                f"{'✓' if significant else '—'} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    paired_csv = Path(args.paired_csv)
    with paired_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No paired rows found in {paired_csv}.")

    reference = rows[0].get("reference", "consensus")
    meta = read_run_meta(paired_csv)
    weights = meta.get("args", {}).get("weights", "model")
    title = args.title or f"Consensus vs. individual methods — {weights}"

    table = render(rows, title=title, reference=reference)
    output_path = Path(args.output) if args.output else paired_csv.parent / "improvement_experiment_table.md"
    output_path.write_text(table, encoding="utf-8")
    print(table)
    print(f"Markdown table written to: {output_path}")


if __name__ == "__main__":
    main()
