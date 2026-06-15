#!/usr/bin/env python3
"""Aggregate the per-case ResNet-50 faithfulness curves from the balanced-40
review diagnostics (iter_47) into one thesis figure per classifier outcome
(tp / fp / tn / fn).

Light, CPU-only post-processing: it reads the per-case ``faithfulness_curves.csv``
files written by the review-diagnostics stage, keeps the positive-view rows,
groups them by the outcome token in each case-folder name, and reuses the
canonical ``plot_faithfulness_curves()`` so the styling matches the other
faithfulness figures. No model inference, no GPU, runs in seconds.

Run from the repo root:

    python3 scripts/aggregate_review_faithfulness.py
"""
from __future__ import annotations

import csv
import glob
import re
import shutil
from pathlib import Path

from explainai_thesis.faithfulness import plot_faithfulness_curves

SRC_GLOB = (
    "outputs/iter_47_resnet_review_diagnostics_balanced40_smoothed_faithfulness/"
    "*/faithfulness_curves.csv"
)
OUT_DIR = Path("outputs/iter_47_resnet_faithfulness_aggregate")
FIG_DIR = Path("thesis/latex/figures")
OUTCOMES = ("tp", "fp", "tn", "fn")
OUTCOME_LABEL = {
    "tp": "true positives",
    "fp": "false positives",
    "tn": "true negatives",
    "fn": "false negatives",
}
OUTCOME_RE = re.compile(r"_balanced_(tp|fp|tn|fn)_")


def main() -> None:
    subsets: dict[str, list[dict[str, float | str]]] = {o: [] for o in OUTCOMES}
    case_ids: dict[str, set[str]] = {o: set() for o in OUTCOMES}
    files = sorted(glob.glob(SRC_GLOB))
    for path in files:
        case_name = Path(path).parent.name
        match = OUTCOME_RE.search(case_name)
        if not match:
            print(f"WARNING: could not parse outcome from {case_name}; skipping")
            continue
        outcome = match.group(1)
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                # Positive-view curves only: one insertion/deletion line per
                # method (incl. consensus), not the positive/negative/magnitude/
                # signed fan-out.
                if row.get("view") != "positive":
                    continue
                subsets[outcome].append(
                    {
                        "method": row["method"],
                        "fraction": float(row["fraction"]),
                        "insertion_probability": float(row["insertion_probability"]),
                        "deletion_probability": float(row["deletion_probability"]),
                    }
                )
        case_ids[outcome].add(case_name)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for outcome in OUTCOMES:
        rows = subsets[outcome]
        n = len(case_ids[outcome])
        if not rows:
            print(f"WARNING: no positive-view rows for {outcome}")
            continue
        src = OUT_DIR / f"faithfulness_curves_{outcome}.png"
        plot_faithfulness_curves(
            rows,
            src,
            f"ResNet-50 faithfulness -- {OUTCOME_LABEL[outcome]} "
            f"({outcome.upper()}, n={n})",
        )
        dest = FIG_DIR / f"iter_47_resnet_faithfulness_{outcome}__faithfulness_curves.png"
        shutil.copyfile(src, dest)
        print(f"{outcome.upper()}: n={n} cases -> {dest}")


if __name__ == "__main__":
    main()
