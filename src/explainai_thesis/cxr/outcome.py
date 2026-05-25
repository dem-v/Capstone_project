"""Classifier-outcome bookkeeping shared by the long-run scan.

AGENT.md freezes the `cases.csv` / `threshold_metrics.csv` / `progress.json`
checkpoint format. The helpers below are the resume-contract surface:
schema drift, sample_index ordering, or skip-predicate semantics here
will fail `tests/test_classifier_outcome_resume.py`.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from ..cli.progress import format_duration


def classifier_outcome(label: int, probability: float, threshold: float) -> str:
    """Map ground-truth label + classifier probability to a tp/fp/tn/fn bucket."""

    prediction = int(probability >= threshold)
    if label == 1 and prediction == 1:
        return "tp"
    if label == 0 and prediction == 1:
        return "fp"
    if label == 0 and prediction == 0:
        return "tn"
    return "fn"


def target_case_count(max_per_outcome: int, candidate_count: int) -> int:
    """Total number of cases the scan aims to keep before stopping."""

    if max_per_outcome > 0:
        return max_per_outcome * 4
    return candidate_count


def case_dir_name(sample_idx: int, outcome: str, source_stem: str) -> str:
    """Per-case output folder name used under tp/fp/tn/fn."""

    return f"case_{sample_idx:03d}_{outcome}_{source_stem}"


def write_rows(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    """Write rows as a CSV. Schema is derived from `rows[0].keys()`.

    Empty `rows` is a no-op so partial-checkpoint paths can call this
    safely on a freshly-started run.
    """
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_existing_rows(path: Path) -> list[dict[str, str]]:
    """Read a previously-written CSV. Returns `[]` for a missing file."""

    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def completed_source_keys(
    case_rows: list[dict[str, str | int | float]],
) -> set[str]:
    """Build the dedup set used by the resume skip predicate.

    Both `image_path` and `filename` columns are indexed so that
    manifest rows whose `filename` matches but whose `image_path`
    differs (e.g. moved data root) are still recognized as completed.
    """
    keys: set[str] = set()
    for row in case_rows:
        image_path = str(row.get("image_path", ""))
        filename = str(row.get("filename", ""))
        if image_path:
            keys.add(image_path)
        if filename:
            keys.add(filename)
    return keys


def write_progress_checkpoint(
    output_dir: Path,
    *,
    candidate_index: int,
    candidate_total: int,
    selected_total: int,
    target_total: int,
    outcome_counts: dict[str, int],
    elapsed_seconds: float,
    eta_seconds: float | None,
    status: str,
) -> None:
    """Write the AGENT.md-frozen `progress.json` schema to `output_dir`."""

    payload = {
        "status": status,
        "candidate_index": candidate_index,
        "candidate_total": candidate_total,
        "selected_total": selected_total,
        "target_total": target_total,
        "outcome_counts": outcome_counts,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "elapsed": format_duration(elapsed_seconds),
        "eta_seconds": round(eta_seconds, 3) if eta_seconds is not None else None,
        "eta": format_duration(eta_seconds),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with (output_dir / "progress.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
