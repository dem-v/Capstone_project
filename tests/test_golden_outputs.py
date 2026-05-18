"""Golden-output regression tests for the synthetic-data smoke pipeline.

These tests run ``scripts/run_smoke_test.py`` end-to-end on the dependency-light
synthetic dataset and assert structural invariants of the produced
``metrics.csv`` and overlay PNGs. They guard against accidental breakage of
the public XAI API (column names, method coverage, value ranges, file layout)
during the Phase 1 ``SignedAttribution`` refactor.

Bit-equal numerical comparison is deliberately avoided: CPU floating-point
output of GradCAM / Integrated Gradients depends on BLAS, torch version, and
host architecture, so we snapshot the *contract*, not the *values*.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest


SMOKE_SCRIPT_REL = "scripts/run_smoke_test.py"

# Frozen public contract of the synthetic smoke run. Update intentionally if
# the underlying script's output schema is consciously changed.
EXPECTED_COLUMNS = [
    "sample_id",
    "method",
    "test_accuracy",
    "iou",
    "dice",
    "pointing_hit",
    "precision_at_fraction",
]
EXPECTED_METHODS = {"grad_cam", "integrated_gradients", "consensus"}
EXPECTED_SAMPLES = 6  # run_smoke_test.py hard-codes a 6-positive-case loop.


@pytest.fixture(scope="module")
def smoke_run_dir(repo_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Execute the synthetic smoke script once per module; reuse outputs."""
    output_dir = tmp_path_factory.mktemp("smoke")
    cmd = [
        sys.executable,
        str(repo_root / SMOKE_SCRIPT_REL),
        "--output-dir", str(output_dir),
        "--epochs", "5",   # 5 is enough to exercise the pipeline deterministically.
        "--seed", "7",
        "--device", "cpu",
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, (
        f"run_smoke_test.py failed (exit={proc.returncode}).\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    return output_dir


@pytest.mark.slow
def test_metrics_csv_schema(smoke_run_dir: Path) -> None:
    """metrics.csv must exist with the frozen column order."""
    metrics_path = smoke_run_dir / "metrics.csv"
    assert metrics_path.is_file(), f"missing {metrics_path}"

    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == EXPECTED_COLUMNS, (
            f"metrics.csv column contract drift: got {reader.fieldnames}"
        )
        rows = list(reader)

    assert rows, "metrics.csv contains no rows"

    methods_seen = {row["method"] for row in rows}
    assert methods_seen == EXPECTED_METHODS, (
        f"method set drift: got {methods_seen}, expected {EXPECTED_METHODS}"
    )

    sample_ids = {int(row["sample_id"]) for row in rows}
    assert sample_ids == set(range(EXPECTED_SAMPLES)), (
        f"sample_id coverage drift: got {sorted(sample_ids)}"
    )

    assert len(rows) == EXPECTED_SAMPLES * len(EXPECTED_METHODS), (
        f"row count drift: got {len(rows)}"
    )


@pytest.mark.slow
def test_metrics_csv_value_ranges(smoke_run_dir: Path) -> None:
    """Numeric metrics must land in their mathematically valid ranges."""
    metrics_path = smoke_run_dir / "metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        for key in ("iou", "dice", "pointing_hit", "precision_at_fraction", "test_accuracy"):
            value = float(row[key])
            assert 0.0 <= value <= 1.0, (
                f"{key}={value} out of [0,1] for sample {row['sample_id']} "
                f"method {row['method']}"
            )


@pytest.mark.slow
def test_overlay_pngs_layout(smoke_run_dir: Path) -> None:
    """One overlay PNG per (sample, method) pair must be emitted."""
    pngs = sorted(p.name for p in smoke_run_dir.glob("sample_*_*.png"))
    assert len(pngs) == EXPECTED_SAMPLES * len(EXPECTED_METHODS), (
        f"overlay count drift: got {len(pngs)} files: {pngs}"
    )
    for sample_idx in range(EXPECTED_SAMPLES):
        for method in EXPECTED_METHODS:
            expected = f"sample_{sample_idx:02d}_{method}.png"
            assert expected in pngs, f"missing overlay: {expected}"
