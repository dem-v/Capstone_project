from __future__ import annotations

import csv
from pathlib import Path

import pytest

# The implementation now lives in the package (migrated verbatim from
# `scripts/run_improvement_experiment.py`, which remains as a thin shim).
from explainai_thesis.cli.commands import run_improvement_experiment as improvement


def test_read_calibrated_fractions_by_metric_filters_metric(tmp_path: Path) -> None:
    calibration_csv = tmp_path / "calibrated_thresholds_v3.csv"
    with calibration_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "selected_fraction", "selection_metric", "selection_metric_mean", "n"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "grad_cam",
                "selected_fraction": "0.10",
                "selection_metric": "iou",
                "selection_metric_mean": "0.01",
                "n": "3",
            }
        )
        writer.writerow(
            {
                "method": "grad_cam",
                "selected_fraction": "0.20",
                "selection_metric": "dice",
                "selection_metric_mean": "0.02",
                "n": "3",
            }
        )
        writer.writerow(
            {
                "method": "consensus",
                "selected_fraction": "0.30",
                "selection_metric": "dice",
                "selection_metric_mean": "0.03",
                "n": "3",
            }
        )

    fractions = improvement.read_calibrated_fractions_by_metric(calibration_csv, "dice")

    assert fractions == {"grad_cam": 0.2, "consensus": 0.3}


def test_read_calibrated_fractions_by_metric_rejects_missing_metric(tmp_path: Path) -> None:
    calibration_csv = tmp_path / "calibrated_thresholds_v3.csv"
    calibration_csv.write_text(
        "method,selected_fraction,selection_metric,selection_metric_mean,n\n"
        "grad_cam,0.10,iou,0.01,3\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="No calibrated fractions"):
        improvement.read_calibrated_fractions_by_metric(calibration_csv, "dice")


def test_paired_rows_outputs_one_row_per_metric_and_comparison() -> None:
    metric_rows = []
    for sample_id, (consensus_dice, gradcam_dice) in enumerate([(0.4, 0.2), (0.6, 0.5)]):
        base = {
            "sample_id": sample_id,
            "filename": f"case_{sample_id}.png",
            "split": "test",
            "weights": "synthetic",
            "view": "positive",
            "family": "synthetic",
            "top_fraction": 0.2,
            "iou": 0.1 + sample_id * 0.1,
            "pointing_hit": 1.0,
            "precision_at_fraction": 0.5,
        }
        metric_rows.append({**base, "method": "consensus", "dice": consensus_dice})
        metric_rows.append({**base, "method": "grad_cam", "dice": gradcam_dice})

    rows = improvement.paired_rows(
        metric_rows,
        reference_method="consensus",
        alpha=0.05,
        bootstrap_resamples=20,
        seed=0,
    )

    assert len(rows) == 4
    assert {row["metric"] for row in rows} == set(improvement.METRICS)
    assert {row["compared"] for row in rows} == {"grad_cam"}
    dice_row = next(row for row in rows if row["metric"] == "dice")
    assert dice_row["n_pairs"] == 2
    assert dice_row["median_diff"] == pytest.approx(0.15, abs=1e-12)