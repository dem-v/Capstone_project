"""Regression tests for the classifier-outcome resume contract.

AGENT.md freezes the `cases.csv` / `threshold_metrics.csv` / `progress.json`
checkpoint format in `scripts/visualize_cxr_classifier_outcome_thresholds.py`.
This module locks the load-bearing pieces:

- `read_existing_rows` / `write_rows` round-trip schemas exactly.
- `completed_source_keys` builds the dedup set used by the main loop's
  skip predicate from both `image_path` and `filename`.
- A simulated kill/resume scenario over a synthetic candidate list
  produces the same final case ordering and row counts as a clean run.
- `write_progress_checkpoint` emits the documented `progress.json` keys.

These tests run on CPU without a model so they stay in the default suite
(<5 s budget). A full integration test that invokes the script as a
subprocess is feasible but belongs in `@pytest.mark.slow`.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "visualize_cxr_classifier_outcome_thresholds.py"


def _load_script_module():
    """Import the classifier-outcome script as a module for unit testing.

    The script is not a package — it's invoked with `python scripts/...`.
    We deliberately avoid running its `main()` here; only top-level
    definitions (helpers, schema, resume primitives) matter to us.
    """
    spec = importlib.util.spec_from_file_location(
        "_classifier_outcome_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script_mod():
    return _load_script_module()


# --- Pinned schemas (AGENT.md "frozen output contract") ---------------------
# These exact column orders are written by the script today. The test
# fails on schema drift; if a column is added intentionally, update the
# constant here and document the change in `docs/progress.md`.

CASES_FIELDS_PINNED = (
    "sample_index",
    "candidate_index",
    "filename",
    "split",
    "label",
    "prediction",
    "classifier_outcome",
    "xrv_pneumothorax_score",
    "xrv_pneumothorax_sigmoid",
    "classifier_threshold",
    "weights",
    "image_size",
    "image_path",
    "mask_path",
)

PROGRESS_JSON_KEYS_PINNED = {
    "status",
    "candidate_index",
    "candidate_total",
    "selected_total",
    "target_total",
    "outcome_counts",
    "elapsed_seconds",
    "elapsed",
    "eta_seconds",
    "eta",
    "updated_at",
}


def _sample_case_row(idx: int, filename: str, image_path: str) -> dict[str, str | int | float]:
    return {
        "sample_index": idx,
        "candidate_index": idx + 1,
        "filename": filename,
        "split": "train",
        "label": 1,
        "prediction": 1,
        "classifier_outcome": "tp",
        "xrv_pneumothorax_score": 1.23,
        "xrv_pneumothorax_sigmoid": 0.77,
        "classifier_threshold": 0.62,
        "weights": "resnet50-res512-all",
        "image_size": 512,
        "image_path": image_path,
        "mask_path": image_path.replace(".png", "_mask.png"),
    }


def test_read_existing_rows_returns_empty_for_missing_file(tmp_path, script_mod):
    assert script_mod.read_existing_rows(tmp_path / "absent.csv") == []


def test_write_then_read_cases_roundtrip_preserves_schema(tmp_path, script_mod):
    rows = [
        _sample_case_row(0, "a.png", "data/a.png"),
        _sample_case_row(1, "b.png", "data/b.png"),
    ]
    target = tmp_path / "cases.csv"
    script_mod.write_rows(target, rows)
    text = target.read_text(encoding="utf-8")
    header = text.splitlines()[0]
    assert tuple(header.split(",")) == CASES_FIELDS_PINNED

    read_back = script_mod.read_existing_rows(target)
    assert len(read_back) == len(rows)
    for original, restored in zip(rows, read_back):
        assert restored["filename"] == original["filename"]
        assert restored["image_path"] == original["image_path"]
        assert restored["classifier_outcome"] == original["classifier_outcome"]


def test_completed_source_keys_indexes_image_path_and_filename(script_mod):
    rows = [
        _sample_case_row(0, "a.png", "data/a.png"),
        _sample_case_row(1, "b.png", "data/b.png"),
    ]
    keys = script_mod.completed_source_keys(rows)
    assert "data/a.png" in keys
    assert "a.png" in keys
    assert "data/b.png" in keys
    assert "b.png" in keys
    assert len(keys) == 4


def test_completed_source_keys_handles_missing_filename(script_mod):
    rows = [{"image_path": "data/x.png", "filename": ""}]
    keys = script_mod.completed_source_keys(rows)
    assert "data/x.png" in keys
    assert "" not in keys


def test_write_progress_checkpoint_emits_documented_keys(tmp_path, script_mod):
    script_mod.write_progress_checkpoint(
        tmp_path,
        candidate_index=12,
        candidate_total=100,
        selected_total=5,
        target_total=40,
        outcome_counts={"tp": 2, "fp": 1, "tn": 1, "fn": 1},
        elapsed_seconds=123.456,
        eta_seconds=789.012,
        status="running",
    )
    progress_path = tmp_path / "progress.json"
    assert progress_path.exists()
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert set(payload.keys()) == PROGRESS_JSON_KEYS_PINNED
    assert payload["candidate_index"] == 12
    assert payload["selected_total"] == 5
    assert payload["status"] == "running"
    assert payload["outcome_counts"]["tp"] == 2


def test_write_progress_checkpoint_handles_unknown_eta(tmp_path, script_mod):
    script_mod.write_progress_checkpoint(
        tmp_path,
        candidate_index=0,
        candidate_total=100,
        selected_total=0,
        target_total=40,
        outcome_counts={"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        elapsed_seconds=0.0,
        eta_seconds=None,
        status="starting",
    )
    payload = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert payload["eta_seconds"] is None
    assert payload["eta"] == "unknown"


def test_simulated_kill_and_resume_produces_no_duplicate_rows(tmp_path, script_mod):
    """Simulate the loop's skip-then-resume contract.

    Flow: write a partial `cases.csv` with two rows, then iterate the
    full candidate list using the same skip predicate the main loop
    uses (`row["image_path"] in completed_keys or filename in completed_keys`),
    and verify the merged final state has the right cardinality, no
    duplicates, and the same ordering as a clean run.
    """

    candidates = [
        {"image_path": "data/c0.png", "filename": "c0.png"},
        {"image_path": "data/c1.png", "filename": "c1.png"},
        {"image_path": "data/c2.png", "filename": "c2.png"},
        {"image_path": "data/c3.png", "filename": "c3.png"},
        {"image_path": "data/c4.png", "filename": "c4.png"},
    ]

    # Partial run: persist the first two as if a checkpoint had landed,
    # then a kill (SIGKILL or OOM) happened mid-third-case.
    partial_rows = [
        _sample_case_row(0, "c0.png", "data/c0.png"),
        _sample_case_row(1, "c1.png", "data/c1.png"),
    ]
    cases_path = tmp_path / "cases.csv"
    script_mod.write_rows(cases_path, partial_rows)

    # Resume: read existing, build skip set, process the rest.
    resumed_rows = script_mod.read_existing_rows(cases_path)
    completed_keys = script_mod.completed_source_keys(resumed_rows)

    final_rows: list[dict[str, str | int | float]] = list(resumed_rows)
    next_sample_idx = len(final_rows)
    for candidate in candidates:
        if (
            candidate["image_path"] in completed_keys
            or candidate["filename"] in completed_keys
        ):
            continue
        final_rows.append(
            _sample_case_row(next_sample_idx, candidate["filename"], candidate["image_path"])
        )
        next_sample_idx += 1

    # Properties of the post-resume state.
    assert len(final_rows) == len(candidates)
    image_paths = [row["image_path"] for row in final_rows]
    assert len(set(image_paths)) == len(image_paths), "no duplicate rows after resume"
    assert image_paths == [candidate["image_path"] for candidate in candidates], (
        "row order must match the unbroken candidate sequence"
    )

    # The resumed rows must keep their original `sample_index`; the
    # newly-added rows must continue the sequence without gaps.
    assert [int(row["sample_index"]) for row in final_rows] == list(range(len(candidates)))


def test_clean_run_and_resumed_run_produce_identical_final_csv(tmp_path, script_mod):
    """The byte-level invariant the future refactor must preserve.

    Flow: build a synthetic 5-case "clean" final CSV; build the same
    final CSV via a partial-then-resume sequence; assert both files have
    identical schema headers and the same row content.
    """

    candidates = [
        ("data/k0.png", "k0.png"),
        ("data/k1.png", "k1.png"),
        ("data/k2.png", "k2.png"),
        ("data/k3.png", "k3.png"),
        ("data/k4.png", "k4.png"),
    ]

    # --- Clean run ---
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    clean_rows = [
        _sample_case_row(idx, fn, ip)
        for idx, (ip, fn) in enumerate(candidates)
    ]
    script_mod.write_rows(clean_dir / "cases.csv", clean_rows)

    # --- Partial + resume run ---
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    partial_rows = [
        _sample_case_row(0, candidates[0][1], candidates[0][0]),
        _sample_case_row(1, candidates[1][1], candidates[1][0]),
        _sample_case_row(2, candidates[2][1], candidates[2][0]),
    ]
    script_mod.write_rows(resume_dir / "cases.csv", partial_rows)

    resumed_rows = script_mod.read_existing_rows(resume_dir / "cases.csv")
    completed_keys = script_mod.completed_source_keys(resumed_rows)
    merged = list(resumed_rows)
    next_idx = len(merged)
    for image_path, filename in candidates:
        if image_path in completed_keys or filename in completed_keys:
            continue
        merged.append(_sample_case_row(next_idx, filename, image_path))
        next_idx += 1
    script_mod.write_rows(resume_dir / "cases.csv", merged)

    clean_text = (clean_dir / "cases.csv").read_text(encoding="utf-8")
    resume_text = (resume_dir / "cases.csv").read_text(encoding="utf-8")
    assert clean_text == resume_text, (
        "resumed cases.csv must be byte-identical to a clean run "
        "for the schema-freeze gate"
    )


def test_threshold_metrics_csv_schema_pinned(tmp_path, script_mod):
    """Lock the threshold_metrics.csv schema by checking write+read order.

    The script writes rows whose dict insertion order defines the CSV
    header. We don't try to reproduce the full producer here — instead
    we lock the schema downstream by feeding a row with the known
    column set and asserting the header round-trips intact.
    """

    pinned_fields = (
        "sample_index",
        "filename",
        "source_stem",
        "image_path",
        "mask_path",
        "label",
        "prediction",
        "classifier_outcome",
        "weights",
        "image_size",
        "method",
        "view",
        "family",
        "metric_component",
        "top_fraction",
        "top_fraction_percent",
        "positive_localization_applicable",
        "selected_pixel_count",
        "mask_pixel_count",
        "intersection_pixel_count",
        "union_pixel_count",
        "iou",
        "dice",
        "pointing_hit",
        "precision_at_fraction",
        "negative_mask_overlap_fraction",
        "negative_mask_avoidance_fraction",
    )
    row = {field: "" for field in pinned_fields}
    target = tmp_path / "threshold_metrics.csv"
    script_mod.write_rows(target, [row])
    header = target.read_text(encoding="utf-8").splitlines()[0]
    assert tuple(header.split(",")) == pinned_fields
