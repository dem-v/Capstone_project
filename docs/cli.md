# Bundled CLI — `explainai-thesis`

Created: 2026-06-06
Updated: 2026-06-06 (standalone-module migration: logic moved into the package;
`scripts/*.py` are now thin shims).

The thesis pipeline is now bundled behind a single installable console entry
point. After an editable install (`pip install -e .`), every command is reachable
as a named subcommand:

```bash
explainai-thesis                 # list all commands
explainai-thesis --list          # same
explainai-thesis <command> ...   # run a command with its native flags
explainai-thesis <command> --help
python -m explainai_thesis ...   # equivalent module form
```

## Architecture (after the standalone-module migration)

Each command's implementation now lives as an importable module under
`src/explainai_thesis/cli/commands/<x>.py`. These modules were migrated
**verbatim** from the historical flat `scripts/*.py` files (no script used
`__file__`/`parents[...]` path resolution, so the move cannot alter any path or
output). Each module keeps its original `argparse` flags/defaults, `main()`, and
`__main__` guard, and is now importable as
`explainai_thesis.cli.commands.<x>` for testing/reuse.

Every `scripts/<x>.py` remains as a **thin shim** that imports `main` from the
matching command module, preserving the documented
`python scripts/<x>.py ...` invocation paths.

## Output-preservation guarantee

A subcommand is dispatched by running its underlying
`src/explainai_thesis/cli/commands/<x>.py` module file with
`run_name="__main__"` (see `src/explainai_thesis/cli/dispatcher.py`). The
module's own `argparse`/`main()` runs unchanged, so behaviour, CLI flags, CSV
schemas, PNG layout, folder names, and numeric results are **byte-for-byte
identical** to the legacy `python scripts/<x>.py ...` invocation (whose shim
imports the same module). Verified: `make-thesis-charts` produces byte-identical
PNGs via both paths, and the GPU smoke script's `--help` body is identical
across both entry points.

The legacy invocations therefore keep working exactly as documented in
`AGENTS.md` and `docs/progress.md`. No documented `scripts/<x>.py` path is
broken by this change.

## Command ↔ script mapping

Command names are the kebab-case form of the module file stem (a leading
underscore is dropped). The mapping is auto-discovered at runtime by
`src/explainai_thesis/cli/_registry.py` from the `commands/` package, so adding a
new `commands/*.py` module automatically exposes a matching subcommand — no
registry edit required. The `Shim` column is the backward-compatible legacy path.

| Command | Module (source of truth) | Shim |
| --- | --- | --- |
| `analyze-metric-correlations` | `cli/commands/analyze_metric_correlations.py` | `scripts/analyze_metric_correlations.py` |
| `analyze-review-scores` | `cli/commands/analyze_review_scores.py` | `scripts/analyze_review_scores.py` |
| `build-ct-manifest` | `cli/commands/build_ct_manifest.py` | `scripts/build_ct_manifest.py` |
| `build-manifest` | `cli/commands/build_manifest.py` | `scripts/build_manifest.py` |
| `build-review-workbook` | `cli/commands/build_review_workbook.py` | `scripts/build_review_workbook.py` |
| `calibrate-cxr-xai-thresholds` | `cli/commands/calibrate_cxr_xai_thresholds.py` | `scripts/calibrate_cxr_xai_thresholds.py` |
| `check-environment` | `cli/commands/check_environment.py` | `scripts/check_environment.py` |
| `ct-gate-probe` | `cli/commands/ct_gate_probe.py` | `scripts/ct_gate_probe.py` |
| `ct-slice-verify` | `cli/commands/ct_slice_verify.py` | `scripts/ct_slice_verify.py` |
| `diagnose-cxr-torchxray-baselines` | `cli/commands/diagnose_cxr_torchxray_baselines.py` | `scripts/diagnose_cxr_torchxray_baselines.py` |
| `embed-report-images` | `cli/commands/embed_report_images.py` | `scripts/embed_report_images.py` |
| `evaluate-cxr-torchxray-model` | `cli/commands/evaluate_cxr_torchxray_model.py` | `scripts/evaluate_cxr_torchxray_model.py` |
| `fix-progress-encoding` | `cli/commands/_fix_progress_encoding.py` | `scripts/_fix_progress_encoding.py` |
| `make-report-figures` | `cli/commands/make_report_figures.py` | `scripts/make_report_figures.py` |
| `make-thesis-charts` | `cli/commands/make_thesis_charts.py` | `scripts/make_thesis_charts.py` |
| `plot-ct-improvement` | `cli/commands/plot_ct_improvement.py` | `scripts/plot_ct_improvement.py` |
| `probe-taheera-vit` | `cli/commands/probe_taheera_vit.py` | `scripts/probe_taheera_vit.py` |
| `probe-taheera-vit-eval` | `cli/commands/probe_taheera_vit_eval.py` | `scripts/probe_taheera_vit_eval.py` |
| `run-ct-improvement-experiment` | `cli/commands/run_ct_improvement_experiment.py` | `scripts/run_ct_improvement_experiment.py` |
| `run-ct-smoke` | `cli/commands/run_ct_smoke.py` | `scripts/run_ct_smoke.py` |
| `run-cxr-torchxray-smoke` | `cli/commands/run_cxr_torchxray_smoke.py` | `scripts/run_cxr_torchxray_smoke.py` |
| `run-improvement-experiment` | `cli/commands/run_improvement_experiment.py` | `scripts/run_improvement_experiment.py` |
| `run-smoke-test` | `cli/commands/run_smoke_test.py` | `scripts/run_smoke_test.py` |
| `select-cxr-review-candidates` | `cli/commands/select_cxr_review_candidates.py` | `scripts/select_cxr_review_candidates.py` |
| `tabulate-improvement-paired` | `cli/commands/tabulate_improvement_paired.py` | `scripts/tabulate_improvement_paired.py` |
| `visualize-ct-xai-maps` | `cli/commands/visualize_ct_xai_maps.py` | `scripts/visualize_ct_xai_maps.py` |
| `visualize-cxr-classifier-outcome-thresholds` | `cli/commands/visualize_cxr_classifier_outcome_thresholds.py` | `scripts/visualize_cxr_classifier_outcome_thresholds.py` |
| `visualize-cxr-threshold-selection` | `cli/commands/visualize_cxr_threshold_selection.py` | `scripts/visualize_cxr_threshold_selection.py` |

## Commands directory resolution

The dispatcher resolves the `commands/` package directory relative to the
installed package location. For non-standard layouts, set
`EXPLAINAI_COMMANDS_DIR` (the legacy `EXPLAINAI_SCRIPTS_DIR` is still honoured) to
the absolute path of the `explainai_thesis/cli/commands/` directory.

## Example

```bash
# Legacy (still works, unchanged):
wsl.exe python3 scripts/make_thesis_charts.py

# Bundled equivalent (identical output):
wsl.exe python3 -m explainai_thesis make-thesis-charts
explainai-thesis make-thesis-charts
```
