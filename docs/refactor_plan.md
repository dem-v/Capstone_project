# Refactor & Optimization Plan (Temporary Working Doc)

Created: 2026-05-18
Owner: Dmytro Valantsevych
Reviewer model: Claude Opus 4.7 (1M context)
Status: Draft, awaiting user go-ahead to start Phase 0.

This file is a temporary working artifact. It mirrors the plan accepted in conversation and is intended for task delegation. Once the work is completed and the resulting decisions are reflected in `docs/progress.md`, this file may be deleted.

---

## Hard Constraints (from `AGENT.md`)

- WSL Ubuntu is the canonical Python environment. All test commands use `wsl.exe python3`.
- Output folder layout is frozen: `outputs/iter_XX_<short>/`, root-level CSVs, per-case folders with source-X-ray-stem in every filename, and `tp/fp/tn/fn/` outcome subfolders.
- CLI flag names are frozen across all current scripts, including the long `iter_27` classifier-outcome command pattern.
- `cases.csv`, `threshold_metrics.csv`, `progress.json` checkpoint format in `scripts/visualize_cxr_classifier_outcome_thresholds.py` is frozen.
- `docs/progress.md` is append-only, chronological.
- `reports/weekly/week_1_report*.md` are frozen and must not be edited.
- Any changed Python file must pass `wsl.exe python3 -m py_compile <file>` before being declared done.
- Default tests must finish under 5 minutes on CPU. Slow tests use `@pytest.mark.slow`.
- Color semantics from `AGENT.md` apply unchanged, plus the new `orange`/`teal` pair for signed scalar maps.

---

## Phase 0 — Foundation

Goal: establish the safety net so later phases can move fast.

- [ ] Add `pyproject.toml` declaring the `explainai_thesis` package; install with `pip install -e .` so `sys.path.insert` is no longer needed.
- [ ] Remove `sys.path.insert(0, str(ROOT / "src"))` blocks from all scripts.
- [ ] Add `requirements-dev.txt` with `pytest`, `pytest-cov`, `ruff`, `mypy`, `scipy`.
- [ ] Create `tests/conftest.py` with reusable fixtures: small deterministic synthetic case (8 samples, seed locked), a known mask/known heatmap pair for metric tests.
- [ ] Add `tests/test_golden_outputs.py`: run smoke on 2 synthetic cases, snapshot `os.listdir(output_dir)` and CSV header to a fixture file. Fail if any future change perturbs the schema.
- [ ] Add `.github/workflows/ci.yml` running `ruff check`, `mypy`, and `pytest -m "not slow and not cuda"` on `ubuntu-latest`.
- [ ] Add a `Makefile` (or `tasks.json`) with targets `test`, `test-fast`, `lint`, `compile-check` that all route through `wsl.exe python3` for local use.

Acceptance: `make test` green, `py_compile` clean across all scripts, golden snapshot recorded.

---

## Phase 1 — Correctness

### 1.1 Fix `grad_cam_plus_plus` polarity double-flip

File: `src/explainai_thesis/xai.py:61-81`

Bug: in the `grad_cam_plus_plus` branch, gradients are sign-flipped when `polarity == "negative"`, and then `cam = F.relu(-cam)` is applied below, which is effectively a second sign-flip.

Fix: pick one place to apply polarity. Recommend keeping the post-weight `cam = F.relu(-cam)` consistent with the standard `grad_cam` branch and removing the pre-weight gradient flip in `grad_cam_plus_plus`.

Tests:
- `test_gradcam_negative_differs_from_positive`: assert `gradcam(x, polarity="negative")` is not numerically equal to `gradcam(x, polarity="positive")` on a synthetic positive case.
- `test_gradcam_negative_does_not_peak_inside_lesion`: argmax of the negative map should not fall inside the lesion mask on average across the synthetic test split (probabilistic, allow > 0.5 of cases outside).

Regression: rerun synthetic smoke and one CXR positive case, attach the before/after overlays to `docs/progress.md` for thesis-defense audit trail.

### 1.2 SignedAttribution: extend the four-view contract to every polarity-supporting method

Scope expansion: the original draft of 1.2 only fixed `integrated_gradients_signed` and `gradient_shap_signed`. The same `positive` / `negative` / `magnitude` / `signed` decomposition is natural and free for `grad_cam`, `grad_cam_plus_plus`, and `occlusion` as well. Each of those methods today calls a full forward+backward (or full occlusion sweep) **separately per polarity**, which is wasted compute and is the same code path that produced the `grad_cam_plus_plus` polarity double-flip bug.

Files: `src/explainai_thesis/xai.py` (all XAI methods), `scripts/run_cxr_torchxray_smoke.py` (method dispatch).

#### New return type

```python
@dataclass(frozen=True)
class SignedAttribution:
    raw: torch.Tensor   # signed map, normalized to [-1, 1] preserving sign

    @property
    def positive(self) -> torch.Tensor:   return self.raw.clamp(min=0)
    @property
    def negative(self) -> torch.Tensor:   return self.raw.clamp(max=0).abs()
    @property
    def magnitude(self) -> torch.Tensor:  return self.raw.abs()
    @property
    def signed(self) -> torch.Tensor:     return self.raw
```

Each XAI method returns a single `SignedAttribution` per case. The four views are derived in microseconds. The `polarity=` keyword argument is removed from the public API; a thin deprecated wrapper preserving the old keyword stays for one phase so `scripts/run_cxr_torchxray_smoke.py` can be refactored independently.

#### Per-method changes

- **Grad-CAM**: return `(weights * activations).sum()` **before** `F.relu`, interpolated to image size, normalized to `[-1, 1]`. Eliminates the post-hoc `F.relu(-cam)` polarity dance. Single forward+backward per case (was two).
- **Grad-CAM++**: same. Drop the `gradients = -self.gradients` pre-weight flip. Bug class disappears.
- **Integrated Gradients**: return raw `(image - baseline) * avg_gradients` per case. Plays well with the batched IG refactor in Phase 3.1. One IG loop per case (was three for magnitude/positive/negative).
- **GradientSHAP**: return Captum's raw signed attribution. One call per case (was three).
- **Occlusion Sensitivity**: return raw `delta = original_score - occluded_score` map. One occlusion sweep per case (was three) — biggest absolute speedup since occlusion is the most expensive method.
- **Consensus**: average signed maps from constituents → emits a signed consensus, not just a magnitude consensus. Adds `consensus_signed` as a first-class output.

#### Localization metrics on the four views

- `positive`, `negative`, `magnitude` use existing top-fraction thresholding.
- `signed` uses `abs(signed)` thresholded at the calibrated top-fraction (consistent with magnitude convention), plus a new diagnostic column `signed_positive_fraction` = fraction of selected pixels where `signed > 0`.

#### Cross-method agreement metric (free, high-value)

Add `agreement_score` per case: cosine similarity between the signed maps of two methods (e.g., Grad-CAM vs IG). Surfaces "do methods agree on direction, not just magnitude" — a strong thesis point. Computed on the same case from already-available `raw` tensors.

#### Overlay rendering

For each polarity-supporting method per case, write four overlays into the case folder:
- `*_positive.png` (red)
- `*_negative.png` (blue)
- `*_magnitude.png` (violet)
- `*_signed.png` (orange ↔ teal diverging)

Per-case output count grows from ~16 to ~24 PNGs. Disk impact roughly +50% per `iter_XX` folder; acceptable.

#### Tests

- `test_signed_map_decomposition`: assert `positive + negative ≈ magnitude` and `positive - negative == signed`.
- `test_signed_map_not_equal_to_positive` for each method: signed ≠ positive numerically.
- `test_signed_consensus_diverging`: on the synthetic dataset, the signed consensus must contain both positive and negative pixels.
- `test_gradcam_single_compute_matches_polarity_pair`: the new single-compute Grad-CAM positive view must equal the old positive-polarity call within tolerance, and same for negative. Regression guard during the refactor.

#### AGENT.md updates

Append new method names to the XAI Method Set: `grad_cam_magnitude`, `grad_cam_signed`, `grad_cam_plus_plus_magnitude`, `grad_cam_plus_plus_signed`, `occlusion_signed`, `consensus_signed`. (`integrated_gradients_signed` and `gradient_shap_signed` are already listed.) The Color Semantics block already covers orange/teal.

### 1.3 Metric unit tests

File: new `tests/test_metrics.py`

Cases:
- Perfect overlap → IoU = Dice = 1.0.
- Disjoint masks → IoU = 0.0.
- Single-pixel argmax inside vs outside mask → `pointing_game_hit` returns 1.0 vs 0.0.
- All-zero heatmap edge case: verify behavior is defined (currently `normalize_map` with ε ≈ 1e-8 produces flat output, `argmax` returns index 0 — document or fix).
- `threshold_top_fraction(fraction=1.0)` returns all-True mask.

### 1.4 Faithfulness sanity test

File: new `tests/test_faithfulness.py`

Assertion: insertion AUC of `grad_cam` on the synthetic dataset > insertion AUC of a random heatmap. If this regresses, downstream thesis claims are unsafe.

Mark `@pytest.mark.slow` if it pushes total suite over 5 minutes.

### 1.5 Manifest label-inference robustness

File: `src/explainai_thesis/manifest.py:34-43`

Bug: substring markers `_1_`, `_0_` would also match `_10_`, `_11_`, `_100_`, `_01_`.

Fix: use word-boundary regex or require exact-token matches.

Tests: `tests/test_manifest.py` with adversarial filenames including `case_10_chest.png`, `image_01_pneumothorax.png`, `study_1_seg.png`.

### 1.6 Faithfulness default baseline switch

File: `scripts/run_cxr_torchxray_smoke.py:117`

Change default `--faithfulness-baseline` from `zero_tensor` to `black`. Keep `zero_tensor` as a valid choice, annotated "historical / not recommended" in argparse help. Add an inline comment citing `AGENT.md` rationale.

---

## Phase 2 — Structural Refactor (Under Frozen Output/CLI Contract)

Files to extract from `scripts/run_cxr_torchxray_smoke.py` (1037 lines → target ~150) into the `src/explainai_thesis/` package:

- [ ] `src/explainai_thesis/faithfulness.py`
  - `faithfulness_baseline_tensor`
  - `faithfulness_curve_rows`
  - `curve_auc`
  - `write_faithfulness_summary`
  - `plot_faithfulness_curves`
  - `plot_faithfulness_summary`
  - `write_faithfulness_plots`
  - `faithfulness_method_family`

- [ ] `src/explainai_thesis/cli/common.py`
  - `resolve_device`
  - Shared argparse parents for `--manifest`, `--split`, `--output-dir`, `--device`, `--seed`. CRITICAL: flag names unchanged, defaults unchanged.

- [ ] `src/explainai_thesis/cxr/io.py`
  - `load_image`
  - `load_mask`
  - `pathology_index`
  - `safe_case_name`
  - `safe_source_stem`
  - `read_positive_rows`
  - `read_calibrated_fractions`
  - `parse_optional_fractions`

- [ ] `src/explainai_thesis/cxr/methods.py`
  - Replace the 16-entry `methods` dict in `main()` with a `MethodSpec(name, fn, kwargs, overlay_color, polarity)` registry.
  - Single dispatch loop computes all methods; eliminates the nested-ternary overlay-parameter blocks at lines 903-913 and 915-933.
  - Preserve method names exactly as listed in `AGENT.md` line 66.

- [ ] `src/explainai_thesis/visualization_cxr.py`
  - `save_selected_threshold_image`
  - `overlay_color_for_method`
  - `NEUTRAL_IMPACT_COLOR` (delete the duplicate in `scripts/run_cxr_torchxray_smoke.py:39`)
  - New: `signed_diverging_overlay` for orange/teal rendering.

- [ ] `src/explainai_thesis/io.py`
  - Single source of truth for output CSV column lists and field orders.

Refactor of `scripts/visualize_cxr_classifier_outcome_thresholds.py` is **gated** by:
- [ ] Write `tests/test_classifier_outcome_resume.py`: run 3 cases, kill, resume, verify no duplicate rows and final counts match a non-interrupted baseline.
- [ ] Verify `cases.csv`, `threshold_metrics.csv`, `progress.json` schemas are byte-identical before and after refactor.

Acceptance per script:
- `wsl.exe python3 -m py_compile <script>` clean.
- Golden-output snapshot from Phase 0 still passes.
- Manual smoke matches a pre-refactor reference run byte-for-byte on the metrics CSV.

---

## Phase 3 — Performance

### 3.1 Batch Integrated Gradients

File: `src/explainai_thesis/xai.py:104-115`

Current: Python loop over `steps`, one forward+backward per step.

New: single batched forward+backward over `[steps, C, H, W]`.

```python
alphas = torch.linspace(1.0 / steps, 1.0, steps, device=image.device).view(steps, 1, 1, 1)
scaled = baseline + alphas * (image - baseline)
scaled.requires_grad_(True)
score = model(scaled)[:, class_idx].sum()
grads = torch.autograd.grad(score, scaled)[0].mean(0, keepdim=True)
attribution = (image - baseline) * grads
```

Expected speedup: 8–15×. VRAM check: `steps=16` × 224×224 × DenseNet-121 ≈ comfortably under 4 GB.

Add `tests/test_ig_batched_matches_loop.py` asserting old-vs-new produce numerically equivalent attributions within tolerance.

### 3.2 Vectorize Occlusion Sensitivity

File: `src/explainai_thesis/xai.py:181-204`

Pre-build occlusion masks `[n_windows, 1, H, W]` once, broadcast-multiply with image, keep `batch_size` chunking for VRAM.

Expected speedup: 3–5×.

Equivalence test with the existing loop implementation.

### 3.3 Strip wasted compute in metrics

File: `src/explainai_thesis/metrics.py`

- `pointing_game_hit`: drop the `normalize_map` call (argmax invariant under monotonic normalization).
- `localization_metrics`: compute `pred_mask` once, reuse for `iou_score`, `dice_score`, and `precision_at_fraction` (currently re-thresholds).

### 3.4 Vectorize `_mask_contour`

File: `src/explainai_thesis/visualization.py:19-26`

Replace nested 3×3 Python loop with `scipy.ndimage.binary_erosion(mask, iterations=1)`. ~50× faster; net runtime impact small but improves readability.

### 3.5 Cache consensus heatmaps

File: `scripts/run_cxr_torchxray_smoke.py` main loop

The neutral-overlay consensus (`consensus_heatmap([ig_map, gradient_shap_map, occlusion_map])`) is recomputed per method iteration. Compute once before the loop.

### 3.6 `torch.inference_mode()` audit

Wrap any forward-only code path that currently has implicit grad tracking (parts of `gradient_shap` post-hoc, occlusion already uses `torch.no_grad`).

---

## Phase 4 — Polish

- [ ] Replace `print()` status lines with `logging` (INFO default; `--quiet` and `--verbose` flags).
- [ ] Add a thin classifier-loading seam: `src/explainai_thesis/cxr/models.py:load_classifier(name) -> (model, target_layer, class_idx, preprocess_fn)`. Today returns only the TorchXRayVision DenseNet. Future: CheXNet variant drops in without touching the XAI loop.
- [ ] Add `run_meta.json` writer: every script writes Python version, PyTorch version, torchxrayvision version, CUDA availability, git short hash, full CLI args, classifier threshold, faithfulness baseline to its output directory. Cheap, thesis-defensible.
- [ ] Add `ruff` + `mypy` configuration to `pyproject.toml`. Aim for `mypy --strict` clean on `src/`.
- [ ] Add `README.md` quickstart at repo root: install, run smoke, run CXR pipeline, run tests. Verify no existing `README.md` is overwritten before writing.

---

## Execution Order

| # | Step | Risk | Notes |
|---|---|---|---|
| 1 | Phase 0 | low | Foundation; gates everything |
| 2 | 1.3, 1.4, 1.5 (tests + manifest fix) | low | Non-controversial, builds the safety net |
| 3 | 1.1 (polarity fix) + regression run | medium | Audit-trail entry to `docs/progress.md` |
| 4 | 1.2 (signed maps) + orange/teal overlay | medium | New AGENT.md color convention applied |
| 5 | 1.6 (faithfulness default switch) | low | One-line default change + tests |
| 5b | Method-dispatch refactor in `run_cxr_torchxray_smoke.py` to consume `SignedAttribution` (Phase 2 prerequisite) | medium | Removes polarity-keyword call sites; preserves CSV/PNG output schema |
| 6 | 3.1 (batch IG) | medium | Biggest performance win; plugs into `SignedAttribution.raw` cleanly |
| 7 | 3.3, 3.4 (dead compute) | low | Quick wins |
| 8 | Phase 2 refactor (excluding classifier-outcome script) | medium | Frozen output contract |
| 9 | classifier-outcome resume regression test + refactor | high | Sacred file; test-first |
| 10 | 3.2 (occlusion vectorization) | medium | |
| 11 | 3.5, 3.6 (caching, inference_mode) | low | |
| 12 | Phase 4 polish | low | Logging, second-model seam, run_meta.json |

Estimated total: ~4-5 working days (extra half-day for the broader `SignedAttribution` refactor, offset by 2-3× speedup on the polarity-supporting methods).

### Performance ledger (expected, before vs after)

Per CXR case, assuming `ig-steps=16`, `gradshap-samples=8`, `occlusion-patch-size=32`, `occlusion-stride=16`:

| Method | Calls today | Calls after 1.2 | Calls after 3.1/3.2 | Net speedup |
|---|---|---|---|---|
| Grad-CAM | 2 | 1 | 1 | ~2× |
| Grad-CAM++ | 2 | 1 | 1 | ~2× |
| Integrated Gradients | 3 | 1 | 1 (batched, 8-15× internal) | ~24-45× |
| GradientSHAP | 3 | 1 | 1 | ~3× |
| Occlusion | 3 | 1 | 1 (vectorized, 3-5× internal) | ~9-15× |
| Consensus | derived | derived | derived | free |

Occlusion alone is the biggest absolute saving; IG is the biggest relative.

---

## Out of Scope (Explicitly Deferred)

- Touching `reports/weekly/week_1_report*.md`.
- Renaming any existing `outputs/iter_XX_*` folder.
- Renaming any CLI flag in the documented `iter_27` long-run command pattern.
- Touching the Week 1 frozen reports.
- LIME (protocol "optional, only if time").

---

## Scope Realignment vs. `docs/experiment_protocol.md` and `docs/supervisor_one_pager.md` (added 2026-05-18)

The protocol promises items the current code does not yet cover. The thesis-draft cutoff is `2026-06-04` (17 days from today), so the refactor scope compresses to make room for protocol completion.

### Refactor scope compression

- **Phase 2**: cut from full architectural decomposition to **only** the `MethodSpec` registry plus `src/explainai_thesis/cxr/io.py`. Plotting/faithfulness helpers stay in `scripts/run_cxr_torchxray_smoke.py` until after the defense. Rationale: the registry unblocks Eigen-CAM/Score-CAM and the CT pilot; everything else is tidiness.
- **Phase 4**: defer `mypy --strict`, the full `logging` migration, and the `README.md` quickstart to post-defense. Keep `run_meta.json` stamping (thesis-required for tool/version disclosure per `AGENT.md` line 34) and the `load_classifier(name)` seam (CT-pilot-required).

### Phase 5 — Protocol Completion

Phase 5 is the new work to close protocol gaps before the draft cutoff. Order matches the revised execution table below.

#### 5.1 Eigen-CAM and Score-CAM

- Decision (2026-05-18): add both.
- Eigen-CAM: PCA of the target-layer activations, project onto the top component. ~30 lines. Fits the `SignedAttribution` contract trivially (signed = principal-component projection; magnitude = absolute value).
- Score-CAM: mask each activation channel into the input, re-score the model, weight activations by score change. Expensive (similar to occlusion). Add a `--score-cam-channels-cap` argument so broad screening runs can subsample channels for speed; thesis-quality reruns use the full set.
- Both register as new entries in the `MethodSpec` table from compressed Phase 2; no script-side glue needed beyond the registry.
- New `AGENT.md` "XAI Method Set" entries: `eigen_cam`, `score_cam` (each with the four-view positive/negative/magnitude/signed family).
- Tests: include both in the metric-sanity and faithfulness-sanity tests.

#### 5.2 Improvement experiment script

- Decision (implicit from protocol "Improvement Experiment" section): formalize the consensus-vs-individual head-to-head as a dedicated script `scripts/run_improvement_experiment.py`.
- Pipeline:
  1. Calibrate per-method top-fraction thresholds on the **validation split** using `scripts/calibrate_cxr_xai_thresholds.py` (already exists).
  2. Freeze thresholds. Run all individual methods + consensus + `consensus_signed` on the **held-out test split**.
  3. Emit `improvement_experiment.csv` with one row per (method, case) for IoU, Dice, pointing_hit, precision_at_fraction, and a paired-test summary CSV `improvement_experiment_paired.csv` (Wilcoxon signed-rank per metric, consensus vs each individual).
  4. Produce Chapter 4 figures: side-by-side box plots per metric.
- Output folder: `outputs/iter_XX_improvement_experiment_<timestamp>/`.
- Discipline: thresholds must be frozen before held-out evaluation. The script refuses to run if any calibration artifact is older than the test smoke it would compare against.

#### 5.3 Radiologist review tooling

- Decision (2026-05-18): hybrid CSV + static HTML index, not a full interactive app.
- New script `scripts/build_review_workbook.py`:
  - Inputs: a smoke-run output directory (e.g., `outputs/iter_27_*`).
  - Outputs to `<run>/review/`:
    - `index.html`: one card per case, embedded thumbnail grid of the four-view overlays per method, the 4-rubric scoring guide, the 7-category failure taxonomy, and the score columns to fill.
    - `scores_template.csv`: prefilled `case_id`, `filename` columns; empty score columns ready for the rater.
    - `INSTRUCTIONS.md`: opening sequence (open `index.html` in browser; review each card; fill `scores.csv` in editor; save as `scores.csv` next to the template; do not edit the template itself).
  - Static-only. No server. Browser opens local PNGs via relative paths.
- Scoring schema (matches `docs/experiment_protocol.md` Radiologist Review section):
  - `localization_score` ∈ {correct, partial, incorrect, none}
  - `usefulness_score` ∈ {useful, potentially_useful, misleading, not_useful}
  - `failure_category` ∈ {correct, partial, anatomically_related, devices_text_artifacts, non_pathological_high_contrast, diffuse_non_specific, clinically_misleading}
  - `artifact_note` (free text)
  - `comment` (free text)
- Tests: schema test verifies `scores.csv` round-trips; smoke test runs the workbook builder on a 2-case synthetic output and asserts the `index.html` references both cases.

#### 5.4 Head CT pilot

- Decision (2026-05-18): **off-the-shelf pretrained CT model + small annotated subset**, not fine-tuning.
- New package layout: `src/explainai_thesis/ct/` mirrors `src/explainai_thesis/cxr/`.
  - `ct/io.py`: HU windowing for DICOM input, slice-level preprocessing, resize to the CT model's expected input size.
  - `ct/models.py`: thin wrapper conforming to the `load_classifier(name)` seam from Phase 4. Initial candidate: a public RSNA-IHD pretrained classifier; alternatives shortlisted below.
- New script `scripts/run_ct_smoke.py`: CT analogue of `run_cxr_torchxray_smoke.py`, sharing the `MethodSpec` registry, XAI methods, faithfulness, calibration, and metrics. The CT-specific code path is only IO and model loading.
- Manifest: `data/ct_hemorrhage_manifest.csv` built by `scripts/build_manifest.py` extended to recognize a `--modality ct` mode (or a separate `build_ct_manifest.py`, decided when implementing).
- Annotation: 20-30 positive cases manually masked by the student-as-radiologist. Masks stored locally (per `data/README.md` rule), referenced from the manifest.
- Output folders: `outputs/iter_XX_ct_<short>/` follow the same ordinal naming.
- Faithfulness baseline: must be re-chosen for CT. `--faithfulness-baseline black` may not mean the same thing in HU space. Add `--faithfulness-baseline soft_tissue_window_zero` as a CT-appropriate option.
- Open subdecisions deferred to implementation time: exact CT model identifier, DICOM-source-of-truth vs PNG export, slice selection rule for IHD (single representative slice vs three-slice stack).
- Tests: CT-specific HU windowing round-trip test; a synthetic CT-like dataset (HU-scaled `SyntheticLesionDataset`) for the smoke pipeline.

#### 5.5 Stronger second pneumothorax model (time-permitting)

- Decision (deferred): only if Phases 0-3 and 5.1-5.4 finish ahead of 2026-05-31. Otherwise document as future work and rely on the TorchXRayVision baseline as the documented weak external baseline (`AGENT.md` "Stronger Second Model Needed" section).
- If pursued: integrate one candidate (CheXNet-style DenseNet or a pneumothorax-specific Kaggle-fine-tuned model) via the `load_classifier(name)` seam. Run the full protocol on it. Add to the comparison table.

#### 5.6 Captum infidelity and sensitivity (optional)

- Protocol-marked optional. Pull in only if Phase 5.5 is skipped and there is time left. Implementation is small (`captum.metrics.infidelity` and `captum.metrics.sensitivity_max`); fits naturally next to the existing deletion/insertion faithfulness writer.

### Revised execution order with deadline anchors

| # | Step | Days | Risk | Cumulative |
|---|---|---|---|---|
| 1 | Phase 0 foundation + golden-output snapshot | 0.5 | low | 2026-05-19 |
| 2 | Phase 1 correctness: tests, polarity fix, signed maps, manifest fix, faithfulness default | 2 | medium | 2026-05-21 |
| 3 | Compressed Phase 2: `MethodSpec` registry + `cxr/io.py` | 1 | medium | 2026-05-22 |
| 4 | Phase 3.1 + 3.2 + 3.3 + 3.4 (batched IG, vectorized occlusion, dead compute, mask contour) | 1 | medium | 2026-05-23 |
| 5 | Phase 5.1 Eigen-CAM + Score-CAM | 0.5 | low | 2026-05-24 (morning) |
| 6 | Phase 5.2 improvement experiment script | 0.5 | low | 2026-05-24 (eve) |
| 7 | Phase 5.4 CT pilot scaffold + first CT smoke | 2-3 | high | 2026-05-27 |
| 8 | Phase 5.3 radiologist review workbook + first scoring pass on CXR | 1.5 | low | 2026-05-28 |
| 9 | Phase 4 minimum: `run_meta.json` stamping + `load_classifier` seam audit | 0.5 | low | 2026-05-29 |
| 10 | Phase 5.5 stronger second CXR model (if time) | 1-2 | medium | 2026-05-31 |
| 11 | Buffer for figures, thesis tables, draft writing | 4-5 | low | 2026-06-04 |

Hard deadline: full thesis draft `2026-06-04`. Final corrections/formatting/defense window `2026-06-05` to `2026-06-21`.

### Items explicitly downgraded

- LIME (protocol optional): drop. Justify as scope adjustment in thesis.
- Captum infidelity/sensitivity (protocol optional): pull in only if Phase 5.5 is dropped.
- Full `logging` migration, `mypy --strict`, full file-split of `run_cxr_torchxray_smoke.py`: defer to post-defense.
