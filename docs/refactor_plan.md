# Refactor & Optimization Plan (Temporary Working Doc)

Created: 2026-05-18
Owner: Dmytro Valantsevych
Reviewer model: Claude Opus 4.7 (1M context)
Status: Phase 0 completed 2026-05-18 (commit `99275dd`). Phases 1+ pending.

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

- [x] Add `pyproject.toml` declaring the `explainai_thesis` package; install with `pip install -e .` so `sys.path.insert` is no longer needed. *(Done 2026-05-18.)*
- [x] Remove `sys.path.insert(0, str(ROOT / "src"))` blocks from all scripts. *(Done 2026-05-18; 8 scripts cleaned.)*
- [x] Add `requirements-dev.txt` with `pytest`, `pytest-cov`, `scipy`. *(Done 2026-05-18. `ruff` and `mypy` intentionally deferred — lint/type tooling not adopted in this thesis pass.)*
- [x] Create `tests/conftest.py` with reusable fixtures. *(Done 2026-05-18: `repo_root` fixture + autouse `torch.manual_seed(0)`. Synthetic-case and known-mask/heatmap fixtures deferred to Phase 1, where the metric/faithfulness tests that need them are written.)*
- [x] Add `tests/test_golden_outputs.py`: snapshot smoke output schema. *(Done 2026-05-18: 3 structural snapshot tests over `run_smoke_test.py` — column contract, value ranges in `[0, 1]`, overlay PNG layout. Bit-equal numerical comparison deliberately avoided — CPU floating-point not portable across BLAS/torch versions.)*
- [ ] ~~Add `.github/workflows/ci.yml`~~ **Deferred to post-Phase-5** (per user decision 2026-05-18: CI workflow is optional for a thesis repo on the 2026-06-04 deadline; local `wsl.exe python3 -m pytest` is the canonical check until then).
- [ ] ~~Add a `Makefile` (or `tasks.json`)~~ **Deferred to post-Phase-5** (per user decision 2026-05-18, same rationale as CI).

Acceptance: `py_compile` clean across all 8 modified scripts ✅, `wsl.exe python3 -m pytest tests/ -v` → 3 passed in 10.42s ✅, editable install resolves `import explainai_thesis` from `src/` ✅. Phase 0 committed as `99275dd "Phase 0 refactor"` (note: `tests/` directory itself was still untracked at commit time; fold into a Phase 0 follow-up commit or roll into Phase 1).

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

### 1.7 Diagnostic second-classifier A/B (promoted ahead of Phase 5 on 2026-05-18)

Scientific motivation: current TorchXRayVision `densenet121-res224-all` localization on SIIM pneumothorax looks clinically weak. Before investing in protocol-completion scope (`Phase 5`), confirm whether the weakness is XAI-side (method/threshold/interpretation) or model-side (training distribution mismatch). If it is model-side, the conclusions of any further methodological work risk being attached to a broken foundation.

#### Stage A — In-family weights plus one out-of-family external model (≈1 day, revised 2026-05-18 after pre-mortem)

Pre-mortem found that the original Stage A (only `torchxrayvision` weights) was structurally blind: the 5 candidate weights all share architecture, input size, normalization, and substantially overlapping training data (CheX, MIMIC, NIH). If all five looked equally weak, the diagnostic could only rule out within-family variance, not training-distribution mismatch. Stage A now includes one out-of-family external model from the start, so the answer is interpretable either way.

- Reuse all current XAI methods, calibration cases, and metrics.
- In-family candidates from `torchxrayvision` (changes only `--weights`):
  - `densenet121-res224-chex`
  - `densenet121-res224-mimic_ch`
  - `densenet121-res224-mimic_nb`
  - `densenet121-res224-rsna`
  - (control) `densenet121-res224-all` — current baseline
- Out-of-family external model (goes through `load_classifier(name)` seam from compressed Phase 2): one candidate from HuggingFace or a public Kaggle pneumothorax-fine-tuned model. Exact identifier decided at implementation time; selection criterion is "different training distribution and architecture from TorchXRayVision DenseNet-121, plus a usable inference API."
- Pipeline: run `scripts/run_cxr_torchxray_smoke.py` (or `scripts/run_cxr_smoke.py` after the seam refactor) with each model on the same 20-30 calibration positive cases at the calibrated top-fractions for each method. Output: `outputs/iter_XX_diagnostic_weights_ab/<model_name>/`.
- Summary table: per-model mean IoU, Dice, pointing_hit, precision_at_fraction across methods. Plus a per-case-per-method agreement view to surface whether one model changes the *spatial* attribution distinctly.

#### Stage B — Outcome decision (≈0 days, decision-only)

Three outcomes:
1. **All weights look similar (poor localization)**: model-architecture or training-distribution is unlikely to be the lever. Issue is methodological / clinical interpretation of saliency. `Phase 5.5` does **not** run. Thesis can confidently frame results as cross-weight-stable XAI behavior. Proceed to Phase 5.
2. **One weight is materially better at localization**: that weight becomes a co-primary baseline. Run Stage C with the better weight as the new candidate. Frame as "training-distribution mismatch was a meaningful factor."
3. **Inconclusive / mixed**: proceed to optional external-model stage.

#### Stage C (optional, only if Stage B is inconclusive or favorable) — External model (≈1-2 days)

- Candidate types (decision deferred to implementation): CheXNet variant from outside `torchxrayvision`, pneumothorax-specific Kaggle-fine-tuned model, or a self-fine-tuned DenseNet on SIIM-train with a published recipe.
- Integration goes through the `load_classifier(name)` seam from compressed Phase 2 so existing scripts work unchanged.
- Run the full diagnostic suite (calibration cases) and the held-out classifier-outcome run from the documented `iter_27` command pattern.

#### Stage B outcome decision rule (refined after pre-mortem)

The three outcomes are now interpretable because Stage A spans both within-family and out-of-family.

1. **All models look similarly poor** (in-family AND external): cross-distribution-stable XAI behavior; thesis frames results as methodological. `Phase 5.5` does not run.
2. **In-family models look similar but the external model is materially better**: training-distribution mismatch is the dominant factor. The external model becomes the co-primary baseline; run the full protocol on it.
3. **One in-family weight is materially better than the others**: within-family variance is meaningful; promote that weight to co-primary baseline.
4. **Inconclusive or mixed**: Stage C is a deeper external-model exploration with 1-2 additional candidates.

#### Tests

- `tests/test_load_classifier_a_b.py`: assert `load_classifier("densenet121-res224-chex")` returns a `(model, target_layer, class_idx, preprocess_fn)` tuple with `class_idx` pointing to a valid pathology head.
- Smoke verification: identical-input parity test — same image into two models, assert outputs are *different* (sanity-check that we are actually loading different weights, not silently caching the same checkpoint).
- Out-of-family loader test: `load_classifier("<external_model_name>")` returns a usable tuple and runs forward without architecture errors.

### 1.2.5 Versioned calibration regeneration (new, added after pre-mortem)

Reason: the `SignedAttribution` refactor in `Phase 1.2` changes how Grad-CAM, IG, GradientSHAP, and Occlusion compute their underlying signed map (no more post-hoc `F.relu(-cam)` polarity dance; IG/SHAP/Occlusion return raw signed attribution; Grad-CAM no longer ReLUs at the end). Existing calibrated top-fractions in earlier `outputs/iter_2*/selected_thresholds.csv` were tuned against the **v1** code and become statistically stale against the **v2** signed-attribution code.

- Immediately after `Phase 1.2` lands and before any `Phase 1.7` / `Phase 5` work, rerun `scripts/calibrate_cxr_xai_thresholds.py` to produce a **versioned** calibration file at `outputs/iter_XX_calibration_v2/calibrated_thresholds_v2.csv`.
- Never overwrite or rename the v1 calibration files. Both v1 and v2 must coexist; downstream scripts pick the correct version explicitly via `--calibrated-fractions`.
- Document both v1 and v2 in `docs/progress.md` so the thesis methodology section can cite the transition cleanly: "we recalibrated XAI top-fractions after the signed-attribution refactor on 2026-05-XX; v1 results were not used in held-out evaluation."
- Budget: 0.5 day, slotted between `Phase 1.2` and `Phase 1.6`.

#### AGENT.md updates

`Phase 1.7` outcome should be recorded in `AGENT.md` under a new "Diagnostic A/B Results" subsection, including: weights tried, calibration set used, per-weight summary metrics, and the Stage B outcome classification. This becomes thesis-defensible evidence for or against the "stronger second model needed" claim already in `AGENT.md`.

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

#### 5.2 Improvement experiment script (pre-draft both narratives before running)

Added after pre-mortem 2026-05-18: the iter_27 evidence already hints at substantial method disagreement. Consensus may or may not outperform the best individual method on held-out IoU/Dice. The thesis explicitly names "low-risk improvement via consensus" as a contribution, so an unfavorable outcome needs a prepared narrative.

- Before running the held-out evaluation, draft **both** Discussion sub-sections in `thesis/`:
  - Narrative A: "Consensus improves localization on held-out cases — implications for clinical XAI deployment."
  - Narrative B: "Consensus does not improve over the best individual method — what method disagreement reveals about model and saliency behavior."
- ~0.5 day of writing. Eliminates panic if results don't favor the preferred story and turns either outcome into a thesis-defensible finding.
- After running the experiment, the actual results select which narrative becomes the final Discussion text; the other becomes a "we considered" footnote.



- Decision (implicit from protocol "Improvement Experiment" section): formalize the consensus-vs-individual head-to-head as a dedicated script `scripts/run_improvement_experiment.py`.
- Pipeline:
  1. Calibrate per-method top-fraction thresholds on the **validation split** using `scripts/calibrate_cxr_xai_thresholds.py` (already exists).
  2. Freeze thresholds. Run all individual methods + consensus + `consensus_signed` on the **held-out test split**.
  3. Emit `improvement_experiment.csv` with one row per (method, case) for IoU, Dice, pointing_hit, precision_at_fraction, and a paired-test summary CSV `improvement_experiment_paired.csv` (Wilcoxon signed-rank per metric, consensus vs each individual).
  4. Produce Chapter 4 figures: side-by-side box plots per metric.
- Output folder: `outputs/iter_XX_improvement_experiment_<timestamp>/`.
- Discipline: thresholds must be frozen before held-out evaluation. The script refuses to run if any calibration artifact is older than the test smoke it would compare against.

#### 5.3 Radiologist review tooling

**Pre-mortem adjustment (2026-05-18)**: the rater (student-as-radiologist) reports a typical CXR reading time of 1-3 minutes per case. With a well-designed workbook that makes the rubric instant to apply without thinking, 100 cases at ~2 minutes average ≈ 3-4 hours of actual scoring. The 1.5-day Phase 5.3 budget covers tooling build (≈1 day) + scoring pass (≈0.5 day). The 100-case target is kept. The binding constraint is **rubric clarity**, not scoring time, so the workbook design rules below are tightened accordingly.


- Decision (2026-05-18): hybrid CSV + static HTML index, not a full interactive app.
- New script `scripts/build_review_workbook.py`:
  - Inputs: a smoke-run output directory (e.g., `outputs/iter_27_*`).
  - Outputs to `<run>/review/`:
    - `index.html`: one card per case, embedded thumbnail grid of the four-view overlays per method, the 4-rubric scoring guide, the 7-category failure taxonomy, and the score columns to fill.
    - `scores_template.csv`: prefilled `case_id`, `filename` columns; empty score columns ready for the rater.
    - `INSTRUCTIONS.md`: opening sequence (open `index.html` in browser; review each card; fill `scores.csv` in editor; save as `scores.csv` next to the template; do not edit the template itself).
  - Static-only. No server. Browser opens local PNGs via relative paths.
- **Rubric clarity rules (binding constraint for keeping scoring at ~2 min/case)**:
  - Each card on `index.html` must inline the **full** 4-rubric definitions plus the 7-category failure taxonomy. The rater must never need to scroll, switch tabs, or re-read `INSTRUCTIONS.md` mid-pass.
  - Each rubric option must show 1-2 example sentences ("`correct`: heatmap peak inside lesion AND overlap region clearly anatomically aligned").
  - Each card must show: the ground-truth mask, the overlay grid (one row per method, four views per row in red/blue/violet/orange-teal), the classifier outcome label (`tp`/`fp`/`tn`/`fn`), and the classifier probability.
  - Failure-category dropdown options must be in a fixed, memorable order matching the protocol; the same order in `scores_template.csv` so the rater's eye-to-keyboard pattern is identical across cases.
  - Cards must be navigable by keyboard (next/prev) so the rater never leaves the keyboard during a pass.
  - `INSTRUCTIONS.md` must include a 3-case warmup section ("score these 3 first to anchor the rubric") so consistency drift is bounded.
- Scoring schema (matches `docs/experiment_protocol.md` Radiologist Review section):
  - `localization_score` ∈ {correct, partial, incorrect, none}
  - `usefulness_score` ∈ {useful, potentially_useful, misleading, not_useful}
  - `failure_category` ∈ {correct, partial, anatomically_related, devices_text_artifacts, non_pathological_high_contrast, diffuse_non_specific, clinically_misleading}
  - `artifact_note` (free text)
  - `comment` (free text)
- Tests: schema test verifies `scores.csv` round-trips; smoke test runs the workbook builder on a 2-case synthetic output and asserts the `index.html` references both cases.

#### 5.4 Head CT pilot

- Decision (2026-05-18): **off-the-shelf pretrained CT model + small annotated subset**, not fine-tuning.
- **Hour-1 model-availability check (added after pre-mortem)**: the first hour of `Phase 5.4` is reserved for validating that a usable public CT hemorrhage classifier actually exists and runs end-to-end on one example slice. If no off-the-shelf model is found, immediately fall back to qualitative external validation (per `docs/experiment_protocol.md` Week-3 fallback rule) instead of sliding into a 4-5 day fine-tuning detour. Decision is binary and made within hour 1; do not let the search-for-a-model phase consume more than half a day.
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

#### 5.5 Stronger second pneumothorax model (time-permitting full protocol run)

- Decision (deferred): only if the diagnostic A/B in `Phase 1.7` identifies an alternate weight or external model materially better at localization than the documented TorchXRayVision baseline. Otherwise the diagnostic-only sweep from `Phase 1.7` is the documented secondary evidence and `Phase 5.5` does not run.
- If pursued: integrate one candidate (CheXNet-style DenseNet or a pneumothorax-specific Kaggle-fine-tuned model) via the `load_classifier(name)` seam. Run the full protocol on it. Add to the comparison table as a co-primary baseline.

#### 5.6 Captum infidelity and sensitivity (conditional)

- Protocol-marked optional. Pull in if Phase 5.5 is skipped and there is time left, or as a parallel low-cost add-on if the rest of Phase 5 lands ahead of schedule. Implementation is small (`captum.metrics.infidelity` and `captum.metrics.sensitivity_max`); fits naturally next to the existing deletion/insertion faithfulness writer.
- Gives the thesis a second faithfulness probe alongside deletion/insertion, which strengthens the H8 "faithfulness vs localization" test by triangulating across two faithfulness families.

#### 5.7 LIME (conditional, low priority)

- Protocol-marked optional ("only if implementation time is low"). Kept on the menu rather than dropped: if the rest of Phase 5 lands by 2026-06-01 and the buffer holds, a small LIME pass on a sub-sampled positive set provides a third explanation family (region-level surrogate) for cross-method comparison.
- Implementation via `lime.lime_image.LimeImageExplainer`; expensive per case because LIME generates many perturbed forward passes. Time-budget controls: cap to ~10-20 representative thesis-quality cases, not the full balanced run. No registry integration needed beyond a thin wrapper that returns a `SignedAttribution`-shaped map.
- If skipped, justify in the thesis methodology as a scope adjustment per the protocol's explicit "only if implementation time is low" clause. The cross-method comparison still holds across Grad-CAM, Grad-CAM++, IG, GradientSHAP, Occlusion, Eigen-CAM, and Score-CAM, which is a stronger method panel than the protocol's required minimum.

### Revised execution order with deadline anchors

| # | Step | Days | Risk | Cumulative |
|---|---|---|---|---|
| 1 | Phase 0 foundation + golden-output snapshot | 0.5 | low | 2026-05-19 | ✅ **Done 2026-05-18** (`99275dd`); CI + Makefile deferred to post-Phase-5.
| 2 | Phase 1 correctness: tests, polarity fix, signed maps, manifest fix, faithfulness default | 2 | medium | 2026-05-21 |
| 2b | Phase 1.2.5 versioned calibration regeneration (v2) | 0.5 | low | 2026-05-22 (AM) |
| 3 | Compressed Phase 2: `MethodSpec` registry + `cxr/io.py` + `load_classifier(name)` seam | 1 | medium | 2026-05-22 (PM) to 2026-05-23 |
| 4 | Phase 3.1 + 3.2 + 3.3 + 3.4 (batched IG, vectorized occlusion, dead compute, mask contour) | 1 | medium | 2026-05-24 |
| 5 | **Phase 1.7 Stage A: in-family weights + 1 out-of-family external model on shared calibration cases** | 1 | medium | 2026-05-25 |
| 6 | Stage B outcome decision; optional Stage C deeper external-model exploration | 0-2 | medium | 2026-05-25 to 2026-05-27 |
| 7 | Phase 5.1 Eigen-CAM + Score-CAM | 0.5 | low | 2026-05-27 |
| 8 | Phase 5.2 improvement experiment script + pre-draft both Discussion narratives | 1 | low | 2026-05-28 |
| 9 | Phase 5.4 CT pilot: **hour-1 model-availability check** then scaffold + first CT smoke (or fallback) | 2-3 | high | 2026-05-31 |
| 10 | Phase 5.3 radiologist review workbook + 100-case scoring pass on CXR | 1.5 | low | 2026-06-02 |
| 11 | Phase 4 minimum: `run_meta.json` stamping + `load_classifier` seam audit | 0.5 | low | 2026-06-02 |
| 12 | Phase 5.5 stronger second CXR model full protocol (conditional on Phase 1.7 outcome) | 1-2 | medium | 2026-06-03 |
| 13 | Final figures, results tables, finalize thesis writing | 1-2 | low | 2026-06-04 |

**Parallel track: thesis writing starts on 2026-05-19**, not in the final buffer. The methodology chapter can be written as soon as `Phase 1.2` lands (signed-attribution semantics decided); the results chapter populates as each phase's CSVs land. Only the Discussion and Conclusions chapters wait for the improvement-experiment results. Net effect: by 2026-06-04, the draft is ready not because there is a writing-only buffer, but because writing has happened concurrently.

Hard deadline: full thesis draft `2026-06-04`. Final corrections/formatting/defense window `2026-06-05` to `2026-06-21`.

### External coordination tasks (added after pre-mortem)

- **AI-tooling disclosure policy**: confirm with the supervisor this week which institutional policy governs disclosure of `GPT-5.5`, `Codex`, `Claude Sonnet 4.6`, `Claude Opus 4.7`, `Junie`, etc. in the thesis (per `AGENT.md` line 34). Result: a methods-section paragraph naming the tools and roles. Done before 2026-05-22 so it does not block draft writing.
- **Polarity-fix supervisor communication**: if the supervisor has seen earlier figures with the buggy `grad_cam_plus_plus_negative` overlays, email proactively with the corrected example after `Phase 1.1` lands, framing it as instrument calibration with before/after.

### Items explicitly downgraded

- Full `logging` migration, `mypy --strict`, full file-split of `run_cxr_torchxray_smoke.py`: defer to post-defense.

### Items kept on the menu but conditional (added 2026-05-18)

- LIME (`Phase 5.7`): kept as a low-priority conditional add-on. Activated only if the rest of Phase 5 lands by 2026-06-01 and the writing buffer holds. If skipped, justify in the thesis methodology under the protocol's "only if implementation time is low" clause.
- Captum infidelity / sensitivity (`Phase 5.6`): kept as a conditional pull-in either when `Phase 5.5` is skipped or as a parallel add-on if the rest of Phase 5 lands ahead of schedule. Strengthens the H8 "faithfulness vs localization" test by triangulating across two faithfulness families.
