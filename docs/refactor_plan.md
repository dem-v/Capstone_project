# Refactor & Optimization Plan (Temporary Working Doc)

Created: 2026-05-18
Owner: Dmytro Valantsevych
Reviewer model: Claude Opus 4.7 (1M context)
Status: Updated 2026-05-21. Phase 0 and Phase 1 are complete; Stage A model diagnostic is complete for TorchXRayVision candidates; Phase 2 structural extraction and Phase 3 performance work remain partially open.

This file is a temporary working artifact. It mirrors the plan accepted in conversation and is intended for task delegation. Once the work is completed and the resulting decisions are reflected in `docs/progress.md`, this file may be deleted.

---

## Current Status Snapshot — 2026-05-21

Source of truth for chronology remains `docs/progress.md`; this section prevents the original plan below from being misread as still fully pending.

### Done

- **Phase 0 foundation**: editable package install, script import cleanup, dev requirements, initial golden-output tests.
- **Phase 1 correctness**: Grad-CAM++ polarity fix, `SignedAttribution` four-view contract, v2 calibration support, `signed_prediction_alignment`, metric/faithfulness/manifest tests, and faithfulness default baseline switch to `black`.
- **Shared v2 XAI method-view cleanup**: `MethodView` / `iter_method_views(...)` in `src/explainai_thesis/xai.py`; active CXR smoke, calibration, single-image visualization, classifier-outcome visualization, review-candidate selection, and synthetic smoke paths now consume the shared v2 method-view contract.
- **Stage A diagnostic A/B for available TorchXRayVision candidates**: seven models evaluated under `outputs/iter_33_stage_a_diagnostic_ab/`; `weights_ab_summary.csv` generated.
- **Review workbook unblock**: missing false-positive diagnostic folders generated and `outputs/iter_28_review_workbook/review/` rebuilt for the current 10-case review set.
- **Metric correlation tooling**: all-model Stage A correlation analysis written under `outputs/iter_35_metric_correlations_iter33_stage_a_all_models/`.

### Current model-selection interpretation

- `resnet50-res512-all` is the strongest tested TorchXRayVision candidate by mean Dice / IoU in Stage A.
- Absolute localization remains weak, so this is a relative improvement only; do not describe it as clinically strong pneumothorax localization.
- The original `densenet121-res224-all` remains a documented weak external baseline.

### Blocked / deferred

- The planned MONAI out-of-family branch is blocked: the checked MONAI Model Zoo CXR bundle was a generative model, not a pneumothorax classifier. Do not add a MONAI loader until a concrete checkpoint with a verified `Pneumothorax` output, license, version, and preprocessing contract is identified.
- CI and task-runner files remain deferred until after Phase 5 per the original 2026-05-18 decision.

### Next recommended work

1. **Thesis-result path**: perform the 10-case radiologist-style scoring pass from `outputs/iter_28_review_workbook/review/`, then join `scores.csv` to v2 metrics for localization/usefulness correlation.
2. **Refactor path**: finish the low-risk Phase 2 extraction that does not change outputs, especially `src/explainai_thesis/faithfulness.py` and shared CLI helpers.
3. **Performance path**: implement and test batched Integrated Gradients and vectorized Occlusion only if runtime remains a blocker for the next thesis run.
4. **Model path**: use `resnet50-res512-all` for targeted qualitative follow-up, while continuing to frame all off-the-shelf localization results conservatively.

---

## Hard Constraints (from `AGENTS.md`)

- WSL Ubuntu is the canonical Python environment. All test commands use `wsl.exe python3`.
- Output folder layout is frozen: `outputs/iter_XX_<short>/`, root-level CSVs, per-case folders with source-X-ray-stem in every filename, and `tp/fp/tn/fn/` outcome subfolders.
- CLI flag names are frozen across all current scripts, including the long `iter_27` classifier-outcome command pattern.
- `cases.csv`, `threshold_metrics.csv`, `progress.json` checkpoint format in `scripts/visualize_cxr_classifier_outcome_thresholds.py` is frozen.
- `docs/progress.md` is append-only, chronological.
- `reports/weekly/week_1_report*.md` are frozen and must not be edited.
- Any changed Python file must pass `wsl.exe python3 -m py_compile <file>` before being declared done.
- Default tests must finish under 5 minutes on CPU. Slow tests use `@pytest.mark.slow`.
- Color semantics from `AGENTS.md` apply unchanged, plus the new `orange`/`teal` pair for signed scalar maps.

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

### 1.1 Fix `grad_cam_plus_plus` polarity double-flip ✅ Done 2026-05-18

File: `src/explainai_thesis/xai.py:61-81`

Bug: in the `grad_cam_plus_plus` branch, gradients are sign-flipped when `polarity == "negative"`, and then `cam = F.relu(-cam)` is applied below, which is effectively a second sign-flip.

Fix: pick one place to apply polarity. Recommend keeping the post-weight `cam = F.relu(-cam)` consistent with the standard `grad_cam` branch and removing the pre-weight gradient flip in `grad_cam_plus_plus`.

Tests:
- `test_gradcam_negative_differs_from_positive`: assert `gradcam(x, polarity="negative")` is not numerically equal to `gradcam(x, polarity="positive")` on a synthetic positive case.
- `test_gradcam_negative_does_not_peak_inside_lesion`: argmax of the negative map should not fall inside the lesion mask on average across the synthetic test split (probabilistic, allow > 0.5 of cases outside).

Regression: rerun synthetic smoke and one CXR positive case, attach the before/after overlays to `docs/progress.md` for thesis-defense audit trail.

**Implementation notes (2026-05-18):** Pre-weight gradient flip removed; polarity now applied exactly once via the post-weight `F.relu(±cam)` block. Tests live in `tests/test_gradcam_polarity.py` and use a briefly-trained tiny CNN (60 SGD steps) so class-1 gradients are principled. Full suite: 5 passed in 12.09s. CXR before/after overlay regression deferred to the supervisor-communication step (`Phase 1.1` pre-mortem item) — synthetic regression is fully covered by `tests/test_gradcam_polarity.py`.

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

#### AGENTS.md updates

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
- Out-of-family external model (goes through `load_classifier(name)` seam from compressed Phase 2): **MONAI Model Zoo CXR bundle (decided 2026-05-18)**. Concrete bundle id is picked at integration time via `python3 -m monai.bundle download <name>` + `configs/metadata.json` inspection to confirm a `Pneumothorax` classification label. Rationale: MONAI bundles ship pneumothorax-aware classifier heads out of the box (no SIIM-train linear probe needed; the "off-the-shelf baseline" rule in `AGENTS.md` is preserved), are MIT-licensed and versioned (thesis-citable by bundle id + version + commit hash), and use a different training mixture and preprocessing pipeline than TorchXRayVision. Honest caveat: several MONAI CXR bundles are DenseNet-121 or EfficientNet, so the architecture axis may be only partially covered; the training-distribution + preprocessing axes are the meaningful out-of-family contrast. Rejected Tier-1 alternative: Google CXR Foundation / ELIXR (would have needed a SIIM-train linear probe -> supervisor sign-off, deferred).
- Pipeline: run `scripts/run_cxr_torchxray_smoke.py` (or `scripts/run_cxr_smoke.py` after the seam refactor) with each model on the same 20-30 calibration positive cases at the calibrated top-fractions for each method. Output: `outputs/iter_XX_diagnostic_weights_ab/<model_name>/`.
- Summary table: per-model mean IoU, Dice, pointing_hit, precision_at_fraction across methods. Plus a per-case-per-method agreement view to surface whether one model changes the *spatial* attribution distinctly.

#### Stage A per-model pipeline (committed 2026-05-18)

Each model in the working set goes through three sub-steps, orchestrated by `scripts/run_stage_a_diagnostic_ab.ps1`:

1. **Classifier-threshold sweep** — `scripts/evaluate_cxr_torchxray_model.py --weights <name>` on the train split. Writes per-model best-F1 / Youden's-J / high-sensitivity operating points under `outputs/iter_33_stage_a_diagnostic_ab/<model>/threshold_sweep/`. The historical `0.62` is DenseNet-`all`-specific and must not be reused for the other weights.
2. **v2 XAI calibration** — `scripts/calibrate_cxr_xai_thresholds.py --weights <name>` on positive masked calibration cases. Writes `outputs/iter_33_stage_a_diagnostic_ab/<model>/calibration_v2/calibrated_thresholds_v2.csv`. v1 calibration files are statistically stale post-`SignedAttribution` refactor and stay frozen.
3. **Smoke + faithfulness** — `scripts/run_cxr_torchxray_smoke.py --weights <name> --calibrated-fractions <v2-csv> --classifier-threshold <best-F1-from-step-1>`. Writes `outputs/iter_33_stage_a_diagnostic_ab/<model>/smoke/`.

Working set: `densenet121-res224-{all, chex, mimic_ch, mimic_nb, nih, pc}` + `resnet50-res512-all` + (once integrated) one MONAI CXR bundle. `densenet121-res224-rsna` and `resnetae-101-elastic` are auto-skipped (no Pneumothorax head / no class head).

Aggregator emits `outputs/iter_33_stage_a_diagnostic_ab/weights_ab_summary.csv`: one row per model with mean IoU, Dice, pointing_hit, precision_at_fraction across methods (positive view, Dice-selected calibrated fraction). Failed models leave a `status=fail` row with NaN metrics rather than aborting the sweep.

Wall-time: ~12-16 h CUDA for the in-family sweep (per-model calibration ~60-90 min, per-model smoke ~30-45 min). Past the 30 min agent-tool budget; the orchestrator is shipped and the user runs it manually per AGENTS.md.

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

#### AGENTS.md updates

`Phase 1.7` outcome should be recorded in `AGENTS.md` under a new "Diagnostic A/B Results" subsection, including: weights tried, calibration set used, per-weight summary metrics, and the Stage B outcome classification. This becomes thesis-defensible evidence for or against the "stronger second model needed" claim already in `AGENTS.md`.

### 1.6 Faithfulness default baseline switch

File: `scripts/run_cxr_torchxray_smoke.py:117`

Change default `--faithfulness-baseline` from `zero_tensor` to `black`. Keep `zero_tensor` as a valid choice, annotated "historical / not recommended" in argparse help. Add an inline comment citing `AGENTS.md` rationale.

---

## Phase 2 — Structural Refactor (Under Frozen Output/CLI Contract)

Files to extract from `scripts/run_cxr_torchxray_smoke.py` (1037 lines → target ~150) into the `src/explainai_thesis/` package:

- [~] `src/explainai_thesis/faithfulness.py`
  - `faithfulness_baseline_tensor` extracted 2026-05-21.
  - `faithfulness_curve_rows` extracted 2026-05-21.
  - `curve_auc` extracted 2026-05-21.
  - `write_faithfulness_summary`
  - `plot_faithfulness_curves`
  - `plot_faithfulness_summary`
  - `write_faithfulness_plots`
  - `faithfulness_method_family`

- [~] `src/explainai_thesis/cli/common.py`
  - `resolve_device` extracted 2026-05-21 and reused by active CXR/smoke scripts.
  - Shared argparse parents for `--manifest`, `--split`, `--output-dir`, `--device`, `--seed`. CRITICAL: flag names unchanged, defaults unchanged.

- [~] `src/explainai_thesis/cxr/io.py`
  - `read_positive_masked_rows`, `load_xray_image`, `load_binary_mask`, `safe_case_name`, and `safe_source_stem` extracted 2026-05-21 from `scripts/run_cxr_torchxray_smoke.py` to reduce script-local CXR IO/case helpers without changing smoke outputs.
  - Pending shared extraction candidates remain where useful: generalized row readers for non-positive cases and any other active-script CXR IO helpers discovered during future size-reduction passes.
  - `read_calibrated_fractions`
  - `parse_optional_fractions`

- [ ] `src/explainai_thesis/cxr/methods.py`
  - Replace the 16-entry `methods` dict in `main()` with a `MethodSpec(name, fn, kwargs, overlay_color, polarity)` registry.
  - Single dispatch loop computes all methods; eliminates the nested-ternary overlay-parameter blocks at lines 903-913 and 915-933.
  - Preserve method names exactly as listed in `AGENTS.md` line 66.

- [~] `src/explainai_thesis/visualization.py`
  - `save_binary_selection` extracted 2026-05-21 from the two threshold visualizer scripts.
  - `overlay_color_for_method` extracted 2026-05-21.
  - `NEUTRAL_IMPACT_COLOR` centralized in the package for the extracted helpers; additional duplicate cleanup may still remain in older script paths.
  - `signed_diverging_overlay` for orange/teal rendering already lives in the package.
  - Pending: broader CXR-specific visualization extraction, including any remaining selected-threshold image helpers from `scripts/run_cxr_torchxray_smoke.py`.

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

### 3.1 Batch Integrated Gradients `[done 2026-05-21]`

File: `src/explainai_thesis/xai.py`

Replaced the Python loop over `steps` with a single batched forward/backward over `[steps, C, H, W]` in `integrated_gradients_signed(...)`. This preserves the previous Riemann-sum sampling points and returns the same normalized `SignedAttribution.raw` map while avoiding one model call per IG step.

Implemented core shape:

```python
alphas = torch.linspace(1.0 / steps, 1.0, steps, device=image.device).view(steps, 1, 1, 1)
scaled = baseline + alphas * (image - baseline)
scaled.requires_grad_(True)
score = model(scaled)[:, class_idx].sum()
grads = torch.autograd.grad(score, scaled)[0].mean(0, keepdim=True)
attribution = (image - baseline) * grads
```

Expected speedup: 8–15×. VRAM check: `steps=16` × 224×224 × DenseNet-121 ≈ comfortably under 4 GB.

Regression coverage lives in `tests/test_signed_attribution.py::test_integrated_gradients_batched_matches_loop_formula`, which recomputes the prior loop formula and asserts numerical equivalence within tolerance.

### 3.2 Vectorize Occlusion Sensitivity `[done 2026-05-21]`

File: `src/explainai_thesis/xai.py:181-204`

Pre-build occlusion masks `[n_windows, 1, H, W]` once, use mask-broadcasted occluded batches, and keep `batch_size` chunking for VRAM. Per-window image mutation and per-window attribution accumulation were replaced by tensor operations while preserving the same window grid, baseline fill value, per-pixel averaging, and signed normalization semantics.

Expected speedup: 3–5×.

Regression coverage lives in `tests/test_signed_attribution.py::test_occlusion_sensitivity_vectorized_matches_loop_formula`, which recomputes the previous per-window loop formula and asserts numerical equivalence within tolerance.

### 3.3 Strip wasted compute in metrics `[done 2026-05-21]`

File: `src/explainai_thesis/metrics.py`

- `pointing_game_hit` now uses the raw `argmax` without `normalize_map` because the maximum location is invariant under the prior min-max normalization.
- `localization_metrics` now computes `pred_mask` once and reuses it for `iou_score`, `dice_score`, and `precision_for_mask`, avoiding duplicate top-fraction thresholding.
- Existing `tests/test_metrics.py` coverage locks the flat-heatmap pointing-game behavior and verifies precision reuse from the precomputed top-fraction mask.

### 3.4 Vectorize `_mask_contour` `[done 2026-05-21]`

File: `src/explainai_thesis/visualization.py:19-26`

Replaced the nested 3×3 Python erosion loop with a NumPy `logical_and.reduce` over shifted padded views. This preserves the no-new-runtime-dependency constraint (`scipy` is dev-only in `pyproject.toml`) while removing the per-pixel loop and keeping output semantics unchanged.

### 3.5 Cache consensus heatmaps `[done 2026-05-21]`

File: `scripts/run_cxr_torchxray_smoke.py` main loop

The original Phase 3.5 note referred to a pre-v2 pattern where the neutral-overlay consensus (`consensus_heatmap([ig_map, gradient_shap_map, occlusion_map])`) could be recomputed during per-method iteration. The current post-Phase-1.2 smoke loop already computes one `consensus_attr = consensus_signed([...])` per case before `iter_method_views(...)`, then expands that cached signed attribution into all consensus views with the same shared dispatch path as the individual methods. Repository search confirms `scripts/run_cxr_torchxray_smoke.py` no longer calls `consensus_heatmap`.

No code change was needed for this item; it is reconciled as completed by the v2 `SignedAttribution` dispatch refactor.

### 3.6 `torch.inference_mode()` audit `[done 2026-05-21]`

Audited package-level forward-only paths and replaced the remaining `torch.no_grad()` contexts with `torch.inference_mode()` where no autograd metadata is needed:

- `src/explainai_thesis/faithfulness.py::model_probability(...)`.
- `src/explainai_thesis/xai.py::occlusion_sensitivity_signed(...)` original and occluded-score forward passes.

Gradient-requiring paths (`GradCAM.signed`, `integrated_gradients_signed`, and `gradient_shap_signed`) intentionally remain outside `inference_mode()`.

---

## Phase 4 — Polish

- [ ] Replace `print()` status lines with `logging` (INFO default; `--quiet` and `--verbose` flags). Deferred by the compressed Phase 4 scope below.
- [x] Add a thin classifier-loading seam: `src/explainai_thesis/cxr/classifier.py:load_classifier(name, device, pathology) -> ClassifierBundle`. Landed during Phase 1.7 and covered by `tests/test_load_classifier.py`; current supported TorchXRayVision branches include DenseNet-121, ResNet-50 classifier, and ResNetAE only when `pathology=None`.
- [~] Add `run_meta.json` writer: `src/explainai_thesis/run_metadata.py` landed 2026-05-21 and the three primary CXR output-producing scripts (`evaluate_cxr_torchxray_model.py`, `calibrate_cxr_xai_thresholds.py`, `run_cxr_torchxray_smoke.py`) now write Python version, PyTorch version, TorchXRayVision version, CUDA availability, git short hash, full CLI args, classifier threshold, faithfulness baseline where applicable, split, and weights. Pending only for secondary/diagnostic scripts if they become thesis-primary outputs.
- [ ] Add `ruff` + `mypy` configuration to `pyproject.toml`. Deferred by the compressed Phase 4 scope below.
- [ ] Add `README.md` quickstart at repo root: install, run smoke, run CXR pipeline, run tests. Deferred by the compressed Phase 4 scope below; `README.md` already exists and must not be overwritten casually.

---

## Execution Order

| # | Step | Risk | Notes |
|---|---|---|---|
| 1 | Phase 0 | low | Foundation; gates everything |
| 2 | 1.3, 1.4, 1.5 (tests + manifest fix) | low | Non-controversial, builds the safety net |
| 3 | 1.1 (polarity fix) + regression run | medium | Audit-trail entry to `docs/progress.md` |
| 4 | 1.2 (signed maps) + orange/teal overlay | medium | New AGENTS.md color convention applied |
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
- **Phase 4**: defer `mypy --strict`, the full `logging` migration, and the `README.md` quickstart to post-defense. Keep `run_meta.json` stamping (thesis-required for tool/version disclosure per `AGENTS.md` line 34) and the `load_classifier(name)` seam (CT-pilot-required).

### Phase 5 — Protocol Completion

Phase 5 is the new work to close protocol gaps before the draft cutoff. Order matches the revised execution table below.

#### 5.1 Eigen-CAM and Score-CAM

- Decision (2026-05-18): add both.
- Eigen-CAM: PCA of the target-layer activations, project onto the top component. ~30 lines. Fits the `SignedAttribution` contract trivially (signed = principal-component projection; magnitude = absolute value).
- Score-CAM: mask each activation channel into the input, re-score the model, weight activations by score change. Expensive (similar to occlusion). Add a `--score-cam-channels-cap` argument so broad screening runs can subsample channels for speed; thesis-quality reruns use the full set.
- Both register as new entries in the `MethodSpec` table from compressed Phase 2; no script-side glue needed beyond the registry.
- New `AGENTS.md` "XAI Method Set" entries: `eigen_cam`, `score_cam` (each with the four-view positive/negative/magnitude/signed family).
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

#### 5.5 Stronger second pneumothorax model (co-primary baseline)

- **Status: superseded by the Phase 5 Implementation Details section below.** Phase 1.7 Stage A (complete 2026-05-20) identified `resnet50-res512-all` as materially better than the original `densenet121-res224-all` baseline by aggregate localization (mean Dice 0.0397 vs 0.0237). ResNet-50 has been promoted to co-primary baseline; the bulk of the protocol has already run on it. The original "Decision (deferred)" framing is no longer accurate.
- Outstanding work: calibration v3 (after Phase 5.1 lands the new methods) and held-out improvement experiment (after Phase 5.2 script lands). See the Phase 5 Implementation Details section for the canonical plan.

#### 5.6 Captum infidelity and sensitivity (conditional)

- Protocol-marked optional. Phase 5.5 is now committed (ResNet-50 promoted to co-primary baseline after Stage A), so Captum metrics are no longer gated on 5.5 being skipped. They are a parallel low-cost add-on pulled in if the rest of Phase 5 lands ahead of schedule. Implementation is small (`captum.metrics.infidelity` and `captum.metrics.sensitivity_max`); fits naturally next to the existing deletion/insertion faithfulness writer. Cut order documented in the Phase 5 Implementation Details § "Cut order if days slip".
- Gives the thesis a second faithfulness probe alongside deletion/insertion, which strengthens the H8 "faithfulness vs localization" test by triangulating across two faithfulness families.

#### 5.7 LIME (conditional, low priority)

- Protocol-marked optional ("only if implementation time is low"). Kept on the menu rather than dropped: if the rest of Phase 5 lands by 2026-06-01 and the buffer holds, a small LIME pass on a sub-sampled positive set provides a third explanation family (region-level surrogate) for cross-method comparison.
- Implementation via `lime.lime_image.LimeImageExplainer`; expensive per case because LIME generates many perturbed forward passes. Time-budget controls: cap to ~10-20 representative thesis-quality cases, not the full balanced run. No registry integration needed beyond a thin wrapper that returns a `SignedAttribution`-shaped map.
- If skipped, justify in the thesis methodology as a scope adjustment per the protocol's explicit "only if implementation time is low" clause. The cross-method comparison still holds across Grad-CAM, Grad-CAM++, IG, GradientSHAP, Occlusion, Eigen-CAM, and Score-CAM, which is a stronger method panel than the protocol's required minimum.

### Revised execution order with deadline anchors

Status reflects the snapshot on 2026-05-27. The "Phase 5 Implementation Details" section below is the live schedule for outstanding work; this table is kept for historical context and completed-item provenance.

| # | Step | Days | Risk | Status |
|---|---|---|---|---|
| 1 | Phase 0 foundation + golden-output snapshot | 0.5 | low | ✅ Done 2026-05-18 (`99275dd`); CI + Makefile deferred to post-Phase-5. |
| 2 | Phase 1 correctness: tests, polarity fix, signed maps, manifest fix, faithfulness default | 2 | medium | ✅ Done 2026-05-21. |
| 2b | Phase 1.2.5 versioned calibration regeneration (v2) | 0.5 | low | ✅ Done 2026-05-22. |
| 3 | Compressed Phase 2: `MethodSpec` registry + `cxr/io.py` + `load_classifier(name)` seam | 1 | medium | ✅ Done 2026-05-22 to 2026-05-25 across Phase 2 + 4 commits (`3bcff3c`, `020c029`, `66c0f4d`). |
| 4 | Phase 3.1 + 3.2 + 3.3 + 3.4 (batched IG, vectorized occlusion, dead compute, mask contour) | 1 | medium | ✅ Done 2026-05-21. |
| 5 | **Phase 1.7 Stage A: in-family weights + 1 out-of-family external model on shared calibration cases** | 1 | medium | ✅ Done 2026-05-20 for in-family weights and ResNet-50; out-of-family slot remains open (MONAI bundle rejected as generative). See AGENTS.md "Diagnostic A/B Results". |
| 6 | Stage B outcome decision; optional Stage C deeper external-model exploration | 0-2 | medium | ✅ Stage B decided: ResNet-50 promoted to co-primary baseline; Stage C deferred. |
| 7 | Phase 5.1 Eigen-CAM + Score-CAM | 0.5 | low | ⏳ Pending. See Phase 5 Implementation Details. |
| 8 | Phase 5.2 improvement experiment script + narrative pre-drafts | 1 | low | ⏳ Pending (script). Narrative pre-drafting is handled by the student outside the agent loop. |
| 9 | Phase 5.4 CT pilot: **hour-1 model-availability check** then scaffold + first CT smoke (or fallback) | 2-3 | high | ⏳ Pending. |
| 10 | Phase 5.3 radiologist review workbook + scoring pass on CXR | 1.5 | low | ✅ Done with revised scope: balanced 40-case ResNet review completed 2026-05-25 (`outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/`); the originally planned 100-case pass was renegotiated to 40 balanced (10 per outcome) to fit the rubric-clarity budget. |
| 11 | Phase 4 minimum: `run_meta.json` stamping + `load_classifier` seam audit | 0.5 | low | ✅ Done. `run_meta.json` stamped on primary scripts 2026-05-21 and secondary/diagnostic scripts 2026-05-25; `load_classifier` seam landed during Phase 1.7. |
| 12 | Phase 5.5 stronger second CXR model full protocol | 1-2 | medium | 🟡 Mostly done. ResNet-50 Stage A, calibration v2, classifier-outcome 1000-case run, and 40-case review all complete. Remaining: ResNet calibration v3 (after 5.1) and ResNet held-out improvement experiment (after 5.2). |
| 13 | Final figures, results tables, finalize thesis writing | 1-2 | low | ⏳ Pending; runs in parallel with 5.1/5.2/5.4. |

**Parallel track: thesis writing starts on 2026-05-19**, not in the final buffer. The methodology chapter can be written as soon as `Phase 1.2` lands (signed-attribution semantics decided); the results chapter populates as each phase's CSVs land. Only the Discussion and Conclusions chapters wait for the improvement-experiment results. Net effect: by 2026-06-04, the draft is ready not because there is a writing-only buffer, but because writing has happened concurrently.

Hard deadline: full thesis draft `2026-06-04`. Final corrections/formatting/defense window `2026-06-05` to `2026-06-21`.

### External coordination tasks (added after pre-mortem)

- **AI-tooling disclosure policy**: confirm with the supervisor this week which institutional policy governs disclosure of the AI/development tools listed in the AI-tooling-disclosure rule in `AGENTS.md` (`GPT-5.5`, `Codex`, `PyCharm`, `Junie`, `VS Code`, `Claude Sonnet 4.6`, `Claude Opus 4.7`). Result: a methods-section paragraph naming the tools and roles. Done before 2026-05-22 so it does not block draft writing.
- **Polarity-fix supervisor communication**: if the supervisor has seen earlier figures with the buggy `grad_cam_plus_plus_negative` overlays, email proactively with the corrected example after `Phase 1.1` lands, framing it as instrument calibration with before/after.

### Items explicitly downgraded

- Full `logging` migration, `mypy --strict`, full file-split of `run_cxr_torchxray_smoke.py`: defer to post-defense.

### Items kept on the menu but conditional (added 2026-05-18; framing updated 2026-05-27)

- LIME (`Phase 5.7`): low-priority conditional add-on. Recommended for cut from the draft scope on 2026-05-27 (see Phase 5 Implementation Details § 5.7). If kept, activate only if the rest of Phase 5 lands by 2026-06-01.
- Captum infidelity / sensitivity (`Phase 5.6`): conditional add-on (parallel pull-in, no longer gated on Phase 5.5 being skipped since 5.5 is now committed). Strengthens the H8 "faithfulness vs localization" triangulation across two faithfulness families. Cut order documented in Phase 5 Implementation Details § "Cut order if days slip".

---

## Phase 5 Implementation Details (added 2026-05-27)

This section expands each Phase 5 item with: scope, files touched, step-by-step plan, tests, gates, risks, and rollback. All cost estimates assume a single working day = 6 focused hours.

Sequencing constraints carried from the discussion on 2026-05-27:

1. **5.1 must land before 5.2.** The improvement experiment's "consensus vs best individual" cannot freeze its candidate set until Eigen-CAM and Score-CAM are either in the registry or explicitly dropped.
2. **5.1 invalidates calibration v2.** Adding new methods means existing top-fractions do not cover them. A calibration v3 regen on the calibration split must precede any held-out evaluation that uses the new methods.
3. **5.2 runs on the held-out test split with frozen thresholds.** The improvement-experiment script must refuse to execute if any calibration artifact is older than the smoke output it would compare against. This is the "frozen thresholds" gate.
4. **5.5 reuses the 5.2 pipeline** on a second model. No new script; parametrize 5.2 by `--weights`.
5. **5.4 decision is binary in hour 1.** Off-the-shelf classifier exists and runs end-to-end on one slice → 2-day build. Doesn't → fallback to qualitative external validation only per the protocol's Week-3 rule. No 4–5 day fine-tuning detour under any circumstances.

Decisions locked on 2026-05-27:

- Statistics: Wilcoxon signed-rank + Holm-Bonferroni FWER control at α=0.05. References: [`REF-WILCOXON-1945`](references.md#ref-wilcoxon-1945), [`REF-HOLM-1979`](references.md#ref-holm-1979), [`REF-AICKIN-GENSLER-1996`](references.md#ref-aickin-gensler-1996), [`REF-DEMSAR-2006`](references.md#ref-demsar-2006). See [`docs/thesis-notes.md` § Statistical Methods for Method-vs-Method Comparison](thesis-notes.md).
- Narrative pre-drafts (consensus-wins vs consensus-loses) handled by the student outside the agent loop.
- CT pilot has no pre-committed claim, so Branch B (qualitative-only) is fully defensible.

---

### Phase 5.1 — Eigen-CAM + Score-CAM

**Goal:** add two CAM-family methods to the `MethodSpec` registry so the improvement experiment and held-out evaluation cover a broader method panel. References: [`REF-EIGEN-CAM`](references.md#ref-eigen-cam), [`REF-SCORECAM`](references.md#ref-scorecam).

**Files touched:**
- `src/explainai_thesis/xai.py` — new `eigen_cam_signed(...)` and `score_cam_signed(...)` functions returning `SignedAttribution`.
- `src/explainai_thesis/cxr/methods.py` — append two `MethodSpec` entries to `DEFAULT_METHOD_SPECS`. Extend `MethodContext` with `score_cam_channels_cap: int = 256`.
- `scripts/run_cxr_torchxray_smoke.py` — add CLI flag `--score-cam-channels-cap` (default 256); wire into `MethodContext`.
- `scripts/visualize_cxr_threshold_selection.py` and `scripts/visualize_cxr_classifier_outcome_thresholds.py` — same CLI flag.
- `AGENTS.md` — append `eigen_cam` and `score_cam` (with their four-view families) to the XAI Method Set; document the Eigen-CAM sign convention.
- `tests/test_eigen_cam.py` and `tests/test_score_cam.py` (new) — synthetic-data sanity tests; also add the two methods to the existing `tests/test_signed_attribution.py` parametrize sweep so the dispatch contract is verified.

**Implementation plan (step by step):**

1. **Eigen-CAM** (`xai.py`):
   - Hook the existing `GradCAM` machinery: it already captures forward activations on `bundle.target_layer`. Reuse the captured activations tensor of shape `[1, C, h, w]`.
   - Flatten to `[C, h*w]`, run `U, S, V = torch.linalg.svd(activations, full_matrices=False)`.
   - The top principal component is `V[0]` (shape `[h*w]`); reshape to `[h, w]`.
   - Sign convention: Eigen-CAM's sign is arbitrary up to a flip. Resolve by computing the inner product `(V[0] * activations.mean(dim=0))` and flipping `V[0]` if the inner product is negative. This anchors the positive side to the dominant activation direction.
   - Interpolate to image size with `F.interpolate(..., mode="bilinear", align_corners=False)`.
   - Normalize via `normalize_signed_map` (already in `metrics.py`).
   - Return a `SignedAttribution(raw=...)`.

2. **Score-CAM** (`xai.py`):
   - Capture forward activations on `bundle.target_layer` (reuse `GradCAM` hook or write a thin `ActivationCapture` class).
   - For each channel `c` in `range(C)` (or top `channels_cap` channels ranked by mean activation magnitude):
     - Upsample channel activation `[1, 1, h, w]` to image size via `F.interpolate`.
     - Normalize the upsampled map to `[0, 1]` via min-max.
     - Mask the original input by elementwise multiplication: `masked_input = input * upsampled_normalized`.
     - Forward `masked_input` through the model. Read the target class logit (or sigmoid).
     - `weight[c] = score(masked_input) - score(baseline)`. Common baseline: zero tensor of input shape.
   - Aggregate: `cam = sum(weight[c] * activation[c] for c in selected_channels)`, ReLU is *not* applied (we keep signed output to match the `SignedAttribution` contract).
   - Normalize via `normalize_signed_map`.
   - Return a `SignedAttribution(raw=cam)`.
   - Wall-time check at implementation time: profile against DenseNet-121-all on a 224×224 input. If a single case exceeds 30 s with `channels_cap=256`, lower the default to 128.

3. **Registry** (`cxr/methods.py`):
   ```python
   def _eigen_cam(ctx): return eigen_cam_signed(ctx.model, ctx.model_input, target_layer=ctx.gradcam.target_layer)
   def _score_cam(ctx): return score_cam_signed(
       ctx.model, ctx.model_input, class_idx=ctx.class_idx,
       target_layer=ctx.gradcam.target_layer,
       channels_cap=ctx.score_cam_channels_cap,
   )
   DEFAULT_METHOD_SPECS = (*existing, MethodSpec("eigen_cam", _eigen_cam), MethodSpec("score_cam", _score_cam))
   ```
   Do **not** add the new methods to `CONSENSUS_CONSTITUENTS`. Consensus stays as the original 4 (Grad-CAM, IG, GradientSHAP, Occlusion). Changing consensus constituents would invalidate every prior consensus result.

4. **Calibration v3 regeneration**:
   - Run `scripts/calibrate_cxr_xai_thresholds.py --weights <name> --calibrated-fractions ...` to produce `outputs/iter_XX_calibration_v3_with_eigen_score/calibrated_thresholds_v3.csv`.
   - Document v3 in `docs/progress.md` and reference it from `AGENTS.md` Calibration Versioning section.
   - Wall time: ~60–90 min per model on CUDA. Past the agent-tool budget; run manually.

**Tests:**
- `test_eigen_cam_signed_decomposition`: round-trip `SignedAttribution.positive + negative ≈ magnitude` on the synthetic dataset.
- `test_eigen_cam_sign_convention_stable`: run Eigen-CAM twice with `torch.manual_seed(0)` and assert the sign of the principal component is reproducible.
- `test_score_cam_with_channels_cap_completes`: synthetic DenseNet target, `channels_cap=8`, finish under 5 s on CPU.
- `test_score_cam_signed_decomposition`: same `positive + negative ≈ magnitude` check.
- Extend `tests/test_signed_attribution.py` parametrize sweep to include both new methods.

**Gates:**
- All new tests pass under `wsl.exe python3 -m pytest tests/ -m 'not slow'` in under 5 s additional.
- Calibration v3 CSV exists for at least DenseNet-121-all and ResNet-50.

**Risks and rollback:**
- Score-CAM wall time blows up at 512×512 on ResNet-50. → Reduce `channels_cap` to 64, document the cap explicitly in the methodology.
- Eigen-CAM sign drifts across cases. → Lock the convention; add the regression test above.
- Either method produces NaN under specific input distributions. → Add a guard in the compute function that returns a zero `SignedAttribution` and logs a warning when SVD or score-weighting fails.

**Cost estimate: 1.0 day total (0.5 Eigen + 0.5 Score + tests). Calibration v3 is 60–90 min wall time per model, scheduled overnight.**

---

### Phase 5.2 — Improvement experiment

**Goal:** Run consensus vs each individual method on the held-out test split with frozen calibration thresholds; report paired Wilcoxon + Holm-corrected p-values per metric. References: [`REF-WILCOXON-1945`](references.md#ref-wilcoxon-1945), [`REF-HOLM-1979`](references.md#ref-holm-1979), [`REF-DEMSAR-2006`](references.md#ref-demsar-2006).

**Files touched:**
- `scripts/run_improvement_experiment.py` (new).
- `src/explainai_thesis/stats.py` (new) — Wilcoxon + Holm-Bonferroni helpers + bootstrap CI.
- `requirements-dev.txt` — add `statsmodels` if not already present.
- `docs/progress.md` — append result entry on the day the experiment runs.

**Implementation plan:**

1. **Script CLI**: mirror `run_cxr_torchxray_smoke.py` flags exactly for compatibility with downstream consumers. New flags:
   - `--calibration-csv <path>` — frozen calibration v3 file. Required.
   - `--reference-method <name>` — default `consensus`. The single method against which all others are tested.
   - `--alpha <float>` — default 0.05.

   **Split discipline:** the manifest has only `split=train|test`. Calibration uses a fixed calibration subset drawn from the train split (the same positive-masked calibration cases that v1/v2/v3 calibration runs use, written via `scripts/calibrate_cxr_xai_thresholds.py`). The held-out evaluation always uses `--split test`. The script never partitions test internally and never adapts thresholds to test outcomes.

2. **Frozen-threshold gate**: at startup, read `<output-dir>/run_meta.json` for any existing run; if the calibration CSV's mtime is newer than the most recent smoke output that would be compared against, exit with a clear error. Force the user to either re-run upstream smoke or accept a stale calibration via `--allow-stale-calibration`.

3. **Pipeline**:
   - Load manifest, filter to `--split test`.
   - For each case: run all methods in `DEFAULT_METHOD_SPECS` plus consensus, using calibration v3 top-fractions.
   - Compute IoU, Dice, pointing_hit, precision_at_fraction per case per method (positive view only — the improvement claim is about positive evidence overlap with the lesion mask).
   - Write `improvement_experiment.csv` with one row per `(case, method, view)`.

4. **Stats** (`src/explainai_thesis/stats.py`):
   - `wilcoxon_paired(reference: np.ndarray, alternatives: dict[str, np.ndarray]) -> dict[str, dict]`: for each method name, drop NaN-paired rows, run `scipy.stats.wilcoxon(reference - alt, zero_method='wilcox')` two-sided. Return per-method dict with `statistic`, `p_raw`, `n_pairs`, `median_diff`, and bootstrap 95% CI.
   - `holm_bonferroni(p_raw: list[float], alpha: float) -> list[bool]`: sort ascending, test against escalating thresholds α/(N−i+1), stop at first failure. Use `statsmodels.stats.multitest.multipletests(p_raw, alpha=alpha, method='holm')` and document the call.
   - `bootstrap_paired_diff_ci(reference, alternative, n_resamples=10000, seed=20260515)`: numpy `default_rng`, percentile bootstrap, two-sided 95% CI.

5. **Output CSVs**:
   - `improvement_experiment.csv`: per-case per-method per-metric rows (long format).
   - `improvement_experiment_paired.csv`: one row per `(metric, method_compared)`. Columns: `metric`, `reference`, `compared`, `n_pairs`, `median_diff`, `bootstrap_ci_low`, `bootstrap_ci_high`, `wilcoxon_stat`, `p_raw`, `p_holm_threshold`, `holm_significant_bool`.
   - `improvement_experiment_summary.md`: short prose summary, including which narrative (A: consensus wins / B: consensus does not improve) the result supports per metric.

6. **Plots**:
   - `improvement_experiment_boxplots.png`: one panel per metric, box plot of per-case values for each method, reference highlighted.
   - `improvement_experiment_paired_diff.png`: paired-difference distribution (consensus − method) per method, with bootstrap CI shown as error bars.

7. **Run on DenseNet held-out test split (Day 4)** and on ResNet held-out test split (same day, second invocation). Output folders: `outputs/iter_XX_improvement_experiment_<weights>/`.

**Tests:**
- `tests/test_stats.py` (new):
  - `test_wilcoxon_paired_known_inputs`: pre-computed scipy result on a fixed seed.
  - `test_holm_bonferroni_known_pvalues`: hand-worked example with 5 p-values, assert correct accept/reject pattern.
  - `test_bootstrap_paired_diff_ci_deterministic`: seed=0, fixed data, assert CI bounds.
- `tests/test_improvement_experiment_smoke.py`: synthetic 5-case dataset, run script end-to-end, assert CSV columns and that no NaN appears in non-CI columns.

**Gates:**
- Frozen-threshold check fires on a stale-calibration test.
- Holm-Bonferroni output matches `statsmodels.stats.multitest.multipletests` reference output on 3+ test cases.
- Bootstrap CI is deterministic with a fixed seed across reruns.

**Risks and rollback:**
- Consensus does not improve over best individual on either model. → Narrative B in the thesis Discussion, framed around method disagreement as a diagnostic finding. The thesis remains defensible.
- N is too small after dropping cases with NaN metrics (e.g. zero mask area). → Document the case-exclusion rule; report N per test.
- statsmodels not yet installed in WSL env. → `pip install statsmodels` is fast and dependency-light.

**Cost estimate: 1.5 days (1.0 script + 0.5 runs on both models).**

---

### Phase 5.4 — CT pilot (binary decision in hour 1)

**Goal:** Test transfer of the validation methodology to a different modality (CT hemorrhage). Decision is binary: if an off-the-shelf classifier with a verifiable hemorrhage class head exists, run a small smoke; if not, fall back to qualitative-only discussion. Reference: [`REF-RSNA-IHD`](references.md#ref-rsna-ihd).

**Hour 0–1: model-availability AND mask-availability check (binding constraint)**

Both axes are checked together because the pilot needs (1) a usable off-the-shelf classifier and (2) usable masks. Manual annotation is a fallback only if the masks track is exhausted.

**Model track** — search candidates in order:
1. **RSNA Intracranial Hemorrhage Detection Kaggle**: public competition; multiple top-N solutions on GitHub. Look for repos with downloadable checkpoints (not just training recipes). Filter on: (a) license, (b) checkpoint URL accessible without competition signup, (c) recognizable PyTorch/Keras `state_dict` load.
2. **MONAI Model Zoo**: re-check post the 2026-05-18 finding that the prior CXR bundle was a generative model. Look for `*hemorrhage*` or `*intracranial*` bundle ids. Confirm `configs/metadata.json` lists a classification head with `hemorrhage` or per-subtype labels.
3. **HuggingFace Hub `transformers` or `timm` model zoo**: search "CT hemorrhage" with `task: image-classification`. Look for cards that cite RSNA-IHD as the training set.
4. **TorchXRayVision's CT branch** (if present): unlikely but worth a 5-minute check.

Model pass criteria: license permits research use; checkpoint URL is stable; one slice loads, preprocesses, and forwards end-to-end producing a meaningful score; class label is recoverable from metadata.

**Mask track** — try in order before falling back to manual annotation:
1. **`vbookshelf/computed-tomography-ct-images` (Kaggle re-host of Hssayeni PhysioNet CT ICH)** — 318 slices with intracranial hemorrhage masks across 82 patients. Cited in `docs/dataset_sources.md` Option A. This is the preferred mask source; if it loads cleanly, manual annotation is skipped entirely.
2. **PhysioNet Hssayeni CT-ICH original** — same data, source-of-truth path. Use if the Kaggle re-host is unavailable.
3. **Manual annotation** — only if both above fail. Cap at 1 hour wall time, 20-30 positive slices, hemorrhage-window viewer preset.

Mask pass criteria: positive slices have a binary mask file at a stable path; mask dimensions match the corresponding image; at least 20 positive slices are usable.

**Decision rule:** if at least one model candidate AND a mask source both pass within hour 1, go to Branch A. Otherwise, go to Branch B. Hard stop on Branch A search after 60 minutes — no exceptions.

**Branch A — model found:**

Files touched:
- `src/explainai_thesis/ct/__init__.py` (new).
- `src/explainai_thesis/ct/io.py` (new) — HU windowing (soft-tissue or hemorrhage window: typically WW=80, WL=40 for brain), DICOM-or-PNG-or-NPZ slice preprocessing, resize to model's expected input.
- `src/explainai_thesis/ct/models.py` (new) — model loader returning a `ClassifierBundle` like the CXR `load_classifier` seam. Extend the seam to dispatch on a `modality` field, OR create a `load_ct_classifier(name)` and don't share with CXR (cleaner if the preprocessing differs significantly).
- `scripts/run_ct_smoke.py` (new) — CT analogue of `run_cxr_torchxray_smoke.py`, reusing the `MethodSpec` registry.
- `data/ct_hemorrhage_manifest.csv` (new) — built from whichever mask source passed the hour-1 check (vbookshelf/Hssayeni preferred; manual annotation only as fallback). Target: 20-30 positive slices.
- `tests/test_ct_io.py` (new) — HU windowing round-trip test on a synthetic HU-scaled tensor; CT-shaped synthetic dataset based on `SyntheticLesionDataset` patterns.

Faithfulness baseline: add `--faithfulness-baseline soft_tissue_window_zero` because `black` (-1024 HU) means air in CT and is clinically meaningful, not neutral. **Implementation note:** the current `src/explainai_thesis/faithfulness.py::faithfulness_baseline_tensor` only supports `zero_tensor`, `black`, `white`, and `case_mean`. The new `soft_tissue_window_zero` branch must be added before any CT smoke runs. Suggested semantics: fill with the HU value that maps to zero in the soft-tissue display window (WW=400, WL=40 → HU=40 ≈ 0.1 in the CT model's normalized input space; defer the exact value to the chosen CT model's preprocessing contract). Document the chosen mapping in `AGENTS.md` alongside the existing CXR baselines.

Output folder: `outputs/iter_XX_ct_smoke_<short>/`.

**Branch B — no model found:**

Files touched:
- `docs/progress.md` — one entry documenting:
  - Which sources were checked (URLs and date).
  - Why each was rejected (no class head, no public weights, license unclear, etc.).
  - Decision to fall back to qualitative external validation only.
- `docs/thesis-notes.md` — already has the framing in the new CT section above; cite the progress entry from there.
- No code changes.

**Tests (Branch A only):**
- `test_ct_io_hu_windowing_roundtrip` (CPU, fast).
- `test_ct_smoke_synthetic_end_to_end` (CPU, fast, mocked model bundle).

**Gates:**
- Branch A: at least one positive case smoke runs end-to-end and produces a non-zero attribution map for at least one method.
- Branch B: progress.md entry exists and references `REF-RSNA-IHD` as the future-work anchor.

**Risks and rollback:**
- Branch A model loads but produces garbage attribution on real CT data. → Document as a qualitative finding in thesis; do not over-claim transfer.
- Branch A annotation takes > 2 hours due to viewer setup. → Cap manual annotation at 1 hour wall time; whatever's done is what's done.
- Branch B is chosen but the supervisor expects a quantitative CT result. → User confirmed on 2026-05-27 that there is no pre-committed CT claim, so this is acceptable.

**Cost estimate: 0.5 day (Branch B) to 2.5 days (Branch A).**

---

### Phase 5.5 — Stronger second CXR model (full protocol)

**Goal:** Re-run the improvement experiment on `resnet50-res512-all` so the thesis can present a head-to-head comparison of consensus vs individual methods across two CXR backbones. Per Stage A outcome, ResNet-50 is the strongest off-the-shelf TorchXRayVision candidate and is the natural second model.

**Most of this is already done.** Stage A (`outputs/iter_33_stage_a_diagnostic_ab/resnet50-res512-all/`) has v2 calibration and smoke. The classifier-outcome 1000-case run (`outputs/iter_36_resnet_classifier_outcome_any1000_all_methods_2/`) is also done. The targeted review (`outputs/iter_47_resnet_review_diagnostics_balanced40_smoothed_faithfulness/`) is complete on 40 balanced cases.

**What's actually missing:**

1. **Calibration v3 on ResNet-50** — must be regenerated after 5.1 lands so it covers Eigen-CAM and Score-CAM.
2. **Improvement experiment on ResNet held-out test split** — one invocation of the 5.2 script: `scripts/run_improvement_experiment.py --weights resnet50-res512-all --calibration-csv <resnet-v3>`.
3. **Head-to-head comparison table** — in the thesis Results section, side-by-side mean Dice / IoU / pointing-hit / Holm-corrected improvement-experiment result for DenseNet-121-all vs ResNet-50-all.

**Files touched:**
- No new scripts. Re-uses 5.1 and 5.2 outputs.
- `thesis/` — head-to-head comparison table added to the Results chapter.
- `docs/progress.md` — entry recording the ResNet improvement-experiment outcome.

**Tests:** None new. Tests added under 5.2 cover the pipeline.

**Gates:**
- ResNet calibration v3 CSV exists.
- ResNet improvement-experiment output folder exists with all expected CSVs.

**Risks and rollback:**
- ResNet calibration v3 takes longer than DenseNet (likely ~90 min at 512×512). → Run overnight.
- ResNet improvement experiment outcome differs materially from DenseNet's. → That is itself a thesis-relevant finding; report both faithfully and discuss model-dependence in Chapter 4.

**Cost estimate: 0.5 day after 5.1 + 5.2 land.**

---

### Phase 5.6 — Captum infidelity + sensitivity

**Goal:** Triangulate H8 ("faithfulness vs localization") by adding two faithfulness metrics from a different faithfulness family (perturbation-prediction linear consistency + attribution stability) alongside the existing deletion/insertion AUC. Reference: [`REF-INFIDELITY-SENSITIVITY`](references.md#ref-infidelity-sensitivity).

**Files touched:**
- `src/explainai_thesis/faithfulness.py` — add `infidelity_score(model, input, attribution, class_idx, perturb_fn, n_samples=10)` and `sensitivity_max_score(model, input, attribution_fn, class_idx, perturb_std=0.02, n_samples=10)`.
- `src/explainai_thesis/io.py` — extend `METRICS_FIELDS` with `infidelity` and `sensitivity_max` columns. Document the schema bump in `docs/progress.md`.
- `scripts/run_improvement_experiment.py` — compute the two metrics per case per method and append columns.
- `tests/test_faithfulness.py` — extend with infidelity/sensitivity sanity tests on the synthetic dataset (`assert 0 <= infidelity` and `assert sensitivity_max >= 0`).

**Implementation plan:**

1. **Perturbation operator**: Gaussian noise added to the input with σ=0.02 on the normalized-image scale. Document this in methodology — perturbation choice matters per [`REF-MEANINGFUL-PERTURBATION`](references.md#ref-meaningful-perturbation).

2. **Infidelity** (`infidelity_score`): for `n_samples` perturbations `δ`:
   - `dot = sum(attribution * δ)`
   - `actual_diff = model(input) - model(input - δ)` (target class scalar).
   - Per-sample squared error: `(dot - actual_diff) ** 2`.
   - Return mean across samples.

3. **Sensitivity-max** (`sensitivity_max_score`): for `n_samples` perturbations `ε` of input:
   - Re-run the attribution function on the perturbed input.
   - Compute `||attribution(input + ε) - attribution(input)||_2 / ||ε||_2`.
   - Return the maximum across samples.

4. **Integration**: call both in `run_improvement_experiment.py` after computing per-case attributions. Wall-time impact: each metric requires `n_samples` extra forward (infidelity) or forward+backward (sensitivity) passes per case. With `n_samples=10` and 100 test cases × 7 methods, expect ~10× the existing per-case cost. Cap `n_samples=10` in the default config; allow higher via CLI.

**Tests:**
- `test_infidelity_score_decreases_under_better_attribution`: on the synthetic dataset, the ground-truth attribution should yield lower infidelity than a random attribution.
- `test_sensitivity_max_score_bounded`: assert the returned value is finite and ≥ 0.
- `test_infidelity_score_deterministic_with_seed`: fixed seed reproducibility.

**Gates:**
- Captum is optional but if installed (already a dependency), reuse `captum.metrics.infidelity` and `captum.metrics.sensitivity_max` directly to avoid re-implementing. Keep our wrappers thin so the methodology can cite Captum implementation.
- Default `n_samples=10` keeps the runtime impact bounded.

**Risks and rollback:**
- Sensitivity computation triggers VRAM OOM at 512×512 on ResNet. → Lower `n_samples` to 5; or run sensitivity on a 30-case subset and report it as a supplementary diagnostic instead of an all-cases metric.
- Captum API drift between minor versions changes the return shape. → Pin captum version in `requirements-dev.txt` and add a single regression test on a known input.

**Cost estimate: 0.5 day.**

---

### Phase 5.7 — LIME (conditional, recommended to cut)

**Goal:** Add a region-level surrogate explanation family as a qualitative third comparator. Reference: [`REF-LIME`](references.md#ref-lime).

**Recommended decision: cut from the draft scope.** Cite the experiment protocol's "only if implementation time is low" clause. The current method panel (Grad-CAM, Grad-CAM++, IG, GradientSHAP, Occlusion, Eigen-CAM, Score-CAM) already spans CAM, gradient, perturbation, and PCA-based families. LIME's region-level framing is conceptually distinct but the per-case wall time makes it impractical at N≥100, and at N=10–20 the sample is too small to support paired-test inclusion.

**If kept (cheapest version):**

Files touched:
- `src/explainai_thesis/xai.py` — `lime_signed(model, input, class_idx, n_samples=1000, segmentation='slic')`. Wraps `lime.lime_image.LimeImageExplainer`.
- `requirements-dev.txt` — add `lime`.
- `scripts/run_lime_supplementary.py` (new) — runs LIME on a 10–20 case subset selected from the existing review-candidate manifest. Output: `outputs/iter_XX_lime_supplementary/`.
- `tests/test_lime.py` (new) — single-case sanity test on a synthetic image.

Implementation plan:
1. Convert input tensor back to `[0, 255]` uint8 image (LIME expects PIL-like).
2. Define a `classifier_fn` that takes a numpy batch of perturbed images, preprocesses each via the classifier bundle's preprocess, and returns logits.
3. Call `explainer.explain_instance(image, classifier_fn, top_labels=1, num_samples=1000)`.
4. Convert the resulting positive-weight region mask into a `SignedAttribution` by setting positive weight pixels to `+weight` and negative weight pixels to `-weight`.
5. Run on the same 10 cases used in the radiologist review workbook. Produce overlays + a brief qualitative comparison figure.

Gates: budget cap of 2 hours wall time. If LIME does not finish on the 10-case set in 2 h, stop.

Risks: LIME results depend on segmentation algorithm (SLIC defaults), num_samples, and random seed. Document all three.

**Cost estimate: 0.5 day + 2 h runtime, only if all of 5.1–5.6 land by 2026-06-01.**

---

### Cross-cutting: docs/progress.md entry discipline

Every Phase 5 item lands with a same-day `docs/progress.md` entry covering:
- What changed in code (paths + commit hash).
- What output folder the run wrote to (if any).
- The headline numerical or qualitative outcome (one or two lines).
- The thesis-defensible interpretation: what the result *means*, not just what it *is*.

This keeps the chronological audit trail intact and prevents thesis-writing-time rediscovery of decisions.

---

### Cut order if days slip

In strict order, first to be cut:

1. **5.7 LIME** — already recommended to cut.
2. **5.6 Captum infidelity + sensitivity** — H8 triangulation still works with deletion/insertion alone.
3. **5.4 Branch A** — fall back to Branch B (qualitative-only) even if a model exists.
4. **Do not cut 5.1, 5.2, or 5.5.** These are thesis-deliverable contributions.

Cut decisions must be recorded in `docs/progress.md` on the day they're made.
