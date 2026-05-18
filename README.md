# master_thesis_draft_explainAI

Master-thesis pipeline for validating explainable-AI (XAI) methods on medical imaging. The primary, documented experiment is **chest X-ray pneumothorax classification + explanation** on the Kaggle/SIIM-ACR pneumothorax dataset, using an off-the-shelf `TorchXRayVision` DenseNet-121 baseline (`densenet121-res224-all`). A secondary head-CT intracranial-hemorrhage pipeline is planned under `src/explainai_thesis/ct/`.

> Authoritative companions: `AGENTS.md` (repo rules and conventions), `docs/progress.md` (append-only lab notebook), `docs/refactor_plan.md` (refactor phases), `docs/experiment_protocol.md` (protocol).

---

## 0. One-time setup

- Use **WSL Ubuntu** for Python runs; native Windows Python is generally not on `PATH`.
- Install the package editably (once per clone):

  ```bash
  wsl.exe --cd /mnt/c/Users/Dmytro.Valantsevych/Downloads/master_thesis_draft_explainAI \
      python3 -m pip install -e . --no-deps --no-build-isolation
  wsl.exe python3 -m pip install -r requirements-dev.txt
  ```

  Requires `setuptools>=68` in the WSL user site (`python3 -m pip install --user --upgrade "setuptools>=68" wheel` if needed). After this, scripts import `explainai_thesis` directly — do not reintroduce any `sys.path.insert(...)` bootstrap.

- Sanity check:
  ```bash
  wsl.exe python3 scripts/check_environment.py
  ```
- Dataset must be present at `data_local/cxr_pneumothorax/siim-acr-pneumothorax` (~12,047 PNG images + masks; 2,669 positive / 9,378 negative). Build/refresh the manifest:
  ```bash
  wsl.exe python3 scripts/build_manifest.py
  ```
- Tests:
  ```bash
  wsl.exe python3 -m pytest tests/ -v          # full
  wsl.exe python3 -m pytest tests/ -m 'not slow'  # fast (~7 s, 44 tests)
  ```

---

## 1. Repository layout

- `src/explainai_thesis/` — the library:
  - `cxr/classifier.py` — `load_classifier(name, device, pathology=...)` seam returning a `ClassifierBundle(model, target_layer, class_idx, preprocess)`.
    - DenseNet-121: `densenet121-res224-{all,chex,mimic_ch,mimic_nb,rsna,nih,pc}` (`rsna` has no Pneumothorax head and is skipped by the orchestrator).
    - ResNet-50: `resnet50-res512-all` — **requires `--image-size 512`**.
    - ResNetAE: `resnetae-101-elastic` — no class head; opt-in only with `pathology=None`; not wired into the pneumothorax XAI pipeline yet.
  - `xai.py` — `MethodSpec` registry: `grad_cam`, `grad_cam_plus_plus`, `integrated_gradients`, `gradient_shap`, `occlusion`, `consensus`. Post-refactor (v2) each method emits one `SignedAttribution` with four views: `positive`, `negative`, `magnitude`, `signed`. Planned additions: `eigen_cam`, `score_cam`, `consensus_attention`.
  - `visualization.py` — overlay rendering. Color semantics: red = positive evidence, blue = negative evidence, violet = magnitude, orange/teal = signed tug-of-war, green = GT mask, yellow/cyan = positive/negative-evidence intersections with the mask.
  - Localization metrics: `IoU`, `Dice`, `pointing_hit`, `precision_at_fraction`. Negative diagnostics: `negative_mask_overlap_fraction`, `negative_mask_avoidance_fraction`. Faithfulness: insertion/deletion curves and AUCs.
- `scripts/` — orchestrators (see next sections).
- `tests/` — pytest suite.
- `outputs/iter_XX_<short_name>/` — every experiment writes here. **Never rename, renumber, or overwrite** previous iteration folders.
- `docs/progress.md` — append-only chronological lab notebook (binding source of truth for what was run/decided).
- `reports/weekly/week_*_report*.md` — weekly summaries; any `*_final.md` is frozen and must not be edited.

---

## 2. End-to-end flow (CXR pneumothorax)

### 2.1 Classifier threshold calibration (per model, train split only)

```bash
wsl.exe python3 scripts/evaluate_cxr_torchxray_model.py --weights densenet121-res224-all --device auto
```

Reports best-`F1`, best-Youden's-`J`, and high-sensitivity operating points. The currently frozen DenseNet-`all` cutoff is **`0.62`**. Every other model needs its own sweep before any held-out comparison. Do **not** tune thresholds on the test split.

### 2.2 XAI top-fraction calibration (per model, v2)

Calibrates per-method top-K pixel fractions on positive masked calibration cases:

```bash
wsl.exe python3 scripts/calibrate_cxr_xai_thresholds.py \
    --weights densenet121-res224-all \
    --fractions 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95 \
    --output-dir outputs/iter_XX_calibration_v2_densenet121_all
```

Writes `calibrated_thresholds_v2.csv` keyed by (method, view, selection metric: Dice/IoU for positive localization, avoidance for negative diagnostics). v1 calibration files are statistically stale after the `SignedAttribution` refactor; **only v2 feeds held-out evaluation**. v1 and v2 outputs coexist; v1 is never overwritten.

### 2.3 Main smoke / localization / faithfulness run

```bash
wsl.exe python3 scripts/run_cxr_torchxray_smoke.py \
    --weights densenet121-res224-all \
    --calibrated-fractions outputs/iter_XX_calibration_v2_densenet121_all/calibrated_thresholds_v2.csv \
    --classifier-threshold 0.62 \
    --ig-steps 8 --gradshap-samples 8 \
    --occlusion-patch-size 56 --occlusion-stride 56 \
    --faithfulness-baseline black \
    --output-dir outputs/iter_XX_smoke_densenet121_all
```

Outputs:
- One folder per source X-ray with all overlays flat inside (every filename includes the X-ray stem so files remain traceable when copied).
- `metrics.csv` (per case, per method, per view), `cases.csv`, faithfulness curves and AUCs.

For broad screening, the `8/8` IG/GradientSHAP settings are fine. For thesis-quality reruns of selected cases, rerun with `--ig-steps 16 --gradshap-samples 64` (or `128` if the map is still unstable), and finer occlusion (`--occlusion-patch-size 32 --occlusion-stride 12`).

**Multi-model orchestration:**
- `scripts/run_cxr_resnet_smoke.ps1` — chains v2 calibration + smoke for ResNet-50 at 512×512.
- `scripts/run_all_models_smoke.ps1` — loops over the working DenseNet/ResNet set (skips `rsna` for missing Pneumothorax head and `resnetae` for being an autoencoder).

### 2.4 Balanced classifier-outcome visualization (long, resumable)

```bash
wsl.exe python3 scripts/visualize_cxr_classifier_outcome_thresholds.py \
    --device auto --split any --max-cases 5000 --random-sample --seed 20260517 \
    --threshold 0.62 --max-per-outcome 100 \
    --ig-steps 8 --gradshap-samples 8 \
    --occlusion-patch-size 56 --occlusion-stride 56 \
    --fractions 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50 \
    --progress-every 10 --checkpoint-every 1 \
    --output-dir outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods
```

Top-level outcome folders `tp/`, `fp/`, `tn/`, `fn/`, one case folder per X-ray inside each. Add `--resume` to restart; resume reads `cases.csv`, `threshold_metrics.csv`, and `progress.json`. `--split any` mixes train and test rows — label such runs exploratory, not held-out.

### 2.5 Targeted diagnostics

- `scripts/visualize_cxr_threshold_selection.py` — single-image threshold sweep for thesis-quality figures.
- `scripts/diagnose_cxr_torchxray_baselines.py` — faithfulness-baseline sanity check. Default baseline is `black`; the historical `zero_tensor` was misleading.
- `scripts/select_cxr_review_candidates.py` — mines cases for the radiologist review pass.

### 2.6 Diagnostic A/B across model weights (Phase 1.7 Stage A)

For each weight in the working set (`densenet121-res224-{all,chex,mimic_ch,mimic_nb,nih,pc}` + `resnet50-res512-all` + one external out-of-family model TBD):

1. Per-model classifier-threshold sweep (§2.1).
2. Per-model v2 XAI calibration (§2.2).
3. Per-model smoke + faithfulness (§2.3).
4. Aggregate per-model means of `IoU`, `Dice`, `pointing_hit`, `precision_at_fraction` into `outputs/iter_33_stage_a_diagnostic_ab/weights_ab_summary.csv`.

Orchestrated by `scripts/run_stage_a_diagnostic_ab.ps1` — chains all three sub-steps per model, skips already-produced artifacts (use `-Force` to rerun), supports `-DryRun`, `-Models <list>`, and `-Skip{Sweep,Calibration,Smoke,Summary}` flags. Out-of-family model: a MONAI Model Zoo CXR bundle (decided 2026-05-18); concrete bundle id picked at integration time after `monai.bundle download` + label inspection. Decision rule recorded in `AGENTS.md` ("Diagnostic A/B Protocol Before Full Second-Model Integration") tells whether a model is promoted to co-primary baseline or the result is framed methodologically.

### 2.7 Radiologist review workflow (planned)

`scripts/build_review_workbook.py` will generate `<run>/review/`:
- `index.html` — one card per case with GT mask + 4-view overlay grid per method, 4-rubric guide and 7-category failure taxonomy inlined, classifier outcome and probability, keyboard navigation.
- `scores_template.csv` — prefilled `case_id`, `filename`.
- `INSTRUCTIONS.md` — workflow plus a 3-case warmup.

Schema (per `docs/experiment_protocol.md`):
- `localization_score` ∈ {correct, partial, incorrect, none}
- `usefulness_score` ∈ {useful, potentially_useful, misleading, not_useful}
- `failure_category` ∈ {correct, partial, anatomically_related, devices_text_artifacts, non_pathological_high_contrast, diffuse_non_specific, clinically_misleading}
- `artifact_note`, `comment` — free text

Save `scores.csv` next to the template; never edit the template.

### 2.8 Improvement experiment (planned)

`scripts/run_improvement_experiment.py` will freeze v2 calibrated thresholds, evaluate each method plus `consensus` and `consensus_signed` on the **held-out test split**, emit `improvement_experiment.csv` and paired Wilcoxon `improvement_experiment_paired.csv` plus Chapter 4 box plots. It refuses to run if any calibration artifact is older than the smoke it would compare against — protecting the val/test contract from accidental peeking.

### 2.9 Metric correlations (planned)

`scripts/analyze_metric_correlations.py` will read any `metrics.csv` and emit Spearman/Pearson heatmaps across `IoU`, `Dice`, `pointing_hit`, `precision_at_fraction`, insertion/deletion AUC, SPA, `agreement_score`; then join `scores.csv` to compute correlations against `localization_score` and `usefulness_score`.

---

## 3. Typical first-time invocation order

1. `scripts/build_manifest.py` (if manifest missing).
2. `scripts/check_environment.py`.
3. `scripts/evaluate_cxr_torchxray_model.py` → pick classifier threshold.
4. `scripts/calibrate_cxr_xai_thresholds.py` → produce v2 calibration.
5. `scripts/run_cxr_torchxray_smoke.py` (or `run_all_models_smoke.ps1` / `run_cxr_resnet_smoke.ps1`) → main XAI/localization/faithfulness pass.
6. `scripts/visualize_cxr_classifier_outcome_thresholds.py` → balanced qualitative grids.
7. *(planned)* `build_review_workbook.py` → radiologist scoring.
8. *(planned)* `run_improvement_experiment.py` → held-out test eval + paired Wilcoxon.
9. *(planned)* `analyze_metric_correlations.py` → metric-metric and metric-vs-radiologist correlations.

---

## 4. Hard rules

- Do **not** tune any threshold on the test split.
- Do **not** overwrite, rename, or renumber `outputs/iter_*` folders.
- v1 XAI outputs/calibrations coexist with v2; **held-out reporting uses v2 only**.
- For runs expected to exceed ~30 minutes, execute manually rather than via short agent tool calls.
- After any change under `src/explainai_thesis/` or anything affecting output schema, run `wsl.exe python3 -m pytest tests/ -v`. For documentation-only changes, tests are not required.
- XAI maps are **model-behavior diagnostics**, not clinical segmentations: keep classifier performance, positive localization, negative-evidence diagnostics, faithfulness, and qualitative case studies separately in any thesis-facing text.

---

## 5. Method/view semantics (post-refactor v2)

Primary method ids: `grad_cam`, `grad_cam_plus_plus`, `integrated_gradients`, `gradient_shap`, `occlusion`, `consensus` (+ planned `eigen_cam`, `score_cam`, `consensus_attention`). Each case produces a single `SignedAttribution` per method with four views:

- `positive` — pixels pushing the model toward the pneumothorax class (red).
- `negative` — pixels pushing the model away from it (blue).
- `magnitude` — `|positive| + |negative|`, impactful pixels regardless of sign (violet).
- `signed` — `positive − negative`, tug-of-war view (orange/teal). For IG and GradientSHAP this is `*_signed`; treat sign interpretation here as supplementary to the dedicated red/blue/violet views.

Positive localization metrics are meaningful only for the positive view on positive-label cases with masks. Negative evidence overlapping the lesion is a concern, not a virtue: prefer `negative_mask_avoidance_fraction`. Faithfulness curves evaluate the model, not clinical correctness; comparable plots must use the same scale.

---

## 6. Reporting

- Append decisions and results to `docs/progress.md` chronologically (do not prepend).
- Weekly summary in `reports/weekly/week_N_report.md`; freeze as `..._final.md` once submitted (frozen files are not edited).
- Final thesis should reference exact versions of AI/development tools used (`GPT-5.5`, `Codex`, `PyCharm`, `Junie`, `VS Code`, `Claude Sonnet 4.6`, `Claude Opus 4.7`) and all research models / XAI methods actually used.
