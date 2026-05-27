# Repository Agent Guide

This file preserves the practical repository context, experiment conventions, and decision rules accumulated during the current thesis work. Future agents should read this before making changes or running experiments.

## Basic rules of code development
- Don't assume. Don't hide confusion. Surface tradeoffs.
- Minimum code that solves the problem. Avoid speculative.
- Touch only what you must, and clean up only your own mess, unless explicitly requested otherwise.
- Define success criteria. Loop until verified.

## Project Context

- This repository is a master thesis project on validating explainable AI (`XAI`) methods for medical imaging.
- Current main experiment: chest X-ray pneumothorax classification/explanation using the Kaggle/SIIM-ACR pneumothorax dataset.
- Current baseline model: external pretrained `TorchXRayVision` `DenseNet`, normally loaded as `xrv.models.DenseNet(weights="densenet121-res224-all")`.
- Important: the `TorchXRayVision` model itself has not been modified, fine-tuned, or locally forked. It is used as an off-the-shelf baseline.
- Current evidence suggests this baseline is clinically weak/mismatched for SIIM pneumothorax localization, even though it has moderate classifier ranking performance.
- Future work should likely add a stronger second pneumothorax model before final thesis conclusions.

## Environment and Execution

- Use `WSL Ubuntu` for Python runs, not native Windows Python.
- Native Windows `python` may not be available on `PATH`; prefer `wsl.exe python3 ...`.
- For commands that need project-root certainty, use `wsl.exe --cd /mnt/c/Users/Dmytro.Valantsevych/Downloads/master_thesis_draft_explainAI python3 ...`.
- Known WSL environment from previous runs: Python `3.10.12`, CUDA-capable PyTorch, and `torchxrayvision==1.4.0`.
- If a run is expected to exceed about `30` minutes, provide the command for manual execution instead of relying on short agent tool execution.
- Package install (added 2026-05-18, Phase 0 refactor): the `explainai_thesis` package is installed editably from `pyproject.toml`. On a fresh clone, run `wsl.exe --cd /mnt/c/Users/Dmytro.Valantsevych/Downloads/master_thesis_draft_explainAI python3 -m pip install -e . --no-deps --no-build-isolation` once. This requires `setuptools>=68` in the WSL user site (upgrade with `python3 -m pip install --user --upgrade "setuptools>=68" wheel` if needed). After this, scripts import `explainai_thesis` directly; the historical `sys.path.insert(0, str(ROOT / "src"))` bootstrap has been removed from all scripts and must not be reintroduced. Dev tools: `python3 -m pip install -r requirements-dev.txt` (pytest, pytest-cov, scipy). Local test command: `wsl.exe python3 -m pytest tests/ -v`.

## Dataset Context

- Main local CXR dataset path: `data_local/cxr_pneumothorax/siim-acr-pneumothorax`.
- Previously documented dataset state: `12,047` PNG images, `12,047` PNG masks, `2,669` positive cases, and `9,378` negative cases.
- For final thesis writing, verify exact dataset citation, access date, Kaggle/SIIM-ACR source details, and all dataset counts again.

## Documentation and Progress Rules

- Keep writing important decisions/results to `docs/progress.md`.
- Preserve chronological/context ordering in `docs/progress.md`; do not prepend new iteration notes above the initial context unless explicitly requested.
- Week 1 reports are treated as frozen/submitted: do not edit `reports/weekly/week_1_report.md` or `reports/weekly/week_1_report_final.md`.
- *_final weekly reports should not be edited unless explicitly requested.
- If adding major tool/model/dataset decisions, keep thesis-safe wording in `docs/progress.md` for later reuse.
- Final thesis should reference AI/development tools and exact versions where applicable, including `GPT-5.5`, `Codex`, `PyCharm`, `Junie`, `VS Code`, `Claude Sonnet 4.6`, and `Claude Opus 4.7`, plus all research models and XAI methods actually used.

## Output Folder Naming and Layout

- New experiment folders under `outputs/` should include a stable ordinal iteration number: `outputs/iter_XX_<short_experiment_name>`.
- Do not rename old output folders after they are referenced in progress notes or reports.
- Keep root-level CSVs and summary plots at the run root.
- For image artifacts, prefer one folder per source X-ray, with all PNGs flat inside that case folder.
- Every generated image filename should include the source X-ray stem so a copied file remains traceable without its parent folder.
- For classifier-outcome visualizations, preserve top-level outcome folders `tp/`, `fp/`, `tn/`, and `fn/`; inside each, use one case folder per X-ray, with all PNGs flat inside.

## Current Core Scripts

- `scripts/evaluate_cxr_torchxray_model.py`: classifier screening/evaluation, random sampling, thresholding, selected operating points, and outcome columns.
- `scripts/run_cxr_torchxray_smoke.py`: main positive-case XAI localization/faithfulness evaluation, grouped case output, calibrated fractions, faithfulness baselines, random sampling, and case targeting.
- `scripts/calibrate_cxr_xai_thresholds.py`: XAI top-fraction calibration on positive masked calibration cases, including all current methods and multiple selection metrics.
- `scripts/visualize_cxr_threshold_selection.py`: single-image threshold sweep/diagnostic visualization for targeted thesis-quality reruns.
- `scripts/visualize_cxr_classifier_outcome_thresholds.py`: balanced `tp`/`fp`/`tn`/`fn` visualization with progress, ETA, checkpointing, six-line live progress, and `--resume`.
- `scripts/diagnose_cxr_torchxray_baselines.py`: diagnostic tool for faithfulness baseline behavior.

## Classifier Threshold Rules

- The earlier `0.61` cutoff was exploratory.
- Current preferred train-calibrated TorchXRayVision cutoff is `0.62`.
- `0.62` came from train-split calibration where best `F1` and best `Youden's J` selected approximately `0.62`.
- For final reporting, threshold calibration should be described as validation/train-derived and frozen before held-out evaluation.
- Report multiple operating points where useful: best `F1`, best `Youden's J`, and high-sensitivity threshold.
- At threshold `0.62`, the test split had only `40` false negatives in a prior full test screen, so `50-100` test-only `FN` examples are impossible without changing threshold or using a broader split.
- `--split any` can provide more outcome diversity but mixes train and test rows; label it exploratory/consolidation, not final held-out reporting.

## XAI Method Set and Semantics

Pre-refactor (v1, current as of 2026-05-18 pre-Phase-1.2) integrated XAI methods are `grad_cam`, `grad_cam_negative`, `grad_cam_plus_plus`, `grad_cam_plus_plus_negative`, `integrated_gradients`, `integrated_gradients_positive`, `integrated_gradients_negative`, `integrated_gradients_signed`, `gradient_shap`, `gradient_shap_positive`, `gradient_shap_negative`, `gradient_shap_signed`, `occlusion`, `occlusion_positive`, `occlusion_negative`, and `consensus`. These names appear in current `outputs/iter_*` CSVs and overlays.

Post-refactor (v2, after Phase 1.2 of `docs/refactor_plan.md` lands) the primary method ids collapse to one entry per family: `grad_cam`, `grad_cam_plus_plus`, `integrated_gradients`, `gradient_shap`, `occlusion`, `consensus`, plus the planned `eigen_cam` and `score_cam`. **Consensus constituents are frozen.** `consensus` (and its signed variant `consensus_signed`) averages the four original methods (`grad_cam`, `integrated_gradients`, `gradient_shap`, `occlusion`). The planned `eigen_cam` and `score_cam` are reported as additional individual methods and are deliberately NOT added to the consensus average; redefining consensus mid-thesis would invalidate prior consensus results and break cross-iteration comparability. The `consensus_attention` variant raised in the 2026-05-18 supervisor sync (an attention-weighted consensus where each method receives a per-method coefficient `alpha_m`, NOT architectural attention inside the classifier) is **out of scope for the 2026-06-04 draft** and is recorded under thesis Chapter 5.3 "Future Work" in `thesis/thesis_skeleton.md`. The polarity-suffix names (`*_positive`, `*_negative`) retire as standalone method ids and become view selectors on a single `SignedAttribution` per case. The four views (`positive`, `negative`, `magnitude`, `signed`) are derived from the same underlying tensor; the `polarity=` keyword in `xai.py` is removed. The new view-suffixed overlay/metric outputs are `grad_cam_magnitude`, `grad_cam_signed`, `grad_cam_plus_plus_magnitude`, `grad_cam_plus_plus_signed`, `integrated_gradients_signed` (kept), `gradient_shap_signed` (kept), `occlusion_signed`, and `consensus_signed`. Cross-method `agreement_score` (cosine similarity between signed maps) is reported when more than one signed-capable method is run on the same case.

Versioning discipline: v1 outputs are never overwritten or renamed. v2 outputs live alongside v1 in new `outputs/iter_XX_*` folders. Held-out evaluation after Phase 1.2 uses only v2.

`Eigen-CAM` and `Score-CAM` were added to the planned method set on 2026-05-18 to close the corresponding gap against `docs/experiment_protocol.md`. Each registers as a `MethodSpec` entry with the same four-view `SignedAttribution` contract. `Score-CAM` accepts a `--score-cam-channels-cap` flag so broad screening runs can subsample activation channels for speed while thesis-quality reruns use the full set.

## Modality Coverage and Pipelines

- Primary pipeline: CXR pneumothorax under `src/explainai_thesis/cxr/` and `scripts/run_cxr_torchxray_smoke.py`. This is the documented primary experiment.
- Secondary pipeline (planned, off-the-shelf model approach decided 2026-05-18): head CT intracranial hemorrhage under `src/explainai_thesis/ct/` and `scripts/run_ct_smoke.py`. CT uses HU windowing rather than 0-255 grayscale normalization, so faithfulness baselines must be re-chosen for CT; a `--faithfulness-baseline soft_tissue_window_zero` option will be added for CT runs. Masks for CT come from a small student-annotated positive subset stored only in `data_local/`. No CT model fine-tuning is planned in scope; an off-the-shelf public CT classifier (e.g., a public RSNA-IHD pretrained model) is the primary candidate.
- Shared layer between modalities: `MethodSpec` registry, the XAI methods themselves, faithfulness curves, calibration, localization metrics, and radiologist review tooling. Only IO and model loading are modality-specific.

## Radiologist Review Workflow

- Tooling: `scripts/build_review_workbook.py` (planned 2026-05-18) generates a static review folder per smoke run under `<run>/review/`.
- Outputs: `index.html` (one card per case, embedded four-view overlay grid per method, four-rubric scoring guide, seven-category failure taxonomy inline), `scores_template.csv` (prefilled `case_id` and `filename` columns), and `INSTRUCTIONS.md` (open `index.html` in a browser, fill `scores.csv` in an editor, save next to the template, do not edit the template itself).
- Scoring schema (matches `docs/experiment_protocol.md`):
  - `localization_score` ∈ `{correct, partial, incorrect, none}`.
  - `usefulness_score` ∈ `{useful, potentially_useful, misleading, not_useful}`.
  - `failure_category` ∈ `{correct, partial, anatomically_related, devices_text_artifacts, non_pathological_high_contrast, diffuse_non_specific, clinically_misleading}`.
  - `artifact_note` free text.
  - `comment` free text.
- The student-as-radiologist scoring pass on the CXR consolidated run is the first protocol-aligned clinical-assessment dataset for the thesis. CT cases will use the same schema once Phase 5.4 lands.
- Rubric clarity is the binding constraint for keeping scoring at ~2 minutes per case across the 100-case CXR set (rater's typical CXR reading time is 1-3 minutes). The workbook must make rubric application automatic and never require scrolling, tab-switching, or re-reading `INSTRUCTIONS.md` mid-pass. Cards inline the full 4-rubric definitions and 7-category failure taxonomy with 1-2 example sentences per option, show the ground-truth mask plus the overlay grid (one row per method, four views per row in red/blue/violet/orange-teal), display the classifier outcome (`tp`/`fp`/`tn`/`fn`) and probability, and are keyboard-navigable next/prev. `INSTRUCTIONS.md` includes a 3-case warmup section to anchor the rubric before the real scoring pass.

## Improvement Experiment Discipline

- Formalized as `scripts/run_improvement_experiment.py` (planned 2026-05-18).
- Pipeline: calibrate per-method top-fraction thresholds on the validation split using `scripts/calibrate_cxr_xai_thresholds.py`, freeze thresholds, evaluate individual methods plus `consensus` plus `consensus_signed` on the held-out test split, emit `improvement_experiment.csv` (per case per method) and `improvement_experiment_paired.csv` (Wilcoxon signed-rank consensus vs each individual) plus Chapter 4 box-plot figures.
- Discipline rule baked into the script: refuses to run if any calibration artifact is older than the test smoke it would compare against. This protects the validation/test split contract against accidental peeking.

### Color Semantics

- `Red`: positive attribution/evidence toward the pneumothorax model output.
- `Blue`: negative attribution/evidence against the pneumothorax model output.
- `Violet`: neutral magnitude/absolute impact, especially for IG/GradientSHAP magnitude maps; it is not positive or negative evidence.
- `Green`: ground-truth mask contour or missed mask area depending on selected-image type.
- `Yellow`: selected positive evidence intersecting the ground-truth mask.
- `Cyan` / blue-green: selected negative evidence intersecting the ground-truth mask.
- `Orange`: positive side of a signed scalar map (`positive - negative > 0`), used for `integrated_gradients_signed` and `gradient_shap_signed` overlays. Distinct from `red` so a reader does not confuse a signed-difference map with a pure positive-evidence map.
- `Teal`: negative side of a signed scalar map (`positive - negative < 0`), paired with `orange` on the same overlay. Distinct from `blue` for the same reason.

### Thesis-Safe Interpretation

- Do not describe heatmaps as direct pathology segmentations or generic attention maps.
- Red regions indicate image areas contributing positively to the model's pneumothorax output; blue regions indicate areas contributing negatively to the same output.
- These maps visualize class-specific attribution with respect to the selected pneumothorax target score and should not be interpreted as anatomical segmentations.
- For signed `Grad-CAM` and signed `Grad-CAM++`, red/blue positive/negative evidence semantics are relatively clean.
- For `Integrated Gradients` and `GradientSHAP`, sign interpretation is useful but should be discussed carefully because baseline choice, sampling, and pixel-level noise matter.
- Magnitude maps answer “which pixels were impactful,” not “which pixels supported pneumothorax.”
- For `integrated_gradients_signed` and `gradient_shap_signed`, the signed scalar is computed as `positive_attribution - negative_attribution` on the same case and rendered with the `orange`/`teal` diverging pair. This is a tug-of-war view: it shows which side wins per pixel after the magnitude/positive/negative views are already inspected separately. It should not replace the dedicated positive (red), negative (blue), and magnitude (violet) overlays; it is an additional fourth view.

## Metrics Interpretation

- Positive localization metrics include `IoU`, `Dice`, `pointing_hit`, and `precision_at_fraction`.
- Positive localization metrics are most meaningful for positive evidence maps and positive-label cases with masks.
- Negative evidence should not be evaluated as if overlap with the lesion were good.
- For negative evidence, overlap with the ground-truth lesion can be concerning, while avoidance of the lesion is often preferable.
- Negative diagnostics include `negative_mask_overlap_fraction` and `negative_mask_avoidance_fraction`.
- `negative_mask_avoidance_fraction` means the selected negative/suppressive evidence stays outside the lesion mask; higher avoidance is usually better for suppressive evidence.
- A method can be faithful to the model while clinically poorly localized. This distinction is thesis-important.

## Faithfulness Evaluation Rules

- Deletion/insertion curves re-evaluate the actual `TorchXRayVision` pneumothorax probability after perturbing input pixels according to each attribution ranking.
- Insertion starts from a baseline image and restores top-attributed pixels; good explanations should restore probability quickly.
- Deletion starts from the original image and removes/replaces top-attributed pixels; good explanations should reduce probability quickly.
- Faithfulness evaluates model behavior, not clinical correctness.
- Use `--faithfulness-baseline black` for current faithfulness runs unless intentionally comparing baselines.
- Historical `zero_tensor` baseline was misleading because it was not a true black image in the normalized TorchXRayVision input space and could still score around `60%` pneumothorax.
- Keep both full-scale and zoomed/family-split faithfulness plots where possible.
- All comparable faithfulness plots should use the same scale when the intent is visual comparison; otherwise they can mislead about degree of probability change.

## GradientSHAP Stability Rules

- `GradientSHAP` is stochastic and depends on baseline/noise samples.
- Low sample counts are useful for speed but can produce unstable, contradictory maps.
- `--gradshap-samples 8` is acceptable for broad exploratory screening only.
- Clinically important, contradictory, or thesis-worthy cases should be rerun with higher-stability settings such as `--ig-steps 16 --gradshap-samples 64`.
- For very important cases, consider `--gradshap-samples 128` if `64` still changes materially.
- Prior `case_019` showed that low-sample `GradientSHAP` worsened attribution quality; at `64` samples, `Grad-CAM++` and `GradientSHAP` no longer appeared contradictory.

## Occlusion Rules

- Occlusion sensitivity is expensive.
- Broad exploratory runs may use coarse settings such as `--occlusion-patch-size 56 --occlusion-stride 56`.
- Targeted diagnostic/thesis cases may use finer settings such as `--occlusion-patch-size 32 --occlusion-stride 12`.
- Occlusion answers a different question from gradient attribution: how model probability changes under direct patch perturbation.
- It may localize a clinically plausible region even when insertion/deletion curves show small probability changes; interpret as complementary evidence, not a contradiction by default.

## Calibration Rules

- Classifier threshold calibration and XAI heatmap threshold calibration are separate.
- Do not tune thresholds on final held-out test results.
- XAI threshold calibration should use positive masked calibration cases.
- Current useful fraction sweep is `0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95`.
- Keep selected thresholds/fractions by metric, not only one global fraction.
- Dice/IoU selection is for positive localization.
- Negative avoidance selection is for suppressive/negative evidence diagnostics.
- Calibration versioning (added 2026-05-18): the `SignedAttribution` refactor changes the underlying signed maps for Grad-CAM, IG, GradientSHAP, and Occlusion, which makes pre-refactor calibrated top-fractions statistically stale. After the refactor, a `v2` calibration must be produced and stored at `outputs/iter_XX_calibration_v2/calibrated_thresholds_v2.csv`. The `v1` files are never overwritten or renamed; both versions coexist. Downstream scripts pick the version explicitly via `--calibrated-fractions`. Only `v2` calibration is used for held-out evaluation after the refactor.

## Long-Run Classifier-Outcome Workflow

- Balanced outcome runs are not hand-picked.
- The script shuffles candidate rows with a fixed seed, classifies each case at the classifier threshold, and keeps cases until each outcome group reaches `--max-per-outcome`.
- Same dataset, same seed, same threshold, and same script version should be reproducible.
- Current broad long-run command pattern:

```powershell
wsl.exe python3 scripts/visualize_cxr_classifier_outcome_thresholds.py --device auto --split any --max-cases 5000 --random-sample --seed 20260517 --threshold 0.62 --max-per-outcome 100 --ig-steps 8 --gradshap-samples 8 --occlusion-patch-size 56 --occlusion-stride 56 --fractions 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50 --progress-every 10 --checkpoint-every 1 --output-dir outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods
```

- For restart after stopping, use the same command and add `--resume`.
- Resume reads existing `cases.csv` and `threshold_metrics.csv` from `--output-dir`, reconstructs existing outcome counts, skips completed source images, continues with new `sample_index` values, and rewrites final CSVs without duplicate completed rows.
- Resume is safe only for runs made with the current checkpoint/resume-capable script. Runs started before checkpoint/resume support are not reliably resumable.
- Check for `cases.csv`, `threshold_metrics.csv`, and `progress.json` before relying on resume.

## Progress and Checkpointing Rules

- For long classifier-outcome runs, use `--progress-every 10 --checkpoint-every 1`.
- The classifier-outcome script uses a six-line live progress display to keep console output readable without losing important state.
- Checkpoint files are the durable record: `cases.csv`, `threshold_metrics.csv`, and `progress.json`.
- `tqdm` was intentionally not used first for the classifier-outcome run because the workload is outcome-balanced, stage-based, and dominated by slow selected-case XAI generation rather than simple candidate-loop progress.
- ETA should be estimated primarily from completed selected cases, not only scanned candidate rows.

## Large-Scale Evaluation Strategy

- Before adding another model, consolidate the current TorchXRayVision baseline with larger random/balanced runs.
- Use broad `8/8` IG/GradientSHAP settings for exploratory diversity discovery, but rerun selected thesis cases with stronger settings.
- A planned/desired consolidation run is at least `1000` cases or balanced `50-100` cases per `tp`/`fp`/`tn`/`fn` group where possible.
- Full all-method faithfulness can be slow; consider localization on broad sets, faithfulness on selected subsets, and high-stability reruns for representative cases.

## Stronger Second Model Needed

- The current TorchXRayVision `densenet121-res224-all` baseline should remain as a documented weak external baseline.
- But many inspected explanations look clinically questionable, with very low localization overlap.
- Future model candidates include CheXNet-style DenseNet, a more recent CXR model with pneumothorax output, a pneumothorax-specific pretrained model, or a custom/fine-tuned SIIM/Kaggle classifier.
- Any second model should go through the same protocol: classifier threshold calibration, XAI threshold calibration, localization metrics, negative evidence diagnostics, faithfulness curves, and qualitative `tp`/`fp`/`tn`/`fn` examples.

## Diagnostic A/B Protocol Before Full Second-Model Integration

- Before committing 1-2 days to a full second-model protocol run, a diagnostic A/B sweep across multiple TorchXRayVision pretrained weights answers the prior question: are the weak-localization results model-specific or method-specific.
- Decision rule from 2026-05-18: run Stage A (same library, different weights) first on the existing calibration positive cases. Candidate weights: `densenet121-res224-chex`, `densenet121-res224-mimic_ch`, `densenet121-res224-mimic_nb`, `densenet121-res224-rsna`, with `densenet121-res224-all` as the control.
- Stage A output folder convention: `outputs/iter_XX_diagnostic_weights_ab/<model_name>/`, with a top-level summary CSV `weights_ab_summary.csv` containing per-model mean IoU, Dice, pointing_hit, precision_at_fraction across methods.
- Pre-mortem revision (2026-05-18): Stage A now includes one out-of-family external model from the start, not as a Stage C contingency. Rationale: the five in-family TorchXRayVision DenseNet-121 weights share architecture, input size, normalization, and substantially overlapping training data (CheX, MIMIC, NIH), so a uniform within-family result would only rule out within-family variance, not training-distribution mismatch. The external candidate is selected at implementation time and integrated through the `load_classifier(name)` seam.
- Stage B outcome classification (refined with external-model context):
  - All in-family and external models look similarly poor: localization weakness is cross-distribution-stable; full second-model integration is not pursued; thesis frames results as methodological.
  - In-family similar but external materially better: training-distribution mismatch is the dominant factor; external model becomes co-primary baseline; full protocol is run on it.
  - One in-family weight is materially better than the others: within-family variance is meaningful; that weight is promoted to co-primary baseline.
  - Inconclusive or mixed: proceed to Stage C deeper external-model exploration.
- The outcome of the Stage A sweep is recorded back into this `AGENTS.md` section once the run completes, as durable thesis evidence.

## Diagnostic A/B Results (Stage A complete 2026-05-20)

- Working set evaluated: `densenet121-res224-{all, chex, mimic_ch, mimic_nb, nih, pc}` (six DenseNet-121 weights) and `resnet50-res512-all`. Auto-skipped from the orchestrator: `densenet121-res224-rsna` (no Pneumothorax head) and `resnetae-101-elastic` (no class head). The originally planned out-of-family MONAI Model Zoo CXR bundle was rejected after inspection because the only relevant bundle (`cxr_image_synthesis_latent_diffusion_model`) is generative, not a pneumothorax classifier; out-of-family coverage remains an unresolved Stage A gap.
- Calibration set: per-model v2 XAI threshold calibration on positive masked calibration cases (`outputs/iter_33_stage_a_diagnostic_ab/<model>/calibration_v2/calibrated_thresholds_v2.csv`); per-model classifier-threshold sweeps for best-F1, Youden's J, and high-sensitivity operating points; smoke + faithfulness on a fixed positive-cases set of `n=180` cases per model.
- Aggregate summary: `outputs/iter_33_stage_a_diagnostic_ab/weights_ab_summary.csv` with one row per model.
- Ranking by mean Dice / IoU on the positive view with Dice-selected calibrated fraction:
  - `resnet50-res512-all`: `mean_dice=0.0397`, `mean_iou=0.0221`, `mean_precision_at_fraction=0.0296` (best).
  - `densenet121-res224-chex`: `mean_dice=0.0284`, `mean_iou=0.0160`.
  - `densenet121-res224-all`: `mean_dice=0.0237`, `mean_iou=0.0130` (original baseline).
  - Remaining DenseNet weights cluster between these values.
- Pointing-hit is near-zero across all Stage A models (711/720 zero on DenseNet-all smoke; similar pattern across other models), confirming that maximum-attribution pixels rarely fall inside the lesion mask regardless of model choice.
- Stage B outcome classification: `resnet50-res512-all` is materially better than `densenet121-res224-all` in this Stage A aggregate but absolute localization remains weak across all tested weights. Per the Stage B refined decision rule, this is **within-family variance is meaningful + cross-distribution-stable weak absolute localization**: ResNet-50 is promoted to co-primary baseline for thesis follow-up (Phase 5.5), while the broader thesis framing remains methodological. The out-of-family Stage A slot remains open because no off-the-shelf pneumothorax-classifier-with-explicit-head was verified.
- Thesis-safe framing: ResNet-50 is the strongest tested TorchXRayVision candidate by aggregate localization. Do not describe it as clinically strong pneumothorax localization. The off-the-shelf model family does not produce lesion-aligned saliency on SIIM pneumothorax masks, and this finding is stable across architecture (DenseNet-121 vs ResNet-50) and pretraining-dataset variation (`all`, `chex`, `mimic_*`, `nih`, `pc`).

## Reporting/Thesis Framing

- Separate classifier performance, positive localization against masks, negative evidence diagnostics, faithfulness/deletion-insertion metrics, and qualitative case studies.
- Use method disagreement as evidence, not just as a problem.
- Thesis-safe core message: XAI maps are model-behavior diagnostics, not direct clinical segmentations. Agreement between methods is stronger evidence than any single heatmap, while disagreement can reveal model reliance on non-lesion or clinically questionable signals.
- Emphasize that pretrained medical models can be moderately predictive but clinically poorly localized; positive and negative evidence must be interpreted separately; mask localization and model faithfulness answer different questions; and explanation validity depends on model quality, preprocessing, thresholds, and perturbation baselines.

## Thesis Submission Format Requirements

Source of truth: `requirements/Шаблон пояснювальної записки дипломної роботи (академічне дослідження) та вимоги до її технічного оформлення .md` (Neoversity template, academic-research variant). Summary below covers the structural and formatting rules that an agent implementing thesis writing must observe; the canonical wording lives in the template.

### Required structural elements (in order)

1. Title page with Student ID, Thesis Supervisor, Co-Supervisor (if applicable), Date of Submission, and the standard copyright line. Not numbered.
2. Thesis Certification page (supervisor sign-off, standard template wording).
3. Declaration of Academic Integrity with start/end dates, AI-tool acknowledgment, candidate signature, and date. The AI-tool acknowledgment must list tools by name; this thesis declares `GPT-5.5`, `Codex`, `PyCharm`, `Junie`, `VS Code`, `Claude Sonnet 4.6`, and `Claude Opus 4.7`.
4. Acknowledgments (optional).
5. Table of Contents (ЗМІСТ) with page numbers, full nested chapter/subsection structure.
6. List of Tables (СПИСОК ТАБЛИЦЬ) — separate from figures, graphs, and charts. Format: 3-column table (Table No., Title, Page No.). Numbering "X.Y" by chapter.
7. List of Figures (СПИСОК РИСУНКІВ) — separate from graphs and charts. Same 3-column format. Numbering "X.Y".
8. List of Graphs (СПИСОК ГРАФІКІВ) — template explicitly distinguishes graphs (line plots) from figures and from charts. Faithfulness deletion/insertion curves and improvement-experiment box plots most likely qualify here.
9. List of Charts (СПИСОК ГРАФІКІВ) — bar charts, pie charts. Faithfulness-AUC bar chart and review-score distribution counts likely qualify.
10. List of Abbreviations (СПИСОК СКОРОЧЕНЬ) — alphabetical, optional if no abbreviations.
11. Abstract / Анотація: 250-300 words covering (1) goal + relevance, (2) methodology, (3) main results, (4) theoretical/practical significance, followed by 5-7 Keywords.
12. Chapter 1. Introduction — subsections 1.1 Research Context, 1.2 Problem Statement / Relevance, 1.3 Aim and Research Objectives, 1.4 Scientific Novelty and Practical Significance, 1.5 Thesis Structure, then **Conclusions to Chapter 1** (~0.5 page) — template requirement.
13. Chapter 2. Literature Review — subsections 2.1, 2.2, ..., then **Conclusions to Chapter 2** (~0.5 page) — template requirement.
14. Chapter 3. Methodology — template defines 3.1 General Approach, 3.2 Methods, 3.3 Justification of Choices, 3.4 Challenges and Ethical Aspects, then **Conclusions to Chapter 3** (~0.5 page) — template requirement. The thesis can include additional subsections beyond 3.4 if they map cleanly to the four required buckets.
15. Chapter 4. Results and Discussion — subsections 4.1 (results presentation), 4.2 Comparative Analysis, 4.3 Practical Applications and Limitations, then **Conclusions to Chapter 4** (~0.5 page) — template requirement.
16. Chapter 5. Conclusions and Recommendations — narrative chapter; template does not require numbered subsections but does not forbid them. Light subdivision (e.g., Main Findings / Practical Recommendations / Future Work) is acceptable.
17. Bibliography — APA or IEEE, consistent throughout. Minimum source count equals the number of pages in Chapters 2+3+4. Each online source must end with `[Online]. Available: <URL>. Accessed: <date>`. Sources sorted alphabetically by author or by reference order.
18. Appendices — labeled А, Б, В (Cyrillic) or I, II, III (Roman). For an English-language thesis, A, B, C (Latin) is conventionally acceptable but **confirm with the supervisor** before final submission. Each appendix has a separate title page and is referenced in the main text.

### Formatting (PDF submission)

- Format: PDF, A4 (210 × 297 mm).
- Margins: left 2.5 cm, top 2 cm, bottom 2 cm, right 1.5 cm. Applies to every page including appendices.
- Font: Times New Roman, black. Body 12 pt; chapter/subsection headings 14 pt bold; chapter headings in ALL CAPS; subsection headings sentence case (only first word capitalized).
- Line spacing: 1.5 for body text; 1.0 inside tables, figures, graphs, and charts.
- Paragraph indent: 1.25 cm.
- Alignment: title page center; chapter titles center; subsection titles left; body justified.
- Page numbering: arabic numerals, center-bottom. Title page not numbered; all subsequent pages including front-matter lists are numbered.
- Spacing between chapter heading and body: 2 blank lines. Between subsection heading and body: 1 blank line.

### Content limits

- Main text (Chapters 1-5) length: 25-50 pages.
- Direct quotes plus paraphrased borrowings: ≤ 20-25% of the main text. ≤ 5% from any single source.
- Abstract: 250-300 words. Keywords: 5-7 terms.
- Chapter conclusions ("Висновки до розділу"): ~0.5 page each.

### Captions and numbering

- Tables: caption **above** the table, left-aligned. Format: "Table X.Y: …" (e.g., Table 3.2). Every table must be referenced in body text ("As shown in Table 3.2 …").
- Figures: caption **below** the figure. Format: "Figure X.Y: …". Every figure must be referenced ("Results are visualized in Figure 4.1 …").
- Graphs and charts: same X.Y chapter-numbered scheme. Captions follow the same above-table / below-figure convention.
- Formulas: centered. Numbering on the right in parentheses, format (X.Y). Every formula must be referenced ("Computed by equation (2.3) …").
- Code in appendices: Courier New, 10 pt.

### Cross-link

- Thesis-prose paraphrases and references live in `docs/thesis-notes.md` and `docs/references.md`.
- Per-chapter outline lives in `thesis/thesis_skeleton.md` with TODOs marking each template-required section, including the four "Conclusions to Chapter N" sub-sections.
- Deviations from the template (e.g., glossary placement, Latin vs Cyrillic appendix labels, Chapter 3 expanded subsection count) are documented in `docs/progress.md` under the template-compliance audit entry.

## Housekeeping

- Do not delete old output folders unless explicitly requested.
- Do not renumber old output folders once referenced.
- Be careful with transient files such as `.output.txt`; if present, treat as a likely tool artifact unless the user says otherwise.
- When changing scripts, run at least WSL syntax checks: `wsl.exe python3 -m py_compile <changed_python_files>`.
- When changing `src/explainai_thesis/` modules or anything that affects script output schema, also run the test suite: `wsl.exe python3 -m pytest tests/ -v`.
- For documentation-only changes, no tests are required.