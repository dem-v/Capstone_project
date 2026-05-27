# Cross-Modality Validation of Explainable AI Methods for Medical Image Classification: Chest X-ray Pneumothorax and Head CT Hemorrhage Case Studies

Author: Dmytro Valantsevych

A Master's Thesis submitted to Neoversity in partial fulfillment of the requirements for the degree of Master of Science in Computer Science

## Abstract

TODO: 250-300 words. Include aim, methodology, main results, conclusions, and practical significance.

Keywords: explainable artificial intelligence, medical image classification, radiology, Grad-CAM, SHAP, pneumothorax, intracranial hemorrhage

## List of Abbreviations

AI - Artificial Intelligence

CLAIM - Checklist for Artificial Intelligence in Medical Imaging

CNN - Convolutional Neural Network

CT - Computed Tomography

CXR - Chest X-ray

DECIDE-AI - Developmental and Exploratory Clinical Investigations of Decision support systems driven by Artificial Intelligence

DL - Deep Learning

FN - False Negative

FP - False Positive

Grad-CAM - Gradient-weighted Class Activation Mapping

HU - Hounsfield Unit

IHD - Intracranial Hemorrhage Detection

IG - Integrated Gradients

IoU - Intersection over Union

ML - Machine Learning

ROC-AUC - Area Under the Receiver Operating Characteristic Curve

SHAP - SHapley Additive exPlanations

SIIM-ACR - Society for Imaging Informatics in Medicine - American College of Radiology

TN - True Negative

TP - True Positive

XAI - Explainable Artificial Intelligence

## Glossary of Methodological Terms

**Attribution map / heatmap** - A two-dimensional visualization derived from a trained classifier that assigns relative importance to image regions for a selected output score. In this thesis, attribution maps are class-specific evidence maps for the selected pneumothorax or hemorrhage target, not automatic anatomical segmentations.

**Positive evidence** - The part of a signed attribution map that contributes positively toward the selected target output. For the CXR pneumothorax experiments, positive evidence supports the model's pneumothorax score.

**Negative evidence** - The part of a signed attribution map that contributes negatively toward the selected target output. Negative evidence is interpreted as suppressive evidence against the target score and is not evaluated as clinically successful merely because it overlaps the lesion mask.

**Magnitude view** - The absolute-value view of a signed attribution map. It answers which pixels were influential, without stating whether their influence increased or decreased the target score.

**Signed view** - A diverging visualization of the normalized signed attribution map. It summarizes the local balance between positive and negative evidence but does not replace inspection of the separate positive, negative, and magnitude views.

**Localization validation** - Evaluation of whether high-attribution pixels spatially correspond to available lesion annotations, using metrics such as IoU, Dice, pointing-hit, and precision-at-fraction.

**Faithfulness validation** - Evaluation of whether the highlighted pixels materially affect the model's own output probability when they are removed, replaced, or inserted. Faithfulness measures model behavior, not clinical correctness.

**Top-fraction thresholding** - Conversion of a continuous heatmap into a binary selected region by keeping a fixed fraction of the highest-valued pixels. This makes heatmaps comparable to masks but introduces a threshold choice that must be calibrated or reported.

**Coverage saturation** - A secondary diagnostic describing how quickly top-fraction thresholding expands to cover nearly the whole image. The proposed `coverage_saturation_fraction_95` records the smallest swept top-fraction at which selected coverage reaches at least `95%`; lower values indicate more diffuse or saturated maps.

**Radiologist-centered review** - A structured qualitative assessment in which a medically trained reader scores explanation localization, usefulness, and failure mode categories under a fixed review task. In this thesis it is an early controlled assessment, not prospective clinical validation.

## Chapter 1. Introduction

### 1.1 Research Context

TODO: Explain the role of deep learning in radiology, the need for interpretability, and the clinical risk of visually plausible but misleading explanations.

### 1.2 Problem Statement and Relevance

TODO: Existing classifiers may achieve high performance, but the clinical reliability of their explanations remains uncertain. Explanation methods must be validated, not only displayed.

### 1.3 Aim and Research Objectives

Aim: To compare, validate, and clinically assess explainable AI methods for medical image classification across chest X-ray pneumothorax and head CT hemorrhage case studies.

Objectives:
- Review recent literature on XAI methods in radiology.
- Build or reuse classification models for pneumothorax and intracranial hemorrhage detection.
- Generate explanation maps using multiple XAI methods.
- Validate explanations against lesion masks using quantitative localization metrics.
- Perform radiologist-centered qualitative assessment of explanation usefulness and failure modes.
- Test a simple improvement strategy for explanation maps.

### 1.4 Scientific Novelty and Practical Significance

TODO: Emphasize cross-modality validation, mask-based explanation assessment, and radiologist-centered interpretation.

### 1.5 Thesis Structure

TODO: Briefly describe Chapters 2-5.

## Chapter 2. Literature Review

### 2.1 Literature Search Strategy

TODO: State databases and search criteria. Main English sources should be from 2016 or newer; main Ukrainian sources from 2021 or newer.

### 2.2 Deep Learning Classification in Radiology

TODO: Cover medical image classifiers, chest X-ray classification, and CT hemorrhage classification.

### 2.3 Explainable AI Methods for Image Classification

TODO: Cover Grad-CAM, Grad-CAM++, Score-CAM/Eigen-CAM, Integrated Gradients, SHAP/GradientSHAP, Occlusion, and LIME if used.

### 2.4 Validation of Explanations

TODO: Discuss localization metrics, faithfulness metrics, human/radiologist evaluation, and limitations of saliency maps.

### 2.5 Research Gap

TODO: Explain why comparing explanation methods with both objective masks and radiologist assessment is needed.

## Chapter 3. Methodology

### 3.1 General Research Design

This thesis uses an experimental validation design for post-hoc explainable AI methods in medical image classification. The central methodological question is not only whether a classifier predicts the correct image-level label, but whether the generated explanations are faithful to the classifier and clinically plausible with respect to available lesion evidence. The study therefore separates four linked but non-identical evaluation layers:

1. **Classifier behavior**: the image-level pneumothorax probability, classifier threshold, and resulting `tp`/`fp`/`tn`/`fn` outcome.
2. **Localization against annotations**: whether selected explanation regions overlap available lesion masks or place their maximum attribution inside the mask.
3. **Faithfulness to the model**: whether deleting or inserting highly attributed pixels changes the classifier probability in the expected direction.
4. **Human-centered usefulness and failure analysis**: whether a radiologist-style reviewer finds the explanation correct, partially useful, misleading, or clinically irrelevant under a defined task.

The primary case study is chest X-ray pneumothorax classification and explanation using the SIIM-ACR pneumothorax dataset and off-the-shelf TorchXRayVision classifiers. Pneumothorax is suitable because the task is clinically meaningful, the public challenge context includes image-level classification and segmentation framing, and positive cases provide masks that can be used as an approximate reference standard for localization-style evaluation.

The secondary case study is planned as a head CT intracranial hemorrhage pilot. It is intentionally described as a pilot because the CT workflow depends on the availability of a suitable off-the-shelf hemorrhage classifier and a small local set of annotated positive cases. CT is methodologically useful because it differs from CXR in image physics, preprocessing, and perturbation baselines: CT uses Hounsfield-unit windowing rather than 0-255 radiograph normalization. If a verified CT classifier is not available, the CT component is treated as limited qualitative or future-work validation rather than a full parallel experiment.

The methodology follows transparent-reporting principles from medical-imaging AI guidance such as CLAIM, TRIPOD+AI, QUADAS-AI, and DECIDE-AI. The practical implication is that the thesis reports the data source, local dataset snapshot, model source, preprocessing, split/calibration strategy, target output, XAI settings, thresholding choices, metrics, qualitative review task, and limitations together. Explanation maps are never described as direct pathology segmentations; they are model-behavior visualizations with respect to the selected target class.

### 3.2 Datasets

#### 3.2.1 Chest X-ray pneumothorax dataset

The primary dataset is the local SIIM-ACR pneumothorax chest X-ray snapshot stored under `data_local/cxr_pneumothorax/siim-acr-pneumothorax`. The working dataset state documented during the project contains `12,047` PNG images and `12,047` corresponding PNG masks, with `2,669` positive pneumothorax cases and `9,378` negative cases. These counts must be recomputed from the final local snapshot before final thesis submission, because the methodology and results should cite the exact data actually used.

The dataset is used in two complementary ways. First, image-level labels are used to screen classifier performance and define outcome groups (`tp`, `fp`, `tn`, `fn`) at a frozen classifier threshold. Second, positive masks are used for localization evaluation of explanation maps. Negative images do not have lesion masks; therefore, they are mainly used for classifier-outcome visualization, false-positive analysis, and qualitative review rather than Dice/IoU localization against a pneumothorax mask.

The project distinguishes calibration, exploratory, and held-out use. Classifier-threshold calibration and XAI top-fraction calibration must not be tuned on final held-out test results. Current CXR work uses a train-calibrated TorchXRayVision pneumothorax cutoff of approximately `0.62`, selected because it was the best or near-best operating point for F1 and Youden's J in the calibration analysis. Outcome-balanced review sets may use `--split any` for exploratory diversity, but any such result must be labelled exploratory because it can mix train and test cases.

#### 3.2.2 Head CT hemorrhage pilot dataset

The planned CT pilot uses local head CT data with a small student-annotated positive subset. Because the CT component is not yet at the same maturity as the CXR pipeline, the methodology treats it as conditional. If a public pretrained intracranial hemorrhage classifier with a verified hemorrhage output is integrated, the CT pilot follows the same high-level protocol: load images, apply CT-appropriate windowing, generate explanations for the hemorrhage target score, compare against available masks where possible, and report qualitative limitations. If no usable classifier is verified, CT remains a future-work or qualitative extension and is not presented as a completed cross-modality validation.

### 3.3 Preprocessing

For CXR experiments, each source image is loaded as a grayscale radiograph and converted into the input format expected by the selected classifier. TorchXRayVision preprocessing maps an `H x W` uint8 array in the range `0..255` through `xrv.datasets.normalize(array, 255)` and returns a single-channel tensor with shape `[1, H, W]`. The tensor is then moved to the selected device. DenseNet-121 TorchXRayVision weights are native to `224 x 224` inputs, while the tested ResNet-50 TorchXRayVision weight `resnet50-res512-all` is native to `512 x 512`; when a script uses a different image size, this must be reported because resizing affects both classification and heatmap resolution.

Masks are loaded as binary reference images for positive pneumothorax cases. For mask-based metrics, heatmaps and masks are brought to the same spatial grid before thresholding and comparison. Continuous attribution maps are normalized only for visualization and selection; normalization does not make them segmentations.

For CT, preprocessing must not reuse CXR normalization. CT images require Hounsfield-unit handling and a clinically chosen window, such as a brain or soft-tissue window. Perturbation baselines also differ: replacing CXR pixels with black is not equivalent to replacing CT voxels with a meaningful HU value. Therefore, the CT pilot requires an explicit CT baseline such as `soft_tissue_window_zero` if faithfulness curves are computed.

All random sampling and balanced outcome selection use fixed seeds where implemented. Stochastic explanation methods, especially GradientSHAP, are reported with their sample count and noise settings. Broad exploratory runs may use faster settings, while thesis-quality selected cases are rerun with higher-stability settings.

### 3.4 Classification Models

The CXR classifiers are loaded off-the-shelf from TorchXRayVision. The original baseline is `xrv.models.DenseNet(weights="densenet121-res224-all")`, used as an external pretrained medical classifier without local modification, fine-tuning, or forking. This distinction is essential: weak SIIM pneumothorax localization should not be interpreted as failure of a locally optimized pneumothorax segmenter, but as evidence about transfer behavior of a pretrained image-level classifier.

The CXR pipeline now uses a classifier-loading seam, `load_classifier(name, device)`, which returns the model, Grad-CAM target layer, pneumothorax class index, and preprocessing function. This allows model comparisons without changing downstream XAI code. Current TorchXRayVision candidates include DenseNet-121 weights such as `densenet121-res224-all`, `densenet121-res224-chex`, `densenet121-res224-mimic_ch`, `densenet121-res224-mimic_nb`, `densenet121-res224-rsna`, `densenet121-res224-nih`, and `densenet121-res224-pc`, plus the architecturally different `resnet50-res512-all` classifier. The DenseNet Grad-CAM target layer is `model.features.denseblock4`; for the TorchXRayVision ResNet-50 wrapper, the target layer is `model.model.layer4`.

The diagnostic model-comparison protocol treats `densenet121-res224-all` as the control and asks whether poor localization is stable across pretrained CXR weights or improves materially for another model. A later Stage A sweep selected `resnet50-res512-all` as the strongest tested TorchXRayVision follow-up candidate by aggregate localization, but absolute overlap remained low; therefore it should be framed as a relative model-side improvement, not as clinically sufficient localization.

The target output for all CXR explanations is the model's `Pneumothorax` head. If a candidate model lacks a pneumothorax class head, it is rejected for this pipeline rather than silently attributing another class. Autoencoder-style weights are not valid classifiers for this methodology unless a separate reconstruction or feature-analysis experiment is explicitly defined.

### 3.5 Explainability Methods

The explanation methods are post-hoc methods applied to an already trained classifier. They are generated with respect to the selected target score and are therefore class-specific attribution maps. The current v2 CXR method registry contains the following primary method families:

- **Grad-CAM** (`grad_cam`): a class-discriminative activation-map method using gradients flowing into a final convolutional layer.
- **Grad-CAM++** (`grad_cam_plus_plus`): a CAM-family variant with modified activation-map weighting, included as a comparator rather than assumed to be universally superior.
- **Integrated Gradients** (`integrated_gradients`): a baseline-dependent path-integration attribution method. The number of integration steps and baseline choice must be reported.
- **GradientSHAP** (`gradient_shap`): a stochastic additive-attribution approximation related to SHAP concepts. The number of samples, noise standard deviation, and stability settings must be reported.
- **Occlusion sensitivity** (`occlusion`): a perturbation method that replaces local patches and measures the target-score response. Patch size, stride, and replacement baseline must be reported.
- **Consensus** (`consensus`): a sign-aware average of available signed attribution maps, used to test whether cross-method agreement produces a more stable explanation.

The project moved from a legacy v1 naming scheme with separate polarity-suffixed method IDs, such as `integrated_gradients_positive` and `gradient_shap_negative`, to a v2 `SignedAttribution` contract. In v2, each method computes one normalized signed map and derives four views from it:

- `positive`: non-negative evidence supporting the target score;
- `negative`: suppressive evidence against the target score;
- `magnitude`: absolute attribution strength, independent of direction;
- `signed`: the normalized signed map used for diverging overlays and signed diagnostics.

This design avoids recomputing the same method separately for each polarity and makes the interpretation of positive, negative, magnitude, and signed outputs explicit. For pixel-level methods such as Integrated Gradients and GradientSHAP, optional smoothing may be applied for review readability; if smoothing is used, metrics should be computed from the same maps shown to the reviewer so that visualization and quantitative evaluation remain aligned.

Future or planned CAM-family extensions include Eigen-CAM and Score-CAM. Score-CAM is expected to be slower because it weights activation maps by additional forward passes. Unless these methods are actually run in the final experiment, they should be described as planned or future extensions rather than as completed evidence.

### 3.6 Explanation Validation Metrics

The validation protocol intentionally combines multiple metrics because explanation quality has no single ground truth. Mask-overlap, peak-localization, negative-evidence, faithfulness, and review-score measures answer different questions.

#### 3.6.1 Top-fraction selection

Continuous heatmaps are converted to binary selected regions using top-fraction thresholding. Given a fraction `f`, the pipeline keeps the highest-valued `f` proportion of pixels after normalization. The selected region is then compared with the binary lesion mask. The fraction can be fixed, swept across values, or selected from calibration. Current useful sweep values are `0.05` to `0.95` in increments of `0.05`. Broad visualization sweeps stop after selected-image coverage reaches approximately `0.95` to avoid redundant whole-image panels.

#### 3.6.2 Positive localization metrics

For positive-label cases with masks, the primary localization metrics are:

- **Intersection over Union (IoU)**: the intersection of selected explanation pixels and mask pixels divided by their union.
- **Dice score**: twice the intersection divided by the sum of selected pixels and mask pixels.
- **Pointing hit**: `1` if the maximum-attribution pixel lies inside the lesion mask, otherwise `0`.
- **Precision at fraction**: the proportion of selected top-fraction pixels that fall inside the mask.

IoU, Dice, and precision-at-fraction primarily measure overlap. Pointing hit measures whether the single strongest attribution point falls inside the lesion and should not be treated as interchangeable with overlap metrics. Current project evidence suggests that overlap metrics can be highly correlated with each other while pointing hit captures a stricter peak-localization property.

#### 3.6.3 Negative-evidence diagnostics

Negative evidence is not evaluated as if lesion overlap were automatically desirable. For a negative view, selected pixels inside the pneumothorax mask can mean that the model is treating lesion-region information as suppressive evidence, which may be clinically concerning. The pipeline therefore reports:

- **negative mask overlap fraction**: the fraction of selected negative-evidence pixels inside the lesion mask;
- **negative mask avoidance fraction**: `1 - negative_mask_overlap_fraction`, interpreted as how much selected negative evidence avoids the lesion.

These metrics are diagnostics rather than universal correctness scores. They are interpreted together with the classifier outcome, positive localization, and visual review.

#### 3.6.4 Faithfulness curves

Faithfulness is evaluated by perturbing the actual model input according to each attribution ranking and re-evaluating the classifier probability. In a deletion curve, pixels are removed or replaced from the original image in descending attribution order; a faithful positive explanation is expected to reduce the target probability quickly. In an insertion curve, pixels are restored into a baseline image in descending attribution order; a faithful explanation is expected to recover the target probability quickly.

For current CXR faithfulness runs, the preferred baseline is a true black-image baseline in the normalized TorchXRayVision input space. Earlier `zero_tensor` baselines were found to be misleading because they could still produce high pneumothorax probabilities. Faithfulness curves are reported as model-behavior tests: a method may be faithful to the classifier while remaining clinically poorly localized against the pneumothorax mask.

#### 3.6.5 Spatial saturation and agreement diagnostics

The proposed secondary coverage metric, `coverage_saturation_fraction_95`, records the smallest swept top-fraction at which the selected explanation region covers at least `95%` of the image. It is used only as a spatial diffuseness or saturation diagnostic. Lower values indicate that the thresholded map becomes whole-image-like quickly; higher values indicate that attribution remains more spatially concentrated over the tested fractions. It does not decide whether positive or negative evidence is clinically correct.

Signed-capable runs may also report method-agreement diagnostics, such as cosine similarity between signed maps. Agreement between methods is treated as stronger evidence than a single overlay, while disagreement is useful for detecting instability, baseline sensitivity, or reliance on non-lesion cues.

### 3.7 Radiologist-Centered Assessment

The radiologist-centered component is implemented as a static review workbook generated from selected rendered cases. The workbook is designed for a controlled thesis-scale review task rather than for clinical deployment. Each case card shows the source CXR, ground-truth mask where available, classifier outcome and probability, and a grid of explanation overlays. Current review workbooks include method rows and four views per method so that positive, negative, magnitude, and signed evidence can be inspected separately.

The scoring schema contains four fields:

- `localization_score` in `{correct, partial, incorrect, none}`;
- `usefulness_score` in `{useful, potentially_useful, misleading, not_useful}`;
- `failure_category` in `{correct, partial, anatomically_related, devices_text_artifacts, non_pathological_high_contrast, diffuse_non_specific, clinically_misleading}`;
- free-text `artifact_note` and `comment` fields.

The review design follows human-centered XAI principles: the intended user is a medically trained reviewer, the task is explanation assessment rather than diagnosis, and all scoring categories are visible in the workbook to reduce context switching. The workbook includes instructions and warmup guidance so that scoring is anchored before the main review pass. Classifier-outcome-balanced selections preserve `tp`, `fp`, `tn`, and `fn` examples where possible, because explanation failures can differ across successful predictions, false alarms, and missed positives.

Review scores are interpreted as structured qualitative evidence. They do not replace quantitative localization or faithfulness metrics, and a single-reader review is not presented as prospective clinical validation. The review is nevertheless important because masks do not capture every clinically relevant pattern: devices, drains, subcutaneous emphysema, image markers, and indirect signs can be clinically related or confounding even when they do not overlap the pneumothorax mask.

### 3.8 Ethical and Practical Considerations

The experiments use public or local retrospective medical-imaging data and do not alter patient care. Public CXR data are treated as de-identified challenge data, and local data are stored under `data_local/` rather than committed as source artifacts. Generated figures should avoid patient-identifiable metadata and should use source image stems only for traceability within the experiment folders.

The study is not a clinical deployment evaluation. The models are off-the-shelf classifiers, the masks are used as available reference annotations rather than perfect clinical truth, and the radiologist-centered workbook is an early structured review artifact. Results should therefore be framed as evidence about explanation behavior, model transfer, and validation methodology, not as a recommendation for automated diagnosis.

Several practical risks are explicitly controlled in the methodology. Classifier thresholds and heatmap fractions are calibrated separately. Final held-out evaluation must not tune on test results. Old output folders are versioned and not overwritten; v1 and v2 attribution outputs are kept separate because the signed-attribution refactor changes the statistical meaning of maps and calibrated fractions. Long exploratory runs use checkpointed output files so that case selection and metrics remain reproducible.

The software environment is part of the methodology. Python runs are performed in WSL Ubuntu, with the project installed editably from `pyproject.toml`; current documented environment notes include Python `3.10.12`, CUDA-capable PyTorch, and `torchxrayvision==1.4.0`. The final thesis should disclose AI/development tools according to institutional policy, including PyCharm/Junie and other assistants used during the research workflow.

### 3.9 Implementation and Reproducibility

The CXR pipeline is organized around reusable scripts and package modules. The main classifier-screening script is `scripts/evaluate_cxr_torchxray_model.py`. The main CXR XAI run script is `scripts/run_cxr_torchxray_smoke.py`. XAI top-fraction calibration is performed by `scripts/calibrate_cxr_xai_thresholds.py`. Single-case threshold diagnostics use `scripts/visualize_cxr_threshold_selection.py`, and outcome-balanced visualizations use `scripts/visualize_cxr_classifier_outcome_thresholds.py`. Review workbooks are generated with `scripts/build_review_workbook.py`.

Experiment outputs are stored under ordinal folders such as `outputs/iter_XX_<short_experiment_name>`. Root-level CSV summaries are kept at the run root, and image artifacts are grouped by case so copied PNG files remain traceable. For classifier-outcome visualizations, top-level folders preserve `tp`, `fp`, `tn`, and `fn` grouping.

When Python source files or output schemas change, syntax checks and relevant tests are run through WSL. The canonical local test command is `wsl.exe python3 -m pytest tests/ -v`; for script-only changes, at least `wsl.exe python3 -m py_compile <changed_python_files>` is required. Documentation-only changes do not require tests, but the Markdown diff should still be reviewed for scope and thesis-safe wording.

## Chapter 4. Results and Discussion

### 4.1 Classification Performance

TODO: Present baseline metrics for X-ray and CT.

Draft notes from current CXR experiments:
- Treat `densenet121-res224-all` as the original weak external TorchXRayVision baseline. It is useful because it demonstrates that an off-the-shelf medical classifier can have moderate ranking/classification behavior while still producing clinically weak pneumothorax localization.
- Stage A TorchXRayVision model comparison selected `resnet50-res512-all` as the strongest tested follow-up candidate by localization aggregate, not as a clinically strong model. In `outputs/iter_33_stage_a_diagnostic_ab/weights_ab_summary.csv`, `resnet50-res512-all` had the highest mean Dice/IoU among tested TorchXRayVision candidates (`mean_dice=0.0397`, `mean_iou=0.0221`), but absolute mask overlap remained very low.
- Thesis-safe wording: ResNet-50 showed a relative improvement over DenseNet-all within this diagnostic sweep, but the result should not be interpreted as reliable lesion segmentation or clinically sufficient pneumothorax localization.

### 4.2 Quantitative Explanation Validation

TODO: Compare XAI methods using mask-based metrics.

Draft notes from current CXR experiments:
- Quantitative overlap metrics and peak-localization metrics answer different questions. Across Stage A all-model correlations (`outputs/iter_35_metric_correlations_iter33_stage_a_all_models/`), IoU, Dice, and precision-at-fraction were almost redundant, while pointing-hit was only weakly associated with those overlap metrics. Use pointing-hit as a strict peak-localization diagnostic, not as an interchangeable replacement for Dice/IoU.
- Signed diagnostics are conceptually separate from positive lesion overlap. Negative attribution should not be scored as successful because it overlaps the pneumothorax mask; for suppressive evidence, lesion avoidance is often more meaningful than lesion overlap.
- Faithfulness and localization must be separated in the discussion. Deletion/insertion curves test whether a heatmap explains the model behavior under perturbation, whereas mask metrics test agreement with annotated pathology location. A method can be faithful to the model while still clinically poorly localized.
- For pixel-level methods (`Integrated Gradients`, `GradientSHAP`), post-processing affects readability. The smoothed ResNet review run used `--pixel-attribution-mask-smoothing 9`, and metrics were computed from the same smoothed maps shown in the review workbook. Present this as a visualization/selection setting, not as a change in the classifier.

### 4.3 Cross-Modality Comparison

TODO: Compare whether the same explanation methods behave similarly on CXR and CT.

### 4.4 Radiologist Assessment and Failure Taxonomy

TODO: Present expert scoring and representative examples.

Draft notes from the 10-case smoothed ResNet review (`outputs/iter_45_resnet_review_analysis_smoothed/`):
- Review score distribution: localization was `correct` in 1/10 cases, `partial` in 6/10, and `incorrect` in 3/10. Usefulness was `useful` in 2/10, `potentially_useful` in 6/10, `misleading` in 1/10, and `not_useful` in 1/10.
- Failure taxonomy counts: `anatomically_related` 3/10, `devices_text_artifacts` 3/10, `partial` 3/10, and `clinically_misleading` 1/10. This supports a nuanced conclusion: explanations are often not random, but they frequently point to indirect or confounded evidence rather than cleanly to the annotated pneumothorax mask.
- Qualitative flags from the completed scoring sheet: devices/tubes were relevant in 7/10 cases, subcutaneous emphysema in 5/10, mask-quality caveats in 4/10, indirect evidence in 8/10, explicit method disagreement in 1/10, and weak or disagreeing IG/GradientSHAP behavior in 4/10.
- Important interpretation: device/tube and subcutaneous emphysema findings should not all be grouped as generic artifacts. Devices and ECG wires are confounders or treatment-related correlates; subcutaneous emphysema can be clinically related to pneumothorax but is not the same as direct mask localization.
- The manual review supports using `usefulness_score` separately from `localization_score`. Some low-overlap maps may still be useful for auditing model failure or identifying clinically adjacent evidence; conversely, a visually plausible map can be misleading if it emphasizes artifacts, bones, or non-lesion regions.
- The review workbook and figures should state that heatmaps visualize model attribution toward the selected pneumothorax output and should not be interpreted as anatomical segmentations.

### 4.5 Explanation Improvement Experiment

TODO: Present consensus heatmap or threshold-calibration results.

Draft notes from current improvement/visualization work:
- The top-fraction sweep now stops after selected-mask coverage reaches approximately full-image coverage (`--stop-fractions-at-coverage 0.95`). This avoids showing redundant high-fraction panels when lower fractions already cover the whole image and prevents over-interpreting visually saturated masks.
- Smoothed IG/GradientSHAP maps improved review readability, but the improvement is best framed as making pixel-level attribution more inspectable, not necessarily more clinically correct.
- Exploratory cross-case pattern analysis found moderately high visual cosine similarity for IG/GradientSHAP maps across the 10 review cases (`mean` roughly `0.53-0.55` across positive, negative, magnitude, and signed views). Higher cross-case similarity tended to associate with lower localization score in this small sample, with strongest observed Spearman around `rho=-0.54` on `n=10` for magnitude views. Treat this as a hypothesis-generating observation: pixel-level methods may sometimes show case-invariant or preprocessing-driven patterns, so qualitative review should check whether maps are case-specific.

### 4.6 Limitations

TODO: Dataset size, annotation limits, model choice, generalizability, heatmap limitations.

Draft limitations from current evidence:
- The radiologist-style ResNet review contains only 10 selected cases, so all review-score correlations and IG/GradientSHAP pattern correlations are exploratory and should not be presented as definitive statistical proof.
- SIIM mask quality affects quantitative conclusions. Cases with missing, incomplete, or clinically questionable masks can underestimate explanation quality if the model highlights plausible pathology-related evidence outside the annotation.
- The tested models are off-the-shelf TorchXRayVision classifiers, not fine-tuned pneumothorax segmenters. Low mask overlap may reflect model/data mismatch as much as explanation-method failure.
- The out-of-family MONAI candidate remains blocked because the checked MONAI CXR bundle is generative, not a pneumothorax classifier. Avoid claiming an external-family classifier comparison unless a real checkpoint with a verified `Pneumothorax` output is later integrated.
- Visualization choices matter: smoothing, selected top-fraction, color mapping, and perturbation baseline can change readability and interpretation. All thesis figures should report these settings.

## Chapter 5. Conclusions and Recommendations

### 5.1 Main Findings

TODO: Summarize results against objectives.

Draft conclusion bullets:
- Off-the-shelf CXR classifiers can produce explanations that are technically attributable to the model but clinically weak against pneumothorax masks.
- The strongest current tested TorchXRayVision candidate (`resnet50-res512-all`) improves over the original DenseNet-all baseline only relatively; qualitative review still shows frequent indirect evidence, device/tube confounding, and mask caveats.
- Agreement between methods is more trustworthy than any single heatmap. Disagreement, especially between CAM/Occlusion and IG/GradientSHAP, is itself useful diagnostic information.
- XAI maps should be framed as model-behavior diagnostics, not direct pathology segmentations.

### 5.2 Practical Recommendations

TODO: State which explanation methods appear safer/more useful and under what conditions.

Draft recommendations:
- Report positive evidence, negative evidence, magnitude, and signed views separately. Do not evaluate negative evidence as if lesion overlap is automatically good.
- Prefer combined quantitative and clinical review: Dice/IoU/precision-at-fraction, pointing-hit, signed diagnostics, and radiologist usefulness/failure categories capture complementary aspects.
- For IG/GradientSHAP, include explicit smoothing/readability settings and inspect for cross-case pattern similarity before using maps as case-specific clinical evidence.
- Preserve native image, mask, continuous views, and threshold sweeps together in review materials so qualitative scoring can distinguish direct lesion localization, indirect clinically related evidence, and artifacts.

### 5.3 Future Work

TODO: Larger CT masks, more pathologies, reader study, prospective validation, publication plan.

## Bibliography

TODO: Use IEEE by default unless supervisor requests APA.

## Appendices

### Appendix A. Experiment Configuration

TODO.

### Appendix B. Radiologist Review Template

TODO.

### Appendix C. Additional Figures

TODO.

