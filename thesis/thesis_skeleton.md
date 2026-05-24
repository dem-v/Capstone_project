# Cross-Modality Validation of Explainable AI Methods for Medical Image Classification: Chest X-ray Pneumothorax and Head CT Hemorrhage Case Studies

Author: Dmytro Valantsevych

A Master's Thesis submitted to Neoversity in partial fulfillment of the requirements for the degree of Master of Science in Computer Science

## Abstract

TODO: 250-300 words. Include aim, methodology, main results, conclusions, and practical significance.

Keywords: explainable artificial intelligence, medical image classification, radiology, Grad-CAM, SHAP, pneumothorax, intracranial hemorrhage

## List of Abbreviations

AI - Artificial Intelligence

CNN - Convolutional Neural Network

CT - Computed Tomography

CXR - Chest X-ray

DL - Deep Learning

IHD - Intracranial Hemorrhage Detection

IoU - Intersection over Union

ML - Machine Learning

SHAP - SHapley Additive exPlanations

XAI - Explainable Artificial Intelligence

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

TODO: Cross-modality experimental study with primary X-ray benchmark and secondary CT pilot.

### 3.2 Datasets

TODO: Describe SIIM-ACR pneumothorax or selected derivative, and RSNA IHD/local head CT subset.

### 3.3 Preprocessing

TODO: DICOM/PNG loading, resizing, normalization, train/validation/test split, anonymization for local data.

### 3.4 Classification Models

TODO: Describe selected architectures and pretrained baselines.

### 3.5 Explainability Methods

TODO: Define the explanation methods and implementation libraries.

### 3.6 Explanation Validation Metrics

TODO: IoU, Dice, pointing game/hit rate, precision-at-k, and optional faithfulness metrics.

### 3.7 Radiologist-Centered Assessment

TODO: Define expert scoring categories and failure taxonomy.

### 3.8 Ethical and Practical Considerations

TODO: Anonymization, retrospective data use, no patient-identifiable examples, limitations.

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

