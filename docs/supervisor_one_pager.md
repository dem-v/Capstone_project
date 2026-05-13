# Thesis Proposal One-Pager

## Proposed Title

Cross-Modality Validation of Explainable AI Methods for Medical Image Classification: Chest X-ray Pneumothorax and Head CT Hemorrhage Case Studies

## Motivation

Deep learning classifiers can achieve strong diagnostic performance in radiology, but their explanations are often presented visually without rigorous validation. In clinical practice, an explanation is useful only if it highlights medically relevant image regions and does not mislead the clinician. This thesis evaluates whether common explainable AI methods localize pathology correctly and whether their behavior is consistent across X-ray and CT imaging tasks.

## Aim

To compare, validate, and clinically assess explainable AI methods for medical image classification using objective lesion masks and radiologist-centered qualitative review.

## Research Objectives

- Review recent literature on XAI in medical imaging.
- Build or reuse classification models for pneumothorax and intracranial hemorrhage detection.
- Generate explanations using several XAI methods.
- Validate explanation maps against lesion masks.
- Compare methods quantitatively and clinically.
- Test a small improvement strategy, such as consensus heatmaps or threshold calibration.

## Datasets

Primary dataset:
- Chest X-ray pneumothorax dataset with masks, preferably SIIM-ACR or a Kaggle derivative.

Secondary dataset:
- RSNA Intracranial Hemorrhage Detection or local anonymized head CT studies.
- Small positive subset manually masked by the student.

## Methods

Candidate classifiers:
- DenseNet, ResNet, EfficientNet, or suitable pretrained radiology models.

Explainability methods:
- Grad-CAM.
- Grad-CAM++.
- Eigen-CAM or Score-CAM.
- Integrated Gradients.
- SHAP/GradientSHAP.
- Occlusion sensitivity.

## Evaluation

Classification metrics:
- AUC, accuracy, sensitivity, specificity, F1-score.

Explanation localization metrics:
- IoU, Dice, pointing game/hit rate, precision-at-k.

Clinical evaluation:
- Radiologist scoring of localization correctness, usefulness, and misleading patterns.
- Failure taxonomy of explanation errors.

## Expected Contribution

The expected contribution is not a new diagnostic model, but a reproducible framework for validating and comparing explanation methods in radiology. The thesis will combine quantitative mask-based metrics with clinical expert judgment and test a simple approach for improving explanation reliability.

## Scope Control

The X-ray experiment is the primary benchmark. The CT experiment is a smaller pilot designed to preserve clinical relevance and test whether explanation-method findings transfer to CT.

## Timeline

- 2026-05-07 to 2026-06-04: experiments and full thesis draft.
- 2026-06-05 to 2026-06-21: corrections, formatting, plagiarism check, presentation, and defense.

