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

### 4.2 Quantitative Explanation Validation

TODO: Compare XAI methods using mask-based metrics.

### 4.3 Cross-Modality Comparison

TODO: Compare whether the same explanation methods behave similarly on CXR and CT.

### 4.4 Radiologist Assessment and Failure Taxonomy

TODO: Present expert scoring and representative examples.

### 4.5 Explanation Improvement Experiment

TODO: Present consensus heatmap or threshold-calibration results.

### 4.6 Limitations

TODO: Dataset size, annotation limits, model choice, generalizability, heatmap limitations.

## Chapter 5. Conclusions and Recommendations

### 5.1 Main Findings

TODO: Summarize results against objectives.

### 5.2 Practical Recommendations

TODO: State which explanation methods appear safer/more useful and under what conditions.

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

