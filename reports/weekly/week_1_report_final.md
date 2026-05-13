# Weekly Progress Report 1

## Current Thesis Direction

Working title:

Cross-Modality Validation of Explainable AI Methods for Medical Image Classification: Chest X-ray Pneumothorax and Head CT Hemorrhage Case Studies

Primary experiment:
- Chest X-ray pneumothorax classification with mask-based explanation validation.

Secondary pilot:
- Head CT intracranial hemorrhage classification/localization with a small manually masked subset.

## Preliminary Problem Analysis

Medical image classifiers can provide useful diagnostic predictions, but visual explanations are often interpreted informally. A heatmap can look plausible while still highlighting the wrong region, a confounder, or a non-diagnostic image artifact. Therefore, this project treats explainability as an object of validation rather than as a visualization-only add-on.

The first practical problem is pneumothorax classification on chest X-rays, because the selected Kaggle dataset provides image-level labels and pixel masks. This allows both model performance and explanation localization to be evaluated. The secondary CT pilot will test whether the same explanation-evaluation logic can transfer to head CT hemorrhage.

## Project Goals

- Compare several explainability methods for medical image classification.
- Evaluate explanations quantitatively against lesion masks.
- Evaluate explanations qualitatively from a radiologist's point of view.
- Identify clinically relevant explanation failure modes.
- Test a simple improvement strategy, such as consensus heatmaps or threshold calibration.
- Prepare the work so it can later evolve toward radiology-aware LLM explanations or knowledge distillation.

## Initial Research Methodology / MVP

Initial MVP:
- a reproducible Python environment;
- synthetic smoke test proving that classification, heatmap generation, overlay export, and localization metrics work end to end;
- manifest builder for image/mask datasets;
- defined primary dataset and next model baseline.

Next experimental MVP:
- run a TorchXRayVision model or DenseNet-style chest X-ray baseline on the Kaggle pneumothorax dataset;
- generate first real-data Grad-CAM / Integrated Gradients / SHAP-style outputs;
- compare heatmaps against pneumothorax masks.

## Plans and Open Questions

Potential future directions:
- Explainable LLM that accounts for the radiological point of view.
- Knowledge distillation: smaller model with equivalent diagnostic quality.

Open questions:
- What will we get from XAI models?
- How will we work with this output?
- Which hypotheses should be prepared before testing?
- Next iteration should use a TorchXRay model and get first results against the Kaggle dataset with masks.

Planned answer:
- XAI methods will produce per-image, per-class heatmaps/attribution maps.
- These maps will be normalized, overlaid on images, thresholded, compared with masks, scored quantitatively, and reviewed clinically.
- The output will support both explanation validation and future work on radiology-aware natural-language explanations.

## Hypotheses Before Testing

- H1: Different XAI methods will produce different localization quality on the same classifier and dataset.
- H2: High classification performance will not guarantee good explanation localization.
- H3: Mask-calibrated thresholding or consensus heatmaps will improve localization compared with at least one individual explanation method.
- H4: Quantitative localization metrics will correlate with radiologist usefulness scores, but not perfectly.
- H5: XAI method rankings may differ between X-ray pneumothorax and CT hemorrhage tasks.

## Risks and Challenges

- CT pilot may require manual annotation time.
- Too many XAI methods may over-expand the scope.
- Local patient data use requires strict anonymization.

Why these risks matter:
- Without scope control, the thesis may become two separate medical AI projects instead of one explainability-validation study.
- Without mask-based validation, explanations remain subjective visual artifacts.

Planned mitigation:
- Start with the Kaggle pneumothorax dataset because it has ready masks.
- Keep CT as a secondary pilot until X-ray results are stable.
- Limit initial XAI methods to 3-4 methods before expanding.
- Use local clinical data only after anonymization and only if it does not delay the core benchmark.

## Mitigation Plan

- Complete the X-ray benchmark first because public masks make objective validation feasible.
- Keep CT as a small pilot unless the workflow becomes very smooth.
- Use IEEE citation style unless supervisor requests APA.
- Do not train large models from scratch unless needed.

## Plan for Next Week

- Configure Kaggle credentials or manually download the selected pneumothorax dataset.
- Build the real CXR image/mask manifest.
- Install/use TorchXRayVision or a DenseNet-style CXR baseline.
- Run baseline X-ray classifier on the Kaggle pneumothorax dataset.
- Generate first real-data Grad-CAM, Integrated Gradients, and SHAP/GradientSHAP-style maps.
- Compute first real-data explanation localization metric.
- Produce first results table for model prediction, XAI method, IoU, Dice, pointing hit, and precision-at-k.
- Build literature matrix.
- Draft Chapter 1 and start Chapter 2.
