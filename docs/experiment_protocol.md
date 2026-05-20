# Experiment Protocol

## Working Title

Cross-Modality Validation of Explainable AI Methods for Medical Image Classification: Chest X-ray Pneumothorax and Head CT Hemorrhage Case Studies

## Research Question

How well do common explainable AI methods localize clinically relevant abnormalities in medical image classification, and can a simple improvement strategy make explanations more reliable across X-ray and CT tasks?

## Study Design

Primary experiment:
- Chest X-ray pneumothorax classification.
- Dataset with lesion masks, preferably SIIM-ACR pneumothorax or a Kaggle derivative.
- Purpose: full quantitative validation of explanation maps.
- Protocol discipline: classifier-threshold calibration, XAI top-fraction calibration, and held-out evaluation are treated as separate steps. Thresholds are selected before held-out interpretation and are not tuned on final test outcomes.
- Model-comparison discipline: the current TorchXRayVision baseline is evaluated as an off-the-shelf clinical-imaging model, then compared against alternative off-the-shelf weights/models when available. A candidate external model is accepted only if its target label, preprocessing contract, checkpoint/license, and citation metadata are explicit.

Secondary pilot:
- Head CT intracranial hemorrhage classification/localization.
- Dataset: RSNA IHD or local anonymized head CT.
- Purpose: test transfer of explanation-method findings to CT.
- Manual masks: small positive subset annotated by the student as radiologist.

## Models

Candidate baseline models:
- DenseNet-121.
- ResNet-50.
- EfficientNet-B0 or B1.
- TorchXRayVision pretrained model for CXR if it fits the pneumothorax task.

Final selection rule:
- Prefer the model that can be trained or reused reproducibly within Week 1-2.
- Classification performance must be adequate but does not need to be state of the art; explanation validation is the central contribution.

## Explainability Methods

Initial method list:
- Grad-CAM.
- Grad-CAM++.
- Eigen-CAM or Score-CAM.
- Integrated Gradients.
- GradientSHAP / SHAP-style attribution.
- Occlusion sensitivity.

Optional:
- LIME, only if implementation time is low.

Signed-attribution convention:
- Each signed-capable method is interpreted through four derived views: positive evidence, negative evidence, magnitude/impact, and signed difference.
- Positive evidence means image regions contributing toward the selected pneumothorax class score; negative evidence means image regions suppressing that same score.
- Magnitude maps answer which pixels were influential, not whether they supported pneumothorax.
- Heatmaps are model-behaviour attributions, not direct pathology segmentations or generic attention maps.

## Quantitative Metrics

Classification:
- AUC.
- Accuracy.
- Sensitivity.
- Specificity.
- F1-score.

Explanation localization:
- IoU after heatmap thresholding.
- Dice after heatmap thresholding.
- Pointing game / hit rate.
- Precision-at-k or top-percent overlap.

Optional faithfulness:
- Deletion/insertion curve.
- Captum infidelity.
- Captum sensitivity.

Metric interpretation rules:
- IoU, Dice, and precision-at-fraction measure overlap between selected attribution regions and the lesion mask on positive masked cases.
- Pointing hit is a stricter peak-localization diagnostic: it is positive only if the single maximum-attribution location lies inside the lesion mask.
- Negative evidence is not scored as better when it overlaps the lesion; separate negative-overlap and negative-avoidance diagnostics are used to describe suppressive evidence.
- Faithfulness curves evaluate whether perturbing highly attributed pixels changes the model score as expected; this is model-behaviour evidence and is distinct from clinical localization.

## Radiologist Review

Each selected positive case should receive:
- localization score: correct, partial, incorrect, no meaningful localization;
- clinical usefulness score: useful, potentially useful, misleading, not useful;
- artifact/confounder note if present;
- free-text radiologist comment.

Failure taxonomy:
- correct lesion localization;
- partial lesion localization;
- attention outside lesion but in anatomically related area;
- attention on devices/text/crop/edge artifacts;
- attention on non-pathological high-contrast structures;
- diffuse non-specific heatmap;
- clinically misleading explanation.

Workbook protocol:
- A static review workbook is generated from selected cases and diagnostic visualizations.
- Each card shows the case metadata, classifier outcome/probability, ground-truth context where available, and the method overlay grid needed for scoring.
- The scoring CSV is kept separate from the immutable template so the review pass is reproducible and auditable.

## Improvement Experiment

Preferred low-risk improvement:
- Normalize explanation maps from top-performing methods.
- Combine them into a consensus heatmap.
- Calibrate threshold on validation masks.
- Compare consensus against individual methods on held-out test masks.

Alternative:
- Select the best explanation method per modality using validation metrics and test whether this selection improves held-out explanation performance.

## Data Handling

No patient data should be committed to the repository.

Local data must be:
- anonymized before use in figures;
- stored outside the repository or in ignored local folders;
- documented only as aggregate counts and non-identifiable examples.

## Minimum Success Criteria

By the end of Week 2:
- X-ray classifier baseline runs.
- At least three explanation methods run.
- At least one localization metric is computed against masks.

By the end of Week 3:
- CT pilot has at least a small annotated positive subset or is explicitly downgraded to qualitative external validation.

By the end of Week 4:
- Full thesis draft contains final tables, representative figures, and limitations.

