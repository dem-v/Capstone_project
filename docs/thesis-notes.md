# Thesis Research Notes

These notes are written as thesis-ready building blocks, not final thesis prose. Every claim should remain linked to a source in `docs/references.md`. Prefer paraphrase in the final thesis; use direct quotes only when the exact wording is needed and keep total quoted material low.

## How Other Works Frame Visual XAI for CNNs

### CAM-family methods are class-specific localization, not segmentation

- Source link: [`REF-GRADCAM`](references.md#ref-gradcam)
- Direct quote candidate: "Grad-CAM uses the gradients of any target concept, flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept."
- Thesis-ready paraphrase: Grad-CAM should be described as a class-discriminative localization method for CNN decisions. In this thesis, the heatmaps are therefore evidence maps for the selected pneumothorax output, not automatic pneumothorax masks.
- Practical pointer: In Chapter 3, explain that `grad_cam` and related methods are computed with respect to the pneumothorax target score, then resized/overlaid for human inspection and thresholded only for evaluation against masks.

### Grad-CAM++ is a method variant worth comparing, not a guaranteed improvement

- Source link: [`REF-GRADCAMPP`](references.md#ref-gradcampp)
- Thesis-ready paraphrase: Grad-CAM++ modifies the weighting of convolutional activation maps to improve visual explanations for deep CNNs. Its inclusion is justified as a related CAM-family method that may behave differently when evidence is spatially distributed or when multiple image regions contribute to the target class.
- Practical pointer: In results, avoid saying Grad-CAM++ is universally better. Report whether it improves pneumothorax mask overlap, pointing hit, or qualitative usefulness in this specific off-the-shelf CXR model.

### Integrated Gradients makes baseline choice central

- Source link: [`REF-INTEGRATED-GRADIENTS`](references.md#ref-integrated-gradients)
- Direct quote candidate: "We identify two fundamental axioms—Sensitivity and Implementation Invariance that attribution methods ought to satisfy."
- Thesis-ready paraphrase: Integrated Gradients has a principled attribution motivation, but the method integrates gradients along a path from a baseline image to the input. For medical images, the baseline is not a neutral technical detail: it affects the clinical meaning of positive, negative, and magnitude attribution.
- Practical pointer: When discussing `integrated_gradients`, explicitly state the baseline and number of integration steps. Explain that signed pixel attributions are interpreted cautiously because X-ray preprocessing and baseline selection can change the sign and magnitude of evidence.

### SHAP/GradientSHAP belongs to additive local attribution framing

- Source link: [`REF-SHAP`](references.md#ref-shap)
- Thesis-ready paraphrase: SHAP frames local explanations as additive feature attributions derived from Shapley-value ideas. The project’s GradientSHAP-style maps should therefore be discussed as local feature attribution approximations, not as direct causal proof that a pixel or region alone caused the diagnosis.
- Practical pointer: Use this source to motivate reporting stochastic settings (`gradshap_samples`) and rerunning thesis-quality cases with higher sample counts.

### Score-CAM is useful because it reduces reliance on raw gradients

- Source link: [`REF-SCORECAM`](references.md#ref-scorecam)
- Direct quote candidate: "Score-CAM gets rid of the dependence on gradients by obtaining the weight of each activation map through its forward passing score on target class."
- Thesis-ready paraphrase: Score-CAM is a relevant extension because it assigns activation-map weights using model output scores rather than gradient weights. This makes it a useful comparator when gradient-based maps appear noisy, saturated, or clinically implausible.
- Practical pointer: If Score-CAM is added to the pipeline, report runtime separately. It is expected to be slower because it requires additional forward passes.

### Occlusion sensitivity is a perturbation diagnostic, not a gradient map

- Source link: [`REF-OCCLUSION-ZEILER`](references.md#ref-occlusion-zeiler)
- Direct quote candidate: "We also perform a sensitivity analysis of the classifier output by occluding portions of the input image, revealing which parts of the scene are important for classification."
- Thesis-ready paraphrase: Occlusion sensitivity tests model behavior by replacing local image patches and measuring how the target probability changes. It therefore asks a different question from gradient attribution: not where the derivative is large, but where direct removal or replacement changes the model output.
- Practical pointer: In Chapter 3, report the occlusion patch size, stride, replacement baseline, and computational cost. In Chapter 4, interpret coarse occlusion maps cautiously because the result depends on patch geometry and may blur small pneumothorax margins.

### Implementation documentation should support, not replace, method citations

- Source link: [`REF-CAPTUM-DOCS`](references.md#ref-captum-docs)
- Thesis-ready paraphrase: Captum is the implementation layer for several attribution methods, so the methodology should cite original method papers for theory and Captum documentation for practical API semantics such as target selection, baselines, sampling, and returned attribution shapes.
- Practical pointer: Use Captum documentation when describing exact implementation settings for `integrated_gradients`, `gradient_shap`, and `occlusion`. This is especially useful for explaining why `GradientShap` is stochastic and why baseline/noise parameters are reported.

## How Other Works Evaluate Whether Explanations Are Trustworthy

### Visual plausibility alone is not enough

- Source link: [`REF-SANITY-CHECKS`](references.md#ref-sanity-checks)
- Direct quote candidate: "Reliance, solely, on visual assessment can be misleading."
- Thesis-ready paraphrase: A central methodological risk is that a heatmap can look plausible to a human while being weakly dependent on the trained model. This supports the thesis decision to combine mask-overlap metrics, faithfulness curves, method agreement, and radiologist review rather than relying on overlay appearance alone.
- Practical pointer: In the limitations section, acknowledge that full parameter-randomization sanity checks may be outside scope, but explain how the current protocol partially mitigates the risk using multiple complementary validation views.

### Deletion and insertion curves evaluate model behavior, not clinical correctness

- Source links: [`REF-RISE`](references.md#ref-rise), [`REF-SAMEK-EVALUATION`](references.md#ref-samek-evaluation)
- Direct quote candidate from RISE: "The deletion metric measures the drop in the probability of a class as important pixels (given by the saliency map) are gradually removed from the image."
- Thesis-ready paraphrase: Perturbation-based faithfulness checks ask whether the model output changes when highly attributed pixels are removed or restored. They do not prove that the highlighted pixels are clinically correct; they measure whether the explanation is faithful to the model’s learned behavior.
- Practical pointer: This distinction is important for Chapter 4. A method can be faithful to the TorchXRayVision classifier while still poorly localized against the pneumothorax mask. That outcome should be framed as a model-behavior finding, not as a failure of the mask metric alone.

## Medical-Imaging-Specific XAI Lessons

### Medical XAI needs clinical users and tasks, not generic interpretability claims

- Source link: [`REF-HUMAN-CENTERED-XAI`](references.md#ref-human-centered-xai)
- Thesis-ready paraphrase: In medical imaging, explanation quality depends on the clinical task and the user who must interpret the output. This supports including a radiologist-style review workbook with explicit rubric categories rather than relying only on algorithmic scores.
- Practical pointer: Use this source to justify the scoring schema: localization correctness, usefulness, failure category, artifact note, and free-text comment.

### Medical-image XAI literature is broad, but evaluation remains a key weakness

- Source link: [`REF-MEDICAL-XAI-REVIEW`](references.md#ref-medical-xai-review)
- Thesis-ready paraphrase: Reviews of deep-learning XAI in medical image analysis show that many explanation methods exist, but their clinical meaning depends on modality, task, model, and validation design. For this thesis, it is safer to present the work as a focused validation study of explanation methods for a specific CXR pneumothorax setting.
- Practical pointer: Chapter 2 can use this review to organize methods into post-hoc saliency/attribution families, while Chapter 3 narrows the scope to methods actually implemented and evaluated.

### XAI can reveal shortcut learning in radiology models

- Source link: [`REF-CLEVER-HANS`](references.md#ref-clever-hans)
- Thesis-ready paraphrase: Published radiology AI audits have shown that models may exploit shortcuts or non-pathology image cues. This is directly relevant when a CXR classifier has moderate ranking performance but heatmaps repeatedly highlight non-lesion structures, devices, image borders, text markers, or diffuse regions.
- Practical pointer: In discussion, treat weak localization as clinically meaningful evidence about the off-the-shelf model’s behavior. Avoid overstating that the XAI method itself is wrong without separating classifier quality, dataset mismatch, preprocessing, and explanation method limitations.

### External validation matters for chest X-ray models

- Source link: [`REF-CXR-GENERALIZATION-ZECH`](references.md#ref-cxr-generalization-zech)
- Thesis-ready paraphrase: Chest X-ray classifiers can show materially different performance when tested across institutions or datasets. This supports treating the TorchXRayVision SIIM-ACR experiment as an external-transfer validation, not as a direct measure of how well the original training datasets were modeled.
- Practical pointer: In Chapter 4, separate classifier ranking performance from explanation localization. If the model remains moderately predictive but poorly localized, this can be discussed as a transfer/generalization problem where disease-label prediction does not guarantee clinically aligned lesion evidence.

### Hidden stratification supports outcome- and subgroup-aware review

- Source link: [`REF-HIDDEN-STRATIFICATION`](references.md#ref-hidden-stratification)
- Thesis-ready paraphrase: A model can have acceptable aggregate performance while failing on clinically important subsets that are not explicitly labeled or evaluated. For pneumothorax, treatment devices, subtle lesions, unusual acquisition patterns, and label noise are plausible hidden factors that may affect both classification and explanation maps.
- Practical pointer: Use this source to justify the balanced `tp`/`fp`/`tn`/`fn` review workflow and the failure taxonomy. A small qualitative review can look for device/text/artifact reliance even when the quantitative dataset does not provide formal subgroup labels.

## Dataset and Baseline Model Framing

### SIIM-ACR pneumothorax is a classification-plus-localization challenge

- Source link: [`REF-SIIM-ACR`](references.md#ref-siim-acr)
- Direct quote candidate: "participants were asked to develop a model to classify (and segment) pneumothorax from a set of chest radiographic images to help aid in the early recognition of pneumothoraces."
- Thesis-ready paraphrase: The selected dataset is appropriate because pneumothorax has both image-level clinical relevance and lesion masks that can support localization-style evaluation of explanation maps.
- Practical pointer: For final thesis writing, do not rely only on online dataset counts. Recompute local counts from the exact dataset snapshot and cite SIIM/Kaggle for challenge context.

### TorchXRayVision is an off-the-shelf baseline, not a task-specific SIIM model

- Source link: [`REF-TORCHXRAYVISION`](references.md#ref-torchxrayvision)
- Direct quote candidate: "TorchXRayVision is an open source software library for working with chest X-ray datasets and deep learning models."
- Thesis-ready paraphrase: TorchXRayVision provides pretrained CXR classifiers and common preprocessing for public chest X-ray datasets. In this thesis, the DenseNet baseline is used as an external pretrained model; it is not locally fine-tuned for SIIM pneumothorax masks.
- Practical pointer: This distinction is crucial for Chapter 4. If the classifier is clinically weak on SIIM localization, that finding should be framed as evidence about transfer/generalization and baseline suitability, not as a claim that all CXR XAI is unreliable.

### Public CXR training sources are related but not interchangeable

- Source links: [`REF-CHESTXRAY8`](references.md#ref-chestxray8), [`REF-CHEXPERT`](references.md#ref-chexpert), [`REF-MIMIC-CXR`](references.md#ref-mimic-cxr)
- Thesis-ready paraphrase: Major public chest X-ray datasets differ in institutions, acquisition workflows, report-derived labels, uncertainty handling, expert-label subsets, and available localization annotations. A pretrained model using one or more of these sources should therefore not be assumed to behave like a pneumothorax-specific SIIM segmentation model.
- Practical pointer: Use this note when introducing the planned diagnostic A/B comparison across TorchXRayVision weights. The comparison is meaningful because these weights share a common library/model family but may reflect different source datasets and label conventions.

### Report-derived CXR labels are useful but clinically imperfect supervision

- Source links: [`REF-CHESTXRAY8`](references.md#ref-chestxray8), [`REF-CHEXPERT`](references.md#ref-chexpert), [`REF-MIMIC-CXR`](references.md#ref-mimic-cxr)
- Thesis-ready paraphrase: Large public CXR datasets enabled scalable model development, but their labels are often derived from radiology reports rather than pixel-level lesion annotation. This means a classifier can learn image-level associations with pneumothorax without necessarily learning a lesion-localizing representation that overlaps the SIIM mask.
- Practical pointer: This supports the thesis separation between image-level classification metrics and mask-overlap explanation metrics. It also explains why a second stronger model, if added, should still undergo the same XAI validation protocol rather than being accepted based only on published dataset performance.

## Candidate Literature Review Structure

1. Medical imaging AI needs explainability because high-stakes clinical predictions require more than classification accuracy.
   - Support: [`REF-MEDICAL-XAI-REVIEW`](references.md#ref-medical-xai-review), [`REF-HUMAN-CENTERED-XAI`](references.md#ref-human-centered-xai)
2. Post-hoc saliency and attribution methods are commonly used, but each answers a different question.
   - Support: [`REF-GRADCAM`](references.md#ref-gradcam), [`REF-GRADCAMPP`](references.md#ref-gradcampp), [`REF-INTEGRATED-GRADIENTS`](references.md#ref-integrated-gradients), [`REF-SHAP`](references.md#ref-shap), [`REF-SCORECAM`](references.md#ref-scorecam), [`REF-OCCLUSION-ZEILER`](references.md#ref-occlusion-zeiler)
3. Explanation validation should combine localization, faithfulness, robustness/sanity checks, and human-centered review.
   - Support: [`REF-SANITY-CHECKS`](references.md#ref-sanity-checks), [`REF-RISE`](references.md#ref-rise), [`REF-SAMEK-EVALUATION`](references.md#ref-samek-evaluation), [`REF-HUMAN-CENTERED-XAI`](references.md#ref-human-centered-xai)
4. Chest X-ray pneumothorax is a good focused case because the dataset has clinically meaningful masks and an official challenge context.
   - Support: [`REF-SIIM-ACR`](references.md#ref-siim-acr), [`REF-TORCHXRAYVISION`](references.md#ref-torchxrayvision), [`REF-CHESTXRAY8`](references.md#ref-chestxray8), [`REF-CHEXPERT`](references.md#ref-chexpert), [`REF-MIMIC-CXR`](references.md#ref-mimic-cxr)
5. A key thesis contribution can be negative or cautionary: off-the-shelf pretrained models may be predictive enough to screen but poorly localized or shortcut-driven.
   - Support: [`REF-CLEVER-HANS`](references.md#ref-clever-hans), [`REF-CXR-GENERALIZATION-ZECH`](references.md#ref-cxr-generalization-zech), [`REF-HIDDEN-STRATIFICATION`](references.md#ref-hidden-stratification), [`REF-TORCHXRAYVISION`](references.md#ref-torchxrayvision), current project results.

## Thesis-Safe Wording Bank

- "The generated maps are class-specific attribution visualizations with respect to the pneumothorax output, not anatomical segmentations."
- "Mask overlap evaluates clinical localization against available lesion annotations, whereas deletion/insertion curves evaluate whether the highlighted pixels affect the model’s own probability."
- "Agreement across methods is stronger evidence than any single overlay, while disagreement is diagnostically useful because it can expose instability, baseline sensitivity, or reliance on non-lesion cues."
- "The off-the-shelf baseline is valuable precisely because it tests transfer behavior: a pretrained medical model can be moderately predictive on a new dataset while still relying on features that are clinically questionable for localization."
- "Radiologist review is treated as a structured qualitative assessment, not as a replacement for quantitative localization or faithfulness metrics."
- "Large report-labeled CXR datasets enable useful pretrained models, but image-level report labels do not guarantee that the learned evidence aligns with pixel-level lesion masks."
- "Occlusion maps are perturbation diagnostics: they show how the model probability changes when image patches are replaced, not where gradients are largest."

## Follow-Up Sources to Consider Later

- Additional peer-reviewed CXR generalization/bias papers if Chapter 4 needs more than the current Zech/generalization, hidden-stratification, and shortcut-learning sources.
- CheXNet or other arXiv-only historical CXR baselines only if the user approves preprint use, or if a peer-reviewed final source is identified.