# Cross-Modality Validation of Explainable AI Methods for Medical Image Classification: Chest X-ray Pneumothorax and Head CT Hemorrhage Case Studies

Author: Dmytro Valantsevych

Draft status: AI-assisted thesis draft prepared on `2026-06-04` for human supervisor/student review. Chapters 1-4 are populated with current experiment results, tables, graph/image references, and thesis-safe interpretations. Remaining `TODO (human review)` markers indicate items that require human confirmation, formatting in the final template, source/citation verification, or final figure selection before submission.

A Master's Thesis submitted to Neoversity in partial fulfillment of the requirements for the degree of Master of Science in Computer Science

Student ID: TODO

Thesis Supervisor: TODO

Co-Supervisor (if applicable): TODO

Date of Submission: TODO (target 2026-06-04 draft cutoff; final defense 2026-06-21)

© The dissertation of Dmytro Valantsevych is approved, and it is acceptable in quality and form for publication electronically.

## Thesis Certification

TODO: Standard supervisor certification statement confirming that the work meets the academic-presentation and breadth/quality criteria for the Master of Science in Computer Science degree. Template wording:

> I confirm that I have reviewed this study and, in my judgment, it adheres to the appropriate standards of academic presentation. I believe it satisfactorily meets the criteria, in terms of both quality and breadth, to serve as a thesis for the attainment of the Master of Science in Computer Science degree. This thesis has been submitted to Neoversity and is deemed sufficient to fulfill the prerequisites for the Master of Science in Computer Science degree.

Supervisor signature line and Co-Supervisor signature line (if applicable).

## Declaration of Academic Integrity

TODO: Required by template. Standard wording from `requirements/Шаблон ... .md`:

> I confirm that this thesis, submitted to fulfill the requirements for the Master of Science in Computer Science degree, completed by me from `<start date>` to `<end date>`, is the result of my own individual endeavor. Any contributions from external sources or individuals, including the use of AI tools, are appropriately acknowledged through citation. Furthermore, I confirm that this material has not been previously submitted, in its entirety or in part, for the completion of a degree at this institution or any other.
>
> By making this declaration, I acknowledge that any violation of this statement constitutes academic misconduct. I understand that such misconduct may lead to expulsion from the program and/or disqualification from receiving the degree.

Name of the Candidate: Dmytro Valantsevych

Signature of Candidate: TODO

Date: TODO

**AI tool acknowledgment** (template-mandated cross-link to `AGENTS.md` AI tooling rule): list of AI/development tools used during the research workflow, including `GPT-5.5`, `Codex`, `PyCharm`, `Junie`, `VS Code`, `Claude Sonnet 4.6`, and `Claude Opus 4.7`. The methodology section of Chapter 3 (Implementation and Reproducibility) describes which tool was used for which class of task; this declaration page only confirms that all such use is acknowledged through citation.

## Acknowledgments

TODO (optional per template): personal acknowledgments to the thesis supervisor, colleagues, and family.

## Table of Contents

TODO: generate from finalized headings before submission. Required by template; must include page numbers. Suggested structure mirrors the chapter outline below:

- LIST OF TABLES
- LIST OF FIGURES
- LIST OF GRAPHS
- LIST OF CHARTS
- LIST OF ABBREVIATIONS
- ABSTRACT
- CHAPTER 1. INTRODUCTION (1.1 … 1.5, Conclusions to Chapter 1)
- CHAPTER 2. LITERATURE REVIEW (2.1 … 2.5, Conclusions to Chapter 2)
- CHAPTER 3. METHODOLOGY (3.1 … 3.9, Conclusions to Chapter 3)
- CHAPTER 4. RESULTS AND DISCUSSION (4.1 … 4.6, Conclusions to Chapter 4)
- CHAPTER 5. CONCLUSIONS AND RECOMMENDATIONS (5.1 … 5.3)
- BIBLIOGRAPHY
- APPENDICES (A, B, C — confirm with supervisor whether English Latin or template Cyrillic А, Б, В is required for an English-language thesis)

## List of Tables

TODO: populate at finalization. Template format requires a table with columns `Table No.`, `Title`, `Page No.`. Tables in the body must be numbered "X.Y" by chapter (e.g., Table 3.2, Table 4.1). Current weekly-report tables use "Table 1, Table 2" sequential numbering and must be renumbered at thesis time.

| Table No. | Title | Page No. |
| :---: | ----- | :---: |
| 2.1 | Summary of XAI method families and validation concerns | TODO |
| 3.1 | Method panel and explanation-view semantics | TODO |
| 3.2 | Calibration and held-out evaluation protocol | TODO |
| 3.3 | Metric interpretation guide for localization, faithfulness, and review scores | TODO |
| 4.1 | CXR classifier performance summary | TODO |
| 4.2 | Stage A TorchXRayVision model-localization comparison | TODO |
| 4.3 | Cross-modality consensus summary | TODO |
| 4.4 | Balanced 40-case radiologist review score distribution | TODO |
| 4.5 | Balanced 40-case review failure taxonomy | TODO |
| 4.6 | Phase 5.2 improvement-experiment paired Dice comparison | TODO |
| … | … | … |

## List of Figures

TODO: populate at finalization. Template format requires a table with `Figure No.`, `Title`, `Page No.`. Figures must be numbered "X.Y" by chapter. Current weekly-report figures (Figure 1, Figure 2, Figure 3, Figures 4a–4j) must be renumbered.

| Figure No. | Title | Page No. |
| :---: | ----- | :---: |
| 2.1 | Conceptual risk of visually plausible but clinically misleading saliency maps | TODO |
| 3.1 | End-to-end validation pipeline: data, classifier, XAI methods, metrics, and review | TODO |
| 3.2 | Four explanation views derived from signed attribution: positive, negative, magnitude, signed | TODO |
| 4.1 | Representative review-workbook explanation case with source image and method/view overlays | TODO |
| 4.2 | Representative radiologist-review failure modes selected from the balanced 40-case workbook | TODO |
| … | … | … |

## List of Graphs

TODO: populate at finalization. Template explicitly distinguishes graphs from figures and from charts. If the thesis ends up with no graphs (only figures and tables), this section is retained but marked as not applicable. Most likely candidates for "graphs": faithfulness deletion/insertion curves and improvement-experiment box plots — which the template would classify as graphs, not figures.

| Graph No. | Title | Page No. |
| :---: | ----- | :---: |
| 4.1 | Deletion and insertion faithfulness curves by method family | TODO |
| 4.2 | DenseNet-chex Phase 5.2 localization metric distributions | TODO |
| 4.3 | ResNet-50 Phase 5.2 localization metric distributions | TODO |
| 4.4 | CT pilot localization metric distributions | TODO |
| 4.5 | Phase 5.2 consensus-vs-individual paired localization differences | TODO |
| … | … | … |

## List of Charts

TODO: populate at finalization. Template distinguishes charts (bar charts, pie charts, etc.) from graphs (line plots) and from figures (image overlays, schematic diagrams). The faithfulness-AUC bar chart and the review-score-distribution counts most likely classify as charts.

| Chart No. | Title | Page No. |
| :---: | ----- | :---: |
| 4.1 | Faithfulness AUC comparison by method and model | TODO |
| 4.2 | Balanced 40-case review score distribution | TODO |
| 4.3 | Balanced 40-case review failure-category distribution | TODO |
| … | … | … |

## Abstract

Draft abstract (final formatting and word-count check still required; target `250-300` words):

This thesis investigates how explainable artificial intelligence methods can be validated for medical image classification rather than used only as visual illustrations. The study focuses primarily on chest X-ray pneumothorax detection using the SIIM-ACR pneumothorax dataset and off-the-shelf TorchXRayVision classifiers, with a completed head CT intracranial hemorrhage pilot used as a cross-modality methodological extension. The aim is to compare several post-hoc explanation methods and assess whether their highlighted evidence is localized, faithful to the model, and clinically useful.

The methodology separates classifier behavior from explanation quality. Pneumothorax classifiers are evaluated at frozen thresholds, while explanation maps are generated with `Grad-CAM`, `Grad-CAM++`, `Integrated Gradients`, `GradientSHAP`, `Occlusion`, `Eigen-CAM`, `Score-CAM`, and a frozen four-method consensus. Each method is interpreted through positive, negative, magnitude, and signed views. Explanations are validated with lesion-mask localization metrics, negative-evidence diagnostics, deletion/insertion faithfulness curves, method-agreement analysis, and a structured radiologist-style review of selected cases.

Current CXR results show that DenseNet-chex supersedes the original DenseNet-all checkpoint as the selected DenseNet baseline, while ResNet-50 remains the strongest tested TorchXRayVision follow-up in relative localization terms; nevertheless, absolute pneumothorax-mask overlap remains weak. The held-out Phase 5.2 improvement experiment shows model-dependent consensus behavior: DenseNet-chex consensus mainly improves over Score-CAM in paired overlap tests, whereas ResNet-50 consensus significantly improves Dice/IoU over several weaker individual methods but not over the strongest CAM comparators. The balanced 40-case review shows mixed usefulness: many maps contain interpretable evidence, yet clinically misleading, indirect, device-related, or high-contrast non-lesion evidence remains common.

The practical significance of the work is a reproducible validation workflow for auditing medical image explanations. The thesis argues that heatmaps should be treated as model-behavior diagnostics, not anatomical segmentations or automatic generators of clinical trust.

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

Deep learning has become a central technology in medical image analysis because convolutional and transformer-based models can learn discriminative image patterns from large radiology datasets. In chest radiography and computed tomography, such models can support classification tasks such as pneumothorax detection or intracranial hemorrhage screening. However, the clinical usefulness of a model cannot be judged only from an image-level prediction score. In high-risk medical settings, a prediction is safer and more informative when the evidence behind it can be inspected and compared with expected clinical findings.

Explainable AI (XAI) methods are often proposed as a way to make deep learning systems more transparent. Saliency maps, class activation maps, occlusion maps, and attribution maps can highlight image regions that contribute to a model output. These visualizations are especially attractive in radiology because they resemble the spatial reasoning used by clinicians. Nevertheless, a visually plausible heatmap is not automatically a clinically valid explanation. A model may highlight devices, image borders, text markers, bone edges, or treatment-related correlates rather than the visible pathology itself.

This thesis is situated in that gap between visual explanation and validated explanation. It treats XAI maps as class-specific model-behavior diagnostics, not as direct pathology segmentations. The primary empirical setting is chest X-ray pneumothorax classification on the SIIM-ACR pneumothorax dataset using off-the-shelf TorchXRayVision classifiers. A completed head CT hemorrhage pilot is included as a methodological extension after verifying a PhysioNet masked CT dataset and a DifeiT Vision Transformer classifier. The CT branch is intentionally narrower than the CXR study and is used to test whether the validation workflow transfers across modality without claiming a full second clinical benchmark.

### 1.2 Problem Statement and Relevance

Medical image classifiers can achieve useful ranking or classification performance while relying on features that are not clinically aligned with the target abnormality. This creates a practical validation problem: if the explanation map is interpreted as evidence that the model "looked at the right place", the explanation itself becomes part of the model's trust argument. Without validation, this trust argument may be misleading.

The problem addressed in this thesis is therefore the clinical and methodological reliability of post-hoc XAI explanations for medical image classification. The key question is not simply whether a heatmap can be generated, but whether it is localized to available lesion evidence, whether it faithfully affects the model output under perturbation, whether different XAI methods agree, and whether a medically trained reviewer judges the explanation useful or misleading.

The thesis is guided by the following research questions:

1. Do off-the-shelf medical image classifiers produce clinically localized explanations on SIIM-ACR pneumothorax cases?
2. How do common XAI methods differ when evaluated by mask-overlap metrics, peak-localization metrics, faithfulness curves, and signed-evidence diagnostics?
3. Does a frozen cross-method consensus improve localization compared with individual methods under held-out paired testing?
4. Can structured radiologist-style review reveal explanation failure modes that are not captured by automatic metrics alone?
5. To what extent does the same validation design transfer to a second modality such as head CT hemorrhage without over-claiming empirical cross-modality results?

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

The scientific novelty of the work lies in treating XAI validation as a layered experimental problem rather than as a set of illustrative heatmaps. The thesis combines classifier-threshold calibration, lesion-mask localization metrics, perturbation-based faithfulness curves, signed positive/negative evidence views, method agreement analysis, and radiologist-centered failure taxonomy within one reproducible pipeline. This design makes it possible to distinguish four ideas that are often conflated: a correct classifier prediction, a lesion-overlapping explanation, a model-faithful explanation, and a clinically useful explanation.

The project also contributes a cautionary empirical case study. The baseline models are not locally fine-tuned pneumothorax segmenters; they are off-the-shelf medical image classifiers. This is practically important because pretrained clinical AI tools are often reused outside their exact development distribution. Showing that a pretrained model can be moderately predictive while producing weakly localized or confounded explanations is a meaningful result: it demonstrates why explanation validation is necessary before heatmaps are used to support trust.

The practical significance is a thesis-ready validation workflow for auditing medical image explanations. The same workflow can support future model comparison, stronger pneumothorax-specific models, larger radiologist studies, and modality extensions. In clinical terms, the intended contribution is not automated diagnosis, but safer interpretation of model evidence: XAI maps can help expose when trust is not justified.

### 1.5 Thesis Structure

Chapter 1 introduces the research context, problem statement, aim, objectives, and thesis contribution. Chapter 2 reviews deep learning in radiology, post-hoc XAI methods, explanation-validation approaches, and known risks such as shortcut learning and visually plausible but unreliable saliency maps. Chapter 3 describes the experimental methodology, including datasets, preprocessing, classifier baselines, XAI methods, threshold calibration, localization and faithfulness metrics, statistical testing, and radiologist-centered review. Chapter 4 presents and discusses the empirical results, including classifier behavior, quantitative explanation validation, model comparison, radiologist review, CXR consensus-improvement experiments, and the completed CT pilot. Chapter 5 summarizes the main findings, practical recommendations, limitations, and future-work directions.

### Conclusions to Chapter 1

Chapter 1 established the motivation for validating explainable AI in medical imaging. Deep learning models can produce useful predictions in radiology, but the clinical meaning of those predictions depends on the evidence used by the model. Post-hoc heatmaps can make this evidence visible, yet visual plausibility alone is insufficient: a heatmap may emphasize non-pathological high-contrast structures, treatment devices, image artifacts, or dataset-specific shortcut cues. For this reason, the thesis frames explanation maps as model-behavior diagnostics rather than anatomical segmentations or automatic trust generators.

The research problem is the reliability of XAI explanations for medical image classification. The primary case study is chest X-ray pneumothorax classification and explanation using SIIM-ACR data and off-the-shelf TorchXRayVision baselines. The head CT hemorrhage pilot is included as a completed but limited modality extension, while the strongest empirical claims remain built around the CXR pipeline because it has a broader method panel, classifier screening, and radiologist-centered review. The next chapter situates this research design in the literature on radiology classification, saliency and attribution methods, explanation faithfulness, localization validation, human-centered XAI, and shortcut learning.

## Chapter 2. Literature Review

### 2.1 Literature Search Strategy

The literature review uses peer-reviewed papers, official dataset/model documentation, and medical-imaging AI reporting guidance. The main source categories are: (1) deep learning for radiology and public chest X-ray datasets; (2) post-hoc XAI methods for image classifiers; (3) explanation-validation methods, including localization, perturbation-based faithfulness, sanity checks, and human-centered evaluation; (4) shortcut learning and clinical applicability risks; and (5) transparent reporting and risk-of-bias guidance for medical AI studies.

The working reference inventory is maintained in `docs/references.md`. Preferred sources include official proceedings pages, journal records, PubMed/PMC entries, CVF/PMLR/NeurIPS/IEEE/ACM records, official dataset pages, and official software documentation. Blogs, informal tutorials, and unverified dataset mirrors are excluded unless explicitly justified. For final submission, online sources should include stable URLs and access dates, and the bibliography should be converted consistently to the selected final style, currently planned as IEEE.

The review emphasizes recent literature from approximately 2016 onward because most widely used post-hoc deep-learning XAI methods and medical-imaging AI reporting frameworks emerged or matured during this period. Older sources are retained where historically necessary, such as occlusion sensitivity and early convolutional-network visualization work.

### 2.2 Deep Learning Classification in Radiology

Deep learning classifiers have become common in radiology research because they can learn hierarchical image features directly from pixel data. Chest X-ray is a particularly active domain: it is widely available, clinically important, and represented in large public datasets such as ChestX-ray14, CheXpert, MIMIC-CXR, and related curated resources. These datasets enabled pretrained CXR models and libraries, including TorchXRayVision, which provide practical baselines for downstream research.

However, public CXR datasets are not interchangeable. They differ in institutions, acquisition workflows, label extraction pipelines, uncertainty handling, disease prevalence, and availability of localization annotations. Many labels are derived from radiology reports rather than pixel-level lesion masks. As a result, a classifier can learn image-level associations that support prediction without necessarily learning spatial evidence that aligns with a lesion annotation. This distinction is central to the present thesis: image-level classifier performance and explanation localization must be reported separately.

Pneumothorax is a useful focused case study because it has image-level clinical relevance and, in the SIIM-ACR challenge context, available segmentation masks for positive cases. These masks make it possible to evaluate whether model explanations overlap the visible abnormality. The thesis uses off-the-shelf TorchXRayVision models as baselines rather than locally fine-tuned pneumothorax segmenters. This choice makes the study a transfer/generalization audit: it asks how pretrained medical classifiers behave when their explanations are tested against a specific pneumothorax localization reference.

Head CT intracranial hemorrhage is methodologically relevant as a second modality because it differs from radiography in image physics, intensity scale, preprocessing, and clinically meaningful perturbation baselines. CT data are typically represented in Hounsfield units and interpreted through diagnostic windows, whereas CXR inputs are 2D radiographs with different normalization assumptions. In this draft, CT is treated as a completed pilot: it strengthens the cross-modality methodology with one verified pretrained classifier and one public masked dataset, but its narrower scope prevents a claim of complete CT clinical validation.

TODO (human review): verify final dataset citations, access dates, and exact bibliographic entries for SIIM-ACR/Kaggle, PhysioNet `ct-ich`, TorchXRayVision, and the DifeiT CT checkpoint before final submission.

### 2.3 Explainable AI Methods for Image Classification

Post-hoc XAI methods for image classifiers can be grouped by the type of signal they use. CAM-family methods use internal convolutional feature maps to produce class-discriminative spatial maps. `Grad-CAM` weights target-layer activations by gradients of the selected class score, while `Grad-CAM++` modifies the gradient weighting scheme to better handle multiple or fine-grained discriminative regions. These methods are attractive for radiology because they produce relatively smooth, spatially coherent maps, but their resolution and interpretation are constrained by the selected feature layer.

`Eigen-CAM` and `Score-CAM` extend the CAM family in different directions. `Eigen-CAM` derives a map from the principal component of target-layer activations and does not require gradients, while `Score-CAM` masks the input with normalized activation maps and weights them according to score changes. In the current project, both are added as individual methods in the expanded Phase 5.1 method panel. They are not added to the frozen consensus average because changing consensus constituents mid-thesis would make previous consensus results incomparable.

Pixel-attribution methods answer a different question. `Integrated Gradients` accumulates gradients along a path from a baseline image to the input, making baseline choice an explicit part of the explanation. `GradientSHAP` approximates Shapley-style attributions through noisy baseline interpolation and is therefore stochastic; sample count and random seed affect stability. In medical images, these methods can reveal fine-grained evidence, but they can also produce noisy or case-invariant patterns if preprocessing, baselines, or smoothing are not handled carefully.

Perturbation methods evaluate model behavior under direct image modification. `Occlusion` replaces patches of the image and measures how the selected class score changes. This makes it conceptually close to faithfulness testing because it asks what happens to the classifier when regions are removed or replaced. However, occlusion is computationally expensive and sensitive to patch size, stride, and replacement baseline. It should therefore be interpreted as a complementary diagnostic rather than as a direct substitute for gradient attribution.

`LIME` represents a region-level surrogate-model family: it perturbs interpretable image regions and fits a local linear approximation to the classifier. In this thesis it remains a lower-priority or future add-on unless explicitly implemented before finalization. Its conceptual value is that it belongs to a different explanation family, but it also introduces segmentation and perturbation hyperparameters that would require separate calibration discipline.

The implemented CXR methods share a `SignedAttribution` contract. Each method can provide positive evidence, negative evidence, magnitude, and signed views. This is important because positive evidence toward pneumothorax, negative evidence against pneumothorax, absolute impact, and signed balance answer different interpretive questions. A negative-evidence map should not be scored as clinically successful merely because it overlaps a lesion mask.

Table 2.1 summarizes how the main XAI method families differ and why none of them is sufficient as a standalone validation signal.

Table 2.1: Summary of XAI method families and validation concerns

| Method family | Examples in thesis | Main question answered | Main validation concern |
| --- | --- | --- | --- |
| CAM-family activation maps | `grad_cam`, `grad_cam_plus_plus`, `eigen_cam`, `score_cam` | Which target-layer activation regions support the selected class score? | Spatial resolution, target-layer choice, and possible focus on discriminative but non-lesion regions |
| Pixel attribution | `integrated_gradients`, `gradient_shap` | Which input pixels contribute positively or negatively relative to a baseline? | Baseline sensitivity, stochasticity, pixel noise, smoothing, and case-invariant patterns |
| Perturbation sensitivity | `occlusion` | How does the model score change when image patches are replaced? | Patch size, stride, replacement baseline, and computational cost |
| Cross-method consensus | `consensus`, `consensus_signed` | Do several explanation families agree on spatial evidence? | Averaging weak or mislocalized maps may not create clinically aligned evidence |
| Region-level surrogate methods | `LIME` if implemented | Which interpretable regions support a local surrogate approximation? | Superpixel definition, perturbation distribution, local-model stability, and separate calibration needs |

### 2.4 Validation of Explanations

Explanation validation has several non-equivalent dimensions. Localization validation compares selected attribution regions with available reference annotations. In this thesis the main positive-localization metrics are `IoU`, `Dice`, `pointing_hit`, and `precision_at_fraction`. `IoU`, `Dice`, and precision-at-fraction summarize overlap between selected heatmap regions and the pneumothorax mask, while `pointing_hit` is stricter: it checks whether the maximum-attribution point falls inside the lesion. These metrics are useful for positive cases with masks, but they do not fully capture clinical usefulness.

Faithfulness validation asks whether the highlighted pixels materially influence the model's own output. Deletion curves start from the original image and replace highly attributed pixels; insertion curves start from a baseline image and restore highly attributed pixels. If an explanation is faithful to the classifier, deleting high-attribution regions should reduce the target probability relatively quickly, while inserting them should restore it quickly. This evaluates model behavior, not clinical correctness. A heatmap may be faithful to a shortcut-driven model while still failing to localize the annotated lesion.

Sanity checks and robustness analyses are also important because saliency methods can sometimes produce visually stable maps even when model parameters or labels are changed. In medical imaging, this concern is heightened by structured preprocessing, strong anatomical priors, and recurring acquisition artifacts. For pixel-level methods such as Integrated Gradients and GradientSHAP, smoothing and cross-case similarity should be reported so that readers can distinguish case-specific evidence from generic image patterns.

Human-centered evaluation complements automatic metrics. A radiologist-style reviewer can identify whether a map highlights direct lesion evidence, indirect clinically related evidence, treatment correlates, devices, text artifacts, non-pathological high-contrast anatomy, or misleading regions. This type of review should be treated as structured qualitative evidence, not as prospective clinical validation. Its value is that it exposes clinically meaningful failure modes that a mask-overlap score may miss.

The central limitation of saliency maps is that they are easy to over-interpret. They can show where attribution is high, but they do not prove causality in the clinical sense, and they do not replace reporting of the model, data, target class, preprocessing, perturbation baseline, thresholding rule, or calibration split. For this reason, the thesis reports explanation maps together with classifier behavior, localization metrics, faithfulness curves, and review scores.

Figure 2.1 is reserved for the final literature-review illustration. It should show the conceptual risk that a heatmap can be visually plausible while highlighting clinically indirect or confounded image regions rather than the annotated lesion. The final figure may use one de-identified project example from the review workbook or a schematic if data-license constraints prevent publication of clinical images.

TODO (human review): select or draw Figure 2.1 and confirm that any clinical image used in Chapter 2 is allowed under the relevant dataset license and institutional submission rules.

### 2.5 Research Gap

The reviewed literature supports three conclusions. First, deep learning classifiers can be useful in radiology, but image-level performance does not guarantee lesion-aligned reasoning. Second, XAI methods are diverse and method-dependent: CAM maps, pixel attributions, Shapley-style approximations, and occlusion diagnostics answer different questions. Third, explanation validation remains difficult because a visually plausible heatmap may be unfaithful, poorly localized, or clinically misleading.

The research gap addressed by this thesis is the lack of a compact but layered validation workflow that compares several XAI methods on the same medical imaging task using mask localization, perturbation faithfulness, method agreement, signed-evidence interpretation, and radiologist-centered failure review. Many studies present heatmaps as qualitative illustrations; fewer test whether those heatmaps align with available lesion masks, whether they affect the model output under controlled perturbation, and whether a medically trained reviewer finds them useful or misleading.

The thesis also focuses on a practically important negative possibility: a pretrained medical classifier may be predictive enough to appear useful while its explanations reveal shortcut learning or weak clinical localization. Demonstrating this failure mode is scientifically valuable because it clarifies what XAI can and cannot justify. The contribution is not the claim that heatmaps automatically create trust, but that validated heatmaps can help decide when trust should be withheld.

### Conclusions to Chapter 2

Chapter 2 reviewed the literature background needed for the thesis. Deep learning classifiers are now common in radiology research, especially in chest X-ray analysis, but public dataset labels, source institutions, and clinical contexts differ substantially. A model trained or pretrained on one data mixture cannot be assumed to localize pathology correctly on another dataset. This motivates the thesis decision to evaluate off-the-shelf TorchXRayVision baselines not only by image-level prediction, but also by explanation localization and qualitative failure modes.

The reviewed XAI methods form complementary families. CAM-family methods provide class-discriminative spatial maps from internal activations; pixel-attribution methods trace gradients or Shapley-style contributions back to the input; occlusion directly perturbs image regions; and surrogate approaches such as LIME explain local behavior through interpretable region perturbations. Because these methods answer different questions, agreement between them is stronger evidence than any single heatmap, while disagreement is itself diagnostically useful.

The validation literature shows that heatmaps must be tested rather than merely displayed. Localization metrics, faithfulness curves, robustness checks, and human-centered review each capture different aspects of explanation quality. In medical imaging, this is especially important because shortcut cues, devices, acquisition artifacts, and non-pathological high-contrast anatomy can all become model evidence. The research design in Chapter 3 follows directly from this gap: it evaluates explanations through a layered protocol that separates classifier behavior, mask localization, faithfulness to the model, signed-evidence semantics, and radiologist-centered usefulness.

## Chapter 3. Methodology

### 3.1 General Research Design

This thesis uses an experimental validation design for post-hoc explainable AI methods in medical image classification. The central methodological question is not only whether a classifier predicts the correct image-level label, but whether the generated explanations are faithful to the classifier and clinically plausible with respect to available lesion evidence. The study therefore separates four linked but non-identical evaluation layers:

1. **Classifier behavior**: the image-level pneumothorax probability, classifier threshold, and resulting `tp`/`fp`/`tn`/`fn` outcome.
2. **Localization against annotations**: whether selected explanation regions overlap available lesion masks or place their maximum attribution inside the mask.
3. **Faithfulness to the model**: whether deleting or inserting highly attributed pixels changes the classifier probability in the expected direction.
4. **Human-centered usefulness and failure analysis**: whether a radiologist-style reviewer finds the explanation correct, partially useful, misleading, or clinically irrelevant under a defined task.

The primary case study is chest X-ray pneumothorax classification and explanation using the SIIM-ACR pneumothorax dataset and off-the-shelf TorchXRayVision classifiers. Pneumothorax is suitable because the task is clinically meaningful, the public challenge context includes image-level classification and segmentation framing, and positive cases provide masks that can be used as an approximate reference standard for localization-style evaluation.

The secondary case study is a head CT intracranial hemorrhage pilot. It is intentionally described as a pilot rather than a full parallel clinical validation because it uses one verified off-the-shelf hemorrhage classifier, one public masked CT dataset, and a narrower input-space explanation panel than the CXR study. CT is methodologically useful because it differs from CXR in image physics, preprocessing, and perturbation baselines: CT uses Hounsfield-unit windowing rather than 0-255 radiograph normalization.

The methodology follows transparent-reporting principles from medical-imaging AI guidance such as CLAIM, TRIPOD+AI, QUADAS-AI, and DECIDE-AI. The practical implication is that the thesis reports the data source, local dataset snapshot, model source, preprocessing, split/calibration strategy, target output, XAI settings, thresholding choices, metrics, qualitative review task, and limitations together. Explanation maps are never described as direct pathology segmentations; they are model-behavior visualizations with respect to the selected target class.

Figure 3.1: End-to-end validation pipeline used in the thesis draft

```text
Medical image + label/mask
        |
        v
Frozen classifier and target score
        |
        v
Post-hoc XAI methods -> positive / negative / magnitude / signed views
        |
        +--> calibrated top-fraction localization metrics
        +--> deletion / insertion faithfulness curves
        +--> signed-evidence and agreement diagnostics
        +--> radiologist-centered review workbook
        |
        v
Separated interpretation of classifier behavior, localization, faithfulness, and usefulness
```

TODO (human review): replace this text schematic with a polished visual diagram before final PDF export, or keep it as a simple figure only if the supervisor accepts Markdown-rendered diagrams.

### 3.2 Datasets

#### 3.2.1 Chest X-ray pneumothorax dataset

The primary dataset is the local SIIM-ACR pneumothorax chest X-ray snapshot stored under `data_local/cxr_pneumothorax/siim-acr-pneumothorax`. The working dataset state documented during the project contains `12,047` PNG images and `12,047` corresponding PNG masks, with `2,669` positive pneumothorax cases and `9,378` negative cases. These counts must be recomputed from the final local snapshot before final thesis submission, because the methodology and results should cite the exact data actually used.

The dataset is used in two complementary ways. First, image-level labels are used to screen classifier performance and define outcome groups (`tp`, `fp`, `tn`, `fn`) at a frozen classifier threshold. Second, positive masks are used for localization evaluation of explanation maps. Negative images do not have lesion masks; therefore, they are mainly used for classifier-outcome visualization, false-positive analysis, and qualitative review rather than Dice/IoU localization against a pneumothorax mask.

The project distinguishes calibration, exploratory, and held-out use. Classifier-threshold calibration and XAI top-fraction calibration must not be tuned on final held-out test results. Current CXR work uses a train-calibrated TorchXRayVision pneumothorax cutoff of approximately `0.62`, selected because it was the best or near-best operating point for F1 and Youden's J in the calibration analysis. Outcome-balanced review sets may use `--split any` for exploratory diversity, but any such result must be labelled exploratory because it can mix train and test cases.

#### 3.2.2 Head CT hemorrhage pilot dataset

The CT pilot uses the credentialed PhysioNet `ct-ich` v1.3.1 dataset stored locally under `data_local/physionet.org/files/ct-ich/1.3.1`. The dataset contains NIfTI CT volumes and matched masks (`ct_scans/NNN.nii` and `masks/NNN.nii`) plus per-slice hemorrhage labels. The verified local snapshot has `2,814` slices, including `318` hemorrhage-positive slices with masks and `2,496` normal slices. The final CT improvement experiment uses the train/test split produced by the CT manifest builder: `213` positive train slices for calibration and `105` positive test slices for held-out evaluation.

When the integrated CT classifier is a Vision Transformer, as in the current pilot, the CT explanation panel is restricted to the input-space methods (`Integrated Gradients`, `GradientSHAP`, `Occlusion`). These methods read only the model input and the output score, so their implementations are model-agnostic and run with identical code on the chest-X-ray and CT pipelines, which keeps the cross-modality comparison controlled. CAM-family methods (`Grad-CAM`, `Grad-CAM++`, `Eigen-CAM`, `Score-CAM`) are not applied on the CT classifier: they weight a two-dimensional convolutional activation map, and a Vision Transformer instead represents the image as a sequence of patch tokens with no such feature map. Running a CAM on a transformer would require a token-to-grid reshape that constitutes a different explanation technique and would confound transfer of the *same* methods; transformer-native spatial-explanation methods such as attention rollout or transformer attribution are therefore noted as future work rather than used here. Because the frozen four-method consensus includes `Grad-CAM`, it cannot be reproduced unchanged on the CT classifier; any CT-side consensus is reported separately as an exploratory aggregate over the three input-space methods and is explicitly distinguished from the frozen chest-X-ray consensus.

### 3.3 Preprocessing

For CXR experiments, each source image is loaded as a grayscale radiograph and converted into the input format expected by the selected classifier. TorchXRayVision preprocessing maps an `H x W` uint8 array in the range `0..255` through `xrv.datasets.normalize(array, 255)` and returns a single-channel tensor with shape `[1, H, W]`. The tensor is then moved to the selected device. DenseNet-121 TorchXRayVision weights are native to `224 x 224` inputs, while the tested ResNet-50 TorchXRayVision weight `resnet50-res512-all` is native to `512 x 512`; when a script uses a different image size, this must be reported because resizing affects both classification and heatmap resolution.

Masks are loaded as binary reference images for positive pneumothorax cases. For mask-based metrics, heatmaps and masks are brought to the same spatial grid before thresholding and comparison. Continuous attribution maps are normalized only for visualization and selection; normalization does not make them segmentations.

For CT, preprocessing does not reuse CXR normalization. CT images are loaded from HU-preserving NIfTI volumes and rendered through a locked brain window (`WL=40`, `WW=80`) before replication into the three-channel format required by the DifeiT ViT image processor. Perturbation baselines also differ: replacing CXR pixels with black is not equivalent to replacing CT voxels with a meaningful HU value. For the integrated DifeiT checkpoint, the image processor uses `image_mean = image_std = 0.5`, so the brain-window midpoint maps to normalized value `0.0`; in this specific model contract, `brain_window_center` is numerically identical to `zero_tensor`, while black/air remains a separate stress-test baseline.

All random sampling and balanced outcome selection use fixed seeds where implemented. Stochastic explanation methods, especially GradientSHAP, are reported with their sample count and noise settings. Broad exploratory runs may use faster settings, while thesis-quality selected cases are rerun with higher-stability settings.

### 3.4 Classification Models

The CXR classifiers are loaded off-the-shelf from TorchXRayVision. The original baseline is `xrv.models.DenseNet(weights="densenet121-res224-all")`, used as an external pretrained medical classifier without local modification, fine-tuning, or forking. This distinction is essential: weak SIIM pneumothorax localization should not be interpreted as failure of a locally optimized pneumothorax segmenter, but as evidence about transfer behavior of a pretrained image-level classifier.

The CXR pipeline now uses a classifier-loading seam, `load_classifier(name, device)`, which returns the model, Grad-CAM target layer, pneumothorax class index, and preprocessing function. This allows model comparisons without changing downstream XAI code. Current TorchXRayVision candidates include DenseNet-121 weights such as `densenet121-res224-all`, `densenet121-res224-chex`, `densenet121-res224-mimic_ch`, `densenet121-res224-mimic_nb`, `densenet121-res224-rsna`, `densenet121-res224-nih`, and `densenet121-res224-pc`, plus the architecturally different `resnet50-res512-all` classifier. The DenseNet Grad-CAM target layer is `model.features.denseblock4`; for the TorchXRayVision ResNet-50 wrapper, the target layer is `model.model.layer4`.

The diagnostic model-comparison protocol treats `densenet121-res224-all` as the control and asks whether poor localization is stable across pretrained CXR weights or improves materially for another model. A later Stage A sweep selected `resnet50-res512-all` as the strongest tested TorchXRayVision follow-up candidate by aggregate localization, but absolute overlap remained low; therefore it should be framed as a relative model-side improvement, not as clinically sufficient localization.

The target output for all CXR explanations is the model's `Pneumothorax` head. If a candidate model lacks a pneumothorax class head, it is rejected for this pipeline rather than silently attributing another class. Autoencoder-style weights are not valid classifiers for this methodology unless a separate reconstruction or feature-analysis experiment is explicitly defined.

### 3.5 Explainability Methods

The explanation methods are post-hoc methods applied to an already trained classifier. They are generated with respect to the selected target score and are therefore class-specific attribution maps. The current v3 CXR method registry contains the following primary method families:

- **Grad-CAM** (`grad_cam`): a class-discriminative activation-map method using gradients flowing into a final convolutional layer.
- **Grad-CAM++** (`grad_cam_plus_plus`): a CAM-family variant with modified activation-map weighting, included as a comparator rather than assumed to be universally superior.
- **Integrated Gradients** (`integrated_gradients`): a baseline-dependent path-integration attribution method. The number of integration steps and baseline choice must be reported.
- **GradientSHAP** (`gradient_shap`): a stochastic additive-attribution approximation related to SHAP concepts. The number of samples, noise standard deviation, and stability settings must be reported.
- **Occlusion sensitivity** (`occlusion`): a perturbation method that replaces local patches and measures the target-score response. Patch size, stride, and replacement baseline must be reported.
- **Eigen-CAM** (`eigen_cam`): a CAM-family method based on the first principal component of target-layer activations, included to broaden the activation-map comparison beyond gradient-weighted CAMs.
- **Score-CAM** (`score_cam`): a CAM-family method that masks the input with normalized activation maps and weights them by signed target-logit changes relative to a baseline. It is computationally expensive because it requires additional forward passes; broad calibration runs therefore report the channel cap used for feasibility.
- **Consensus** (`consensus`): a sign-aware average of the frozen original constituent methods (`grad_cam`, `integrated_gradients`, `gradient_shap`, and `occlusion`), used to test whether cross-method agreement produces a more stable explanation. Eigen-CAM and Score-CAM are deliberately reported as additional individual methods rather than being folded into the consensus definition, so that consensus results remain comparable across thesis iterations.

The project moved from a legacy v1 naming scheme with separate polarity-suffixed method IDs, such as `integrated_gradients_positive` and `gradient_shap_negative`, to a v2 `SignedAttribution` contract. In v2, each method computes one normalized signed map and derives four views from it:

- `positive`: non-negative evidence supporting the target score;
- `negative`: suppressive evidence against the target score;
- `magnitude`: absolute attribution strength, independent of direction;
- `signed`: the normalized signed map used for diverging overlays and signed diagnostics.

This design avoids recomputing the same method separately for each polarity and makes the interpretation of positive, negative, magnitude, and signed outputs explicit. For pixel-level methods such as Integrated Gradients and GradientSHAP, optional smoothing may be applied for review readability; if smoothing is used, metrics should be computed from the same maps shown to the reviewer so that visualization and quantitative evaluation remain aligned.

Figure 3.2 illustrates the four-view contract used throughout the CXR review workbook. The example paths below point to one balanced-review case and should be replaced by the final selected thesis-quality case after human figure selection.

Figure 3.2: Example four-view explanation contract for one review case (`case_31`, source `101_test_1_`)

| Positive evidence | Negative evidence | Magnitude view | Signed view |
| --- | --- | --- | --- |
| ![Consensus positive evidence example](../outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/review/assets/case_31/101_test_1__consensus_continuous_heatmap.png) | ![Consensus negative evidence example](../outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/review/assets/case_31/101_test_1__consensus_negative_continuous_heatmap.png) | ![Consensus magnitude example](../outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/review/assets/case_31/101_test_1__consensus_magnitude_continuous_heatmap.png) | ![Consensus signed example](../outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/review/assets/case_31/101_test_1__consensus_signed_continuous_heatmap.png) |

TODO (human review): confirm that `case_31` is the desired final Figure 3.2 example; otherwise replace the paths with a case that better illustrates the four views without overstating clinical correctness.

Table 3.1 summarizes the method panel and the semantic role of each output view. The table is intended as a final-writing scaffold; exact run parameters such as `ig_steps`, `gradshap_samples`, occlusion patch/stride, and `score_cam_channels_cap` must be filled from the final run metadata.

Table 3.1: Method panel and explanation-view semantics

| Method family | Method id | Primary evidence type | Four-view support | Final parameter fields to report |
| --- | --- | --- | --- | --- |
| Grad-CAM | `grad_cam` | target-layer gradient-weighted activation evidence | `positive`, `negative`, `magnitude`, `signed` | target layer, input size |
| Grad-CAM++ | `grad_cam_plus_plus` | CAM variant with modified gradient weighting | `positive`, `negative`, `magnitude`, `signed` | target layer, input size |
| Integrated Gradients | `integrated_gradients` | baseline-dependent pixel attribution | `positive`, `negative`, `magnitude`, `signed` | baseline, `ig_steps`, smoothing if used |
| GradientSHAP | `gradient_shap` | stochastic baseline/noise pixel attribution | `positive`, `negative`, `magnitude`, `signed` | samples, stdevs, seed, smoothing if used |
| Occlusion | `occlusion` | patch perturbation sensitivity | `positive`, `negative`, `magnitude`, `signed` | patch size, stride, baseline |
| Eigen-CAM | `eigen_cam` | activation principal-component map | `positive`, `negative`, `magnitude`, `signed` | target layer, sign-orientation rule |
| Score-CAM | `score_cam` | activation-mask forward-pass scoring | `positive`, `negative`, `magnitude`, `signed` | target layer, channel cap, baseline |
| Frozen consensus | `consensus` | average of original four constituent methods | `positive`, `negative`, `magnitude`, `signed` | constituent list, signed-map normalization |

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

#### 3.6.6 Calibration and paired statistical comparison

Classifier thresholds and explanation thresholds are calibrated separately. The classifier threshold converts the pneumothorax probability into an image-level outcome (`tp`, `fp`, `tn`, or `fn`). In contrast, the XAI top-fraction threshold selects a spatial region from each heatmap for mask comparison. Tuning one threshold does not tune the other, and neither should be optimized on the final held-out test results.

For the expanded Phase 5.1 method panel, XAI top fractions are recalibrated as `v3` artifacts on positive masked train-split cases. The canonical calibration output is `calibrated_thresholds_v3.csv`, generated separately for each selected classifier because the model architecture, native input size, and attribution maps differ between DenseNet-chex and ResNet-50. The held-out Phase 5.2 improvement experiment then uses the frozen `v3` calibration file and evaluates positive masked cases from the test split.

The improvement experiment compares the frozen four-method consensus against each individual method using paired case-level metrics. The primary statistical test is the Wilcoxon signed-rank test, chosen because localization scores are bounded, often zero-inflated, and not safely assumed to be normally distributed. Multiple consensus-vs-method comparisons are corrected with Holm-Bonferroni family-wise error control at `alpha = 0.05`. The effect-size companion is the median paired difference with a bootstrap 95% confidence interval under a fixed seed. This framing separates statistical evidence for paired improvement from visual or clinical usefulness, which is still assessed through representative cases and the radiologist-centered review.

Table 3.2 records the intended calibration and held-out evaluation discipline.

Table 3.2: Calibration and held-out evaluation protocol

| Stage | Data used | Output artifact | Purpose | Must not be changed after |
| --- | --- | --- | --- | --- |
| Classifier threshold calibration | train/calibration subset | selected probability cutoff per model | define `tp`/`fp`/`tn`/`fn` outcomes | before held-out reporting |
| XAI top-fraction calibration v3 | positive masked train-split cases | `calibrated_thresholds_v3.csv` per model | choose method/view fractions for spatial metrics | before Phase 5.2 test evaluation |
| Held-out improvement experiment | positive masked test-split cases | `improvement_experiment.csv`, paired statistics, plots | compare consensus against individual methods | no threshold tuning on these results |
| Qualitative review | balanced selected cases | review workbook and scored `scores.csv` | assess usefulness and failure modes | scoring rubric fixed before review |

Table 3.3 provides a compact metric interpretation guide for final writing. It should be kept close to the methodology text so the reader does not confuse localization, faithfulness, and clinical usefulness.

Table 3.3: Metric interpretation guide for localization, faithfulness, and review scores

| Evidence layer | Metric or artifact | What higher/better means | What it does not prove |
| --- | --- | --- | --- |
| Positive localization | `IoU`, `Dice`, `precision_at_fraction` | selected positive evidence overlaps the pneumothorax mask more strongly | the model is clinically safe or the heatmap is causal |
| Peak localization | `pointing_hit` | the maximum-attribution pixel lies inside the lesion mask | broad map quality or overall lesion coverage |
| Negative diagnostics | `negative_mask_avoidance_fraction` | selected negative evidence avoids the lesion mask | negative evidence is clinically correct in all contexts |
| Faithfulness | deletion/insertion curves and AUC | highlighted pixels affect the model probability under the chosen perturbation | highlighted pixels are clinically appropriate pathology evidence |
| Method agreement | signed-map cosine similarity or qualitative agreement | multiple methods highlight similar signed evidence | the shared evidence is necessarily lesion-aligned |
| Human-centered review | localization/usefulness/failure taxonomy | a medically trained reviewer finds the explanation useful or identifies a failure mode | prospective deployment safety or inter-rater reliability |

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

The final draft currently uses the following result-producing artifacts as Chapter 4 sources: DenseNet-chex classifier screen `outputs/iter_56_chex_classifier_eval_test/`, DenseNet-chex v3 XAI calibration `outputs/iter_57_chex_calibration_v3/`, DenseNet-chex held-out improvement `outputs/iter_58_chex_improvement_v3/`, ResNet-50 classifier screen `outputs/iter_55_resnet_classifier_eval_test/`, ResNet-50 v3 XAI calibration `outputs/iter_50_resnet_calibration_v3/`, ResNet-50 held-out improvement `outputs/iter_52_resnet_improvement_v3/`, balanced review workbook and analysis `outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/` and `outputs/iter_48_resnet_review_analysis_balanced40_smoothed_faithfulness/`, CT smoke test `outputs/iter_53_ct_smoke_test/`, and CT improvement run `outputs/iter_54_ct_improvement_test/`.

TODO (human review): before submission, copy the final chosen figures into a stable thesis figure folder if required by the document build process; the current Markdown links intentionally point to reproducible output artifacts for traceability.

Experiment outputs are stored under ordinal folders such as `outputs/iter_XX_<short_experiment_name>`. Root-level CSV summaries are kept at the run root, and image artifacts are grouped by case so copied PNG files remain traceable. For classifier-outcome visualizations, top-level folders preserve `tp`, `fp`, `tn`, and `fn` grouping.

When Python source files or output schemas change, syntax checks and relevant tests are run through WSL. The canonical local test command is `wsl.exe python3 -m pytest tests/ -v`; for script-only changes, at least `wsl.exe python3 -m py_compile <changed_python_files>` is required. Documentation-only changes do not require tests, but the Markdown diff should still be reviewed for scope and thesis-safe wording.

### Conclusions to Chapter 3

The methodology defines XAI validation as a layered experimental problem rather than as a visual-inspection exercise. The classifier layer measures whether an off-the-shelf medical image classifier assigns the correct image-level pneumothorax label. The localization layer tests whether selected attribution regions overlap available pneumothorax masks. The faithfulness layer asks whether perturbing highly attributed pixels changes the actual model probability in the expected direction. The human-centered layer then checks whether a medically trained reviewer finds the explanations useful, misleading, or clinically incomplete under a defined scoring rubric. Keeping these layers separate is essential because a heatmap can be faithful to the model while still being poorly localized to clinically expected pathology.

The CXR pipeline uses TorchXRayVision classifiers without local fine-tuning. DenseNet-all remains the original historical external baseline, but the Stage A diagnostic sweep identified DenseNet-chex as the stronger DenseNet-121 checkpoint, and ResNet-50 as the strongest tested TorchXRayVision follow-up overall. Explanation methods are implemented through the `SignedAttribution` contract, which provides positive, negative, magnitude, and signed views from one underlying attribution map. The method panel includes CAM-family, pixel-attribution, perturbation, and consensus approaches; the consensus remains frozen to the original four constituents so that cross-iteration comparison is not invalidated by the later Eigen-CAM and Score-CAM additions.

The evaluation protocol protects the train/test boundary by separating classifier-threshold calibration, XAI top-fraction calibration, and held-out testing. Phase 5.1 recalibrates top fractions as `v3` artifacts for the expanded method panel, and Phase 5.2 uses those frozen artifacts for paired test-set comparisons. Wilcoxon signed-rank tests, Holm-Bonferroni correction, and bootstrap confidence intervals provide a non-parametric statistical framework for consensus-vs-individual comparisons. The scripts, output folders, seeds, and run metadata make the experiments reproducible, while Chapter 4 returns to the main limitations: off-the-shelf model mismatch, mask-reference imperfections, perturbation-baseline sensitivity, and the exploratory scale of the radiologist-centered review.

## Chapter 4. Results and Discussion

### 4.1 Classification Performance

Classifier screening confirms that the selected CXR baselines differ substantially in image-level ranking performance. DenseNet-chex replaces DenseNet-all as the current DenseNet baseline because Stage A showed better aggregate localization among DenseNet-121 checkpoints; the full CheX test screen in `outputs/iter_56_chex_classifier_eval_test/` shows moderate image-level performance but not a uniformly better classifier-ranking profile than the original DenseNet-all. ResNet-50 is a stronger co-primary TorchXRayVision follow-up: the full test screen in `outputs/iter_55_resnet_classifier_eval_test/` gives higher AUC, average precision, specificity, and F1 at the Stage A threshold. These classifier values are reported only as image-level behavior. They do not imply that either model localizes pneumothorax well.

The CT branch is not reported as a conventional binary classifier-performance table because the integrated DifeiT ViT checkpoint is a multi-class ICH subtype classifier and the thesis target is defined as `1 - P(normal)` for positive masked hemorrhage slices. Its role in this chapter is therefore cross-modality explanation validation on verified positive slices rather than a complete CT screening benchmark.

Key interpretation points for the current CXR experiments:
- Treat `densenet121-res224-all` as the original historical TorchXRayVision baseline, not as the final selected DenseNet baseline. It remains useful because it demonstrates that an off-the-shelf medical classifier can have moderate ranking/classification behavior while still producing clinically weak pneumothorax localization.
- Treat `densenet121-res224-chex` as the selected DenseNet baseline after Stage A because it had better aggregate localization than DenseNet-all among DenseNet-121 checkpoints (`mean_dice=0.0284`, `mean_iou=0.0160` versus DenseNet-all `mean_dice=0.0237`, `mean_iou=0.0130`).
- Stage A TorchXRayVision model comparison selected `resnet50-res512-all` as the strongest tested follow-up candidate by localization aggregate, not as a clinically strong model. In `outputs/iter_33_stage_a_diagnostic_ab/weights_ab_summary.csv`, `resnet50-res512-all` had the highest mean Dice/IoU among tested TorchXRayVision candidates (`mean_dice=0.0397`, `mean_iou=0.0221`), but absolute mask overlap remained very low.
- Thesis-safe wording: ResNet-50 showed a relative improvement over the DenseNet branch within this diagnostic sweep, but the result should not be interpreted as reliable lesion segmentation or clinically sufficient pneumothorax localization.

Table 4.1 keeps classifier performance separate from explanation quality: a model can rank cases well while producing attribution maps that are weakly localized to the lesion mask.

VERIFIED (2026-06-06): the Table 4.1 thresholds and metrics match the frozen sources exactly. CheX `0.565` is the `default_threshold` operating point in `outputs/iter_56_chex_classifier_eval_test/classification_metrics.csv` (AUC `0.745962`, acc `0.682216`, sens `0.713793`, spec `0.673752`, F1 `0.487059`, `207`/`353`/`729`/`83` TP/FP/TN/FN). ResNet `0.525` is the `default_threshold` operating point in `outputs/iter_55_resnet_classifier_eval_test/classification_metrics.csv` (AUC `0.916290`, acc `0.860058`, sens `0.710345`, spec `0.900185`, F1 `0.682119`, `206`/`108`/`974`/`84`). Note for the human pass: these are the frozen Stage A operating points, distinct from each model's `best_f1`/`best_youden_j` in `selected_thresholds.csv` (CheX `0.57`, ResNet `0.515`); confirm only that the "Threshold source" wording matches the final methods text.

Table 4.1: CXR classifier performance summary

| Model | Split | Threshold source | Threshold | AUC | Accuracy | Sensitivity | Specificity | F1 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `densenet121-res224-chex` | test | Stage A / CheX rerun operating point | `0.565` | `0.7460` | `0.6822` | `0.7138` | `0.6738` | `0.4871` | selected DenseNet baseline after Stage A localization; `207` TP, `353` FP, `729` TN, `83` FN |
| `resnet50-res512-all` | test | Stage A threshold sweep | `0.525` | `0.9163` | `0.8601` | `0.7103` | `0.9002` | `0.6821` | co-primary TorchXRayVision follow-up; `206` TP, `108` FP, `974` TN, `84` FN |

Table 4.2 summarizes the already completed Stage A localization signal that motivated keeping ResNet-50 as a co-primary baseline. Values are aggregate positive-view localization means from `outputs/iter_33_stage_a_diagnostic_ab/weights_ab_summary.csv`.

Table 4.2: Stage A TorchXRayVision model-localization comparison

| Model | Mean Dice | Mean IoU | Mean precision at fraction | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `resnet50-res512-all` | `0.0397` | `0.0221` | `0.0296` | best tested TorchXRayVision candidate by aggregate localization, but still weak in absolute terms |
| `densenet121-res224-chex` | `0.0284` | `0.0160` | `0.0216` | strongest DenseNet-121 comparator in the Stage A summary |
| `densenet121-res224-all` | `0.0237` | `0.0130` | `0.0182` | original baseline; lower localization than ResNet-50 |
| `densenet121-res224-nih` | `0.0194` | `0.0108` | `0.0151` | weaker DenseNet-121 comparator |
| `densenet121-res224-pc` | `0.0180` | `0.0097` | `0.0127` | weaker DenseNet-121 comparator |
| `densenet121-res224-mimic_ch` | `0.0139` | `0.0073` | `0.0095` | weaker DenseNet-121 comparator |
| `densenet121-res224-mimic_nb` | `0.0128` | `0.0070` | `0.0098` | weakest tested DenseNet-121 comparator |

### 4.2 Quantitative Explanation Validation

The final Phase 5.2 held-out improvement experiments were run after freezing the expanded-method `v3` calibration artifacts. Both CXR baselines used `200` randomly sampled positive masked test-split cases (`seed=20260515`) and the same positive-view localization metric panel: `IoU`, `Dice`, `pointing_hit`, and `precision_at_fraction`. The quantitative results support three thesis-safe conclusions.

First, absolute pneumothorax-mask overlap remains low for all methods. For DenseNet-chex, the highest mean Dice values are `occlusion` (`0.0411`) and `grad_cam_plus_plus` (`0.0409`), with consensus close behind (`0.0383`) and the remaining methods in the `0.026-0.036` range. For ResNet-50, `grad_cam` has the highest mean Dice (`0.0540`), followed by `score_cam` (`0.0513`) and consensus (`0.0488`). These values are useful for relative method comparison, but they are not clinically strong segmentation-like localization.

Second, consensus behavior is model-dependent. In the DenseNet-chex held-out run (`outputs/iter_58_chex_improvement_v3/`), consensus significantly improves Dice, IoU, and precision-at-fraction over `score_cam`, but it does not significantly beat the stronger aggregate Dice methods such as `occlusion` or `grad_cam_plus_plus`. In the ResNet-50 run (`outputs/iter_52_resnet_improvement_v3/`), consensus significantly improves Dice/IoU over `grad_cam_plus_plus`, `integrated_gradients`, `gradient_shap`, `occlusion`, and `eigen_cam`, but not over `grad_cam` or `score_cam`. Therefore, consensus should be framed as a stabilizing method in some settings, not as a universally superior explanation.

Third, `pointing_hit` remains a strict and sparse diagnostic. Most methods have median `pointing_hit=0.0`; significant differences in this metric should be interpreted cautiously because the maximum-attribution pixel rarely falls inside the pneumothorax mask. This confirms the earlier Stage A pattern that selected-region overlap metrics and peak-localization metrics should not be treated as interchangeable.

Key interpretation points for quantitative CXR explanation validation:
- Quantitative overlap metrics and peak-localization metrics answer different questions. Across Stage A all-model correlations (`outputs/iter_35_metric_correlations_iter33_stage_a_all_models/`), IoU, Dice, and precision-at-fraction were almost redundant, while pointing-hit was only weakly associated with those overlap metrics. Use pointing-hit as a strict peak-localization diagnostic, not as an interchangeable replacement for Dice/IoU.
- Signed diagnostics are conceptually separate from positive lesion overlap. Negative attribution should not be scored as successful because it overlaps the pneumothorax mask; for suppressive evidence, lesion avoidance is often more meaningful than lesion overlap.
- Faithfulness and localization must be separated in the discussion. Deletion/insertion curves test whether a heatmap explains the model behavior under perturbation, whereas mask metrics test agreement with annotated pathology location. A method can be faithful to the model while still clinically poorly localized.
- For pixel-level methods (`Integrated Gradients`, `GradientSHAP`), post-processing affects readability. The smoothed ResNet review run used `--pixel-attribution-mask-smoothing 9`, and metrics were computed from the same smoothed maps shown in the review workbook. Present this as a visualization/selection setting, not as a change in the classifier.

These results distinguish calibration-dependent Phase 5.2 evidence from the earlier diagnostic Stage A evidence. The interpretation is intentionally conservative: averaging weak or mislocalized method families does not automatically create clinically aligned evidence, but under the stronger ResNet-50 baseline the frozen consensus can improve localization relative to several weaker individual methods.

Graph 4.2: DenseNet-chex Phase 5.2 localization metric distributions

![DenseNet-chex Phase 5.2 localization metric boxplots](../outputs/iter_58_chex_improvement_v3/improvement_experiment_boxplots.png)

Graph 4.3: ResNet-50 Phase 5.2 localization metric distributions

![ResNet-50 Phase 5.2 localization metric boxplots](../outputs/iter_52_resnet_improvement_v3/improvement_experiment_boxplots.png)

TODO (human review): decide whether Graphs 4.2-4.3 should be kept at full `0-1` y-axis scale for honest comparability or supplemented with zoomed versions in the appendix for readability.

### 4.3 Cross-Modality Comparison

Phase 5.4 Branch A was completed, so the thesis includes a real CT pilot rather than only a methodological extension. The verified PhysioNet `ct-ich` v1.3.1 branch uses NIfTI volumes with matched masks, HU-preserving preprocessing, the DifeiT ViT checkpoint, and the binary target `1 - P(normal)`. The full CT smoke run is stored in `outputs/iter_53_ct_smoke_test/`, and the calibrated held-out CT improvement experiment is stored in `outputs/iter_54_ct_improvement_test/`.

The cross-modality result is deliberately asymmetric. The CXR experiments evaluate the frozen four-method CXR consensus across CAM, input-gradient, and perturbation families. The CT pilot evaluates `consensus_input3`, an input-space consensus of `integrated_gradients`, `gradient_shap`, and `occlusion`, because the convolutional CAM implementation does not transfer directly to the Vision Transformer checkpoint. Therefore, CT results are valid evidence that the validation framework transfers across modalities, but they are not a one-to-one replication of the CXR method panel.

Table 4.3 summarizes the corrected cross-modality consensus reading. The important negative control is that there is no general law that consensus improves localization. DenseNet-chex shows significant overlap gains mainly over `score_cam`, while stronger aggregate Dice methods remain competitive or better descriptively. ResNet-50 shows small significant overlap gains over weaker methods while tying the strongest CAM methods. CT shows the cleanest consensus benefit, but only for peak localization (`pointing_hit`), while broad overlap metrics remain non-significant.

Table 4.3: Cross-modality consensus summary

| Setting | Reference consensus | Test set | Main significant result | Aggregate consensus metrics | Thesis interpretation |
| --- | --- | --- | --- | --- | --- |
| CXR DenseNet-chex | frozen four-method CXR `consensus` | `200` SIIM test positives | Dice/IoU/precision higher than `score_cam`; pointing-hit differs from several methods with zero median effects | IoU `0.0203`, Dice `0.0383`, pointing `0.050`, precision `0.0213` | consensus is competitive but not best by aggregate Dice; not universally superior |
| CXR ResNet-50 | frozen four-method CXR `consensus` | `200` SIIM test positives | Dice/IoU/precision higher than five weaker methods; not significant vs `grad_cam` or `score_cam` | IoU `0.0265`, Dice `0.0488`, pointing `0.010`, precision `0.0289` | consensus stabilizes weak/noisy methods but remains comparable to the best CAM alternatives |
| CT DifeiT ViT | `consensus_input3` from input-space methods | `105` PhysioNet CT positive slices | pointing-hit significantly higher than IG, GradientSHAP, and occlusion; overlap metrics non-significant | IoU `0.0411`, Dice `0.0754`, pointing `0.343`, precision `0.0458` | strongest consensus signal is peak localization on CT, not broad overlap |

The CT finding should be reported with four caveats. First, the CT calibration selected fraction `0.05` for all methods, which is the floor of the sweep and suggests that smaller fractions may be relevant for tiny hemorrhage masks. Second, `pointing_hit` is binary per case, so the proper interpretation combines per-method rates, win/tie/loss counts, and the paired rank test; median paired differences alone are not informative. Third, the CT pilot uses one classifier and one dataset. Fourth, CAM-family transformer explanations remain future work because adopting transformer-specific CAM or attention methods would introduce a different implementation path from the CXR convolutional code.

Graph 4.4: CT pilot localization metric distributions

![CT pilot localization metric boxplots](../outputs/iter_54_ct_improvement_test/ct_improvement_experiment_boxplots.png)

TODO (human review): verify whether this CT graph should remain in the main Chapter 4 cross-modality section or be moved to an appendix if the supervisor wants the main results to stay CXR-focused.

### 4.4 Radiologist Assessment and Failure Taxonomy

This section combines the automatic metrics with the structured balanced 40-case ResNet review. The numerical distributions are already available below; the remaining human-review task is to choose the final 2-4 representative visual examples. The candidate figures should connect the counts to direct localization, indirect clinically related evidence, device/tube confounding, high-contrast non-lesion evidence, and clinically misleading maps.

Figure 4.1: Candidate review-workbook case for explanation visual inspection

![Candidate review source image](../outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/review/assets/case_31/101_test_1_.png)

| Consensus positive view | Consensus threshold sweep | Consensus signed view |
| --- | --- | --- |
| ![Candidate consensus positive heatmap](../outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/review/assets/case_31/101_test_1__consensus_continuous_heatmap.png) | ![Candidate consensus threshold sweep](../outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/review/assets/case_31/101_test_1__consensus_threshold_sweep_panel.png) | ![Candidate consensus signed heatmap](../outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/review/assets/case_31/101_test_1__consensus_signed_continuous_heatmap.png) |

TODO (human review): replace Figure 4.1 with the final selected good/partial/misleading examples from the balanced workbook and ensure the caption states the classifier outcome, probability, source split, method settings, and that the overlays are attribution maps, not segmentations.

Key interpretation points from the balanced 40-case ResNet review (`outputs/iter_48_resnet_review_analysis_balanced40_smoothed_faithfulness/`, scored 2026-05-25). This is the canonical radiologist-review evidence for the thesis; the earlier 10-case smoothed review at `outputs/iter_45_resnet_review_analysis_smoothed/` is preserved as a methodological pilot but is superseded by the balanced outcome-stratified 40-case scoring.
- Review score distribution (`n=40`, with 10 cases per `tp`/`fp`/`tn`/`fn` outcome): localization was `correct` in 11/40 cases, `partial` in 15/40, and `incorrect` in 14/40. Usefulness was `useful` in 12/40, `potentially_useful` in 13/40, `misleading` in 14/40, and `not_useful` in 1/40. The split between useful + potentially useful (25/40) and misleading + not useful (15/40) supports a nuanced Chapter 4 framing: many maps carry some interpretable signal, but clinically misleading or poorly localized explanations remain common.
- Failure taxonomy counts: `correct` 10/40, `partial` 8/40, `non_pathological_high_contrast` 13/40, `clinically_misleading` 7/40, and `devices_text_artifacts` 2/40. The dominant failure mode at this stage of the off-the-shelf TorchXRayVision baseline is attention on non-pathological high-contrast structures (bones, rib edges, lung apex) rather than on device/text artifacts.
- Qualitative flags from the completed scoring sheet: devices/tubes were relevant in 19/40 cases, indirect evidence in 8/40, method disagreement in 8/40, weak pixel attribution in 4/40, subcutaneous emphysema in 4/40, and mask-quality issue in 3/40. The high devices/tubes flag rate is a structural confounder of the off-the-shelf classifier on SIIM pneumothorax: devices co-occur with positive cases and the classifier's evidence frequently latches onto treatment correlates rather than pneumothorax-specific findings.
- Interpretation guard: device/tube and subcutaneous-emphysema findings should not all be grouped as generic artifacts. Devices and ECG wires are treatment-related confounders that the classifier exploits; subcutaneous emphysema is sometimes clinically related to pneumothorax and is not equivalent to direct mask localization. The thesis must keep these categories distinct.
- The 40-case balanced review supports using `usefulness_score` separately from `localization_score`. Low-overlap maps can still be useful for auditing model failure or identifying clinically adjacent evidence; conversely, a visually plausible map can be misleading if it emphasizes non-lesion regions.
- The review workbook and figures should state that heatmaps visualize model attribution toward the selected pneumothorax output and should not be interpreted as anatomical segmentations.
- Exploratory score-metric correlations on the balanced 40-case set are modest: strongest absolute Spearman associations are about `|rho| <= 0.42`. Frame these as supporting evidence alongside the structured radiologist categories, not as standalone proof that any automatic metric captures clinical usefulness.

Table 4.4 gives the canonical balanced 40-case review score distribution.

Chart 4.2 visualizes Table 4.4 as review-score counts. The chart is generated from the canonical counts in `outputs/iter_48_resnet_review_analysis_balanced40_smoothed_faithfulness/review_score_counts.csv` by `scripts/make_thesis_charts.py`.

Chart 4.2: Balanced 40-case review score distribution (localization and usefulness)

![Chart 4.2 review score distribution](../outputs/iter_61_thesis_charts/chart_4_2_review_scores.png)

TODO (human review): confirm Chart 4.2 styling/caption and decide whether the table-only or chart-plus-table presentation is preferred for the final document.

Table 4.4: Balanced 40-case radiologist review score distribution

| Review dimension | Category | Count / 40 | Thesis interpretation |
| --- | --- | ---: | --- |
| Localization | `correct` | `11` | clear lesion-aligned evidence in a minority of cases |
| Localization | `partial` | `15` | some plausible evidence, but incomplete or indirect |
| Localization | `incorrect` | `14` | substantial clinically weak or mislocalized explanations |
| Usefulness | `useful` | `12` | directly helpful for model-audit interpretation |
| Usefulness | `potentially_useful` | `13` | informative but requiring caution/context |
| Usefulness | `misleading` | `14` | clinically misleading evidence remains common |
| Usefulness | `not_useful` | `1` | no practically useful signal in the scored view |

Table 4.5 summarizes the failure taxonomy for the same review set.

Chart 4.3 visualizes Table 4.5 as failure-taxonomy counts alongside the qualitative review flags. The chart is generated from `outputs/iter_48_resnet_review_analysis_balanced40_smoothed_faithfulness/review_score_counts.csv` and `outputs/iter_48_resnet_review_analysis_balanced40_smoothed_faithfulness/review_flag_counts.csv` by `scripts/make_thesis_charts.py`.

Chart 4.3: Balanced 40-case review failure taxonomy and qualitative flags

![Chart 4.3 failure taxonomy and flags](../outputs/iter_61_thesis_charts/chart_4_3_failure_taxonomy.png)

TODO (human review): choose 1-2 visual examples for Figure 4.2 that illustrate the dominant failure category `non_pathological_high_contrast` without implying that all non-mask evidence is clinically irrelevant.

Table 4.5: Balanced 40-case review failure taxonomy

| Failure category | Count / 40 | Interpretation |
| --- | ---: | --- |
| `correct` | `10` | explanation pattern judged clinically aligned |
| `partial` | `8` | partly aligned or incomplete explanation |
| `non_pathological_high_contrast` | `13` | dominant failure mode; model evidence often follows high-contrast structures rather than pneumothorax mask |
| `clinically_misleading` | `7` | explanation emphasizes regions that could mislead clinical interpretation |
| `devices_text_artifacts` | `2` | explicit device/text artifact category; separate from clinically related tubes or subcutaneous emphysema notes |

### 4.5 Explanation Improvement Experiment

The Phase 5.2 explanation-improvement experiment evaluates whether the frozen four-method consensus improves positive-lesion localization over individual methods on held-out positive CXR test cases. The selected-baseline run uses train-split top-fraction calibration only: `outputs/iter_57_chex_calibration_v3/calibrated_thresholds_v3.csv` for DenseNet-chex and `outputs/iter_50_resnet_calibration_v3/calibrated_thresholds_v3.csv` for ResNet-50. The held-out outputs are stored in `outputs/iter_58_chex_improvement_v3/` and `outputs/iter_52_resnet_improvement_v3/`. The earlier DenseNet-all outputs remain historical/original-baseline continuity artifacts, not the current selected DenseNet baseline.

Each run produced `1,600` per-case metric rows (`200` cases x `8` positive-view methods), paired consensus-vs-method Wilcoxon signed-rank tests, Holm-Bonferroni adjusted p-values, bootstrap confidence intervals, and aggregate plots. Table 4.6 reports the Dice-focused paired comparison because Dice is the most interpretable overlap summary for selected attribution regions. Full IoU, pointing-hit, and precision-at-fraction results remain available in `improvement_experiment_paired.csv` for both runs.

Key interpretation points for the improvement and visualization work:
- The top-fraction sweep now stops after selected-mask coverage reaches approximately full-image coverage (`--stop-fractions-at-coverage 0.95`). This avoids showing redundant high-fraction panels when lower fractions already cover the whole image and prevents over-interpreting visually saturated masks.
- Smoothed IG/GradientSHAP maps improved review readability, but the improvement is best framed as making pixel-level attribution more inspectable, not necessarily more clinically correct.
- Exploratory cross-case pattern analysis found moderately high visual cosine similarity for IG/GradientSHAP maps across the 10 review cases (`mean` roughly `0.53-0.55` across positive, negative, magnitude, and signed views). Higher cross-case similarity tended to associate with lower localization score in this small sample, with strongest observed Spearman around `rho=-0.54` on `n=10` for magnitude views. Treat this as a hypothesis-generating observation: pixel-level methods may sometimes show case-invariant or preprocessing-driven patterns, so qualitative review should check whether maps are case-specific.

Final figure candidates for this section are the aggregate boxplots and paired-difference plots from `outputs/iter_58_chex_improvement_v3/improvement_experiment_boxplots.png`, `outputs/iter_58_chex_improvement_v3/improvement_experiment_paired_diff.png`, `outputs/iter_52_resnet_improvement_v3/improvement_experiment_boxplots.png`, and `outputs/iter_52_resnet_improvement_v3/improvement_experiment_paired_diff.png`. The CT improvement run provides matching figures generated with the same plotting style, `outputs/iter_54_ct_improvement_test/ct_improvement_experiment_boxplots.png` and `outputs/iter_54_ct_improvement_test/ct_improvement_experiment_paired_diff.png`, so the CT panel can be shown alongside the CXR ones without mixing incomparable visual encodings.

Graph 4.5: Phase 5.2 paired consensus-vs-individual localization differences

| DenseNet-chex paired differences | ResNet-50 paired differences | CT paired differences |
| --- | --- | --- |
| ![DenseNet-chex paired differences](../outputs/iter_58_chex_improvement_v3/improvement_experiment_paired_diff.png) | ![ResNet-50 paired differences](../outputs/iter_52_resnet_improvement_v3/improvement_experiment_paired_diff.png) | ![CT paired differences](../outputs/iter_54_ct_improvement_test/ct_improvement_experiment_paired_diff.png) |

TODO (human review): decide whether the CT paired-difference graph should share Graph 4.5 with CXR or be kept as a separate cross-modality graph, because the CT consensus definition differs from the frozen CXR consensus.

Table 4.6: Phase 5.2 improvement-experiment paired Dice comparison

| Model | Method compared with frozen consensus | Metric | Consensus median | Method median | Median paired difference | 95% bootstrap CI | Wilcoxon p | Holm-adjusted p | Interpretation |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `densenet121-res224-chex` | `grad_cam` | Dice | `0.0211` | `0.0222` | `0.0000` | `[0.0000, 0.0016]` | `0.0441` | `0.2205` | not significant |
| `densenet121-res224-chex` | `grad_cam_plus_plus` | Dice | `0.0211` | `0.0127` | `0.0000` | `[0.0000, 0.0009]` | `0.8024` | `0.8024` | not significant |
| `densenet121-res224-chex` | `integrated_gradients` | Dice | `0.0211` | `0.0226` | `-0.0005` | `[-0.0033, 0.0041]` | `0.1972` | `0.3945` | not significant |
| `densenet121-res224-chex` | `gradient_shap` | Dice | `0.0211` | `0.0213` | `-0.0002` | `[-0.0024, 0.0052]` | `0.0499` | `0.2205` | not significant |
| `densenet121-res224-chex` | `occlusion` | Dice | `0.0211` | `0.0191` | `0.0000` | `[-0.0017, 0.0000]` | `0.1111` | `0.3333` | not significant |
| `densenet121-res224-chex` | `eigen_cam` | Dice | `0.0211` | `0.0186` | `0.0004` | `[-0.0013, 0.0036]` | `0.0309` | `0.1855` | not significant |
| `densenet121-res224-chex` | `score_cam` | Dice | `0.0211` | `0.0190` | `0.0015` | `[-0.0002, 0.0068]` | `0.0005` | `0.0033` | significant, small effect |
| `resnet50-res512-all` | `grad_cam` | Dice | `0.0218` | `0.0172` | `0.0001` | `[0.0000, 0.0010]` | `0.6000` | `0.9789` | not significant |
| `resnet50-res512-all` | `grad_cam_plus_plus` | Dice | `0.0218` | `0.0085` | `0.0030` | `[0.0007, 0.0054]` | `0.0002` | `0.0006` | significant |
| `resnet50-res512-all` | `integrated_gradients` | Dice | `0.0218` | `0.0184` | `0.0039` | `[-0.0001, 0.0098]` | `<0.0001` | `<0.0001` | significant |
| `resnet50-res512-all` | `gradient_shap` | Dice | `0.0218` | `0.0183` | `0.0040` | `[0.0004, 0.0097]` | `<0.0001` | `<0.0001` | significant |
| `resnet50-res512-all` | `occlusion` | Dice | `0.0218` | `0.0252` | `0.0001` | `[0.0000, 0.0024]` | `0.0139` | `0.0416` | significant, very small effect |
| `resnet50-res512-all` | `eigen_cam` | Dice | `0.0218` | `0.0139` | `0.0051` | `[0.0011, 0.0076]` | `<0.0001` | `0.0002` | significant |
| `resnet50-res512-all` | `score_cam` | Dice | `0.0218` | `0.0144` | `0.0004` | `[0.0000, 0.0024]` | `0.4894` | `0.9789` | not significant |

The Dice table should be read together with the aggregate means. For DenseNet-chex, consensus is statistically better than `score_cam` by Dice/IoU/precision, but it is not the best method by aggregate mean Dice (`occlusion` and `grad_cam_plus_plus` are slightly higher descriptively). For ResNet-50, consensus is statistically better than several methods by Dice/IoU, but `grad_cam` and `score_cam` remain competitive and the absolute overlap remains weak. This supports a nuanced conclusion: consensus can stabilize localization when individual methods are noisy or weak, but it is not a substitute for a clinically aligned classifier.

### 4.6 Limitations

This study should be interpreted through explicit threats to validity rather than as a clinical-deployment validation.

**Internal validity.** The results depend on preprocessing, image size, classifier threshold, XAI hyperparameters, top-fraction calibration, smoothing, perturbation baseline, and random sampling. The project controls these risks by freezing classifier thresholds, using fixed seeds, versioning calibration artifacts, and reporting method settings. However, small changes in baseline choice or smoothing can alter pixel-level attribution maps, especially for `Integrated Gradients` and `GradientSHAP`. Long-running experiments also use practical speed settings such as capped `Score-CAM` channels or coarse occlusion patches; thesis-quality reruns should report whether such approximations were used.

**Construct validity.** Pneumothorax masks are useful reference annotations, but they are imperfect proxies for explanation quality. A model may highlight clinically related indirect evidence such as subcutaneous emphysema or treatment devices that does not overlap the mask. Conversely, a heatmap may overlap the mask for non-causal or visually diffuse reasons. Therefore, mask metrics are interpreted as localization evidence, not as complete clinical validity. Negative evidence is also a separate construct: overlap between suppressive evidence and the lesion is not automatically good.

**External validity.** The tested CXR models are off-the-shelf TorchXRayVision classifiers, not SIIM-specific pneumothorax localization models. Low mask overlap may reflect model/data mismatch, report-label pretraining, or transfer-distribution shift as much as explanation-method failure. The Stage A model sweep improves this limitation by comparing several TorchXRayVision weights and architectures, but the out-of-family external-model slot remains unresolved because the checked MONAI CXR candidate was generative rather than a pneumothorax classifier. The results should therefore be framed as evidence about this off-the-shelf model family on SIIM pneumothorax, not as a universal claim about all CXR AI systems.

**Statistical validity.** The balanced ResNet review is a single-rater, 40-case, outcome-stratified study with 10 `tp`, 10 `fp`, 10 `tn`, and 10 `fn` cases. It is exploratory by design and is not a powered multi-rater reader study. There is no inter-rater reliability statistic, no formal clinical-deployment validation, and no prospective workflow assessment. Review-score correlations and cross-case pattern-similarity observations should therefore be described as supporting qualitative evidence rather than definitive statistical proof. The earlier 10-case smoothed review is retained only as a methodological pilot; the balanced 40-case set is the canonical review evidence.

**Visualization and interpretation limits.** Heatmap readability depends on color mapping, selected view, selected top fraction, smoothing, and overlay opacity. All thesis figures should state these settings and should remind the reader that attribution maps are class-specific model-evidence visualizations, not anatomical segmentations. Faithfulness curves have a related limitation: they evaluate sensitivity of the model output to a perturbation design, so their clinical meaning depends on whether the replacement baseline is appropriate for the modality.

**Scope limits.** The CT component is a completed pilot, not a full second clinical validation study. It uses one PhysioNet CT hemorrhage dataset, one DifeiT ViT checkpoint, positive masked slices, and input-space explanation methods only. The CXR and CT consensus definitions also differ: CXR uses the frozen four-method consensus, whereas CT uses `consensus_input3` because the convolutional CAM implementations do not directly apply to the Vision Transformer. Similarly, `LIME`, Captum infidelity/sensitivity, attention-weighted consensus, and transformer-specific CAM/attention methods should be reported only if implemented and verified; otherwise they remain future-work items.

### Conclusions to Chapter 4

Chapter 4 shows that the evaluation of explanations must be separated from the evaluation of classifier predictions. The original DenseNet-all baseline remains useful as a historical weak external reference, while DenseNet-chex is the selected DenseNet baseline after Stage A and `resnet50-res512-all` is the strongest tested TorchXRayVision follow-up by aggregate localization. However, the absolute localization values remain low, so the model comparison supports a relative improvement claim rather than a clinical localization claim.

The quantitative explanation results should be interpreted through complementary metrics. `IoU`, `Dice`, and `precision_at_fraction` describe selected-region overlap with the pneumothorax mask, while `pointing_hit` is a stricter peak-localization diagnostic. Signed diagnostics and negative-evidence measures answer a different question and should not be collapsed into positive lesion-overlap scores. Faithfulness curves add another layer by testing whether highlighted pixels affect the model output under deletion or insertion, but they do not prove that the highlighted evidence is clinically appropriate.

The balanced 40-case ResNet review provides the strongest qualitative CXR evidence currently available. It shows that explanations can be useful or potentially useful in many cases, but clinically misleading and poorly localized maps remain common. Non-pathological high-contrast evidence, device/tube confounding, indirect clinically related signs, method disagreement, weak pixel attribution, subcutaneous emphysema, and mask-quality concerns all affect interpretation. The Phase 5.2 and Phase 5.4 improvement experiments add that consensus behavior is model-, metric-, and modality-dependent: for DenseNet-chex it mainly improves over Score-CAM in paired overlap tests, for ResNet-50 it improves Dice/IoU over several weaker methods while remaining comparable to strong CAM alternatives, and for CT it gives its clearest benefit on pointing-hit rather than broad overlap. Together, these findings support the broader thesis framing: XAI maps are valuable when they are validated as model-behavior diagnostics, not when they are displayed as automatic evidence of trustworthiness.

## Chapter 5. Conclusions and Recommendations

### 5.1 Main Findings

The thesis supports a cautious interpretation of explainable AI in medical image classification. The main finding is that heatmaps can be useful for auditing model behavior, but they should not be treated as automatic evidence of clinical correctness. In the CXR pneumothorax case study, the tested off-the-shelf TorchXRayVision models can produce class-specific attribution maps, yet many maps remain weakly localized to the pneumothorax masks or highlight indirect, confounded, or high-contrast non-lesion structures.

The Stage A model comparison shows that model choice matters. `densenet121-res224-chex` improves over the original `densenet121-res224-all` checkpoint within the DenseNet-121 branch, while `resnet50-res512-all` is the strongest tested TorchXRayVision candidate by aggregate localization in the completed diagnostic sweep. However, the absolute localization values remain low. This means the CheX and ResNet-50 results should be interpreted as better relative baselines, not as clinically strong pneumothorax localizers.

The radiologist-centered review reinforces the quantitative caution. In the balanced 40-case ResNet review, explanation usefulness was mixed: many cases contained some interpretable signal, but misleading or poorly localized maps remained common. Non-pathological high-contrast evidence, device/tube confounding, indirect evidence, method disagreement, weak pixel attribution, subcutaneous emphysema, and mask-quality issues all appeared as clinically relevant interpretation factors. These findings support the thesis claim that XAI is most valuable when used to expose model limitations rather than to create trust automatically.

The Phase 5.2 consensus-improvement experiment shows that consensus is not universally superior. In the DenseNet-chex baseline, the frozen consensus significantly improves overlap mainly over `score_cam`, while `occlusion` and `grad_cam_plus_plus` remain slightly higher by aggregate mean Dice. In the ResNet-50 baseline, consensus significantly improves Dice/IoU over several weaker methods, but not over `grad_cam` or `score_cam`. This supports a model-dependent conclusion: cross-method consensus can stabilize localization under some baselines, but averaging weak or mislocalized evidence does not automatically produce clinically aligned explanations. The methodological contribution remains that explanation claims should be evaluated through classifier behavior, localization, faithfulness, method agreement, and clinical review together.

The completed CT pilot extends this conclusion across modality without turning it into a universal consensus claim. On positive masked CT hemorrhage slices, `consensus_input3` significantly improves `pointing_hit` over all three input-space constituent methods, but its overlap metrics are not Holm-significantly better than individual methods. The result is useful because it shows that the same validation framework can be executed beyond CXR and can reveal a different consensus signature; it remains limited by a single CT model/dataset, a calibration-floor effect at top fraction `0.05`, and the absence of transformer-specific CAM methods in the current draft.

### 5.2 Practical Recommendations

The first practical recommendation is to report classifier performance and explanation quality separately. A model's `AUC`, sensitivity, specificity, or F1 score should not be used as a substitute for evidence that the model localizes pathology. Conversely, a poor localization score should be interpreted in relation to the model, dataset, target class, and reference annotation rather than blamed automatically on one XAI method.

The second recommendation is to keep explanation views semantically separate. Positive evidence, negative evidence, magnitude, and signed views answer different questions. Positive evidence can be evaluated against positive lesion masks; negative evidence should be interpreted as suppressive evidence and should not be rewarded merely for overlapping the lesion. Magnitude maps answer where the image was influential, not whether the influence supported the disease class. Signed maps summarize the balance between positive and negative evidence but should not replace the separate views.

The third recommendation is to use multiple validation layers. For thesis-scale XAI evaluation, the minimum useful panel is: mask-overlap metrics (`IoU`, `Dice`, `precision_at_fraction`), strict peak localization (`pointing_hit`), faithfulness curves, signed-evidence diagnostics, method agreement or disagreement, and structured radiologist-style review. These layers should be interpreted together because each has a different failure mode.

The fourth recommendation is to handle pixel-level attribution methods carefully. `Integrated Gradients` and `GradientSHAP` should be reported with baseline choice, step/sample count, random seed, smoothing/readability settings, and any cross-case pattern checks. If such methods produce visually similar maps across unrelated cases, they should be treated as hypothesis-generating or diagnostic rather than as strong case-specific clinical evidence.

The fifth recommendation is to design review materials for auditability. Each selected case should preserve the source image, mask or mask contour, classifier outcome, probability, continuous heatmap views, thresholded selections, and method settings. This allows a reviewer to distinguish direct lesion localization, indirect clinically related evidence, treatment/device confounding, non-pathological high-contrast structures, and map artifacts.

The final practical recommendation is to avoid over-claiming XAI. Heatmaps are best used as part of a validation and model-audit workflow. They can identify when a classifier is behaving plausibly, but their more important thesis role is to show when a model's apparent success is not supported by clinically expected evidence.

### 5.3 Future Work

Several extensions follow directly from the limitations of the current work.

First, future work should evaluate stronger pneumothorax-specific or externally trained classifiers. The current off-the-shelf TorchXRayVision baselines are useful for auditing transfer behavior, but a full clinical claim would require models whose development target, labels, preprocessing, and validation data are closer to SIIM pneumothorax localization. Any stronger model should undergo the same protocol: classifier-threshold calibration, XAI top-fraction calibration, held-out localization metrics, negative-evidence diagnostics, faithfulness curves, and radiologist-centered review. A preliminary drop-in probe of an off-the-shelf Vision Transformer ChestX-ray14 classifier confirmed the same weak-detection and weak, off-lesion localization pattern observed for the convolutional baselines, so a controlled transformer comparison (a probe-trained transformer encoder paired with a token-to-grid CAM adapter so the full method panel transfers) is left to future work rather than treated as a result here.

Second, the CT pilot should be expanded beyond the current Branch A scope. The present draft verifies one hemorrhage classifier and one mask source, but a full CT study would add broader test screening, larger or multi-source annotated hemorrhage masks, subtype-aware analysis, multi-rater review, and a calibration sweep that extends below top fraction `0.05` for small hemorrhage masks. Candidate future sources include RSNA-IHD-derived classifier work and additional public CT hemorrhage datasets with masks, but every extension must verify class heads, licenses, preprocessing, and data compatibility before results are claimed.

Third, the radiologist review should be scaled beyond the current single-rater balanced review. A stronger reader study would include multiple radiologists, inter-rater reliability statistics, a larger and prospectively defined case sample, blinded scoring, and clearer separation between direct lesion localization and indirect clinically relevant evidence. DECIDE-AI-style reporting would be appropriate for any study that moves closer to workflow evaluation.

Fourth, additional explanation methods and robustness metrics can be added. `LIME` would provide a region-level surrogate-family comparator. Captum infidelity and sensitivity metrics could triangulate faithfulness and robustness beyond deletion/insertion curves. These additions should remain secondary unless they are implemented with the same calibration and held-out evaluation discipline as the existing methods.

Fifth, an attention-weighted consensus (`consensus_attention`) can be studied after the unweighted consensus result is known. This variant would assign a coefficient `alpha_m` to each constituent method `m`, using possible policies such as calibration-set mask-IoU, agreement with the method panel, or perturbation stability. The coefficients would be attention weights across explanation methods, not architectural attention inside the classifier. The method is deferred because it introduces a new hyperparameter family and would require its own calibration protocol to avoid test-set peeking.

Sixth, the CT pilot can be extended to the CAM-family explanation methods by adopting transformer-specific spatial-explanation techniques. Because the integrated CT classifier is a Vision Transformer, `Grad-CAM`, `Grad-CAM++`, `Eigen-CAM`, and `Score-CAM` cannot be applied with the convolutional-feature-map implementation reused across the chest-X-ray models. Concrete adaptation paths exist, and their exclusion from the current draft is a deliberate methodological choice rather than an oversight: the widely-used `pytorch-grad-cam` library exposes a token-to-grid reshape that runs CAM-family methods on Vision Transformers; `ViT-CX` provides a transformer-specific, Score-CAM-style causal explanation built on patch embeddings; and transformer-native methods such as attention rollout and transformer attribution explain Vision Transformers through attention flow and relevance propagation. Each of these is, however, a different implementation from the convolutional chest-X-ray code path, so adopting one would re-introduce an implementation confound into a comparison whose strength is that the explanation algorithm is held constant across modalities; for this reason the current draft restricts the CT pilot to the input-space methods whose code is identical on both pipelines. Implementing and validating one of these transformer-specific approaches in a controlled way would let the full method panel, including a four-method consensus comparable to the chest-X-ray definition, be evaluated across both modalities.

Finally, future work should broaden the clinical scope. Additional pathologies, larger multi-institutional datasets, stronger segmentation references, and prospective evaluation would test whether the current cautionary findings generalize. A publication-ready version of the work should include a cleaned code release, documented experiment configurations, frozen calibration artifacts, and a reproducible subset of de-identified review figures where data licenses permit.

## Bibliography

Use IEEE by default unless the supervisor requests APA. Before final submission, convert all entries from `docs/references.md` into one consistent bibliography style. Every online source should include a stable URL and an access date in the required form, for example `[Online]. Available: <URL>. Accessed: <date>`. The final bibliography must be large enough for the template rule that the number of sources should be at least the number of pages in Chapters 2-4.

Final bibliography checklist:

- verify all method papers: `Grad-CAM`, `Grad-CAM++`, `Integrated Gradients`, `SHAP`, `Score-CAM`, `Eigen-CAM`, occlusion sensitivity, and `LIME` if used;
- verify medical-imaging AI and reporting sources: CXR datasets, TorchXRayVision, CLAIM, TRIPOD+AI, QUADAS-AI, DECIDE-AI, and RSNA-IHD if CT is discussed;
- verify explanation-validation sources: faithfulness, sanity checks, saliency-map trustworthiness, shortcut learning, hidden stratification, and radiology shortcut bias;
- ensure citation keys in the prose map cleanly to final numbered bibliography entries;
- avoid citing unverified blogs, informal tutorials, or unofficial mirrors unless explicitly approved.

## Appendices

### Appendix A. Experiment Configuration

Include reproducibility information that would interrupt the main text if placed in Chapter 3. Suggested contents:

- final environment snapshot: WSL Ubuntu, Python version, PyTorch version, CUDA availability, `torchxrayvision` version, and project editable-install note;
- final dataset-count verification for the local SIIM-ACR snapshot: number of PNG images, masks, positive cases, negative cases, and split counts;
- final DenseNet-chex v3 calibration command and output folder;
- final ResNet-50 v3 calibration command and output folder;
- final DenseNet-chex Phase 5.2 improvement-experiment command and output folder;
- final ResNet-50 Phase 5.2 improvement-experiment command and output folder;
- final CT smoke and improvement-experiment commands and output folders: `outputs/iter_53_ct_smoke_test/` and `outputs/iter_54_ct_improvement_test/`;
- final DenseNet-chex full test classifier screen: `outputs/iter_56_chex_classifier_eval_test/`;
- final ResNet-50 full test classifier screen: `outputs/iter_55_resnet_classifier_eval_test/`;
- classifier threshold sources, including `0.565` for `densenet121-res224-chex`, `0.525` for `resnet50-res512-all`, and the historical `0.62` operating point for `densenet121-res224-all`;
- random seeds, `ig_steps`, `gradshap_samples`, occlusion patch/stride, `score_cam_channels_cap`, faithfulness baseline, smoothing settings, and selected calibrated-fraction files;
- note that historical v1/v2 output folders are preserved and not overwritten.

Example command block template to fill with final paths:

```powershell
wsl.exe --cd /mnt/c/Users/Dmytro.Valantsevych/Downloads/master_thesis_draft_explainAI python3 scripts/run_improvement_experiment.py `
  --weights <model_weights> `
  --image-size <224_or_512> `
  --split test `
  --max-positive <n> `
  --random-sample `
  --seed 20260515 `
  --ig-steps 16 `
  --gradshap-samples 8 `
  --occlusion-patch-size 32 `
  --occlusion-stride 16 `
  --score-cam-channels-cap 256 `
  --calibration-csv outputs/<calibration_run>/calibrated_thresholds_v3.csv `
  --device auto `
  --output-dir outputs/<improvement_run>
```

Final command/path checklist to preserve in the appendix:

| Purpose | Output artifact |
| --- | --- |
| DenseNet-chex full test classifier screen | `outputs/iter_56_chex_classifier_eval_test/` |
| DenseNet-chex CXR calibration v3 | `outputs/iter_57_chex_calibration_v3/calibrated_thresholds_v3.csv` |
| ResNet-50 CXR calibration v3 | `outputs/iter_50_resnet_calibration_v3/calibrated_thresholds_v3.csv` |
| DenseNet-chex CXR held-out improvement | `outputs/iter_58_chex_improvement_v3/` |
| ResNet-50 CXR held-out improvement | `outputs/iter_52_resnet_improvement_v3/` |
| CT smoke test | `outputs/iter_53_ct_smoke_test/` |
| CT calibrated improvement test | `outputs/iter_54_ct_improvement_test/` |
| ResNet-50 full test classifier screen | `outputs/iter_55_resnet_classifier_eval_test/` |

### Appendix B. Radiologist Review Template

Include the review-task instructions and scoring schema used for the balanced review workbook. The appendix should make the rater task auditable without requiring the reader to open the HTML workbook.

Suggested contents:

- source workbook path and scored output path for the canonical balanced 40-case review;
- case selection rule: balanced `tp`/`fp`/`tn`/`fn` design with 10 cases per outcome group;
- fields shown to the reviewer: source image, mask or mask contour where available, classifier outcome, probability, and explanation overlay grid;
- explanation-view semantics: positive evidence, negative evidence, magnitude, and signed view;
- scoring categories:
  - `localization_score` in `{correct, partial, incorrect, none}`;
  - `usefulness_score` in `{useful, potentially_useful, misleading, not_useful}`;
  - `failure_category` in `{correct, partial, anatomically_related, devices_text_artifacts, non_pathological_high_contrast, diffuse_non_specific, clinically_misleading}`;
  - free-text `artifact_note` and `comment`;
- warmup-case procedure and note that the review is a structured qualitative assessment, not prospective clinical validation;
- final review-score distribution table or a pointer to Table 4.4 and Table 4.5.

### Appendix C. Additional Figures

Reserve this appendix for supporting visual examples that are useful but too detailed for Chapter 4.

Suggested contents:

- additional representative `tp`, `fp`, `tn`, and `fn` cases with all method rows and four views per method;
- threshold-sweep panels showing how top-fraction selection changes selected regions;
- examples of direct lesion localization, indirect clinically related evidence, device/tube confounding, non-pathological high-contrast evidence, and clinically misleading explanations;
- comparison panels for DenseNet-chex versus ResNet-50 on matched or similar cases if final artifacts support this;
- faithfulness-curve details and family-split plots not included in the main chapter;
- final Phase 5.2 consensus-vs-individual figures from `outputs/iter_58_chex_improvement_v3/` and `outputs/iter_52_resnet_improvement_v3/`;
- optional CT summary plot generated from `outputs/iter_54_ct_improvement_test/ct_improvement_experiment.csv`, if included with the same caption caveats as Table 4.3.

Every appendix figure should include the source image stem in the filename or caption, the model, method, view, classifier probability, threshold/fraction, and whether the case is `tp`, `fp`, `tn`, or `fn`.

