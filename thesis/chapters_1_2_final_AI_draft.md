# Chapters 1-2 Final AI Draft

Standalone AI-assisted draft generated on `2026-06-04` from `thesis/thesis_skeleton.md`. This file intentionally contains only Chapters 1 and 2 plus embedded reference/source notes needed for standalone PDF rendering. Human review is still required, including for final citations, template formatting, and any `TODO (human review)` markers.
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

Figure 2.1: Conceptual risk of visually plausible but clinically misleading saliency maps

```text
Image-level classifier says "positive"
             |
             v
      Saliency / attribution map
             |
   +---------+---------+
   |                   |
   v                   v
Lesion-aligned     Visually plausible but
model evidence     clinically misleading evidence
(mask overlap,     (device, border, text marker,
faithful score      high-contrast anatomy, or
change, reviewer    shortcut cue)
support)
   |                   |
   +---------+---------+
             |
             v
Validation is required before interpreting the map as clinical evidence:
localization metrics + faithfulness curves + method agreement + human review.
```

The schematic emphasizes the core literature-review point used in this thesis draft: a visually plausible region of high attribution is not equivalent to a clinically validated lesion explanation. The map must be interpreted together with classifier behavior, localization, faithfulness, and human-centered review.

TODO (human review): select or draw Figure 2.1 and confirm that any clinical image used in Chapter 2 is allowed under the relevant dataset license and institutional submission rules.

### 2.5 Research Gap

The reviewed literature supports three conclusions. First, deep learning classifiers can be useful in radiology, but image-level performance does not guarantee lesion-aligned reasoning. Second, XAI methods are diverse and method-dependent: CAM maps, pixel attributions, Shapley-style approximations, and occlusion diagnostics answer different questions. Third, explanation validation remains difficult because a visually plausible heatmap may be unfaithful, poorly localized, or clinically misleading.

The research gap addressed by this thesis is the lack of a compact but layered validation workflow that compares several XAI methods on the same medical imaging task using mask localization, perturbation faithfulness, method agreement, signed-evidence interpretation, and radiologist-centered failure review. Many studies present heatmaps as qualitative illustrations; fewer test whether those heatmaps align with available lesion masks, whether they affect the model output under controlled perturbation, and whether a medically trained reviewer finds them useful or misleading.

The thesis also focuses on a practically important negative possibility: a pretrained medical classifier may be predictive enough to appear useful while its explanations reveal shortcut learning or weak clinical localization. Demonstrating this failure mode is scientifically valuable because it clarifies what XAI can and cannot justify. The contribution is not the claim that heatmaps automatically create trust, but that validated heatmaps can help decide when trust should be withheld.

### Conclusions to Chapter 2

Chapter 2 reviewed the literature background needed for the thesis. Deep learning classifiers are now common in radiology research, especially in chest X-ray analysis, but public dataset labels, source institutions, and clinical contexts differ substantially. A model trained or pretrained on one data mixture cannot be assumed to localize pathology correctly on another dataset. This motivates the thesis decision to evaluate off-the-shelf TorchXRayVision baselines not only by image-level prediction, but also by explanation localization and qualitative failure modes.

The reviewed XAI methods form complementary families. CAM-family methods provide class-discriminative spatial maps from internal activations; pixel-attribution methods trace gradients or Shapley-style contributions back to the input; occlusion directly perturbs image regions; and surrogate approaches such as LIME explain local behavior through interpretable region perturbations. Because these methods answer different questions, agreement between them is stronger evidence than any single heatmap, while disagreement is itself diagnostically useful.

The validation literature shows that heatmaps must be tested rather than merely displayed. Localization metrics, faithfulness curves, robustness checks, and human-centered review each capture different aspects of explanation quality. In medical imaging, this is especially important because shortcut cues, devices, acquisition artifacts, and non-pathological high-contrast anatomy can all become model evidence. The research design in Chapter 3 follows directly from this gap: it evaluates explanations through a layered protocol that separates classifier behavior, mask localization, faithfulness to the model, signed-evidence semantics, and radiologist-centered usefulness.
## Built-in References and Source Notes for Standalone Rendering

This standalone AI draft embeds the working thesis reference inventory below so that the file can be rendered independently for supervisor review. The bibliography still requires human verification, final style conversion, and access-date checks before formal submission.

# Thesis Research References

This file lists reliable, verifiable sources used for thesis background notes. Reference IDs are stable cross-links for `docs/thesis-notes.md`.

Access date for URLs checked: 2026-05-24, 2026-05-25, and 2026-05-27 (the latter added: `REF-WILCOXON-1945`, `REF-HOLM-1979`, `REF-AICKIN-GENSLER-1996`, `REF-DEMSAR-2006`, `REF-EIGEN-CAM`, `REF-LIME`, `REF-INFIDELITY-SENSITIVITY`, `REF-RSNA-IHD`).

## Source Selection Rules

- Preferred sources: peer-reviewed conference/journal papers, proceedings pages, PubMed/PMC/Nature/PMLR/CVF/NeurIPS/IEEE records, official dataset pages, and official model/library documentation.
- Excluded unless explicitly approved later: blogs, unsourced tutorials, social media posts, unofficial dataset mirrors, and competition writeups without a stable official or peer-reviewed context.
- Citation style: IEEE-like compact entries for working notes; final thesis bibliography can be converted to the university-required final style.

## XAI Methods

### <a id="ref-gradcam"></a>`REF-GRADCAM` — Grad-CAM

- IEEE-style entry: R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization," in *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2017.
- Source type: peer-reviewed conference paper.
- Verified URLs:
  - CVF proceedings page: https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
  - arXiv record: https://arxiv.org/abs/1610.02391
- Reliability status: high; official CVF proceedings and arXiv record.
- Thesis relevance: core CAM-family method for class-discriminative localization in CNN classifiers; directly supports the project’s `grad_cam` implementation and the explanation that heatmaps are class-specific attribution maps, not segmentations.

### <a id="ref-gradcampp"></a>`REF-GRADCAMPP` — Grad-CAM++

- IEEE-style entry: A. Chattopadhay, A. Sarkar, P. Howlader, and V. N. Balasubramanian, "Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks," in *2018 IEEE Winter Conference on Applications of Computer Vision (WACV)*, 2018, pp. 839-847.
- Source type: peer-reviewed conference paper.
- Verified URLs:
  - arXiv record: https://arxiv.org/abs/1710.11063
  - IEEE Xplore record: https://ieeexplore.ieee.org/document/8354201
- Reliability status: high; IEEE conference publication with arXiv preprint.
- Thesis relevance: motivates comparison between `grad_cam` and `grad_cam_plus_plus`, especially for cases where localization extent or multiple relevant regions differ.

### <a id="ref-integrated-gradients"></a>`REF-INTEGRATED-GRADIENTS` — Integrated Gradients

- IEEE-style entry: M. Sundararajan, A. Taly, and Q. Yan, "Axiomatic Attribution for Deep Networks," in *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR, vol. 70, 2017, pp. 3319-3328.
- Source type: peer-reviewed conference paper.
- Verified URLs:
  - PMLR proceedings PDF/page: https://proceedings.mlr.press/v70/sundararajan17a.html
  - arXiv record: https://arxiv.org/abs/1703.01365
- Reliability status: high; official ICML/PMLR record.
- Thesis relevance: supports baseline-dependent attribution analysis and the need to discuss baseline choice carefully for pixel-space medical images.

### <a id="ref-shap"></a>`REF-SHAP` — SHAP / Shapley Additive Explanations

- IEEE-style entry: S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Advances in Neural Information Processing Systems*, 2017.
- Source type: peer-reviewed conference paper.
- Verified URLs:
  - NeurIPS proceedings page: https://proceedings.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
  - Semantic Scholar record: https://www.semanticscholar.org/paper/A-Unified-Approach-to-Interpreting-Model-Lundberg-Lee/442e10a3c6640ded9408622005e3c2a8906ce4c2
- Reliability status: high; NeurIPS proceedings and bibliographic record.
- Thesis relevance: conceptual basis for Shapley-value attribution and the project’s `gradient_shap` family; useful for explaining additive local attribution assumptions.

### <a id="ref-scorecam"></a>`REF-SCORECAM` — Score-CAM

- IEEE-style entry: H. Wang, Z. Wang, M. Du, F. Yang, Z. Zhang, S. Ding, P. Mardziel, and X. Hu, "Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks," in *2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2020, pp. 111-119.
- Source type: peer-reviewed workshop paper.
- Verified URLs:
  - arXiv record: https://arxiv.org/abs/1910.01279
  - CVF PDF: https://openaccess.thecvf.com/content_CVPRW_2020/papers/w1/Wang_Score-CAM_Score-Weighted_Visual_Explanations_for_Convolutional_Neural_Networks_CVPRW_2020_paper.pdf
  - DOI record visible via NJIT/IEEE metadata: https://doi.org/10.1109/CVPRW50498.2020.00020
- Reliability status: high; IEEE/CVF workshop publication and DOI.
- Thesis relevance: planned additional CAM-family method; useful as a gradient-free bridge between activation mapping and perturbation-style scoring.

### <a id="ref-occlusion-zeiler"></a>`REF-OCCLUSION-ZEILER` — Occlusion Sensitivity

- IEEE-style entry: M. D. Zeiler and R. Fergus, "Visualizing and Understanding Convolutional Networks," in *Computer Vision -- ECCV 2014*, Lecture Notes in Computer Science, vol. 8689, Springer, 2014, pp. 818-833.
- Source type: peer-reviewed conference paper.
- Verified URLs:
  - Springer chapter page: https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53
  - Author-hosted PDF: https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf
- Reliability status: high; official Springer proceedings page plus author-hosted copy.
- Thesis relevance: historical basis for occlusion sensitivity: systematically masking parts of an image and observing classifier-output changes. This supports the project’s `occlusion` method as a perturbation-style diagnostic distinct from gradient attribution.

### <a id="ref-eigen-cam"></a>`REF-EIGEN-CAM` — Eigen-CAM

- IEEE-style entry: M. B. Muhammad and M. Yeasin, "Eigen-CAM: Class Activation Map using Principal Components," in *2020 International Joint Conference on Neural Networks (IJCNN)*, 2020, pp. 1-7.
- Source type: peer-reviewed conference paper.
- Verified URLs:
  - arXiv record: https://arxiv.org/abs/2008.00299
  - IEEE Xplore record: https://ieeexplore.ieee.org/document/9206626
  - DOI: https://doi.org/10.1109/IJCNN48605.2020.9206626
- Reliability status: high; IJCNN conference publication with DOI.
- Thesis relevance: gradient-free CAM variant based on the principal component of target-layer activations. Useful in this thesis as a third CAM-family comparator alongside Grad-CAM and Grad-CAM++ that does not depend on backward passes, complementing Score-CAM's score-weighted forward variant.

### <a id="ref-lime"></a>`REF-LIME` — Local Interpretable Model-Agnostic Explanations

- IEEE-style entry: M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why Should I Trust You?': Explaining the Predictions of Any Classifier," in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*, 2016, pp. 1135-1144.
- Source type: peer-reviewed conference paper.
- Verified URLs:
  - ACM DOI: https://doi.org/10.1145/2939672.2939778
  - arXiv record: https://arxiv.org/abs/1602.04938
- Reliability status: high; ACM KDD publication.
- Thesis relevance: region-level surrogate-model explanation framework. Included as a conditional add-on (Phase 5.7) for cross-family comparison against gradient and perturbation attribution. Justifies presenting LIME as a third explanation family rather than a substitute for gradient or occlusion methods.

### <a id="ref-captum-docs"></a>`REF-CAPTUM-DOCS` — Captum Implementation Documentation

- IEEE-style entry: Meta PyTorch, "Captum: Model Interpretability for PyTorch," official documentation and API reference.
- Source type: official software documentation.
- Verified URLs:
  - Captum introduction: https://captum.ai/docs/introduction
  - Integrated Gradients tutorial: https://captum.ai/docs/extension/integrated_gradients
  - GradientShap API: https://captum.ai/api/gradient_shap.html
  - Attribution API index including `Occlusion`: https://captum.ai/api/attribution.html
- Reliability status: high for implementation semantics; official documentation for the library used to implement several attribution methods.
- Thesis relevance: supports method-implementation details such as Captum’s baseline handling, target-class attribution, GradientShap sampling/noise semantics, and occlusion API behavior. Use alongside original method papers rather than as a replacement for theoretical citations.

## XAI Evaluation and Faithfulness

### <a id="ref-sanity-checks"></a>`REF-SANITY-CHECKS` — Sanity Checks for Saliency Maps

- IEEE-style entry: J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, and B. Kim, "Sanity Checks for Saliency Maps," in *Advances in Neural Information Processing Systems*, 2018.
- Source type: peer-reviewed conference paper.
- Verified URLs:
  - NeurIPS proceedings page: https://papers.nips.cc/paper/8160-sanity-checks-for-saliency-maps
  - NeurIPS PDF: https://papers.neurips.cc/paper/2018/file/294a8ed24b1ad22ec2e7efea049b8737-Paper.pdf
- Reliability status: high; official NeurIPS proceedings.
- Thesis relevance: supports the caution that visually plausible maps are insufficient evidence; methods should be checked for sensitivity to model parameters and labels where feasible.

### <a id="ref-infidelity-sensitivity"></a>`REF-INFIDELITY-SENSITIVITY` — Infidelity and Sensitivity Faithfulness Metrics

- IEEE-style entry: C.-K. Yeh, C.-Y. Hsieh, A. S. Suggala, D. I. Inouye, and P. Ravikumar, "On the (In)fidelity and Sensitivity of Explanations," in *Advances in Neural Information Processing Systems*, vol. 32, 2019.
- Source type: peer-reviewed conference paper.
- Verified URLs:
  - NeurIPS proceedings page: https://papers.nips.cc/paper/2019/hash/a7471fdc77b3435276507cc8f2dc2569-Abstract.html
  - arXiv record: https://arxiv.org/abs/1901.09392
  - Captum API: https://captum.ai/api/metrics.html
- Reliability status: high; NeurIPS publication with available Captum implementation.
- Thesis relevance: infidelity measures the expected squared difference between attribution-explained perturbation effects and actual model output change; sensitivity-max measures the largest attribution change under small input perturbations. Both triangulate against the deletion/insertion faithfulness pipeline already in this project, strengthening the H8 "faithfulness vs localization" cross-test.

### <a id="ref-rise"></a>`REF-RISE` — RISE and Deletion/Insertion Metrics

- IEEE-style entry: V. Petsiuk, A. Das, and K. Saenko, "RISE: Randomized Input Sampling for Explanation of Black-box Models," in *British Machine Vision Conference (BMVC)*, 2018.
- Source type: peer-reviewed conference paper.
- Verified URLs:
  - BMVC PDF: http://bmvc2018.org/contents/papers/1064.pdf
  - arXiv record: https://arxiv.org/abs/1806.07421
  - DBLP record: https://dblp.org/rec/conf/bmvc/PetsiukDS18
- Reliability status: high; BMVC paper with arXiv and DBLP records.
- Thesis relevance: provides a standard explanation-faithfulness framing using deletion and insertion curves; directly aligns with the project’s perturb-and-reevaluate faithfulness pipeline.

### <a id="ref-samek-evaluation"></a>`REF-SAMEK-EVALUATION` — Perturbation-Based Relevance Evaluation

- IEEE-style entry: W. Samek, A. Binder, G. Montavon, S. Lapuschkin, and K.-R. Muller, "Evaluating the Visualization of What a Deep Neural Network Has Learned," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 28, no. 11, pp. 2660-2673, 2017.
- Source type: peer-reviewed journal article.
- Verified URLs:
  - IEEE Xplore search/title record: https://ieeexplore.ieee.org/document/7552539
  - DOI: https://doi.org/10.1109/TNNLS.2016.2599820
- Reliability status: high; IEEE journal article.
- Thesis relevance: supports perturbation/removal evaluation as a way to test whether highly relevant regions materially affect model predictions.

### <a id="ref-meaningful-perturbation"></a>`REF-MEANINGFUL-PERTURBATION` — Meaningful Perturbation Explanations

- IEEE-style entry: R. C. Fong and A. Vedaldi, "Interpretable Explanations of Black Boxes by Meaningful Perturbation," in *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2017, pp. 3449-3457.
- Source type: peer-reviewed conference paper.
- Verified URLs:
  - CVF PDF: https://openaccess.thecvf.com/content_ICCV_2017/papers/Fong_Interpretable_Explanations_of_ICCV_2017_paper.pdf
  - arXiv record with ICCV reference: https://arxiv.org/abs/1704.03296
  - DOI: https://doi.org/10.1109/ICCV.2017.371
- Reliability status: high; ICCV paper with DOI and official CVF/arXiv records.
- Thesis relevance: strengthens the perturbation-based explanation background. Useful for discussing why deletion/insertion and occlusion results depend on the perturbation operator and replacement baseline, not only on the saliency ranking.

### <a id="ref-right-reasons"></a>`REF-RIGHT-REASONS` — Right for the Right Reasons

- IEEE-style entry: A. S. Ross, M. C. Hughes, and F. Doshi-Velez, "Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations," in *Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI-17)*, 2017, pp. 2662-2670.
- Source type: peer-reviewed conference paper.
- Verified URLs:
  - IJCAI proceedings page: https://www.ijcai.org/proceedings/2017/371
  - IJCAI PDF: https://www.ijcai.org/proceedings/2017/0371.pdf
  - DOI: https://doi.org/10.24963/ijcai.2017/371
- Reliability status: high; official IJCAI proceedings and DOI.
- Thesis relevance: provides a concise conceptual frame for separating correct predictions from clinically acceptable evidence. It supports the thesis language that a model can be predictive while still using the wrong image regions or non-clinical reasons.

### <a id="ref-saliency-trust-medical"></a>`REF-SALIENCY-TRUST-MEDICAL` — Trustworthiness of Medical-Imaging Saliency Maps

- IEEE-style entry: N. Arun et al., "Assessing the Trustworthiness of Saliency Maps for Localizing Abnormalities in Medical Imaging," *Radiology: Artificial Intelligence*, vol. 3, no. 6, Art. no. e200267, 2021.
- Source type: peer-reviewed radiology AI journal article.
- Verified URLs:
  - RSNA article page: https://pubs.rsna.org/doi/10.1148/ryai.2021200267
  - PubMed record: https://pubmed.ncbi.nlm.nih.gov/34870212/
  - PMC full-text record: https://pmc.ncbi.nlm.nih.gov/articles/PMC8637231
  - DOI: https://doi.org/10.1148/ryai.2021200267
- Reliability status: high; RSNA journal article indexed in PubMed with PMC full text.
- Thesis relevance: directly supports caution when using saliency maps for abnormality localization in medical imaging. Particularly useful for arguing that classification explanations should be validated against localization evidence and should not be treated as substitutes for detection or segmentation models.

### <a id="ref-cxr-saliency-benchmark"></a>`REF-CXR-SALIENCY-BENCHMARK` — CXR Saliency Benchmarking Against Human Localization

- IEEE-style entry: A. Saporta et al., "Benchmarking saliency methods for chest X-ray interpretation," *Nature Machine Intelligence*, vol. 4, pp. 867-878, 2022.
- Source type: peer-reviewed journal article.
- Verified URLs:
  - Nature article page: https://www.nature.com/articles/s42256-022-00536-x
  - DOI: https://doi.org/10.1038/s42256-022-00536-x
- Reliability status: high; Nature Machine Intelligence article with DOI.
- Thesis relevance: highly aligned with the current project because it evaluates saliency methods for CXR interpretation, compares localization against human benchmarks, and reports that saliency failures vary by clinical/pathology conditions. This supports reporting pneumothorax size/shape/subtlety and not relying on heatmaps as clinical evidence without validation.

### <a id="ref-chexlocalize"></a>`REF-CHEXLOCALIZE` — CheXlocalize CXR Localization Benchmark

- IEEE-style entry: Stanford Center for Artificial Intelligence in Medicine and Imaging, "CheXlocalize," official dataset page; associated with A. Saporta et al., "Benchmarking saliency methods for chest X-ray interpretation," *Nature Machine Intelligence*, 2022.
- Source type: official dataset page tied to a peer-reviewed benchmark paper.
- Verified URLs:
  - Stanford AIMI dataset page: https://aimi.stanford.edu/datasets/chexlocalize
  - Official code/dataset repository: https://github.com/rajpurkarlab/cheXlocalize
  - Associated Nature Machine Intelligence article: https://www.nature.com/articles/s42256-022-00536-x
- Reliability status: high; official Stanford AIMI dataset page and peer-reviewed associated publication.
- Thesis relevance: gives a concrete example of how another CXR saliency study used radiologist pixel-level segmentations and most-representative points to benchmark localization. Useful when justifying this thesis's mask-overlap metrics and pointing-hit style measures.

## Medical Imaging XAI and Human-Centered Validation

### <a id="ref-medical-xai-review"></a>`REF-MEDICAL-XAI-REVIEW` — XAI in Deep Learning-Based Medical Image Analysis

- IEEE-style entry: B. H. M. van der Velden, H. J. Kuijf, K. G. A. Gilhuijs, and M. A. Viergever, "Explainable artificial intelligence (XAI) in deep learning-based medical image analysis," *Medical Image Analysis*, vol. 79, 2022, Art. no. 102470.
- Source type: peer-reviewed journal review.
- Verified URLs:
  - DOI: https://doi.org/10.1016/j.media.2022.102470
  - PubMed record: https://pubmed.ncbi.nlm.nih.gov/35316819/
- Reliability status: high; Medical Image Analysis review indexed in PubMed.
- Thesis relevance: background source for taxonomy and medical-imaging-specific XAI challenges; useful in Chapter 2 literature review.

### <a id="ref-human-centered-xai"></a>`REF-HUMAN-CENTERED-XAI` — Human-Centered Medical Imaging XAI

- IEEE-style entry: H. Chen, C. Gomez, C.-M. Huang, and M. Unberath, "Explainable medical imaging AI needs human-centered design: guidelines and evidence from a systematic review," *npj Digital Medicine*, vol. 5, 2022, Art. no. 156.
- Source type: peer-reviewed systematic review / guideline paper.
- Verified URLs:
  - Nature article page: https://www.nature.com/articles/s41746-022-00699-2
  - DOI: https://doi.org/10.1038/s41746-022-00699-2
  - arXiv record: https://arxiv.org/abs/2112.12596
- Reliability status: high; Nature Portfolio journal article with DOI.
- Thesis relevance: supports the project’s radiologist-review workflow, human-centered rubric design, and warning that explanation usefulness depends on user tasks and empirical evaluation.

### <a id="ref-clever-hans"></a>`REF-CLEVER-HANS` — Spurious Correlations in Medical AI via XAI Audit

- IEEE-style entry: A. DeGrave, J. D. Janizek, and S.-I. Lee, "AI for radiographic COVID-19 detection selects shortcuts over signal," *Nature Machine Intelligence*, vol. 3, pp. 610-619, 2021.
- Source type: peer-reviewed journal article.
- Verified URLs:
  - Nature article page: https://www.nature.com/articles/s42256-021-00338-7
  - DOI: https://doi.org/10.1038/s42256-021-00338-7
- Reliability status: high; Nature Machine Intelligence article.
- Thesis relevance: strong cautionary example for radiology AI: a classifier can perform well while relying on non-pathology shortcuts, which is central to interpreting weak pneumothorax localization.

### <a id="ref-cxr-generalization-zech"></a>`REF-CXR-GENERALIZATION-ZECH` — CXR Dataset Shift and External Generalization

- IEEE-style entry: J. R. Zech, M. A. Badgeley, M. Liu, A. B. Costa, J. J. Titano, and E. K. Oermann, "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study," *PLOS Medicine*, vol. 15, no. 11, Art. no. e1002683, 2018.
- Source type: peer-reviewed journal article.
- Verified URLs:
  - PLOS Medicine article page: https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683
  - PMC full-text record: https://pmc.ncbi.nlm.nih.gov/articles/PMC6219764
  - DOI: https://doi.org/10.1371/journal.pmed.1002683
- Reliability status: high; peer-reviewed open-access journal article with DOI and PMC record.
- Thesis relevance: supports the caution that CXR model performance may drop or change under external validation across institutions/datasets. This is relevant when interpreting a TorchXRayVision baseline transferred to SIIM-ACR pneumothorax images.

### <a id="ref-hidden-stratification"></a>`REF-HIDDEN-STRATIFICATION` — Hidden Stratification in Medical Imaging

- IEEE-style entry: L. Oakden-Rayner, J. Dunnmon, G. Carneiro, and C. Re, "Hidden stratification causes clinically meaningful failures in machine learning for medical imaging," in *Proceedings of the ACM Conference on Health, Inference, and Learning (CHIL)*, 2020, pp. 151-159.
- Source type: peer-reviewed conference paper.
- Verified URLs:
  - ACM DOI: https://doi.org/10.1145/3368555.3384468
  - PMC full-text record: https://pmc.ncbi.nlm.nih.gov/articles/PMC7665161
- Reliability status: high; peer-reviewed ACM CHIL paper with DOI and PMC record.
- Thesis relevance: supports subgroup-oriented analysis and qualitative failure review. Particularly relevant to pneumothorax because hidden treatment/device subsets, such as chest drains, can change apparent model performance and explanation behavior.

### <a id="ref-radiology-interpretability"></a>`REF-RADIOLOGY-INTERPRETABILITY` — Interpretability in Radiology

- IEEE-style entry: M. Reyes, R. Meier, S. Pereira, C. A. Silva, F.-M. Dahlweid, H. von Tengg-Kobligk, R. M. Summers, and R. Wiest, "On the interpretability of artificial intelligence in radiology: challenges and opportunities," *Radiology: Artificial Intelligence*, vol. 2, no. 3, Art. no. e190043, 2020.
- Source type: peer-reviewed radiology review article.
- Verified URLs:
  - PubMed record: https://pubmed.ncbi.nlm.nih.gov/32510054/
  - PMC full-text record: https://pmc.ncbi.nlm.nih.gov/articles/PMC7259808/
  - DOI: https://doi.org/10.1148/ryai.2020190043
- Reliability status: high; RSNA journal article indexed in PubMed with DOI and PMC full text.
- Thesis relevance: radiology-specific source for distinguishing interpretability goals, user needs, and validation challenges. Useful for framing the project as radiologist-centered validation rather than generic saliency-map generation.

### <a id="ref-false-hope-xai"></a>`REF-FALSE-HOPE-XAI` — Limits of Current Health-Care XAI

- IEEE-style entry: M. Ghassemi, L. Oakden-Rayner, and A. L. Beam, "The false hope of current approaches to explainable artificial intelligence in health care," *The Lancet Digital Health*, vol. 3, no. 11, pp. e745-e750, 2021.
- Source type: peer-reviewed health-care AI viewpoint article.
- Verified URLs:
  - PubMed record: https://pubmed.ncbi.nlm.nih.gov/34711379/
  - Lancet article page: https://www.thelancet.com/journals/landig/article/PIIS2589-7500(21)00208-9/fulltext
  - DOI: https://doi.org/10.1016/S2589-7500(21)00208-9
- Reliability status: high; Lancet Digital Health article indexed in PubMed with DOI.
- Thesis relevance: supports cautious wording that explanations should not be treated as automatic guarantees of safety, trust, fairness, or patient-level clinical correctness. Strong support for pairing XAI with validation and error analysis.

### <a id="ref-shortcut-learning"></a>`REF-SHORTCUT-LEARNING` — General Shortcut Learning in Deep Neural Networks

- IEEE-style entry: R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and F. A. Wichmann, "Shortcut learning in deep neural networks," *Nature Machine Intelligence*, vol. 2, pp. 665-673, 2020.
- Source type: peer-reviewed journal review/perspective article.
- Verified URLs:
  - Nature article page: https://www.nature.com/articles/s42256-020-00257-z
  - DOI: https://doi.org/10.1038/s42256-020-00257-z
- Reliability status: high; Nature Machine Intelligence article with DOI.
- Thesis relevance: provides general ML framing for models that solve training or benchmark objectives using unintended decision rules. Useful for interpreting external-transfer CXR results where classifier performance and lesion-localizing evidence diverge.

### <a id="ref-radiology-shortcuts"></a>`REF-RADIOLOGY-SHORTCUTS` — Shortcut Bias in Radiology AI

- IEEE-style entry: I. Banerjee et al., "'Shortcuts' Causing Bias in Radiology Artificial Intelligence: Causes, Evaluation, and Mitigation," *Journal of the American College of Radiology*, vol. 20, no. 9, pp. 842-851, 2023.
- Source type: peer-reviewed radiology review article.
- Verified URLs:
  - PubMed record: https://pubmed.ncbi.nlm.nih.gov/37506964/
  - PMC full-text record: https://pmc.ncbi.nlm.nih.gov/articles/PMC11192466/
  - DOI: https://doi.org/10.1016/j.jacr.2023.06.025
- Reliability status: high; JACR article indexed in PubMed with PMC author manuscript.
- Thesis relevance: supports describing device markers, acquisition artifacts, demographic proxies, institution-specific cues, and other non-pathology signals as possible shortcut mechanisms in radiology AI. Directly useful for the review-workbook failure taxonomy.

### <a id="ref-race-medical-imaging"></a>`REF-RACE-MEDICAL-IMAGING` — Hidden Signals in Medical Images

- IEEE-style entry: J. W. Gichoya et al., "AI recognition of patient race in medical imaging: a modelling study," *The Lancet Digital Health*, vol. 4, no. 6, pp. e406-e414, 2022.
- Source type: peer-reviewed medical AI modelling study.
- Verified URLs:
  - Lancet article DOI: https://doi.org/10.1016/S2589-7500(22)00063-2
  - NIBIB summary with citation: https://www.nibib.nih.gov/news-events/newsroom/study-finds-artificial-intelligence-can-determine-race-medical-images
- Reliability status: high; Lancet Digital Health article with DOI; NIBIB summary used only for accessible citation confirmation.
- Thesis relevance: supports the broader warning that medical images contain latent signals that human experts may not consciously use or detect. This strengthens the argument that a CXR model can rely on hidden or clinically unintended cues even when the heatmap appears plausible.

## Medical AI Reporting, Quality, and Validation Guidance

### <a id="ref-claim"></a>`REF-CLAIM` — Checklist for Artificial Intelligence in Medical Imaging

- IEEE-style entry: J. Mongan, L. Moy, and C. E. Kahn Jr., "Checklist for Artificial Intelligence in Medical Imaging (CLAIM): A Guide for Authors and Reviewers," *Radiology: Artificial Intelligence*, vol. 2, no. 2, Art. no. e200029, 2020.
- Source type: peer-reviewed medical-imaging AI reporting guideline.
- Verified URLs:
  - DOI: https://doi.org/10.1148/ryai.2020200029
  - PubMed record: https://pubmed.ncbi.nlm.nih.gov/33937821/
  - EQUATOR Network guideline/history page: https://www.equator-network.org/reporting-guidelines/checklist-for-artificial-intelligence-in-medical-imaging-claim-a-guide-for-authors-and-reviewers
- Reliability status: high; RSNA journal guideline with DOI, PubMed indexing, and EQUATOR guideline record.
- Thesis relevance: supports transparent reporting of data sources, ground truth, preprocessing, partitions, model details, evaluation metrics, and limitations in the thesis methodology.

### <a id="ref-tripod-ai"></a>`REF-TRIPOD-AI` — TRIPOD+AI Reporting Guidance

- IEEE-style entry: G. S. Collins et al., "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods," *BMJ*, vol. 385, Art. no. e078378, 2024.
- Source type: peer-reviewed clinical prediction model reporting guideline.
- Verified URLs:
  - PubMed record: https://pubmed.ncbi.nlm.nih.gov/38626948/
  - DOI: https://doi.org/10.1136/bmj-2023-078378
  - EQUATOR Network guideline page: https://www.equator-network.org/reporting-guidelines/tripod-statement
- Reliability status: high; BMJ guideline article with DOI, PubMed indexing, and EQUATOR record.
- Thesis relevance: supports reporting classifier thresholding, model inputs, validation design, and performance metrics as prediction-model evaluation details rather than treating XAI outputs in isolation.

### <a id="ref-quadas-ai"></a>`REF-QUADAS-AI` — QUADAS-AI Quality Assessment

- IEEE-style entry: V. Sounderajah et al., "A quality assessment tool for artificial intelligence-centered diagnostic test accuracy studies: QUADAS-AI," *Nature Medicine*, vol. 27, no. 10, pp. 1663-1665, 2021.
- Source type: peer-reviewed diagnostic-AI quality assessment article.
- Verified URLs:
  - Nature Medicine article page: https://www.nature.com/articles/s41591-021-01517-0
  - DOI: https://doi.org/10.1038/s41591-021-01517-0
- Reliability status: high; Nature Medicine article with DOI.
- Thesis relevance: supports discussing risk of bias, applicability, patient/data selection, reference standards, and clinical deployment limitations when evaluating medical-imaging AI outputs.

### <a id="ref-decide-ai"></a>`REF-DECIDE-AI` — Early-Stage Clinical Evaluation Reporting for AI Decision Support

- IEEE-style entry: B. Vasey et al. and the DECIDE-AI expert group, "Reporting guideline for the early-stage clinical evaluation of decision support systems driven by artificial intelligence: DECIDE-AI," *Nature Medicine*, vol. 28, no. 5, pp. 924-933, 2022.
- Source type: peer-reviewed AI clinical-evaluation reporting guideline.
- Verified URLs:
  - Nature Medicine article page: https://www.nature.com/articles/s41591-022-01772-9
  - PubMed record: https://pubmed.ncbi.nlm.nih.gov/35585198/
  - EQUATOR Network guideline page: https://www.equator-network.org/reporting-guidelines/reporting-guideline-for-the-early-stage-clinical-evaluation-of-decision-support-systems-driven-by-artificial-intelligence-decide-ai
  - DOI: https://doi.org/10.1038/s41591-022-01772-9
- Reliability status: high; Nature Medicine guideline with PubMed and EQUATOR records.
- Thesis relevance: useful if the radiologist-review workbook is framed as an early, human-in-the-loop evaluation artifact. It supports reporting intended users, workflow context, human-AI interaction, errors, and limitations rather than only algorithmic metrics.

## Dataset and Model Context

### <a id="ref-siim-acr"></a>`REF-SIIM-ACR` — SIIM-ACR Pneumothorax Challenge

- IEEE-style entry: Society for Imaging Informatics in Medicine (SIIM), "SIIM-ACR Pneumothorax Segmentation Kaggle Challenge," official challenge information page.
- Source type: official challenge/institutional page.
- Verified URLs:
  - SIIM page: https://siim.org/research-journal/siim-machine-learning-challenges/pneumothorax-kaggle-challenge
  - Kaggle competition page: https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation
- Reliability status: high for challenge description and organizer facts; Kaggle access may require account/login.
- Thesis relevance: official context for the pneumothorax classification/segmentation task, organizer institutions, and challenge participation.

### <a id="ref-torchxrayvision"></a>`REF-TORCHXRAYVISION` — TorchXRayVision Library and Models

- IEEE-style entry: J. P. Cohen et al., "TorchXRayVision: A library of chest X-ray datasets and models," in *Proceedings of the 5th International Conference on Medical Imaging with Deep Learning*, PMLR, vol. 172, 2022, pp. 231-249.
- Source type: peer-reviewed conference paper plus official documentation.
- Verified URLs:
  - PMLR proceedings page: https://proceedings.mlr.press/v172/cohen22a.html
  - Official documentation: https://mlmed.org/torchxrayvision
  - GitHub repository: https://github.com/mlmed/torchxrayvision
- Reliability status: high; PMLR conference paper and official project documentation.
- Thesis relevance: source for the external pretrained CXR model family used as the current baseline and for the claim that the model is used off-the-shelf rather than locally modified.

### <a id="ref-chestxray8"></a>`REF-CHESTXRAY8` — NIH ChestX-ray8 / ChestX-ray14 Dataset Paper

- IEEE-style entry: X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, and R. M. Summers, "ChestX-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 3462-3471.
- Source type: peer-reviewed conference paper plus official public-dataset documentation.
- Verified URLs:
  - CVF PDF: https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf
  - arXiv record with CVPR reference: https://arxiv.org/abs/1705.02315
  - Google Cloud public-dataset documentation: https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest
- Reliability status: high; CVPR publication and official dataset-access documentation.
- Thesis relevance: background for one major public CXR source represented in pretrained CXR model ecosystems. Important caveat: labels are report/NLP-derived and localization annotations are limited, which affects how models trained on this source should be interpreted.

### <a id="ref-chexnext"></a>`REF-CHEXNEXT` — CheXNeXt CXR Classifier and Radiologist Comparison

- IEEE-style entry: P. Rajpurkar et al., "Deep learning for chest radiograph diagnosis: A retrospective comparison of the CheXNeXt algorithm to practicing radiologists," *PLOS Medicine*, vol. 15, no. 11, Art. no. e1002686, 2018.
- Source type: peer-reviewed journal article.
- Verified URLs:
  - PLOS Medicine article page: https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002686
  - PubMed record: https://pubmed.ncbi.nlm.nih.gov/30457988/
  - DOI: https://doi.org/10.1371/journal.pmed.1002686
- Reliability status: high; peer-reviewed open-access PLOS Medicine article with DOI and PubMed record.
- Thesis relevance: peer-reviewed CXR classifier baseline literature showing radiologist-comparison evaluation and pneumothorax as one of several CXR labels. Useful as a safer alternative to relying on arXiv-only CheXNet material for historical CXR deep-learning context.

### <a id="ref-chexpert"></a>`REF-CHEXPERT` — CheXpert Dataset

- IEEE-style entry: J. Irvin et al., "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 33, no. 1, 2019, pp. 590-597.
- Source type: peer-reviewed conference paper.
- Verified URLs:
  - AAAI proceedings page: https://ojs.aaai.org/index.php/AAAI/article/view/3834
  - AAAI PDF: https://ojs.aaai.org/index.php/AAAI/article/view/3834/3712
  - DOI: https://doi.org/10.1609/aaai.v33i01.3301590
- Reliability status: high; AAAI proceedings article with DOI.
- Thesis relevance: background for a major public CXR dataset with uncertainty labels and expert comparison. Useful when discussing pretrained CXR weights, label uncertainty, and why dataset-specific validation remains necessary.

### <a id="ref-mimic-cxr"></a>`REF-MIMIC-CXR` — MIMIC-CXR Dataset

- IEEE-style entry: A. E. W. Johnson et al., "MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports," *Scientific Data*, vol. 6, Art. no. 317, 2019.
- Source type: peer-reviewed dataset descriptor plus official PhysioNet dataset page.
- Verified URLs:
  - Nature Scientific Data article: https://www.nature.com/articles/s41597-019-0322-0
  - PMC full-text record: https://pmc.ncbi.nlm.nih.gov/articles/PMC6908718
  - PhysioNet dataset page: https://physionet.org/content/mimic-cxr
  - DOI: https://doi.org/10.1038/s41597-019-0322-0
- Reliability status: high; peer-reviewed Scientific Data article and official PhysioNet release.
- Thesis relevance: background for another major public CXR source represented in pretrained model families. It reinforces that public CXR models often learn from large report-labeled datasets whose label definitions, population, acquisition workflow, and preprocessing differ from SIIM-ACR pneumothorax segmentation data.

### <a id="ref-vindr-cxr"></a>`REF-VINDR-CXR` — VinDr-CXR Dataset

- IEEE-style entry: H. Q. Nguyen et al., "VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations," *Scientific Data*, vol. 9, Art. no. 429, 2022.
- Source type: peer-reviewed dataset descriptor plus official PhysioNet dataset page.
- Verified URLs:
  - Nature Scientific Data article: https://www.nature.com/articles/s41597-022-01498-w
  - PhysioNet dataset page: https://physionet.org/content/vindr-cxr
  - DOI: https://doi.org/10.1038/s41597-022-01498-w
- Reliability status: high; peer-reviewed Scientific Data dataset article and official PhysioNet release.
- Thesis relevance: demonstrates that CXR datasets with radiologist localization annotations and pneumothorax labels exist beyond SIIM-ACR. Useful as context for future external validation or alternative localization benchmarks, while not replacing the current SIIM mask-based protocol.

## Statistical Methods for Paired XAI Comparison

### <a id="ref-wilcoxon-1945"></a>`REF-WILCOXON-1945` — Wilcoxon Signed-Rank Test (original)

- IEEE-style entry: F. Wilcoxon, "Individual Comparisons by Ranking Methods," *Biometrics Bulletin*, vol. 1, no. 6, pp. 80-83, 1945.
- Source type: peer-reviewed journal article (foundational).
- Verified URLs:
  - JSTOR record: https://www.jstor.org/stable/3001968
  - DOI: https://doi.org/10.2307/3001968
- Reliability status: high; foundational paper, widely re-cited.
- Thesis relevance: original statistical reference for the non-parametric paired test used to compare consensus against each individual XAI method in the improvement experiment. The paired-difference + signed-rank framing is exactly what the thesis needs because per-case localization residuals (IoU, Dice) are typically non-normal.

### <a id="ref-holm-1979"></a>`REF-HOLM-1979` — Holm Step-Down Multiple-Test Procedure

- IEEE-style entry: S. Holm, "A Simple Sequentially Rejective Multiple Test Procedure," *Scandinavian Journal of Statistics*, vol. 6, no. 2, pp. 65-70, 1979.
- Source type: peer-reviewed journal article (foundational).
- Verified URLs:
  - JSTOR record: https://www.jstor.org/stable/4615733
- Reliability status: high; foundational paper for Holm-Bonferroni step-down family-wise error control.
- Thesis relevance: original reference for the multiple-comparison correction used across the family of "consensus vs each individual method" tests. Holm uniformly dominates plain Bonferroni at the same family-wise error rate, so this is the citation for the methodology choice to apply Holm rather than Bonferroni.

### <a id="ref-aickin-gensler-1996"></a>`REF-AICKIN-GENSLER-1996` — Bonferroni vs Holm in Medical Research

- IEEE-style entry: M. Aickin and H. Gensler, "Adjusting for Multiple Testing When Reporting Research Results: The Bonferroni vs Holm Methods," *American Journal of Public Health*, vol. 86, no. 5, pp. 726-728, 1996.
- Source type: peer-reviewed methodological article.
- Verified URLs:
  - PMC full-text record: https://pmc.ncbi.nlm.nih.gov/articles/PMC1380484
  - DOI: https://doi.org/10.2105/ajph.86.5.726
- Reliability status: high; AJPH article with PMC full text.
- Thesis relevance: accessible medical-research framing of why Holm should be preferred over plain Bonferroni when family-wise error control is needed. Useful when justifying the choice in language that is closer to the thesis's clinical-evaluation context than the original Holm (1979) paper.

### <a id="ref-demsar-2006"></a>`REF-DEMSAR-2006` — Statistical Comparison of Classifiers Across Datasets

- IEEE-style entry: J. Demšar, "Statistical Comparisons of Classifiers over Multiple Data Sets," *Journal of Machine Learning Research*, vol. 7, pp. 1-30, 2006.
- Source type: peer-reviewed journal article.
- Verified URLs:
  - JMLR article page: https://www.jmlr.org/papers/v7/demsar06a.html
  - JMLR PDF: https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf
- Reliability status: high; canonical JMLR reference for non-parametric paired tests in ML method comparison.
- Thesis relevance: load-bearing ML-side citation for using Wilcoxon signed-rank tests with step-down (Holm-style) correction for pairwise method comparisons. Anchors the thesis methodology in mainstream ML evaluation practice rather than only in medical statistics.

## Computed Tomography Hemorrhage Context

### <a id="ref-rsna-ihd"></a>`REF-RSNA-IHD` — RSNA Intracranial Hemorrhage Detection Challenge

- IEEE-style entry: A. E. Flanders et al., "Construction of a Machine Learning Dataset through Collaboration: The RSNA 2019 Brain CT Hemorrhage Challenge," *Radiology: Artificial Intelligence*, vol. 2, no. 3, Art. no. e190211, 2020.
- Source type: peer-reviewed radiology AI journal article tied to an official challenge.
- Verified URLs:
  - RSNA article page: https://pubs.rsna.org/doi/10.1148/ryai.2020190211
  - PubMed record: https://pubmed.ncbi.nlm.nih.gov/33937827/
  - PMC full-text record: https://pmc.ncbi.nlm.nih.gov/articles/PMC8082297
  - Kaggle competition page: https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection
  - DOI: https://doi.org/10.1148/ryai.2020190211
- Reliability status: high; peer-reviewed journal article describing the dataset, ground truth, and challenge protocol, plus official Kaggle competition page for dataset access.
- Thesis relevance: provides the canonical reference for the CT pilot modality and dataset. The 2019 RSNA challenge released ~870k labeled CT slices with per-subtype hemorrhage labels (epidural, intraparenchymal, intraventricular, subarachnoid, subdural, any). Useful as evidence that off-the-shelf public CT hemorrhage classifiers exist and have a clinically meaningful class head. If the Phase 5.4 hour-1 model-availability check finds an off-the-shelf classifier, the thesis cites this dataset/challenge as the training distribution; if not, this reference frames the qualitative-only fallback as scoped against a recognized public CT task.

### <a id="ref-vit"></a>`REF-VIT` — Vision Transformer (ViT)

- IEEE-style entry: A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale," in *Proc. International Conference on Learning Representations (ICLR)*, 2021.
- Verified URLs:
  - OpenReview: https://openreview.net/forum?id=YicbFdNTTy
  - arXiv: https://arxiv.org/abs/2010.11929
- Reliability status: high; peer-reviewed ICLR paper, the foundational Vision Transformer reference.
- Thesis relevance: the Phase 5.4 CT hemorrhage classifier is a ViT-base-patch16-224 backbone. Establishes why CAM-family methods do not transfer directly to the CT pilot: a ViT represents an image as a sequence of patch tokens plus a class token, with no native two-dimensional convolutional feature map for activation-based CAMs to weight and upsample.

### <a id="ref-attention-rollout"></a>`REF-ATTENTION-ROLLOUT` — Attention Rollout

- IEEE-style entry: S. Abnar and W. Zuidema, "Quantifying Attention Flow in Transformers," in *Proc. 58th Annual Meeting of the Association for Computational Linguistics (ACL)*, 2020, pp. 4190-4197.
- Verified URLs:
  - ACL Anthology: https://aclanthology.org/2020.acl-main.385/
  - arXiv: https://arxiv.org/abs/2005.00928
- Reliability status: high; peer-reviewed ACL paper.
- Thesis relevance: a transformer-specific explanation technique. Cited as the type of architecture-native method that would be required to explain a Vision Transformer spatially in place of CAM-family methods, supporting the framing of a CAM-on-ViT CT extension as future work.

### <a id="ref-transformer-attribution"></a>`REF-TRANSFORMER-ATTRIBUTION` — Transformer Interpretability Beyond Attention

- IEEE-style entry: H. Chefer, S. Gur, and L. Wolf, "Transformer Interpretability Beyond Attention Visualization," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021, pp. 782-791.
- Verified URLs:
  - CVF: https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html
  - arXiv: https://arxiv.org/abs/2012.09838
- Reliability status: high; peer-reviewed CVPR paper.
- Thesis relevance: a gradient-based relevance-propagation method designed for transformers. Together with attention rollout it represents the transformer-native alternative to CAM-family methods, justifying the exclusion of CAM methods from the controlled cross-modality comparison and the deferral of ViT-specific spatial explanation to future work.

### <a id="ref-vit-cx"></a>`REF-VIT-CX` — ViT-CX (Causal Explanation of Vision Transformers)

- IEEE-style entry: W. Xie, X.-H. Li, C. C. Cao, and N. L. Zhang, "ViT-CX: Causal Explanation of Vision Transformers," in *Proc. 32nd International Joint Conference on Artificial Intelligence (IJCAI)*, 2023.
- Verified URLs:
  - IJCAI proceedings: https://www.ijcai.org/proceedings/2023/0174
  - arXiv: https://arxiv.org/abs/2211.03064
- Reliability status: high; peer-reviewed IJCAI 2023 paper.
- Thesis relevance: a Vision-Transformer-specific, Score-CAM-style causal explanation method built on patch-embedding masks rather than convolutional activations. Concrete evidence that CAM-style spatial explanation of a ViT is possible, but only via a transformer-specific method rather than the convolutional CAM implementation reused across the CXR pipeline. Cited in future work as a candidate for a CAM-comparable CT explanation that would let the full method panel (and a four-method consensus) be evaluated on the CT classifier.

### <a id="ref-pytorch-gradcam"></a>`REF-PYTORCH-GRADCAM` — pytorch-grad-cam (software library)

- IEEE-style entry: J. Gildenblat and contributors, *PyTorch library for CAM methods (pytorch-grad-cam)*, GitHub repository, 2021.
- Verified URLs:
  - Repository: https://github.com/jacobgil/pytorch-grad-cam
- Reliability status: medium; widely-used open-source software library, not a peer-reviewed publication. Cited as a software tool only (consistent with the source-reliability working notes), not as empirical evidence.
- Thesis relevance: the de-facto open-source CAM implementation. It exposes a `reshape_transform` that folds ViT patch tokens into a two-dimensional grid so Grad-CAM/Grad-CAM++/Eigen-CAM/Score-CAM can be run on transformer backbones. Demonstrates that running CAM-family methods on a ViT is feasible via an architecture-specific adapter — i.e., a different code path from the convolutional CXR implementation — which supports the controlled-comparison rationale for excluding CAMs from the CT transfer experiment in the current draft.

## Working Notes on Source Reliability

- `Dataset Ninja` and similar mirrors were not used as primary references, even when they contain convenient dataset counts, because the user requested reliable, verifiable sources. Official SIIM/Kaggle pages and local dataset verification should be preferred for final counts.
- Kaggle discussion posts and winning-solution repositories can be useful for engineering ideas, but they are not included here as thesis evidence unless the user explicitly approves them.
- YouTube talks, vendor/press articles, and arXiv-only historical CXR baselines were not used as primary sources in this pass.

