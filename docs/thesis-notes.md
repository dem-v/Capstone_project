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

### Eigen-CAM uses principal-component analysis of activations instead of gradients

- Source link: [`REF-EIGEN-CAM`](references.md#ref-eigen-cam)
- Direct quote candidate: "Eigen-CAM ... computes and visualizes the principle [sic] components of the learned features/representations from the convolutional layers."
- Thesis-ready paraphrase: Eigen-CAM is a gradient-free CAM-family method that projects target-layer activations onto their top principal component, producing a coarse localization map without requiring a backward pass. In this thesis, it complements Grad-CAM and Grad-CAM++ as a third CAM variant that does not depend on gradient flow, which is useful when gradient-based maps are noisy, saturated, or unstable across architectures.
- Practical pointer: When discussing `eigen_cam` in Chapter 3, report the sign-convention choice (Eigen-CAM's sign is arbitrary up to a flip; this project fixes it via alignment with the predicted-class logit gradient sign). In Chapter 4, treat it as a method-class diagnostic rather than as a claimed improvement over Grad-CAM.

### Score-CAM is useful because it reduces reliance on raw gradients

- Source link: [`REF-SCORECAM`](references.md#ref-scorecam)
- Direct quote candidate: "Score-CAM gets rid of the dependence on gradients by obtaining the weight of each activation map through its forward passing score on target class."
- Thesis-ready paraphrase: Score-CAM is a relevant extension because it assigns activation-map weights using model output scores rather than gradient weights. This makes it a useful comparator when gradient-based maps appear noisy, saturated, or clinically implausible.
- Practical pointer: If Score-CAM is added to the pipeline, report runtime separately. It is expected to be slower because it requires additional forward passes.

### Consensus constituents are frozen at the original four methods

- Thesis-ready paraphrase: The `consensus` and `consensus_signed` methods used in this thesis are the unweighted average of four signed attribution maps: `grad_cam`, `integrated_gradients`, `gradient_shap`, and `occlusion`. The added CAM-family methods `eigen_cam` and `score_cam` are reported as additional individual methods but are deliberately NOT included in the consensus average. The rationale is comparability: redefining consensus mid-thesis would invalidate all prior consensus results, break cross-iteration figures, and conflate "consensus contribution" with "method-panel expansion". The improvement experiment (Phase 5.2) tests consensus-of-four against each individual method including `eigen_cam` and `score_cam`, which is the cleaner contrast.
- Practical pointer: When the thesis Discussion talks about consensus, state explicitly that the consensus is over the original four methods and that the inclusion of `eigen_cam` and `score_cam` as additional individuals strengthens (not weakens) the comparison because consensus is now tested against a broader pool of individual baselines. The `consensus_attention` variant raised in the 2026-05-18 supervisor sync (attention-weighted consensus, NOT architectural attention) is recorded under thesis Chapter 5.3 "Future Work" — defer to post-defense follow-up.

### LIME is a region-level surrogate explanation, included as a third family if scope allows

- Source link: [`REF-LIME`](references.md#ref-lime)
- Thesis-ready paraphrase: LIME explains an individual classifier prediction by fitting a sparse linear surrogate over interpretable input perturbations (image super-pixels). It belongs to a different explanation family than gradient attribution (Grad-CAM, IG, GradientSHAP) or perturbation attribution (Occlusion, Score-CAM). Including LIME on a sub-sampled positive case set tests whether a region-level surrogate identifies the same evidence as pixel-level and CAM-family methods.
- Practical pointer: LIME is conditional in this thesis. If implementation time is low (per the experiment protocol's explicit clause) and the rest of Phase 5 lands on schedule, run LIME on 10–20 representative cases as a qualitative comparator. Do not include LIME in the paired Wilcoxon comparison unless N is large enough to support it. If skipped, justify in methodology by citing the protocol clause and noting that the panel already spans CAM, gradient, perturbation, and PCA-based families.

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

### Infidelity and sensitivity triangulate the deletion/insertion faithfulness pipeline

- Source link: [`REF-INFIDELITY-SENSITIVITY`](references.md#ref-infidelity-sensitivity)
- Thesis-ready paraphrase: Infidelity measures the expected squared difference between (a) the attribution-dot-perturbation predicted output change and (b) the actual classifier output change under that perturbation. Sensitivity-max measures the largest change in attribution under small input perturbations. Together they test a different aspect of faithfulness than deletion/insertion curves: infidelity asks "does the attribution linearly predict the model's response to perturbations?", while sensitivity asks "is the attribution stable to small input noise?".
- Practical pointer: Add these as supplementary columns on the held-out test metrics rows. They reuse the existing per-case heatmap and model without additional model retraining. In Chapter 4, treat the joint reading of "deletion/insertion AUC + infidelity + sensitivity-max" as a triangulation across faithfulness families rather than as three measurements of the same quantity. Document the perturbation operator (e.g. Gaussian noise σ=0.02) and the number of perturbation samples in the methodology.

### Perturbation explanations depend on the replacement operation

- Source link: [`REF-MEANINGFUL-PERTURBATION`](references.md#ref-meaningful-perturbation)
- Thesis-ready paraphrase: Perturbation-style explanations identify image regions whose masking or replacement changes a target score, but the clinical meaning depends on how the image is perturbed. A black, blurred, noisy, or dataset-average replacement can each create different artifacts and different model responses.
- Practical pointer: Use this when explaining faithfulness baselines. The current `black` baseline should be described as a deliberate perturbation choice, not as a universally neutral image; baseline comparisons are useful diagnostics rather than final clinical truth.

### "Right for the right reasons" is a useful discussion frame

- Source link: [`REF-RIGHT-REASONS`](references.md#ref-right-reasons)
- Thesis-ready paraphrase: High classification performance is not enough if the model relies on evidence that is clinically irrelevant or inappropriate. For this thesis, the phrase "right for the right reasons" can be used carefully as a conceptual frame: the project tests whether pneumothorax predictions are supported by lesion-aligned evidence, not only whether the binary label is correct.
- Practical pointer: This is most useful in Chapter 5 and future work. If a stronger second model is trained or fine-tuned later, explanation constraints or region-aware training could be introduced as a possible next step after the validation study identifies recurrent failure patterns.

### Medical saliency maps need localization validation before clinical use

- Source link: [`REF-SALIENCY-TRUST-MEDICAL`](references.md#ref-saliency-trust-medical)
- Direct quote candidate: "The use of saliency maps in the high-risk domain of medical imaging warrants additional scrutiny and recommend that detection or segmentation models be used if localization is the desired output of the network."
- Thesis-ready paraphrase: Medical-imaging saliency maps should not be accepted as reliable abnormality localizers simply because they are visually plausible or attached to a classifier. If the desired claim is localization, the maps need explicit localization validation and should be compared against task-appropriate alternatives such as segmentation or detection models.
- Practical pointer: This source strongly supports the current thesis direction: the project evaluates whether classifier explanations align with pneumothorax masks, rather than claiming that saliency maps are lesion segmentations.

### Chest X-ray saliency methods can underperform human localization benchmarks

- Source link: [`REF-CXR-SALIENCY-BENCHMARK`](references.md#ref-cxr-saliency-benchmark)
- Thesis-ready paraphrase: CXR saliency benchmarking shows that explanation quality varies by method, model, and clinical condition, and that saliency maps may fall short of human localization benchmarks. This supports treating pneumothorax heatmap evaluation as an empirical question rather than assuming that a standard XAI method will localize the clinically relevant region.
- Practical pointer: In Chapter 4, discuss lesion size, lesion shape, subtle pneumothorax, chest tubes/devices, and classifier confidence as plausible factors affecting explanation localization. This also supports the planned balanced outcome review and targeted qualitative case studies.

### CheXlocalize shows a concrete CXR benchmark design for localization

- Source link: [`REF-CHEXLOCALIZE`](references.md#ref-chexlocalize)
- Thesis-ready paraphrase: CheXlocalize is useful as a nearby example of CXR explanation validation because it pairs image-level disease tasks with radiologist localization annotations, including pixel-level segmentations and most-representative points. This supports the thesis choice to report both overlap-style metrics and pointing-hit behavior.
- Practical pointer: In Chapter 3, explicitly state that SIIM masks play the role of the localization reference standard for this thesis. CheXlocalize can be cited as evidence that expert segmentations and representative-point metrics are accepted ways to benchmark CXR explanation localization, while noting that the current data source and task are SIIM pneumothorax-specific.

### XAI should complement validation, not substitute for it

- Source link: [`REF-FALSE-HOPE-XAI`](references.md#ref-false-hope-xai)
- Thesis-ready paraphrase: Health-care XAI literature warns that current explanation methods are not sufficient by themselves to guarantee clinical safety, fairness, or reliable patient-level decision support. For this thesis, explanation maps should therefore be presented as diagnostic evidence about model behavior, not as proof that the model is safe for deployment.
- Practical pointer: In Chapter 5, avoid promising that adding heatmaps makes the classifier trustworthy. State that the study uses XAI to expose limitations and generate clinically interpretable evidence, while final trust still depends on validation design, reference standards, dataset representativeness, and prospective clinical evaluation.

## Medical-Imaging-Specific XAI Lessons

### Medical XAI needs clinical users and tasks, not generic interpretability claims

- Source link: [`REF-HUMAN-CENTERED-XAI`](references.md#ref-human-centered-xai)
- Thesis-ready paraphrase: In medical imaging, explanation quality depends on the clinical task and the user who must interpret the output. This supports including a radiologist-style review workbook with explicit rubric categories rather than relying only on algorithmic scores.
- Practical pointer: Use this source to justify the scoring schema: localization correctness, usefulness, failure category, artifact note, and free-text comment.

### Medical-image XAI literature is broad, but evaluation remains a key weakness

- Source link: [`REF-MEDICAL-XAI-REVIEW`](references.md#ref-medical-xai-review)
- Thesis-ready paraphrase: Reviews of deep-learning XAI in medical image analysis show that many explanation methods exist, but their clinical meaning depends on modality, task, model, and validation design. For this thesis, it is safer to present the work as a focused validation study of explanation methods for a specific CXR pneumothorax setting.
- Practical pointer: Chapter 2 can use this review to organize methods into post-hoc saliency/attribution families, while Chapter 3 narrows the scope to methods actually implemented and evaluated.

### Radiology interpretability must be tied to the clinical task

- Source link: [`REF-RADIOLOGY-INTERPRETABILITY`](references.md#ref-radiology-interpretability)
- Thesis-ready paraphrase: Radiology interpretability is not a single technical property. A useful explanation depends on the imaging task, the target user, the model output, and the decision context. In this thesis, the relevant question is therefore not whether a heatmap is generally interpretable, but whether it helps a radiology-trained reader judge pneumothorax evidence and model failure patterns.
- Practical pointer: Use this source to motivate the radiologist-style workbook and the separation between algorithmic localization metrics and qualitative clinical usefulness categories.

### XAI can reveal shortcut learning in radiology models

- Source link: [`REF-CLEVER-HANS`](references.md#ref-clever-hans)
- Thesis-ready paraphrase: Published radiology AI audits have shown that models may exploit shortcuts or non-pathology image cues. This is directly relevant when a CXR classifier has moderate ranking performance but heatmaps repeatedly highlight non-lesion structures, devices, image borders, text markers, or diffuse regions.
- Practical pointer: In discussion, treat weak localization as clinically meaningful evidence about the off-the-shelf model’s behavior. Avoid overstating that the XAI method itself is wrong without separating classifier quality, dataset mismatch, preprocessing, and explanation method limitations.

### Shortcut learning gives a general explanation for predictive-but-misaligned models

- Source links: [`REF-SHORTCUT-LEARNING`](references.md#ref-shortcut-learning), [`REF-RADIOLOGY-SHORTCUTS`](references.md#ref-radiology-shortcuts)
- Thesis-ready paraphrase: Shortcut learning describes the tendency of deep models to solve an objective using easy or unintended cues that work in the training data but may fail under distribution shift or clinical scrutiny. In radiology AI, possible shortcuts include acquisition markers, scanner/site patterns, devices, text, demographic proxies, and other non-pathology signals.
- Practical pointer: Use this framing when explaining why the current ResNet-50 follow-up can be relatively better than DenseNet-all yet still clinically weak in absolute localization. The key research question becomes whether explanations show pneumothorax evidence or merely model-useful correlates.

### Medical images can contain hidden signals that clinicians do not consciously perceive

- Source link: [`REF-RACE-MEDICAL-IMAGING`](references.md#ref-race-medical-imaging)
- Thesis-ready paraphrase: Medical-imaging models can learn predictive signals that are not obvious to human readers and may not be clinically intended for the diagnostic task. This supports caution when interpreting both high classifier performance and plausible-looking heatmaps: the model may be using latent acquisition, population, or image-property signals rather than lesion evidence.
- Practical pointer: This source is useful in limitations and ethics/fairness discussion. The current thesis does not need to perform demographic fairness analysis, but it should acknowledge that hidden signals and subgroup behavior remain outside the available SIIM mask-based evaluation.

### External validation matters for chest X-ray models

- Source link: [`REF-CXR-GENERALIZATION-ZECH`](references.md#ref-cxr-generalization-zech)
- Thesis-ready paraphrase: Chest X-ray classifiers can show materially different performance when tested across institutions or datasets. This supports treating the TorchXRayVision SIIM-ACR experiment as an external-transfer validation, not as a direct measure of how well the original training datasets were modeled.
- Practical pointer: In Chapter 4, separate classifier ranking performance from explanation localization. If the model remains moderately predictive but poorly localized, this can be discussed as a transfer/generalization problem where disease-label prediction does not guarantee clinically aligned lesion evidence.

### Hidden stratification supports outcome- and subgroup-aware review

- Source link: [`REF-HIDDEN-STRATIFICATION`](references.md#ref-hidden-stratification)
- Thesis-ready paraphrase: A model can have acceptable aggregate performance while failing on clinically important subsets that are not explicitly labeled or evaluated. For pneumothorax, treatment devices, subtle lesions, unusual acquisition patterns, and label noise are plausible hidden factors that may affect both classification and explanation maps.
- Practical pointer: Use this source to justify the balanced `tp`/`fp`/`tn`/`fn` review workflow and the failure taxonomy. A small qualitative review can look for device/text/artifact reliance even when the quantitative dataset does not provide formal subgroup labels.

## Statistical Methods for Method-vs-Method Comparison

### Wilcoxon signed-rank is the right paired test for non-normal localization residuals

- Source links: [`REF-WILCOXON-1945`](references.md#ref-wilcoxon-1945), [`REF-DEMSAR-2006`](references.md#ref-demsar-2006)
- Thesis-ready paraphrase: The Wilcoxon signed-rank test is the non-parametric paired analog of the paired t-test. For each case it ranks the absolute paired differences (e.g. consensus IoU minus Grad-CAM IoU), sums the ranks of positive and negative differences, and tests whether the median paired difference is zero. It does not assume the per-case difference distribution is normal, which matters because IoU and Dice residuals on tail-heavy XAI benchmarks rarely are, and it is less sensitive than the paired t-test to outliers from one or two anomalously good or bad explanations. This is also the test recommended by Demšar (2006) for paired classifier-method comparisons in machine-learning evaluation literature.
- Practical pointer: In methodology, state explicitly that the test is two-sided unless a directional alternative is justified by H1 prior to seeing the data. Report the test statistic, p-value, median paired difference, and a 95% bootstrap confidence interval on the paired difference (10 000 resamples) as the effect-size companion. Avoid reporting p-values alone, since N is small enough that small p-values can still correspond to small effects.

### Holm-Bonferroni is uniformly more powerful than plain Bonferroni at the same family-wise error rate

- Source links: [`REF-HOLM-1979`](references.md#ref-holm-1979), [`REF-AICKIN-GENSLER-1996`](references.md#ref-aickin-gensler-1996), [`REF-DEMSAR-2006`](references.md#ref-demsar-2006)
- Thesis-ready paraphrase: When the improvement experiment compares consensus against each of the N individual XAI methods, running N tests at α=0.05 inflates the family-wise probability of falsely rejecting at least one true null hypothesis to roughly 1 − (1 − α)^N (~30% for N=7). Both Bonferroni and Holm-Bonferroni control the family-wise error rate at α, but they do so differently. Plain Bonferroni rejects each test only if its p-value clears α / N. Holm-Bonferroni instead sorts the N p-values ascending and tests them sequentially against escalating thresholds α / N, α / (N−1), …, α / 1, stopping at the first failure. Holm therefore rejects every hypothesis Bonferroni rejects, and possibly more, while maintaining the same FWER guarantee — it is uniformly more powerful at the same α. For paired XAI method comparison with N ≈ 5–7 tests and confirmatory framing, Holm is the recommended choice in both medical-statistics (Aickin & Gensler, 1996) and ML evaluation (Demšar, 2006) literature.
- Practical pointer: In methodology, write: "Family-wise error was controlled across the N consensus-vs-individual paired tests per metric using the Holm-Bonferroni step-down procedure at α=0.05." Report both the raw p-value and the Holm-adjusted threshold each test was compared against, so the reader can audit the decision. If even one paired comparison clears the corrected threshold for any metric, frame the consensus contribution accordingly in Discussion; if none clear it, frame as Narrative B with method-disagreement analysis taking the foreground.

### Stats discipline in code: scipy + statsmodels, no new heavy dependencies

- Practical pointer: `scipy.stats.wilcoxon` and `statsmodels.stats.multitest.multipletests(..., method="holm")` are both already callable in the current environment (scipy is required; add statsmodels to `requirements-dev.txt` if not present). Effect sizes are computed by bootstrap-resampling per-case paired differences with `numpy.random.default_rng(seed)` to stay deterministic. Document RNG seeds in methodology alongside the standard `seed=20260515` discipline already used elsewhere in the project.

## Reporting and Validation Discipline

### CLAIM supports transparent medical-imaging AI reporting

- Source link: [`REF-CLAIM`](references.md#ref-claim)
- Thesis-ready paraphrase: Medical-imaging AI studies should clearly report data sources, eligibility criteria, preprocessing, ground truth, partitions, model details, evaluation metrics, and limitations. This directly supports the thesis habit of documenting dataset paths, local counts, train/test split rules, calibration choices, and output schemas.
- Practical pointer: Before final submission, use CLAIM as a checklist for Chapter 3: dataset source and local snapshot, mask reference standard, preprocessing, classifier threshold calibration, XAI method settings, metrics, hardware/software versions, and limitations.

### TRIPOD+AI helps frame the classifier as a prediction model

- Source link: [`REF-TRIPOD-AI`](references.md#ref-tripod-ai)
- Thesis-ready paraphrase: When the pneumothorax network is treated as a clinical prediction model, its inputs, intended use, validation data, thresholding, and performance measures should be reported transparently. This matters even though the thesis focus is XAI, because explanation quality cannot be interpreted independently from classifier behavior.
- Practical pointer: Use this source in Chapter 3 or limitations when explaining why classifier screening metrics, threshold calibration, and `tp`/`fp`/`tn`/`fn` sampling are documented before explanation-map analysis.

### QUADAS-AI supports risk-of-bias and applicability discussion

- Source link: [`REF-QUADAS-AI`](references.md#ref-quadas-ai)
- Thesis-ready paraphrase: Diagnostic-AI evaluation should consider not only numerical accuracy but also patient selection, data sources, reference standards, and applicability to the intended clinical context. For the current project, the major applicability issue is that an off-the-shelf CXR model is being evaluated on SIIM pneumothorax images rather than on the exact distribution used for model development.
- Practical pointer: Use QUADAS-AI language in the limitations section: retrospective dataset, public challenge data, report-derived pretraining labels, missing clinical context, uncertain deployment population, and single-reader qualitative review are all sources of residual bias or limited applicability.

### DECIDE-AI can frame the review workbook as early human-centered evaluation

- Source link: [`REF-DECIDE-AI`](references.md#ref-decide-ai)
- Thesis-ready paraphrase: If the radiologist-style workbook is discussed as more than qualitative illustration, it should be framed as early-stage, controlled evaluation rather than deployment evidence. The useful output is structured information about explanation usefulness, errors, and workflow fit under a clearly defined review task.
- Practical pointer: Use DECIDE-AI to report the intended user, task, review environment, inputs shown to the rater, scoring categories, and limitations. Do not imply prospective clinical validation; this is a thesis-scale human-centered assessment artifact.

### CT pilot scope is gated by the hour-1 model-availability check

- Source link: [`REF-RSNA-IHD`](references.md#ref-rsna-ihd)
- Thesis-ready paraphrase: The CT pilot tests whether the validation findings on CXR pneumothorax transfer to a different modality and task. Because the thesis does not pre-commit to a CT performance claim, the scope is scoped by the hour-1 model-availability check at the start of Phase 5.4: if a verifiable off-the-shelf RSNA-IHD-derived classifier with a usable hemorrhage class head, license, version, and preprocessing contract is found, the pilot runs as a small qualitative + quantitative smoke on 20-30 manually annotated slices; if no such model exists, the pilot collapses to qualitative external validation only per the experiment protocol's Week-3 fallback rule. Either outcome is thesis-defensible because no prior claim has been written that requires a CT smoke to land.
- Practical pointer: In methodology, frame the CT contribution honestly per outcome. Branch A (model found): describe the CT pipeline as deliberate scope-minimization (off-the-shelf classifier, small manual annotation, no calibration regen), explicitly avoiding over-claiming localization. Branch B (no model): describe the CT contribution as a qualitative external-validation note pointing future work toward CT XAI on a pre-trained head, and cite the RSNA-IHD challenge as the natural training distribution for that future work. In both cases the thesis methodology cites `REF-RSNA-IHD` so the CT discussion is anchored to a peer-reviewed challenge rather than to a generic claim about "CT".

### Coverage saturation can be a future secondary heatmap diagnostic

- Project-derived metric idea: `coverage_saturation_fraction_95` / `top_fraction_at_95_coverage` would record the smallest swept top-fraction at which the selected attribution mask covers at least `95%` of the image. Lower values indicate that the thresholded map becomes whole-image-like quickly; higher values indicate a more spatially concentrated or less saturated map across the tested fractions.
- Thesis-ready paraphrase: This metric should be framed as a spatial diffuseness/saturation diagnostic, not as evidence that a heatmap is clinically correct. It can explain why some threshold panels become visually redundant and can compare whether positive, negative, magnitude, or signed views differ in how quickly they expand over the image.
- Practical pointer: Use it only as a secondary or future-work metric alongside `IoU`, `Dice`, `pointing_hit`, `precision_at_fraction`, negative-evidence diagnostics, faithfulness curves, and review scores. If a negative view reaches `95%` coverage later than a positive view, the safe interpretation is that the negative evidence is more spatially concentrated, not automatically that the negative explanation is better or clinically correct.

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

### Radiologist-localized CXR datasets are useful future validation context

- Source link: [`REF-VINDR-CXR`](references.md#ref-vindr-cxr)
- Thesis-ready paraphrase: VinDr-CXR shows that public CXR resources can include radiologist localization annotations, not only image-level report labels. Although the current thesis uses SIIM-ACR because it directly targets pneumothorax masks, VinDr-CXR is useful context for future external validation or for testing whether the conclusions generalize beyond one public challenge dataset.
- Practical pointer: Mention VinDr-CXR only as future-work/context unless it is actually used. Do not merge its counts, labels, or annotation style with SIIM-ACR results; keep dataset-specific conclusions separate.

### Peer-reviewed CXR baselines can replace arXiv-only historical context

- Source link: [`REF-CHEXNEXT`](references.md#ref-chexnext)
- Thesis-ready paraphrase: CheXNeXt is a peer-reviewed example of multi-label CXR classification evaluated against practicing radiologists on a held-out annotated set. It is useful background for showing that CXR deep-learning classifiers can reach strong image-level performance, while also reminding the reader that such studies usually evaluate classification rather than lesion-mask-aligned explanations.
- Practical pointer: Prefer citing CheXNeXt for historical CXR classifier/radiologist-comparison context unless an arXiv-only CheXNet citation is explicitly needed and approved. Do not use it as evidence that the current TorchXRayVision baseline is clinically validated on SIIM pneumothorax localization.

## Candidate Literature Review Structure

1. Medical imaging AI needs explainability because high-stakes clinical predictions require more than classification accuracy.
   - Support: [`REF-MEDICAL-XAI-REVIEW`](references.md#ref-medical-xai-review), [`REF-HUMAN-CENTERED-XAI`](references.md#ref-human-centered-xai), [`REF-RADIOLOGY-INTERPRETABILITY`](references.md#ref-radiology-interpretability)
2. Post-hoc saliency and attribution methods are commonly used, but each answers a different question.
   - Support: [`REF-GRADCAM`](references.md#ref-gradcam), [`REF-GRADCAMPP`](references.md#ref-gradcampp), [`REF-INTEGRATED-GRADIENTS`](references.md#ref-integrated-gradients), [`REF-SHAP`](references.md#ref-shap), [`REF-SCORECAM`](references.md#ref-scorecam), [`REF-OCCLUSION-ZEILER`](references.md#ref-occlusion-zeiler)
3. Explanation validation should combine localization, faithfulness, robustness/sanity checks, and human-centered review.
   - Support: [`REF-SANITY-CHECKS`](references.md#ref-sanity-checks), [`REF-RISE`](references.md#ref-rise), [`REF-SAMEK-EVALUATION`](references.md#ref-samek-evaluation), [`REF-MEANINGFUL-PERTURBATION`](references.md#ref-meaningful-perturbation), [`REF-SALIENCY-TRUST-MEDICAL`](references.md#ref-saliency-trust-medical), [`REF-CXR-SALIENCY-BENCHMARK`](references.md#ref-cxr-saliency-benchmark), [`REF-CHEXLOCALIZE`](references.md#ref-chexlocalize), [`REF-HUMAN-CENTERED-XAI`](references.md#ref-human-centered-xai), [`REF-FALSE-HOPE-XAI`](references.md#ref-false-hope-xai)
4. Chest X-ray pneumothorax is a good focused case because the dataset has clinically meaningful masks and an official challenge context.
   - Support: [`REF-SIIM-ACR`](references.md#ref-siim-acr), [`REF-TORCHXRAYVISION`](references.md#ref-torchxrayvision), [`REF-CHESTXRAY8`](references.md#ref-chestxray8), [`REF-CHEXNEXT`](references.md#ref-chexnext), [`REF-CHEXPERT`](references.md#ref-chexpert), [`REF-MIMIC-CXR`](references.md#ref-mimic-cxr), [`REF-VINDR-CXR`](references.md#ref-vindr-cxr)
5. A key thesis contribution can be negative or cautionary: off-the-shelf pretrained models may be predictive enough to screen but poorly localized or shortcut-driven.
   - Support: [`REF-CLEVER-HANS`](references.md#ref-clever-hans), [`REF-RIGHT-REASONS`](references.md#ref-right-reasons), [`REF-SHORTCUT-LEARNING`](references.md#ref-shortcut-learning), [`REF-RADIOLOGY-SHORTCUTS`](references.md#ref-radiology-shortcuts), [`REF-RACE-MEDICAL-IMAGING`](references.md#ref-race-medical-imaging), [`REF-CXR-GENERALIZATION-ZECH`](references.md#ref-cxr-generalization-zech), [`REF-HIDDEN-STRATIFICATION`](references.md#ref-hidden-stratification), [`REF-QUADAS-AI`](references.md#ref-quadas-ai), [`REF-TORCHXRAYVISION`](references.md#ref-torchxrayvision), current project results.
6. The methodology chapter should report enough detail for reproducibility and risk-of-bias assessment.
   - Support: [`REF-CLAIM`](references.md#ref-claim), [`REF-TRIPOD-AI`](references.md#ref-tripod-ai), [`REF-QUADAS-AI`](references.md#ref-quadas-ai), [`REF-DECIDE-AI`](references.md#ref-decide-ai)
7. Input-space attribution methods transfer across architectures and modalities with identical implementations; CAM-family methods do not, because they depend on a convolutional feature map the backbone may not have.
   - Integrated Gradients, GradientSHAP, and Occlusion read only the model input and the output score, so the *same* implementation runs unchanged on a convolutional chest-X-ray classifier and on a Vision-Transformer CT classifier — only the model and the wrapped target scalar (`1 - P(normal)` for hemorrhage) differ. CAM-family methods (Grad-CAM, Grad-CAM++, Eigen-CAM, Score-CAM) instead weight a two-dimensional convolutional activation map of shape (channels, height, width) and upsample it; their localization is inherited from the retinotopic layout of that grid. A Vision Transformer has no such tensor — the image becomes a sequence of patch tokens plus a non-spatial class token — so a CAM can only be run after a token→grid reshape that introduces new modeling choices (which tensor/layer, how to handle the class token, global vs local attention) and is therefore a *different, contested* method, not the same one. Restricting the CT pilot to the input-space methods keeps the cross-modality comparison controlled (identical explanation code; only model and modality vary); adapting CAM-family methods to the ViT via transformer-native techniques is future work.
   - Support: [`REF-GRADCAM`](references.md#ref-gradcam), [`REF-VIT`](references.md#ref-vit), [`REF-ATTENTION-ROLLOUT`](references.md#ref-attention-rollout), [`REF-TRANSFORMER-ATTRIBUTION`](references.md#ref-transformer-attribution), [`REF-INTEGRATED-GRADIENTS`](references.md#ref-integrated-gradients), [`REF-OCCLUSION-ZEILER`](references.md#ref-occlusion-zeiler)

## Thesis-Safe Wording Bank

- "The generated maps are class-specific attribution visualizations with respect to the pneumothorax output, not anatomical segmentations."
- "Mask overlap evaluates clinical localization against available lesion annotations, whereas deletion/insertion curves evaluate whether the highlighted pixels affect the model’s own probability."
- "Agreement across methods is stronger evidence than any single overlay, while disagreement is diagnostically useful because it can expose instability, baseline sensitivity, or reliance on non-lesion cues."
- "The off-the-shelf baseline is valuable precisely because it tests transfer behavior: a pretrained medical model can be moderately predictive on a new dataset while still relying on features that are clinically questionable for localization."
- "Radiologist review is treated as a structured qualitative assessment, not as a replacement for quantitative localization or faithfulness metrics."
- "Input-space attribution methods (Integrated Gradients, GradientSHAP, Occlusion) depend only on the model input and output score, so the identical implementation transfers from the convolutional chest-X-ray classifier to the Vision-Transformer CT classifier; this keeps the cross-modality comparison controlled rather than confounded by re-implementation."
- "CAM-family methods require a two-dimensional convolutional feature map and therefore do not apply directly to a Vision Transformer, whose internal representation is a sequence of patch tokens; adapting them would require a transformer-specific reshape that constitutes a different method, which is left to future work."
- "Large report-labeled CXR datasets enable useful pretrained models, but image-level report labels do not guarantee that the learned evidence aligns with pixel-level lesion masks."
- "Occlusion maps are perturbation diagnostics: they show how the model probability changes when image patches are replaced, not where gradients are largest."
- "Heatmaps can support model critique, but they do not replace rigorous validation, transparent reporting, or clinical applicability assessment."
- "The classifier and the explanation pipeline should be reported together because an explanation can only be interpreted in relation to the model output, threshold, target class, preprocessing, and validation data."
- "A predictive model can be right for dataset-specific or model-useful reasons while still being clinically misaligned with the visible pathology."
- "The thesis contribution is not that heatmaps create trust automatically; it is that validated heatmaps and failure review can expose when trust is not justified."
- "A perturbation faithfulness curve is only as clinically meaningful as the perturbation design; replacement baselines and artifacts must be reported."

## Follow-Up Sources to Consider Later

- Additional peer-reviewed CXR generalization/bias papers if Chapter 4 needs more than the current Zech/generalization, hidden-stratification, and shortcut-learning sources.
- CheXNet or other arXiv-only historical CXR baselines only if the user approves preprint use, or if a peer-reviewed final source is identified.