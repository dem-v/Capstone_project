# Thesis Research References

This file lists reliable, verifiable sources used for thesis background notes. Reference IDs are stable cross-links for `docs/thesis-notes.md`.

Access date for URLs checked in this pass: 2026-05-24.

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

## Working Notes on Source Reliability

- `Dataset Ninja` and similar mirrors were not used as primary references, even when they contain convenient dataset counts, because the user requested reliable, verifiable sources. Official SIIM/Kaggle pages and local dataset verification should be preferred for final counts.
- Kaggle discussion posts and winning-solution repositories can be useful for engineering ideas, but they are not included here as thesis evidence unless the user explicitly approves them.
- YouTube talks, vendor/press articles, and arXiv-only historical CXR baselines were not used as primary sources in this pass.