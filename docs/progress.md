# Master Thesis Progress Memory

This file keeps the project context, decisions, constraints, and next steps for the master thesis work.

## 2026-05-07 - Initial Thesis Planning Context

### Thesis Direction

Working concept: use Explainable AI techniques to evaluate existing tools/models for CT and/or X-ray segmentation and/or classification, identify what drives their predictions/segmentations, and provide expert interpretation as a practising radiologist.

Academic framing: Master of Science in Computer Science, Data Science and Machine Learning, intentionally positioned at the border of computer science and radiology.

Preferred modality: CT is personally more important; X-ray is acceptable if it makes the thesis safer and stronger within the deadline.

Preferred thesis type: academic research, not startup/enterprise. Written in English.

### Required Outputs

- Written thesis.
- Code repository if experiments require code.
- Weekly experiment/progress demonstrations.
- Defense presentation after thesis submission.
- Possible publication after defense.

### University Requirements Extracted From Local Files

Source files:
- `requirements/Покрокова інструкція для кожного етапу виконання дипломної роботи.md`
- `requirements/Шаблон пояснювальної записки дипломної роботи (академічне дослідження) та вимоги до її технічного оформлення .md`

Key process requirements:
- Weekly progress reports are expected.
- Weekly report should include completed work, risks/challenges, next-week plan, and artifact links.
- Weekly reports are due by Sunday 23:59 unless otherwise agreed with supervisor.
- Demo checkpoints in the standard plan:
  - Week 2: first demo, up to 15 minutes.
  - Week 4: second demo, 10-15 minutes.
  - Week 6: final demo, 15 minutes.
- Thesis plagiarism check is expected before defense; originality requirement is more than 55%.
- For plagiarism check, the final searchable PDF includes title through bibliography, without appendices.
- Defense talk for individual work is up to 7 minutes.
- Presentation maximum is 20 slides.

Key thesis format requirements:
- Main thesis body: 25-50 pages, chapters 1-5.
- Abstract: 250-300 words.
- Bibliography count for academic research should be at least equal to the number of pages in chapters 2, 3, and 4.
- Citation style: APA or IEEE, consistent throughout.
- Quotes/borrowed text should not exceed 20-25% of the main body and no more than 5% from one source.
- Final submission format: searchable PDF.
- A4, Times New Roman, 12 pt body, 14 pt bold headings, 1.5 line spacing for text.

Required thesis structure:
- Abstract.
- Chapter 1: Introduction.
- Chapter 2: Literature Review / Problem Analysis.
- Chapter 3: Methodology.
- Chapter 4: Results and Discussion.
- Chapter 5: Conclusions and Recommendations.
- Bibliography.
- Appendices.

### Data and Resources

Available data:
- CT and X-ray DICOM studies.
- Body regions vary widely.
- Depending on body region, available study count ranges from tens to tens of thousands.
- No structured labels currently.
- Reports are available as free text.
- No segmentation masks currently.
- Data can be anonymized.
- Retrospective use is possible, with approval if needed.

Compute:
- MSI GS66 laptop with RTX 3050.
- Google Colab available.
- Other resources may be available.
- Python/PyTorch notebooks are comfortable.
- No package installation restrictions.

### Explainability Interests

Strong interest:
- Heatmaps.
- Clinician-centered qualitative evaluation.
- Segmentation uncertainty/error analysis.

Possible additional methods:
- Perturbation/occlusion sensitivity.
- Integrated Gradients.
- SHAP/LIME if time allows and if they add value.

Evaluation interests:
- Quantitative metrics if feasible.
- Explainability comparison.
- Radiologist qualitative critique.
- Failure-case taxonomy.

### Major Risks

- Scope uncertainty.
- Low-quality or unconvincing result.
- Unexpected costs.
- Missing compressed deadlines.
- Lack of labels and segmentation masks.
- Ethics/anonymization/approval delays.

### Current Planning Judgment

Because only 4 weeks are available, the thesis should avoid requiring new model training or large manual annotation. The safest path is to evaluate existing pretrained models/tools and use a small, carefully selected, anonymized radiologist-reviewed dataset.

Likely strong direction:
- Existing CT segmentation tools, with radiologist-guided qualitative and quantitative/error analysis.

Likely safer fallback:
- Existing chest X-ray classification models with heatmap explainability and radiologist-centered failure analysis.

Important decision still pending:
- Choose one primary path and one fallback path after quick feasibility checks of available tools, data export/anonymization, and report/label extraction.

### Immediate Next Steps

1. Identify the exact 4-week calendar deadlines and map them to required artifacts.
2. Choose a primary thesis topic and a fallback topic.
3. Run a feasibility scan of open-source CT segmentation and X-ray classification/explainability tools.
4. Define a minimal dataset protocol:
   - modality/body region,
   - inclusion/exclusion criteria,
   - anonymization process,
   - number of cases,
   - expert review template.
5. Create the thesis skeleton in English using the required chapter structure.
6. Create the first weekly progress report template.

## 2026-05-07 - Initial Scope Recommendation

### Recommended Primary Topic

Working title:

Explainable Evaluation of Automated CT Segmentation Tools: A Radiologist-Centered Error and Uncertainty Analysis

Rationale:
- CT matches the student's personal preference and clinical expertise.
- Existing segmentation tools can produce results without manually creating full ground-truth masks.
- Radiologist review can become the main expert-evaluation contribution.
- The work can still satisfy Computer Science / Data Science / Machine Learning expectations through model evaluation, explainability/error analysis, reproducible pipeline design, and quantitative summaries.

Core research question:

How clinically reliable and interpretable are existing automated CT segmentation tools when applied to retrospective DICOM studies, and what types of segmentation errors are most relevant from a radiologist's perspective?

Minimum viable experiment:
- 20-40 anonymized CT studies from one body region/protocol.
- One primary tool: TotalSegmentator.
- Optional second tool if feasibility permits: MedSAM/MedSAM2 or a MONAI bundle.
- Target structures chosen for visibility and clinical relevance, e.g. liver, spleen, kidneys, lungs/lobes, vertebrae, aorta, depending on available CT region.
- Quantitative outputs:
  - segmentation availability/failure rate,
  - runtime,
  - volume statistics,
  - Dice/IoU only on a small manually corrected subset if feasible,
  - expert ordinal scores for boundary accuracy and clinical usability.
- Qualitative outputs:
  - radiologist error taxonomy,
  - representative visual examples,
  - discussion of clinically meaningful vs clinically irrelevant errors.

### Fallback Topic

Working title:

Explainable AI Evaluation of Chest X-ray Classification Models: Heatmap Faithfulness and Radiologist-Centered Failure Analysis

Use if CT tooling, DICOM conversion, runtime, or anonymization becomes too risky.

Minimum viable experiment:
- 50-200 anonymized chest X-rays.
- Existing pretrained TorchXRayVision model.
- Grad-CAM / Score-CAM / Eigen-CAM heatmaps.
- Compare model predictions with report-derived weak labels and radiologist review.
- Failure taxonomy: wrong region attention, device/crop confounding, projection/positioning issues, label noise from reports, clinically nonspecific findings.

### Compressed 4-Week Plan

Assumption: only 4 calendar weeks remain, so the standard 8-week university process must be compressed. Priority is early demoable output, then thesis writing in parallel with experiments.

Week 1:
- Finalize topic with supervisor.
- Select CT body region and 20-40 candidate studies.
- Confirm anonymization workflow.
- Build DICOM-to-NIfTI/test pipeline.
- Run first tool on 3-5 cases.
- Produce first screenshots and preliminary feasibility report.
- Start Chapter 1 and literature source collection.

Week 2:
- Run primary experiment on full selected subset.
- Define and use radiologist review template.
- Produce preliminary result tables and visual examples.
- Decide whether a second tool is feasible.
- Draft Chapter 2 and Chapter 3.
- Prepare first/second demo material depending on supervisor schedule.

Week 3:
- Complete experiments.
- Complete expert review.
- Build error taxonomy and quantitative summaries.
- Draft Chapter 4 Results and Discussion.
- Draft Chapter 5.
- Prepare publication-oriented framing notes.

Week 4:
- Finalize thesis text, bibliography, figures, tables, appendices.
- Format according to requirements.
- Prepare searchable PDF.
- Prepare plagiarism-check version without appendices if required.
- Build defense presentation, max 20 slides.
- Rehearse 7-minute individual defense talk.

### Supervisor Questions To Resolve

1. Is an academic research thesis evaluating existing models/tools acceptable without training a new model?
2. Is retrospective anonymized patient imaging allowed for thesis figures if all identifiers are removed?
3. Can radiologist expert review be treated as the principal validation component?
4. Does the supervisor prefer CT segmentation as primary scope, or X-ray classification as a safer scope?
5. What exact dates apply for the next weekly demo, thesis draft, plagiarism check, final submission, and defense?
6. Which citation style should be used: IEEE is recommended for CS/ML unless the supervisor prefers APA.

## 2026-05-07 - Clarifications From Student

Updated constraints:
- The effective thesis completion window is 4 weeks.
- The following 2 weeks should be reserved for defense preparation and defense execution.
- Absolute finish date for thesis completion plus defense handling is 2026-06-21.
- 20-40 CT studies can likely be exported/anonymized.
- Kaggle datasets are also acceptable if local data access becomes slow or risky.
- No strict body-region preference; choose whichever is technically and academically strongest.
- Head CT is often available, but the site receives about 30 CT studies per day across different regions.
- Local real patient data and public/Kaggle datasets are both feasible; a mixed design may be possible.
- Supervisor is a math professor; evaluation of existing models/tools with rigorous methodology is likely acceptable, but should be confirmed.
- Citation style should follow the local template/instructions. Since the template allows APA or IEEE, use IEEE by default for CS/ML unless the supervisor requests APA.
- Student is comfortable creating a small manual expert review table, e.g. cases x structures x scoring categories.

Planning adjustment:
- The thesis must be treated as a 4-week research sprint with no dependency on large annotation, model training, or ethics delays.
- The 2 defense weeks should not contain core experiment work except small corrections requested by the supervisor.
- Primary path remains CT segmentation evaluation.
- Fallback path remains X-ray classification/explainability using public datasets if local DICOM extraction becomes a blocker.

Research-gap judgment:
- Attention heatmaps for classification, especially chest X-ray classification with Grad-CAM-like methods, are heavily researched and have existing benchmark papers.
- Radiologist-centered expert evaluation of automated CT segmentation is less saturated when framed as clinical usability, error taxonomy, and explanation of model behavior on local retrospective data.
- Automated segmentation itself is researched, but the clinical expert-opinion layer is still a stronger thesis novelty angle than another heatmap classification comparison.

## 2026-05-07 - Supervisor Suggestion: Explanation-Focused Variant

Supervisor suggested:
- Use something that sounded like "Shym"; likely SHAP (SHapley Additive exPlanations), but confirm spelling.
- Bring the topic closer to explainability.
- Inspect datasets, models, and Kaggle solutions.
- Combine models with explanation methods.
- Test different explanation approaches.
- Validate explainability of explanation approaches/models.
- Possible final goal: improve classification, or improve explanation approaches/techniques/algorithms themselves.

Additional literature restriction:
- Main Ukrainian-language articles should be no older than 5 years.
- Main English-language articles should be no older than 10 years.
- As of 2026-05-07, this means:
  - Ukrainian main sources: 2021 or newer.
  - English main sources: 2016 or newer.
- Foundational English XAI methods still fit if published in 2016 or newer:
  - LIME: 2016.
  - Grad-CAM: 2017.
  - SHAP: 2017.
  - Integrated Gradients: 2017.
- U-Net 2015 is older than the English 10-year main-source window, so avoid relying on it as a main article unless the supervisor allows historical/foundational exceptions.

Revised best-fit thesis direction:

Validation and Improvement of Explainability Methods for Medical Image Classification: A Radiologist-Centered Study

Most practical dataset direction:
- SIIM-ACR pneumothorax chest X-ray dataset or a cleaned Kaggle derivative.
- Reason: it provides both classification labels and pixel-level pneumothorax masks, which makes objective explanation validation possible.
- Use local retrospective data as external qualitative validation if feasible, not as the core dependency.

Candidate methods:
- Classification model: DenseNet/ResNet/EfficientNet, preferably pretrained or quickly fine-tuned.
- Explainability methods:
  - Grad-CAM.
  - Grad-CAM++.
  - Score-CAM or Eigen-CAM.
  - Integrated Gradients.
  - GradientSHAP / SHAP-style attribution.
  - Occlusion sensitivity.
  - Optional LIME if time permits.

Validation metrics:
- Classification: AUC, accuracy, sensitivity, specificity, F1.
- Explanation localization against masks: mIoU, Dice, hit rate / pointing game, precision-at-k.
- Explanation faithfulness: deletion/insertion curves or Captum infidelity/sensitivity if feasible.
- Clinical evaluation: radiologist score of whether the explanation highlights the clinically relevant region and whether it would be misleading.

Possible contribution/improvement:
- Do not attempt a large new model architecture.
- More feasible contribution: create a simple explanation-consensus method, e.g. normalize and combine the best-performing attribution maps, then compare it against individual methods.
- Alternative contribution: select explanation thresholds using validation masks and show improved localization/reduced misleading heatmaps.
- Alternative classification improvement: explanation-guided crop or mask-based auxiliary experiment, but only if time remains.

Current planning implication:
- If the supervisor strongly wants SHAP/explainability validation, the X-ray pneumothorax path becomes safer than CT segmentation because it has public masks for objective evaluation.
- CT segmentation remains a clinically interesting backup or a discussion/future-work track.

## 2026-05-07 - Mixed Modality Option

Student is willing to create lesion masks manually for a test subset. This makes RSNA Intracranial Hemorrhage Detection or local head CT feasible for explanation-localization validation, not only classification.

Potential combined design:
- Primary experiment: one modality/task with full quantitative explanation validation.
- Secondary pilot: another modality/task to test whether explanation-method conclusions transfer across modality.

Candidate mixed design:
- X-ray: pneumothorax classification using SIIM-ACR masks for objective validation.
- CT: intracranial hemorrhage classification using RSNA IHD or local head CT, with student-created masks on a small test subset.

Important scope guard:
- Avoid making both X-ray and CT equally large. With 4 weeks, the thesis should have one primary dataset and one secondary pilot.
- If both are included, the research question should be about explainability method validation across medical imaging tasks, not about separately solving two clinical problems.

Possible final title:

Cross-Modality Validation of Explainable AI Methods for Medical Image Classification: Chest X-ray Pneumothorax and Head CT Hemorrhage Case Studies

Possible core contribution:
- Compare several explanation methods on two clinically different imaging tasks.
- Validate heatmaps against lesion masks.
- Add radiologist-centered qualitative scoring.
- Test a simple explanation-consensus or thresholding improvement.

## 2026-05-07 - Agreed Working Plan

Aligned direction:

Cross-modality validation of explainable AI methods for medical image classification, with:
- Primary quantitative benchmark: chest X-ray pneumothorax classification using SIIM-ACR-style masks.
- Secondary CT pilot: head CT intracranial hemorrhage classification/localization using RSNA IHD or local anonymized head CT, with student-created masks on a small subset.

Core thesis idea:
- Train, fine-tune, or reuse classification models.
- Generate explanation maps using several XAI methods.
- Validate explanation maps against lesion masks.
- Add radiologist-centered qualitative assessment.
- Propose a small improvement such as explanation consensus, threshold calibration, or method-selection rules.

Preferred working title:

Cross-Modality Validation of Explainable AI Methods for Medical Image Classification: Chest X-ray Pneumothorax and Head CT Hemorrhage Case Studies

Alternative shorter title:

Validation and Improvement of Explainable AI Methods for Medical Image Classification in Radiology

### Timeline

Absolute finish: 2026-06-21.

Thesis completion sprint:
- 2026-05-07 to 2026-06-04: core experiments, thesis writing, final thesis draft.

Defense reserve:
- 2026-06-05 to 2026-06-21: supervisor corrections, plagiarism check, formatting, presentation, rehearsal, defense.

### Week 1: 2026-05-07 to 2026-05-14

Goal: lock protocol and prove feasibility.

Tasks:
- Confirm topic with supervisor.
- Create thesis skeleton in English.
- Create first weekly report.
- Select datasets:
  - Primary: SIIM-ACR pneumothorax or Kaggle derivative with masks.
  - Secondary: RSNA IHD or local anonymized head CT.
- Build environment and data loaders.
- Run baseline model on a small X-ray sample.
- Generate first Grad-CAM and Integrated Gradients maps.
- Define radiologist review and mask-validation schema.
- Decide whether CT pilot uses public RSNA IHD or local DICOM first.

Deliverables:
- Topic/protocol one-pager.
- Thesis skeleton.
- First experiment notebook/script.
- First heatmap examples.
- Weekly report 1.

### Week 2: 2026-05-15 to 2026-05-21

Goal: primary X-ray benchmark working end to end.

Tasks:
- Train/fine-tune or establish a reproducible pretrained baseline for pneumothorax classification.
- Generate explanations with Grad-CAM, Grad-CAM++, Integrated Gradients, GradientSHAP/SHAP-style attribution, and Occlusion if feasible.
- Convert masks and heatmaps to comparable binary/continuous formats.
- Compute explanation metrics:
  - IoU/Dice after thresholding.
  - Pointing game / hit rate.
  - Precision-at-k or top-percent overlap.
  - Optional deletion/insertion or Captum infidelity/sensitivity.
- Start Chapter 1 and Chapter 2.

Deliverables:
- Baseline classification metrics.
- First explanation-comparison table.
- Literature matrix.
- Draft Chapter 1 and partial Chapter 2.
- Weekly report 2/demo.

### Week 3: 2026-05-22 to 2026-05-28

Goal: CT pilot and improvement component.

Tasks:
- Prepare CT pilot dataset.
- Select small CT subset for manual masks.
- Run CT classifier baseline or slice-level model.
- Generate explanation maps for CT.
- Manually annotate limited CT positive subset.
- Compare explanation localization on CT.
- Implement simple explanation improvement:
  - consensus heatmap,
  - threshold calibration,
  - or method-selection rule based on validation metrics.
- Draft Chapter 3.

Deliverables:
- CT pilot results.
- Manual mask subset.
- Improvement-method results.
- Draft Chapter 3.
- Updated Chapter 2.
- Weekly report 3/demo.

### Week 4: 2026-05-29 to 2026-06-04

Goal: finish thesis-grade results and full draft.

Tasks:
- Finalize all experiments.
- Complete radiologist qualitative scoring.
- Build failure taxonomy:
  - correct localization,
  - partial localization,
  - attention outside lesion,
  - device/crop/text/edge artifacts,
  - confounding by non-pathological high-contrast structures,
  - clinically misleading explanation.
- Finalize figures and tables.
- Write Chapter 4 and Chapter 5.
- Complete abstract, bibliography, appendices.
- Prepare supervisor-ready full draft.

Deliverables:
- Full thesis draft.
- Final result tables and figures.
- Reproducible code/notebooks.
- Weekly report 4.

### Defense Reserve: 2026-06-05 to 2026-06-21

Tasks:
- Address supervisor corrections.
- Format thesis according to template.
- Prepare searchable PDF.
- Prepare plagiarism-check version without appendices if requested.
- Build defense slides, max 20.
- Prepare 7-minute speech.
- Rehearse defense and questions.
- Prepare publication plan after defense.

### Immediate Next Actions

1. Create thesis skeleton and working documents.
2. Create experiment protocol.
3. Create weekly report template.
4. Set up repository folders:
   - `data/README.md` only; no patient data committed.
   - `notebooks/`.
   - `src/`.
   - `reports/weekly/`.
   - `thesis/`.
   - `figures/`.
5. Verify dataset access and environment.
6. Run first X-ray explainability smoke test.

## 2026-05-07 - Repository Structure and Starter Documents Created

Created folders:
- `thesis/`
- `reports/weekly/`
- `notebooks/`
- `src/`
- `figures/`
- `data/`
- `docs/`

Created files:
- `thesis/thesis_skeleton.md` - English thesis skeleton following the required academic structure.
- `docs/experiment_protocol.md` - experiment design, metrics, methods, and minimum success criteria.
- `docs/supervisor_one_pager.md` - concise proposal for supervisor review.
- `reports/weekly/week_1_report.md` - first weekly progress report template.
- `data/README.md` - patient/public data handling rules.
- `.gitignore` - excludes local data, images, medical imaging files, generated office/PDF files, and Python caches.

Next technical step:
- Set up Python environment and run first X-ray explainability smoke test on a tiny sample.

## 2026-05-11 - Executed First Technical Setup

Environment:
- Python 3.10.12.
- PyTorch 2.10.0+cu128 and torchvision 0.25.0+cu128 already installed.
- Installed project dependencies from `requirements.txt` with approved network access:
  - pandas, scikit-learn, matplotlib, pydicom, nibabel, OpenCV, Captum, pytorch-grad-cam, SHAP, Kaggle, Jupyter, tqdm.
- `scripts/check_environment.py` passes for the expected data/XAI stack.
- Kaggle package is installed, but credentials are not present yet at `.kaggle/kaggle.json`.

Created technical files:
- `requirements.txt` - project dependency list.
- `src/explainai_thesis/metrics.py` - IoU, Dice, pointing game, precision-at-fraction.
- `src/explainai_thesis/xai.py` - minimal Grad-CAM, Integrated Gradients, consensus heatmap.
- `src/explainai_thesis/synthetic.py` - synthetic lesion dataset for smoke testing.
- `src/explainai_thesis/models.py` - tiny CNN for smoke testing.
- `src/explainai_thesis/visualization.py` - heatmap/mask overlay export.
- `src/explainai_thesis/manifest.py` - generic PNG image/mask manifest builder.
- `scripts/run_smoke_test.py` - end-to-end synthetic XAI smoke test.
- `scripts/check_environment.py` - package/import verification.
- `scripts/build_manifest.py` - manifest creation for downloaded PNG image/mask datasets.
- `docs/dataset_sources.md` - selected dataset sources and Kaggle commands.
- `.env.example` - local runtime path variables.

Smoke test result:
- Command: `python3 scripts/run_smoke_test.py --device auto`
- Result: completed successfully on CPU.
- Synthetic classification accuracy: 1.000.
- Outputs:
  - `outputs/smoke_test/metrics.csv`
  - `outputs/smoke_test/sample_*_grad_cam.png`
  - `outputs/smoke_test/sample_*_integrated_gradients.png`
  - `outputs/smoke_test/sample_*_consensus.png`

Dataset decision:
- Primary real dataset target: Kaggle `vbookshelf/pneumothorax-chest-xray-images-and-masks`.
- CT pilot target: Kaggle `vbookshelf/computed-tomography-ct-images`, with RSNA IHD or local head CT as secondary alternatives.

Immediate next task:
- Add Kaggle credentials locally as `.kaggle/kaggle.json` or manually place downloaded data into `data_local/`.
- Then run:
  - `PATH="$HOME/.local/bin:$PATH" KAGGLE_CONFIG_DIR="$PWD/.kaggle" kaggle datasets download -d vbookshelf/pneumothorax-chest-xray-images-and-masks -p data_local/cxr_pneumothorax --unzip`
  - `python3 scripts/build_manifest.py data_local/cxr_pneumothorax --output data/cxr_pneumothorax_manifest.csv`

## 2026-05-13 - Supervisor Feedback and Next Planning Iteration

Supervisor discussed possible long-term perspectives:
- Explainable LLM that accounts for the radiological point of view.
- Knowledge distillation: smaller model with equivalent diagnostic quality.

Interpretation:
- These are best treated as future-evolution directions, not Week 1 implementation targets.
- The current thesis can produce artifacts useful for both:
  - structured XAI outputs + radiologist comments can later feed an LLM explanation system;
  - model/explanation metrics can later support distilling a large model into a smaller model while preserving classification and explanation behavior.

Open questions from supervisor:
1. What will we get from XAI models?
2. How will we work with this output?
3. What hypotheses will be prepared before testing?
4. Next iteration should use a TorchXRay model and produce first results on a Kaggle dataset with masks.

Planned answers:

What XAI outputs provide:
- Per-image, per-target-class attribution heatmaps.
- Continuous importance values normalized to `[0, 1]`.
- Optional binary explanation masks after thresholding or top-k selection.
- Visual overlays for radiologist review.
- Quantitative localization metrics against lesion masks.
- Failure patterns, such as attention outside the lesion, device/crop artifacts, diffuse heatmaps, or clinically misleading explanations.

How outputs will be used:
- Normalize heatmaps.
- Overlay heatmaps on radiographs.
- Threshold heatmaps into explanation masks.
- Compare explanation masks with ground-truth lesion masks using IoU, Dice, pointing game/hit rate, and precision-at-k.
- Score clinically as a radiologist: correct, partially correct, misleading, or not useful.
- Build an error taxonomy.
- Test whether a consensus/threshold-calibrated explanation improves localization over individual methods.

Pre-test hypotheses:
- H1: Different XAI methods will produce significantly different localization quality on the same model and dataset.
- H2: High classification quality will not necessarily imply high explanation-localization quality.
- H3: Mask-calibrated thresholding or consensus heatmaps will improve explanation localization compared with at least one individual baseline method.
- H4: Quantitative localization metrics will correlate with radiologist usefulness scores, but not perfectly.
- H5: Explanation methods that perform well on X-ray pneumothorax may not rank identically on CT hemorrhage, because modality and pathology morphology differ.

Immediate technical iteration:
- Install/use TorchXRayVision if feasible.
- Run TorchXRayVision DenseNet-style pretrained model on the pneumothorax Kaggle dataset or use it as a feature/model baseline.
- Generate first real-data Grad-CAM or Captum attribution maps.
- Validate against Kaggle PNG masks.
- Produce the first table:
  - model prediction,
  - mask availability,
  - XAI method,
  - IoU,
  - Dice,
  - pointing hit,
  - precision-at-k.

Week 1 report requirements received:
- completed tasks, achievements, and project changes;
- identified risks/challenges, including what the problem is, why it matters, and mitigation;
- next-week plan;
- links/references to artifacts;
- expected result: preliminary problem analysis, defined project goals, initial MVP or research methodology, first progress report.

## 2026-05-13 - Week 1 Report MVP/Smoke-Test Section Added

Updated:
- `reports/weekly/week_1_report.md`
- `reports/weekly/week_1_report_final.md`

Added:
- concise initial MVP flow description;
- smoke-test command;
- synthetic smoke-test result summary;
- aggregate localization metrics for Grad-CAM, Integrated Gradients, and consensus heatmap;
- embedded sample overlay images from `outputs/smoke_test/`.

## 2026-05-13 - Restored Real-Data TorchXRayVision Iteration

Repository/runtime status:
- Primary Kaggle pneumothorax dataset is present locally under `data_local/cxr_pneumothorax/siim-acr-pneumothorax`.
- Dataset layout:
  - `png_images/`: 12,047 files.
  - `png_masks/`: 12,047 files.
  - `stage_1_train_images.csv` and `stage_1_test_images.csv`.
- The active Python runtime is WSL Ubuntu, not native Windows Python:
  - command pattern: `wsl.exe --cd /mnt/c/Users/Dmytro.Valantsevych/Downloads/master_thesis_draft_explainAI python3 ...`
  - WSL Python: 3.10.12.
  - PyTorch: 2.10.0+cu128.
  - torchvision: 0.25.0+cu128.
- Native Windows `python`/`python3` points to Microsoft Store aliases; native `py -3.10` works but does not have the ML stack installed.
- Installed `torchxrayvision==1.4.0` into the WSL user environment.

Code updates:
- `src/explainai_thesis/manifest.py` now detects the SIIM/Kaggle pneumothorax layout and writes split, image ID, and filename fields.
- `scripts/build_manifest.py` now calls the layout-aware `build_manifest`.
- Added `scripts/run_cxr_torchxray_smoke.py` for a small real-data TorchXRayVision explainability pass.

Generated artifact:
- Command:

```bash
wsl.exe --cd /mnt/c/Users/Dmytro.Valantsevych/Downloads/master_thesis_draft_explainAI python3 scripts/run_cxr_torchxray_smoke.py --device auto --max-positive 6 --ig-steps 16
```

- Manifest:
  - `data/cxr_pneumothorax_manifest.csv`
  - 12,047 rows.
  - 12,047 rows with masks.
  - Labels: 9,378 negative and 2,669 positive.
- Real-data smoke-test outputs:
  - `outputs/cxr_torchxray_smoke/metrics.csv`
  - `outputs/cxr_torchxray_smoke/sample_*_grad_cam.png`
  - `outputs/cxr_torchxray_smoke/sample_*_integrated_gradients.png`
  - `outputs/cxr_torchxray_smoke/sample_*_consensus.png`

Initial real-data observation:
- TorchXRayVision DenseNet (`densenet121-res224-all`) runs on CUDA and produces pneumothorax scores for positive SIIM-style samples.
- On the first 6 positive test cases, uncalibrated Grad-CAM / Integrated Gradients / consensus localization against masks is low, with pointing-game hits equal to 0 in this tiny sample.
- This supports the thesis motivation: plausible classifier outputs do not automatically imply clinically/localization-valid explanations.

Immediate next technical steps:
1. Run a larger evaluation subset and aggregate metrics by method.
2. Add a train/fine-tuned pneumothorax-specific baseline or select a more task-aligned TorchXRayVision weight configuration.
3. Add Grad-CAM++ / Score-CAM or Captum GradientSHAP as the next explanation method.
4. Calibrate heatmap thresholds on a validation split, then compare against held-out test masks.
