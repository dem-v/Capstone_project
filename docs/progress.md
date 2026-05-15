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

Reporting note:
- Week 1 reports are already submitted and should be treated as frozen.
- Do not update `reports/weekly/week_1_report.md` or `reports/weekly/week_1_report_final.md` for new work.
- New progress-report updates should go into Week 2 report files.

## 2026-05-13 - Week 2 Larger Real-Data XAI Evaluation

Executed the next technical step on the real pneumothorax dataset:

```bash
wsl.exe --cd /mnt/c/Users/Dmytro.Valantsevych/Downloads/master_thesis_draft_explainAI python3 scripts/run_cxr_torchxray_smoke.py --device auto --max-positive 50 --ig-steps 16 --max-overlays 12 --output-dir outputs/cxr_torchxray_week2_50
```

Code update:
- `scripts/run_cxr_torchxray_smoke.py` now writes both per-case metrics and `metrics_summary.csv`.
- The script now limits overlay export with `--max-overlays` so larger metric runs do not create too many images.
- Fixed local import path ordering so `src/` is inserted before importing `explainai_thesis`.

Output artifacts:
- `outputs/cxr_torchxray_week2_50/metrics.csv`
- `outputs/cxr_torchxray_week2_50/metrics_summary.csv`
- overlays for the first 12 positive cases.

Aggregate result on 50 positive test cases:
- Grad-CAM: mean IoU 0.0213, mean Dice 0.0400, pointing hit rate 0.0000, mean precision-at-15% 0.0234.
- Integrated Gradients: mean IoU 0.0147, mean Dice 0.0282, pointing hit rate 0.0200, mean precision-at-15% 0.0168.
- Consensus: mean IoU 0.0213, mean Dice 0.0400, pointing hit rate 0.0000, mean precision-at-15% 0.0234.

Interpretation:
- The pretrained TorchXRayVision model runs end to end, but uncalibrated heatmap localization is weak on the first 50 positive cases.
- This supports the thesis premise that XAI outputs require quantitative and clinical validation.
- The next best technical step is to add a stronger third XAI method and threshold calibration before deciding whether model fine-tuning is needed.

## 2026-05-13 - TorchXRayVision Classification Baseline Evaluation

Question tested:
- Are weak XAI localization results caused only by untuned XAI, or is the pretrained TorchXRayVision model itself weak on this Kaggle/SIIM-style pneumothorax dataset?

Added:
- `scripts/evaluate_cxr_torchxray_model.py`

Commands:

```bash
wsl.exe --cd /mnt/c/Users/Dmytro.Valantsevych/Downloads/master_thesis_draft_explainAI python3 scripts/evaluate_cxr_torchxray_model.py --device auto --split test --batch-size 64 --output-dir outputs/cxr_torchxray_model_eval_test
wsl.exe --cd /mnt/c/Users/Dmytro.Valantsevych/Downloads/master_thesis_draft_explainAI python3 scripts/evaluate_cxr_torchxray_model.py --device auto --split train --batch-size 64 --output-dir outputs/cxr_torchxray_model_eval_train
```

Test split classification result:
- N = 1,372; positives = 290; negatives = 1,082.
- ROC AUC = 0.7711.
- Average precision = 0.4120.
- Mean sigmoid score: positives 0.6191, negatives 0.5818.
- Default threshold 0.5:
  - accuracy 0.2114;
  - sensitivity 1.0000;
  - specificity 0.0000;
  - F1 0.3490.
- Best F1 threshold in sweep: 0.61.
  - accuracy 0.5940;
  - sensitivity 0.8931;
  - specificity 0.5139;
  - F1 0.4819.

Train split classification result:
- N = 10,675; positives = 2,379; negatives = 8,296.
- ROC AUC = 0.7720.
- Average precision = 0.4280.
- Default threshold 0.5:
  - accuracy 0.2229;
  - sensitivity 1.0000;
  - specificity 0.0000;
  - F1 0.3645.
- Best F1 threshold in sweep: 0.62.
  - accuracy 0.6145;
  - sensitivity 0.8932;
  - specificity 0.5346;
  - F1 0.5081.

Interpretation:
- TorchXRayVision is not random on this dataset: AUC around 0.77 means it has moderate ranking signal.
- However, the default sigmoid threshold is unusable here because it predicts every case as positive.
- The model is also not a strong pneumothorax classifier for this dataset without calibration/fine-tuning.
- Therefore the weak XAI localization is likely a combined issue:
  - the classifier is only moderately matched to the dataset and poorly calibrated;
  - the explanation maps are also uncalibrated and not optimized for lesion-mask localization.
- Next decision: either fine-tune/calibrate a pneumothorax-specific classifier before final XAI comparison, or clearly frame TorchXRayVision as a pretrained external baseline whose limitations are part of the evaluation.

Important XAI interpretation note:
- Grad-CAM, Integrated Gradients, SHAP-style methods, Occlusion, and similar XAI methods are normally not trained as separate diagnostic models.
- They are post-hoc explanation methods applied to an already trained classifier.
- They can have configuration choices, such as target layer, target class, baseline image/background distribution, heatmap normalization, top-k fraction, threshold, patch size, or number of integration steps.
- Poor explanation localization can mean:
  - the classifier did not learn clinically relevant features for this dataset;
  - the classifier is poorly calibrated or domain-mismatched;
  - the explanation method configuration is not suitable;
  - or the method faithfully reveals that the model is relying on non-lesion features.
- Therefore, classifier performance and explanation localization must be evaluated separately before drawing conclusions about XAI method quality.

## 2026-05-13 - XAI Heatmap Threshold Calibration Workflow Added

Goal:
- Add validation-derived heatmap threshold calibration so explanation masks are not evaluated only with a fixed arbitrary top-fraction.
- Keep TorchXRayVision as an unchanged pretrained external baseline; calibration affects only XAI mask binarization.

Code updates:
- Added `scripts/calibrate_cxr_xai_thresholds.py`.
- Updated `scripts/run_cxr_torchxray_smoke.py` to accept `--calibrated-fractions` and write `top_fraction` in `metrics.csv`.

Calibration workflow:
- Use positive masked calibration cases, by default from the train split.
- Generate Grad-CAM, Integrated Gradients, and consensus heatmaps.
- Sweep top-fractions, defaulting to `0.05,0.10,0.15,0.20,0.25,0.30`.
- Select the best fraction per method by a validation metric, defaulting to mean Dice.
- Write:
  - `calibration_metrics.csv` for per-case/per-fraction metrics;
  - `calibration_summary.csv` for aggregate metrics by method and fraction;
  - `selected_fractions.csv` for frozen method-specific fractions.

Held-out evaluation workflow:
- Pass the frozen calibration file to `scripts/run_cxr_torchxray_smoke.py` with `--calibrated-fractions`.
- The evaluation script then applies each method's selected top-fraction instead of the default `--top-fraction`.
- Final test metrics should be reported only after thresholds are selected on calibration data.

Verification completed:
- WSL syntax check passed:

```bash
wsl.exe python3 -m py_compile scripts/calibrate_cxr_xai_thresholds.py scripts/run_cxr_torchxray_smoke.py
```

- CUDA smoke calibration completed on 1 train positive case:

```bash
wsl.exe python3 scripts/calibrate_cxr_xai_thresholds.py --device auto --split train --max-positive 1 --ig-steps 2 --fractions 0.10,0.20 --output-dir outputs/cxr_xai_threshold_calibration_smoke
```

- CUDA frozen-fraction evaluation completed on 1 test positive case:

```bash
wsl.exe python3 scripts/run_cxr_torchxray_smoke.py --device auto --split test --max-positive 1 --ig-steps 2 --max-overlays 0 --calibrated-fractions outputs/cxr_xai_threshold_calibration_smoke/selected_fractions.csv --output-dir outputs/cxr_xai_calibrated_eval_smoke
```

Recommended next technical run:
- Run calibration on a larger train subset, for example 100-200 positive cases.
- Then run held-out calibrated XAI evaluation on a separate test subset, for example 100-200 positive cases.
- Compare calibrated results against the previous fixed 15% top-fraction baseline.

## 2026-05-13 - First Larger Calibrated XAI Outputs Generated

Calibration run:

```bash
wsl.exe --cd /mnt/c/Users/Dmytro.Valantsevych/Downloads/master_thesis_draft_explainAI python3 scripts/calibrate_cxr_xai_thresholds.py --device auto --split train --max-positive 100 --ig-steps 16 --fractions 0.05,0.10,0.15,0.20,0.25,0.30 --selection-metric dice --output-dir outputs/cxr_xai_threshold_calibration_train100_dice
```

Selected train-derived Dice-optimal top-fractions:
- Grad-CAM: `0.10` with validation mean Dice `0.048577`.
- Integrated Gradients: `0.30` with validation mean Dice `0.024299`.
- Consensus: `0.10` with validation mean Dice `0.048237`.

Generated calibration artifacts:
- `outputs/cxr_xai_threshold_calibration_train100_dice/calibration_metrics.csv`
- `outputs/cxr_xai_threshold_calibration_train100_dice/calibration_summary.csv`
- `outputs/cxr_xai_threshold_calibration_train100_dice/selected_fractions.csv`

Held-out calibrated evaluation run:

```bash
wsl.exe --cd /mnt/c/Users/Dmytro.Valantsevych/Downloads/master_thesis_draft_explainAI python3 scripts/run_cxr_torchxray_smoke.py --device auto --split test --max-positive 100 --ig-steps 16 --max-overlays 20 --calibrated-fractions outputs/cxr_xai_threshold_calibration_train100_dice/selected_fractions.csv --output-dir outputs/cxr_xai_calibrated_eval_test100_dice
```

Generated held-out evaluation artifacts:
- `outputs/cxr_xai_calibrated_eval_test100_dice/metrics.csv`
- `outputs/cxr_xai_calibrated_eval_test100_dice/metrics_summary.csv`
- overlay PNGs for the first 20 positive test cases.

Held-out calibrated evaluation results on 100 positive test cases:
- Grad-CAM: mean IoU `0.024297`, mean Dice `0.045134`, pointing hit rate `0.000000`, mean precision-at-fraction `0.027763`.
- Integrated Gradients: mean IoU `0.013209`, mean Dice `0.025625`, pointing hit rate `0.010000`, mean precision-at-fraction `0.013960`.
- Consensus: mean IoU `0.024163`, mean Dice `0.044858`, pointing hit rate `0.010000`, mean precision-at-fraction `0.027630`.

Interpretation:
- The calibrated top-fractions slightly improve/standardize the evaluation protocol but do not make localization strong.
- Grad-CAM remains the best of the current methods by held-out mean Dice and IoU.
- Consensus remains very close to Grad-CAM, so the current consensus strategy is still not a meaningful improvement over Grad-CAM alone.
- Integrated Gradients selected a much broader top-fraction (`0.30`), but held-out overlap remains weaker than Grad-CAM.

## 2026-05-13 - Overlay Visualization Clarified

Code update:
- `src/explainai_thesis/visualization.py` now draws the ground-truth lesion mask as a green contour instead of a filled green mask.
- The contour is alpha-blended over the existing red heatmap overlay so red attribution remains visible where it overlaps the contour.

Interpretation note:
- This changes only exported PNG visualization, not localization metrics.
- The red overlay represents method-specific positive attribution for the selected pneumothorax class in the current implementation, not a generic eye-tracking/attention signal.

## 2026-05-13 - Negative Signed Grad-CAM Added

Code update:
- `src/explainai_thesis/xai.py` now supports Grad-CAM polarity: positive Grad-CAM keeps the usual ReLU class-supporting map, while negative Grad-CAM applies ReLU to the negated raw CAM to show regions that suppress the selected pneumothorax score.
- `scripts/run_cxr_torchxray_smoke.py` and `scripts/calibrate_cxr_xai_thresholds.py` now include `grad_cam_negative` as an additional method.
- `src/explainai_thesis/visualization.py` can render negative Grad-CAM as a blue heatmap; the green ground-truth contour is still alpha-blended on top so overlap remains visible.

Interpretation note:
- Red Grad-CAM highlights image regions that increase/support the target pneumothorax output.
- Blue negative Grad-CAM highlights image regions whose signed Grad-CAM contribution goes in the opposite direction, i.e. regions that move evidence away from the target output under this Grad-CAM approximation.

## 2026-05-13 - Future Output Folder Naming Rule

Future note:
- Every new experiment output folder should include an ordinal iteration number in its name, so results are easier to navigate chronologically.
- Recommended pattern: `outputs/iter_XX_<short_experiment_name>`, for example `outputs/iter_01_cxr_xai_calibration_train100_dice` and `outputs/iter_02_cxr_xai_eval_test100_dice`.
- Keep the ordinal number stable once results are generated; do not renumber old folders after reports or progress notes reference them.

## 2026-05-13 - Consensus Overlay Now Includes Blue Negative Evidence

Code update:
- `src/explainai_thesis/visualization.py` now accepts an optional `negative_heatmap` and blends it as a blue channel before drawing the positive heatmap and green mask contour.
- `scripts/run_cxr_torchxray_smoke.py` now passes `grad_cam_negative` as the blue channel for `consensus` overlay PNGs.

Interpretation note:
- Consensus metrics remain based on the positive consensus heatmap, so existing quantitative comparisons are unchanged.
- Consensus PNGs are now qualitative signed overlays: red shows the positive consensus map, blue shows negative signed Grad-CAM evidence against the pneumothorax target, and overlap can appear purple/mixed before the green ground-truth contour is blended on top.

Verification:
- WSL syntax check passed for `src/explainai_thesis/visualization.py` and `scripts/run_cxr_torchxray_smoke.py`.
- One-case CUDA smoke output generated at `outputs/iter_01_consensus_signed_overlay_smoke`.

## 2026-05-13 - Single-Image Threshold Selection Visualization Added

Code update:
- Added `scripts/visualize_cxr_threshold_selection.py` for inspecting what the top-fraction thresholding step does on one positive masked CXR case.
- The script generates the same backend heatmaps as the calibration/evaluation workflow: `grad_cam`, `grad_cam_negative`, `integrated_gradients`, and `consensus`.
- For each method and requested top-fraction, it saves a binary threshold-selection PNG where red means selected outside the true mask, yellow means selected inside the true mask, and green means missed true-mask pixels.
- It also saves each method's continuous overlay, a per-method `threshold_sweep_panel.png`, `threshold_metrics.csv`, and `case_metadata.csv`.

Generated iteration output:

```bash
wsl.exe python3 scripts/visualize_cxr_threshold_selection.py --device auto --split train --case-index 0 --ig-steps 16 --fractions 0.05,0.10,0.15,0.20,0.25,0.30 --output-dir outputs/iter_02_threshold_selection_single_image
```

Verification:
- WSL syntax check passed for `scripts/visualize_cxr_threshold_selection.py`.
- CUDA smoke visualization completed at `outputs/iter_02_threshold_selection_single_image_smoke`.
- Full single-image visualization completed at `outputs/iter_02_threshold_selection_single_image` for case `2_train_1_.png`.

## 2026-05-13 - Selected Threshold Images Added to Main XAI Output

Code update:
- `scripts/run_cxr_torchxray_smoke.py` now writes selected-threshold PNGs for every method/sample that receives a standard overlay export.
- The selected images use the same style as `scripts/visualize_cxr_threshold_selection.py`: red means selected outside the true mask, yellow means selected inside the true mask, and green means missed true-mask pixels.
- File naming pattern: `sample_XX_<method>_selected.png`, placed next to the existing continuous overlay `sample_XX_<method>.png`.
- The selected image uses the same `top_fraction` used for metrics, including per-method calibrated fractions when `--calibrated-fractions` is supplied.

Verification:
- WSL syntax check passed for `scripts/run_cxr_torchxray_smoke.py`.
- CUDA smoke evaluation completed at `outputs/iter_03_main_selected_images_smoke`.
- Confirmed selected images exist for `grad_cam`, `grad_cam_negative`, `integrated_gradients`, and `consensus`.

## 2026-05-13 - Negative Selected Images and Consensus Diagnostics Clarified

Code update:
- `grad_cam_negative` selected-threshold PNGs now use blue for selected negative evidence outside the true mask, and cyan/green+blue for selected negative evidence that intersects the true mask.
- Positive methods keep the previous selected-image colors: red for selected outside the mask, yellow for selected inside the mask, and green for missed mask pixels.
- `scripts/run_cxr_torchxray_smoke.py` now adds negative evidence diagnostics to `metrics.csv` and `metrics_summary.csv`: `negative_mask_overlap_fraction` and `negative_mask_avoidance_fraction`.

Interpretation note:
- Standard Dice/IoU for `consensus` still measures only positive consensus localization against the lesion mask.
- The blue negative evidence shown on consensus overlays is not folded into Dice/IoU as a reward; instead, the new negative avoidance columns separately report whether the negative Grad-CAM selected area avoids the lesion mask, where higher `negative_mask_avoidance_fraction` is better.

Verification:
- WSL syntax check passed for `scripts/run_cxr_torchxray_smoke.py` and `scripts/visualize_cxr_threshold_selection.py`.
- CUDA smoke evaluation completed at `outputs/iter_04_negative_selected_consensus_metrics_smoke`.

## 2026-05-13 - Classifier Outcome Threshold Sweep Visualization for 100 Test CXRs

Code update:
- Added `scripts/visualize_cxr_classifier_outcome_thresholds.py` to visualize XAI threshold selection across mixed classifier outcomes (`tp`, `fp`, `tn`, `fn`).
- The script samples mixed-label manifest rows, applies a classifier probability threshold, groups each case by classifier outcome, and generates threshold-selection images for `grad_cam`, `grad_cam_negative`, `integrated_gradients`, and `consensus`.
- Default threshold fractions are `0.05` through `0.50` in `0.05` steps.
- Positive selected images keep red/yellow/green semantics; `grad_cam_negative` selected images use blue/cyan/green semantics.

Generated iteration output:

```bash
wsl.exe --cd /mnt/c/Users/Dmytro.Valantsevych/Downloads/master_thesis_draft_explainAI python3 scripts/visualize_cxr_classifier_outcome_thresholds.py --device auto --split test --max-cases 100 --threshold 0.61 --ig-steps 16 --fractions 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50 --output-dir outputs/iter_05_classifier_outcome_thresholds_test100
```

Generated artifacts:
- `outputs/iter_05_classifier_outcome_thresholds_test100/cases.csv`
- `outputs/iter_05_classifier_outcome_thresholds_test100/threshold_metrics.csv`
- `outputs/iter_05_classifier_outcome_thresholds_test100/outcome_summary.csv`
- Outcome-grouped image folders under `tp`, `fp`, and `tn`; no `fn` folder was generated because there were no false negatives in this first 100-case subset at threshold `0.61`.

Classifier outcome counts at threshold `0.61` on the first 100 test rows:
- TP: `20`
- FP: `35`
- TN: `45`
- FN: `0`

Verification:
- WSL syntax check passed for `scripts/visualize_cxr_classifier_outcome_thresholds.py`.
- CUDA smoke run completed at `outputs/iter_05_classifier_outcome_thresholds_smoke`.
- Full CUDA run completed at `outputs/iter_05_classifier_outcome_thresholds_test100` with `4000` threshold metric rows (`100` cases x `4` methods x `10` fractions).

## 2026-05-13 - Consensus Stepped Selection Images Include Blue Negative Evidence

Code update:
- `scripts/visualize_cxr_classifier_outcome_thresholds.py` now renders thresholded `grad_cam_negative` evidence inside `consensus` selected-step images, not only in the continuous consensus overlay.
- `scripts/visualize_cxr_threshold_selection.py` received the same behavior for the single-image threshold sweep.
- Consensus selected images now combine blue/cyan thresholded negative evidence with red/yellow positive consensus selection, while green still indicates true-mask pixels missed by the positive consensus selection.

Verification:
- WSL syntax check passed for both threshold visualization scripts.
- CUDA smoke run completed at `outputs/iter_06_consensus_selected_negative_smoke`; consensus `selected_top_05.png` files were generated under `tp` and `tn` case folders.

## 2026-05-13 - Thesis Interpretation Note for Red and Blue Attribution Maps

Future thesis wording:
- In attribution overlays, red regions indicate image areas that contribute positively to the model's pneumothorax output, while blue regions indicate areas that contribute negatively to the same output.
- Red should be described as evidence supporting the pneumothorax prediction; blue should be described as evidence suppressing or arguing against the pneumothorax prediction.
- These maps should not be described as direct pneumothorax segmentations or as generic model attention maps. They are class-specific attribution maps with respect to the selected pneumothorax target score.

Method-specific caveat:
- For signed `Grad-CAM`, the red/blue interpretation is relatively clean because positive and negative components are separated from the signed class activation map: red supports the target score, blue suppresses it.
- For the current `Integrated Gradients` visualization, interpretation is less strictly signed because it is used as a target-class attribution magnitude-style heatmap unless signed positive/negative IG is implemented later.
- For `consensus`, red represents positive consensus evidence, while blue represents the separate negative signed `Grad-CAM` evidence included for qualitative inspection.

Metric interpretation:
- For positive red evidence, overlap with the green ground-truth pneumothorax contour is desirable and can be evaluated using positive localization metrics such as Dice, IoU, and pointing hit.
- For blue negative evidence, the interpretation is reversed: blue outside the lesion is acceptable or expected, while blue inside the lesion may indicate that the model treats a true lesion region as evidence against pneumothorax.
- Therefore, blue evidence should be treated as a diagnostic complement, not as another lesion-localization method to maximize with Dice/IoU; use negative avoidance/overlap diagnostics separately.

Suggested figure-caption wording:
- Red overlay denotes positive attribution for the pneumothorax target class, indicating regions that increase the model's pneumothorax score. Blue overlay denotes negative signed attribution, indicating regions that decrease or suppress the pneumothorax score. The green contour represents the ground-truth pneumothorax mask. Overlap between red attribution and the green contour suggests localization of positive model evidence within the annotated lesion, whereas blue attribution inside the contour may indicate that the model treats part of the annotated lesion as evidence against the target class.

## 2026-05-13 - Signed Integrated Gradients Expansion

Code update:
- `src/explainai_thesis/xai.py` now supports `integrated_gradients(..., polarity="magnitude" | "positive" | "negative")`.
- The existing `integrated_gradients` default remains a magnitude/impact map, meaning it highlights the strongest input-attribution portions affecting the target output regardless of sign.
- New method outputs are available across the main evaluation, calibration, and threshold visualization scripts: `integrated_gradients_positive` and `integrated_gradients_negative`.
- Negative-signed IG selected-threshold images use the same negative evidence style as negative Grad-CAM: blue for selected negative evidence outside the true mask and cyan where selected negative evidence intersects the true mask.

Interpretation note:
- `integrated_gradients` should be described as strongest-impact target attribution.
- `integrated_gradients_positive` can be described as input-level evidence increasing/supporting the pneumothorax target score relative to the baseline.
- `integrated_gradients_negative` can be described as input-level evidence decreasing/suppressing the pneumothorax target score relative to the baseline.
- IG sign remains baseline-dependent, so the thesis should still mention the baseline choice when interpreting signed IG maps.

Verification:
- WSL syntax check passed for `src/explainai_thesis/xai.py`, `scripts/run_cxr_torchxray_smoke.py`, `scripts/calibrate_cxr_xai_thresholds.py`, `scripts/visualize_cxr_threshold_selection.py`, and `scripts/visualize_cxr_classifier_outcome_thresholds.py`.
- CUDA smoke main evaluation completed at `outputs/iter_07_signed_ig_smoke` and produced `integrated_gradients_positive` and `integrated_gradients_negative` rows/images.
- CUDA smoke single-image threshold visualization completed at `outputs/iter_07_signed_ig_threshold_smoke` and produced signed IG threshold sweep panels.
- CUDA smoke classifier-outcome threshold visualization completed at `outputs/iter_07_signed_ig_classifier_threshold_smoke`.

## 2026-05-13 - Integrated Gradients Visual Semantics Clarified

Code update:
- The default `integrated_gradients` output is now rendered as a neutral impact map, not as red/blue/green evidence. Brighter pixels mean larger absolute IG attribution magnitude regardless of sign.
- Signed IG is now represented by three visual structures: `integrated_gradients_positive` in red, `integrated_gradients_negative` in blue, and `integrated_gradients_signed` as a combined red/blue overlay on the same X-ray.
- Thresholded selected images follow the same semantics: magnitude IG uses a neutral violet selection color, positive IG uses red/yellow, negative IG uses blue/cyan, and combined signed IG overlays thresholded positive and negative selections together.
- Calibration and threshold-sweep scripts now include `integrated_gradients_signed`, so multiple threshold sets can be generated for the combined signed IG view as well as the separate positive/negative maps.

Follow-up correction:
- Neutral magnitude IG must not be black, white, gray, red, green, blue, yellow, or cyan. Continuous and selected-threshold magnitude IG images now use violet as the distinct neutral impact color.

Verification:
- WSL syntax check passed for `src/explainai_thesis/visualization.py`, `scripts/run_cxr_torchxray_smoke.py`, `scripts/calibrate_cxr_xai_thresholds.py`, `scripts/visualize_cxr_threshold_selection.py`, and `scripts/visualize_cxr_classifier_outcome_thresholds.py`.
- CUDA smoke main evaluation completed at `outputs/iter_08_ig_visual_semantics_smoke` and produced neutral magnitude, positive, negative, and combined signed IG images.
- CUDA smoke single-image threshold visualization completed at `outputs/iter_08_ig_visual_semantics_threshold_smoke`.
- Follow-up CUDA smoke evaluation completed at `outputs/iter_09_ig_neutral_violet_smoke` and produced violet neutral magnitude IG continuous and selected-threshold images.

## 2026-05-13 - Main Evaluation Output Grouped by Source X-ray

Code update:
- `scripts/run_cxr_torchxray_smoke.py` now writes per-case visual artifacts into one folder per source X-ray instead of placing all `sample_*` PNGs flat in the output root.
- Root-level files remain reserved for run-wide CSV artifacts such as `metrics.csv` and `metrics_summary.csv`.
- Each exported case folder uses the pattern `case_XXX_<source_xray_stem>` and contains same-level method files such as `grad_cam.png`, `grad_cam_selected.png`, `integrated_gradients.png`, `integrated_gradients_signed_selected.png`, `consensus.png`, and `consensus_selected.png`.

Reason:
- This makes the main smoke/evaluation output easier to browse and more consistent with the source-Xray-oriented threshold visualization outputs.

Verification:
- WSL syntax check passed for `scripts/run_cxr_torchxray_smoke.py`.
- CUDA smoke evaluation completed at `outputs/iter_10_grouped_main_output_smoke`.
- Confirmed the smoke output root contains `metrics.csv` and `metrics_summary.csv`, while the PNG artifacts are grouped under `case_000_0_test_1` with all method images as sibling files.

## 2026-05-13 - Consensus Visualization Includes IG Magnitude Layer

Code update:
- General `consensus` visualization now explicitly includes the neutral violet `integrated_gradients` magnitude map as a separate visual layer, instead of rendering the combined consensus heatmap only as red positive evidence.
- Continuous consensus overlays now combine red positive consensus evidence, violet IG magnitude impact, blue negative Grad-CAM evidence, and the green ground-truth contour.
- Consensus selected-threshold images now also include thresholded violet IG magnitude selections, while preserving the existing red/yellow positive selection and blue/cyan negative evidence semantics.
- The same consensus visual semantics were applied to the main grouped evaluation output and both threshold visualization scripts.

Interpretation note:
- Consensus Dice/IoU still use the combined positive consensus heatmap for the reported localization score.
- The violet IG magnitude layer is a qualitative impact layer showing where IG contributes strongly regardless of sign; it should not be described as positive pneumothorax evidence.

Verification:
- WSL syntax check passed for `src/explainai_thesis/visualization.py`, `scripts/run_cxr_torchxray_smoke.py`, `scripts/visualize_cxr_threshold_selection.py`, and `scripts/visualize_cxr_classifier_outcome_thresholds.py`.
- CUDA smoke evaluation completed at `outputs/iter_11_consensus_ig_magnitude_smoke`.
- Confirmed `consensus.png` and `consensus_selected.png` were generated under the grouped case folder.

## 2026-05-13 - Future Thesis References for Tools, Data, and Models

Future thesis/reference note:
- The final thesis methodology/acknowledgements/appendix should explicitly reference the software and AI-assisted development tools used during the work, including current `GPT-5.5`, `Codex`, `VS Code`, `PyCharm`, `Junie`, `Claude Sonnet 4.6`, and `Claude Opus 4.7 1M context`, with final version/access details checked near submission time.
- The data section should cite the Kaggle/SIIM-ACR pneumothorax chest X-ray dataset and include the exact dataset details used in this project, including local source path, manifest/split construction, image and mask counts, positive/negative counts, preprocessing steps, and any sampled calibration/evaluation subsets.
- The model/methods section should list all models and explainability methods actually used in the research, including the unchanged pretrained `TorchXRayVision` `DenseNet` baseline with `densenet121-res224-all` weights, classifier thresholding details, `Grad-CAM`, signed negative `Grad-CAM`, `Integrated Gradients` magnitude, signed positive/negative/combined `Integrated Gradients`, consensus overlays, and calibrated heatmap top-fraction thresholding.
- Before final writing, verify exact tool names, versions, dates, package versions, dataset citation text, and model weight identifiers from the environment and generated artifacts instead of relying only on memory.

## 2026-05-13 - Full Future Work Explanation: XAI Methods and Faithfulness Metrics

This section preserves the full explanation of optional faithfulness metrics and candidate XAI methods because it will be useful later for thesis writing, future-work discussion, method selection, and figure interpretation.

### Optional Faithfulness Metrics

These are not new explanation methods. They are evaluation tests that ask whether an attribution map is actually faithful to the model's behavior.

#### Deletion / Insertion Curve

- Purpose: tests whether the highlighted pixels are truly important for the model output.
- `Deletion`: progressively remove or blur the most-attributed pixels first.
  - If the explanation is good, the pneumothorax score should drop quickly.
- `Insertion`: start from a blank/blurred image and progressively add the most-attributed pixels back.
  - If the explanation is good, the pneumothorax score should rise quickly.
- What it tells us:
  - Whether the heatmap identifies pixels that actually affect the model score.
- Thesis value:
  - Very useful because it evaluates attribution against the model itself, not only against the segmentation mask.

Example interpretation:

```text
A method may overlap poorly with the ground-truth mask but still strongly affect the model score. That means it is faithful to the model, but the model may be relying on non-lesion evidence.
```

#### Captum Infidelity

- Purpose: measures whether the attribution map correctly predicts the model's output change when the input is perturbed.
- Idea:
  - perturb the image;
  - observe how much the model output changes;
  - compare that real output change to what the attribution map predicted.
- Lower infidelity is better.
- What it tells us:
  - Whether attribution values are numerically consistent with model behavior.
- Thesis value:
  - Good for saying whether an explanation is mathematically faithful, even if it is not clinically localized.

#### Captum Sensitivity

- Purpose: measures explanation stability under small input perturbations.
- Idea:
  - slightly perturb the image;
  - recompute or compare attribution behavior;
  - check how much the explanation changes.
- Lower sensitivity generally means more stable explanations.
- What it tells us:
  - Whether the explanation is robust or fragile.
- Thesis value:
  - Important for medical imaging because a clinically useful explanation should not change drastically from tiny image noise.

### Important Distinction

There are two evaluation families here:

| Evaluation Type | Question Asked |
|---|---|
| Mask localization metrics | Does the explanation overlap the annotated pneumothorax mask? |
| Faithfulness metrics | Does the explanation reflect what the model actually uses? |

This distinction is very important for the thesis.

A method can be:

- clinically well-localized but not very faithful;
- faithful to the model but clinically wrong;
- stable but not localized;
- localized but unstable.

That is exactly why comparing several metrics is stronger than relying only on `Dice` or visual overlays.

### What Each XAI Method Brings

#### Grad-CAM

- Already in the workflow.
- Uses gradients of the target class with respect to late convolutional feature maps.
- Produces coarse class-discriminative regions.
- Strength:
  - easy to interpret visually;
  - good for saying which broad region supports the class output;
  - signed version lets us separate red positive evidence and blue negative evidence.
- Weakness:
  - low spatial resolution;
  - may miss small pneumothorax regions;
  - depends on the chosen convolutional layer.

Thesis role:

```text
Baseline class-discriminative localization method.
```

#### Grad-CAM++

- Extension of Grad-CAM.
- Uses a more refined weighting of gradients, designed to better handle multiple object regions or small discriminative regions.
- Potential advantage over Grad-CAM:
  - can produce sharper or more localized maps;
  - may better detect small pneumothorax regions.
- Weakness:
  - slightly more complex;
  - still feature-map based and upsampled;
  - not guaranteed to improve medical localization.

What different information it brings:

```text
Whether a more refined CAM weighting improves lesion localization compared with standard Grad-CAM.
```

For this thesis, this is probably the most natural next CAM method to add.

#### Eigen-CAM

- CAM-style method that does not require gradients.
- Uses principal components of feature activations to highlight dominant activation patterns.
- It shows where the model has strong internal feature activity, but it is less class-specific unless adapted carefully.
- Strength:
  - gradient-free;
  - often visually smooth and stable;
  - useful as an activation-structure baseline.
- Weakness:
  - may show what the network activates on, not necessarily what supports pneumothorax specifically;
  - weaker for signed red/blue target-evidence interpretation.

What different information it brings:

```text
Whether dominant internal CNN activations align with the lesion, independent of target-class gradients.
```

Thesis role:

```text
A gradient-free activation localization comparison.
```

#### Score-CAM

- CAM-style method that avoids gradients.
- It masks the input image using activation maps and checks how each masked image changes the class score.
- Strength:
  - more directly score-based than Grad-CAM;
  - can be more faithful in some cases;
  - avoids noisy gradients.
- Weakness:
  - much slower because it requires many forward passes;
  - expensive for 100+ X-rays and multiple thresholds;
  - may be impractical unless used on small subsets.

What different information it brings:

```text
Whether score-based perturbation of activation regions agrees with gradient-based CAM evidence.
```

Thesis role:

```text
Potentially stronger but computationally expensive CAM alternative.
```

If time is limited, choose `Grad-CAM++` before `Score-CAM`.

#### Integrated Gradients

- Already expanded in the workflow.
- Attributes the model output back to input pixels by integrating gradients from a baseline image to the actual X-ray.
- Conceptually there are now:
  - magnitude IG: neutral violet impact map;
  - positive IG: red positive contribution;
  - negative IG: blue negative contribution;
  - signed IG: red/blue combined.
- Strength:
  - pixel-level attribution;
  - theoretically grounded;
  - captures contributions relative to a baseline.
- Weakness:
  - depends heavily on baseline choice;
  - can be noisy;
  - sign interpretation needs careful wording.

What different information it brings:

```text
Which input-level pixels contribute most to the pneumothorax output, compared with a baseline image.
```

Thesis role:

```text
Input-level attribution complement to Grad-CAM's feature-map-level attribution.
```

#### GradientSHAP / SHAP-style Attribution

- Combines ideas from Integrated Gradients and SHAP.
- Instead of using one fixed baseline, it samples multiple baselines/noisy variants and estimates attribution more like an expected contribution.
- Strength:
  - better accounts for baseline uncertainty;
  - more SHAP-like feature contribution interpretation;
  - useful if IG baseline choice is questionable.
- Weakness:
  - more computationally expensive than IG;
  - still sensitive to baseline distribution;
  - may be noisy unless enough samples are used.

What different information it brings:

```text
Whether attribution remains consistent when baseline uncertainty is modeled rather than fixed to one reference image.
```

Thesis role:

```text
Robustness check for IG-style pixel attribution.
```

This is scientifically useful, but probably second priority after `Grad-CAM++`.

#### Occlusion Sensitivity

- Perturbation-based method.
- Slides a patch over the image and masks/occludes that region.
- Measures how much the pneumothorax score changes.
- If occluding a region drops the score, that region was important positive evidence.
- If occluding a region increases the score, that region was suppressing the target.
- Strength:
  - very intuitive;
  - directly measures causal effect on model output;
  - does not rely on gradients.
- Weakness:
  - slow;
  - patch size matters;
  - lower spatial precision depending on stride and patch size.

What different information it brings:

```text
Which regions are causally important according to direct input removal, not gradients.
```

Thesis role:

```text
Perturbation-based sanity check for attribution maps.
```

This would be very useful for selected cases, especially `TP`, `FP`, `TN`, and suspicious examples.

#### LIME

- Perturbation-based local surrogate method.
- Splits the image into superpixels, perturbs them on/off, and fits a simple interpretable model around that one prediction.
- Strength:
  - intuitive;
  - model-agnostic;
  - can explain local prediction behavior.
- Weakness:
  - superpixels are often awkward for X-rays;
  - results depend heavily on segmentation settings;
  - can be unstable;
  - may require tuning to avoid misleading medical-image explanations.

What different information it brings:

```text
Which image regions a simple local surrogate model thinks are responsible for the prediction.
```

Thesis role:

```text
Optional model-agnostic comparison, but only if implementation is cheap.
```

Do not prioritize `LIME` unless there is spare time.

### Recommended Priority for This Thesis

#### Priority 1 - Add Grad-CAM++

- Most natural extension of current Grad-CAM.
- Keeps the red/blue class-evidence discussion coherent.
- Likely easy to compare with current metrics and selected-threshold images.

#### Priority 2 - Add faithfulness metrics

Especially:

- deletion curve;
- insertion curve;
- maybe Captum infidelity/sensitivity on a smaller subset.

This is highly thesis-relevant because it separates:

```text
Does the map overlap the lesion?
```

from:

```text
Does the map actually explain the model score?
```

#### Priority 3 - Add Occlusion Sensitivity for selected cases

- Very useful for qualitative validation.
- Especially for confusing `FP` and suspicious `TP` cases.
- Maybe not necessary for all 100 images if runtime is high.

#### Priority 4 - Add GradientSHAP

- Good robustness comparison for IG.
- Useful if discussing baseline sensitivity.

#### Lower Priority - Score-CAM / Eigen-CAM / LIME

- `Score-CAM`: useful but slow.
- `Eigen-CAM`: useful but less class-specific.
- `LIME`: optional, but may be unstable on CXR images.

### What Different Information They Bring Together

A strong thesis framing would be:

| Method / Metric | Main Question |
|---|---|
| `Grad-CAM` | Which high-level regions support or suppress pneumothorax? |
| `Grad-CAM++` | Does refined CAM weighting improve localization, especially for small lesions? |
| `Integrated Gradients` | Which input pixels contribute most relative to a baseline? |
| `GradientSHAP` | Are pixel attributions robust to baseline uncertainty? |
| `Occlusion Sensitivity` | Which regions causally change the model score when removed? |
| `Score-CAM` | Which activation regions increase the score via forward-pass masking? |
| `Eigen-CAM` | Where are dominant internal CNN activations located, gradient-free? |
| `LIME` | What local region-level surrogate explanation approximates this prediction? |
| `Deletion / Insertion` | Do highlighted pixels actually control the model score? |
| `Infidelity` | Do attribution values predict output changes under perturbation? |
| `Sensitivity` | Are explanations stable under small input changes? |

### Best Thesis-Safe Conclusion

The most important message is:

```text
Different XAI methods answer different questions. Some localize class-discriminative CNN regions, some attribute the output to input pixels, some test causal score changes through perturbation, and faithfulness metrics evaluate whether the explanations actually reflect the model's behavior. Therefore, agreement between methods is stronger evidence than any single heatmap, while disagreement is itself an important validation finding.
```

For the next implementation step, the selected direction was:

```text
Grad-CAM++ + deletion/insertion faithfulness curves
```

That combination adds both a stronger visual method and a more rigorous explanation-quality evaluation.

## 2026-05-13 - Iteration 12: Grad-CAM++ and Deletion/Insertion Faithfulness Started

- Implemented the first prioritized future-work item:
  - added `Grad-CAM++` positive evidence maps;
  - added `Grad-CAM++` negative signed evidence maps;
  - added optional deletion/insertion faithfulness curve output in the main CXR evaluation script.
- `scripts/run_cxr_torchxray_smoke.py` now accepts `--faithfulness-fractions`, for example `0.0,0.5,1.0`, and writes `faithfulness_curves.csv` with insertion and deletion pneumothorax probabilities per method/fraction.
- `Grad-CAM++` is now included in:
  - main evaluation metrics and grouped PNG output;
  - heatmap threshold calibration;
  - single-image threshold selection visualization;
  - classifier-outcome threshold visualization.
- Smoke outputs generated:
  - `outputs/iter_12_gradcampp_faithfulness_smoke`
  - `outputs/iter_12_gradcampp_calibration_smoke`
  - `outputs/iter_12_gradcampp_threshold_smoke`
- Verification:
  - WSL `py_compile` passed for changed Python files;
  - CUDA main smoke confirmed `grad_cam_plus_plus`, `grad_cam_plus_plus_negative`, and `faithfulness_curves.csv`;
  - CUDA calibration smoke confirmed selected fractions include the new methods;
  - CUDA threshold smoke confirmed Grad-CAM++ sweep panels are generated.

## 2026-05-14 - Iteration 13: Classifier Outcome Output Layout Corrected

- Corrected `scripts/visualize_cxr_classifier_outcome_thresholds.py` output layout after the main output refactor.
- Desired classifier-outcome layout convention:
  - keep top-level classifier outcome folders: `tp`, `fp`, `tn`, `fn`;
  - inside each outcome folder, create one folder per source X-ray/case;
  - inside each case folder, keep all generated PNGs flat as sibling files, without per-method subfolders.
- Added the source X-ray stem to each generated PNG filename so an exported/copied individual image remains traceable to its original case.
- Smoke output generated:
  - `outputs/iter_13_classifier_outcome_grouped_flat_smoke`
- Verification:
  - WSL `py_compile` passed for `scripts/visualize_cxr_classifier_outcome_thresholds.py`;
  - CUDA smoke with 2 test cases produced `tp` and `tn` folders;
  - each outcome folder contains per-case folders such as `case_000_tp_0_test_1`;
  - case folders contain flat files such as `0_test_1_consensus.png`, `0_test_1_consensus_selected_top_05.png`, and `0_test_1_consensus_threshold_sweep_panel.png` with no nested method directories.

## 2026-05-14 - Iteration 14: Threshold Metrics CSV Clarified

- Investigated confusing apparent duplicate rows in `threshold_metrics.csv` from `scripts/visualize_cxr_classifier_outcome_thresholds.py`.
- Finding:
  - rows are not duplicate samples;
  - the intended row key is `sample_index` + `method` + `top_fraction`;
  - repeated rows for the same `sample_index` are expected because each source X-ray is evaluated for every method and every threshold fraction.
- Fixed misleading epsilon-only metric values:
  - no-overlap `IoU`, `Dice`, and `precision_at_fraction` now return exact `0.0` instead of tiny values such as `2.94e-12`;
  - this makes no-overlap cases visually and numerically unambiguous.
- Added traceability and backend-debug columns to classifier-outcome `threshold_metrics.csv`:
  - `source_stem`, `image_path`, `mask_path`;
  - `metric_component`;
  - `top_fraction_percent`;
  - `positive_localization_applicable`;
  - `selected_pixel_count`, `mask_pixel_count`, `intersection_pixel_count`, `union_pixel_count`.
- Positive lesion-localization metrics are now blank for negative-label/no-mask cases in classifier-outcome threshold runs, because `IoU`/`Dice` against an empty mask are not meaningful lesion-localization scores.
- Negative evidence diagnostics remain separate:
  - `negative_mask_overlap_fraction` shows how much selected negative evidence intersects the mask;
  - `negative_mask_avoidance_fraction` is the complementary avoidance score.
- Smoke output generated:
  - `outputs/iter_14_threshold_metrics_clarity_smoke`
- Verification:
  - WSL `py_compile` passed for `src/explainai_thesis/metrics.py` and `scripts/visualize_cxr_classifier_outcome_thresholds.py`;
  - CUDA smoke with 2 test cases completed successfully;
  - inspected `threshold_metrics.csv` and confirmed the new columns, blank positive metrics for `tn`, and exact zero no-overlap behavior.

## 2026-05-14 - Iteration 15: Faithfulness Curves Visualized

- Expanded the deletion/insertion faithfulness output from CSV-only values into thesis-readable visual artifacts.
- `scripts/run_cxr_torchxray_smoke.py` now writes the following whenever `--faithfulness-fractions` is provided:
  - `faithfulness_curves.csv`: raw per-case/per-method/per-fraction insertion and deletion probabilities;
  - `faithfulness_summary.csv`: method-level mean insertion AUC, deletion AUC, and deletion-drop AUC;
  - `faithfulness_curves.png`: aggregate line plot comparing insertion and deletion curves across methods;
  - `case_XXX_<source_stem>/faithfulness_curves.png`: per-case line plot inside each exported case folder.
- Interpretation reminder:
  - strong insertion behavior means pneumothorax probability rises quickly when top-attributed pixels are restored from a blank baseline;
  - strong deletion behavior means pneumothorax probability drops quickly when top-attributed pixels are removed from the original image;
  - these curves evaluate faithfulness to the model score, not direct lesion-mask localization.
- Smoke output generated:
  - `outputs/iter_15_faithfulness_plots_smoke`
- Verification:
  - WSL `py_compile` passed for `scripts/run_cxr_torchxray_smoke.py`;
  - CUDA smoke with 1 positive test case and `--faithfulness-fractions 0.0,0.5,1.0` completed successfully;
  - confirmed root `faithfulness_curves.csv`, `faithfulness_summary.csv`, root `faithfulness_curves.png`, and case-level `faithfulness_curves.png` exist.

### Faithfulness Chart Readability and Interpretation Note

- The first generated chart at `outputs/iter_15_faithfulness_plots_smoke/faithfulness_curves.png` is readable enough for an internal smoke-test check, but not yet thesis/report quality.
- Readability limitations:
  - too many methods are plotted together and many curves overlap almost exactly;
  - the legend is large and far from the plots;
  - the smoke run used only three fraction points, `0.0`, `0.5`, and `1.0`, making the curve shape too coarse;
  - the y-axis is fixed to `0–1`, while the observed probabilities are compressed around `0.50–0.64`;
  - positive, negative, magnitude, and consensus methods are mixed in one figure even though they answer partly different questions.
- Meaning of the chart:
  - it is an aggregate deletion/insertion faithfulness curve plot;
  - it asks whether pixels ranked as important by an attribution method actually control the TorchXRayVision pneumothorax score.
- Insertion panel interpretation:
  - x-axis = fraction of top-attributed pixels restored into a baseline/blank image;
  - y-axis = pneumothorax probability after restoration;
  - a faithful explanation should ideally rise quickly, because restoring only the most important pixels should recover much of the original pneumothorax score.
- Deletion panel interpretation:
  - x-axis = fraction of top-attributed pixels removed from the original image;
  - y-axis = pneumothorax probability after removal;
  - a faithful explanation should ideally drop quickly, because removing truly important pixels should reduce the pneumothorax score.
- Safe interpretation of the smoke chart:
  - it proves the plotting pipeline works;
  - it is not a thesis result because it used only one positive X-ray and three coarse fraction points;
  - no method clearly shows ideal insertion/deletion behavior in that smoke chart;
  - the model probability remains roughly in a narrow `0.50–0.64` range across perturbations.
- Recommended thesis-quality improvements:
  - run on `50–100` positive test cases or another clearly defined evaluation subset;
  - use finer fractions such as `0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,0.90,1.0`;
  - separate method families in plots, for example CAM methods vs Integrated Gradients variants;
  - use `faithfulness_summary.csv` AUC values for method comparison;
  - consider a zoomed y-axis version if probabilities remain compressed.
- Thesis interpretation guidance:
  - higher insertion AUC means the method restores the model score faster from important pixels;
  - lower deletion AUC, or higher deletion-drop AUC, means the method removes score-supporting evidence more effectively;
  - disagreement between localization metrics and faithfulness metrics is important, because a heatmap may be faithful to the model but not clinically localized, or visually/clinically plausible but not actually controlling the model score.

## 2026-05-14 - Iteration 16: Finer Faithfulness Evaluation on 50 Positive Test X-rays

- Generated a more meaningful deletion/insertion faithfulness output beyond the initial smoke chart.
- Command:

```bash
wsl.exe --cd /mnt/c/Users/Dmytro.Valantsevych/Downloads/master_thesis_draft_explainAI python3 scripts/run_cxr_torchxray_smoke.py --device auto --split test --max-positive 50 --ig-steps 16 --max-overlays 10 --faithfulness-fractions 0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,0.90,1.0 --output-dir outputs/iter_16_faithfulness_test50_fine
```

- Output folder:
  - `outputs/iter_16_faithfulness_test50_fine`
- Main artifacts:
  - `metrics.csv`
  - `metrics_summary.csv`
  - `faithfulness_curves.csv`
  - `faithfulness_summary.csv`
  - `faithfulness_curves.png`
  - per-case folders for the first 10 overlay-exported cases, each including `faithfulness_curves.png`.
- Faithfulness summary from 50 positive test cases:

| Method | Insertion AUC mean | Deletion AUC mean | Deletion-drop AUC mean |
|---|---:|---:|---:|
| `consensus` | `0.625316` | `0.619961` | `0.380039` |
| `grad_cam` | `0.627488` | `0.622606` | `0.377394` |
| `grad_cam_negative` | `0.622913` | `0.626269` | `0.373731` |
| `grad_cam_plus_plus` | `0.627415` | `0.621973` | `0.378027` |
| `grad_cam_plus_plus_negative` | `0.626356` | `0.622891` | `0.377109` |
| `integrated_gradients` | `0.567037` | `0.550506` | `0.449494` |
| `integrated_gradients_negative` | `0.545448` | `0.552268` | `0.447732` |
| `integrated_gradients_positive` | `0.546898` | `0.552045` | `0.447955` |
| `integrated_gradients_signed` | `0.546898` | `0.552045` | `0.447955` |

- Initial reading:
  - CAM-family methods have slightly higher insertion AUC, meaning they restore the model score faster under the current insertion protocol;
  - IG-family methods have lower deletion AUC and higher deletion-drop AUC, meaning removing IG-ranked pixels suppresses the model score more strongly under the current deletion protocol;
  - this suggests method disagreement between insertion and deletion faithfulness, which is thesis-relevant and should be discussed together with localization metrics rather than treated as a single winner.
- Verification:
  - CUDA run completed successfully on WSL;
  - confirmed root `faithfulness_curves.csv`, `faithfulness_summary.csv`, `faithfulness_curves.png`, `metrics.csv`, and `metrics_summary.csv` exist;
  - confirmed at least one exported case folder contains a case-level `faithfulness_curves.png`.

### Deletion/Insertion Probability Source and Curve Behavior Interpretation

- Deletion/insertion faithfulness evaluates the `TorchXRayVision` model's pneumothorax probability, not a separate metric model.
- The implementation calls the same loaded model and target class after perturbing the input image:

```python
output = model(image)
probability = torch.sigmoid(output[0, class_idx])
```

- For each method heatmap, pixels are sorted from highest attribution to lowest attribution.
- Insertion:
  - starts from a zero baseline image;
  - restores the top-fraction pixels from the original X-ray;
  - re-evaluates the `TorchXRayVision` pneumothorax probability.
- Deletion:
  - starts from the original X-ray;
  - replaces the top-fraction pixels with the zero baseline;
  - re-evaluates the `TorchXRayVision` pneumothorax probability.
- Therefore, every insertion/deletion point is a real forward pass through `TorchXRayVision`.
- If `Grad-CAM` causes a noticeable probability drop at `0.1`, this means the top `10%` `Grad-CAM`-ranked pixels contain regions that the model uses strongly for the pneumothorax score.
- For deletion, this is the expected faithfulness behavior: removing top-ranked evidence should reduce the model's target probability.
- For insertion, if restoring the top `10%` already changes the probability strongly, the ranked region carries high score-relevant information.
- A steep `Grad-CAM` curve suggests that `Grad-CAM` is relatively good at identifying score-controlling regions for this model, even if those regions do not always overlap well with the pneumothorax mask.
- Behavior near `0.9` should be interpreted carefully:
  - at deletion `0.9`, `90%` of the image has been removed and only `10%` remains original;
  - at insertion `0.9`, `90%` of the image has been restored and only `10%` remains baseline.
- Strong probability changes near `0.9` may reflect broad image context, background, lung field structure, or artifacts caused by replacing large image areas with zero baseline, not only lesion-specific evidence.
- `Grad-CAM++` may fluctuate because it uses different weighting logic from standard `Grad-CAM` and may rank sharper, smaller, or different discriminative regions.
- Non-monotonic `Grad-CAM++` behavior can occur because:
  - removing a small region can expose other evidence or remove suppressive evidence;
  - the perturbed image is not a natural X-ray;
  - the pretrained `TorchXRayVision` output is only moderately calibrated for this dataset;
  - the model response to artificial masking does not have to be perfectly monotonic.
- Important thesis point: deletion/insertion curves are directionally informative, but they are not expected to be perfectly monotonic in real CNNs.
- `Integrated Gradients` variants can fluctuate more because they rank input pixels rather than broad feature-map regions.
- IG-ranked top pixels can be scattered, edge-like, high-frequency, signed, and baseline-sensitive, so hard deletion/insertion can create more artificial-looking perturbations than CAM-based deletion/insertion.
- Current IG variant interpretation:
  - `integrated_gradients` = absolute/magnitude impact, regardless of sign;
  - `integrated_gradients_positive` = pixels pushing toward pneumothorax;
  - `integrated_gradients_negative` = pixels pushing away from pneumothorax;
  - `integrated_gradients_signed` = positive IG localization with negative IG diagnostics/visual pairing.
- The magnitude IG map can mix high positive attribution and high negative attribution, so deleting the most impactful IG pixels may remove both score-supporting and score-suppressing evidence.
- If deletion removes suppressive evidence, pneumothorax probability can increase or fluctuate instead of decreasing smoothly.
- If individual IG maps fluctuate but `consensus` is smoother, this likely means the consensus is dominated by more spatially coherent CAM-like regions while individual IG variants include finer signed pixel effects.
- Thesis-safe interpretation:

```text
Deletion/insertion evaluates faithfulness to the model score, not clinical correctness. A steep curve means the heatmap identifies pixels that control the TorchXRayVision pneumothorax probability. It does not automatically mean those pixels correspond to the true pneumothorax mask.
```

- If `Grad-CAM` has stronger deletion/insertion behavior but weak `Dice`/`IoU`, that is thesis-important: `Grad-CAM` may be faithful to what the model uses, while the model itself may rely on non-lesion or distributed image evidence.
- If IG fluctuates, this suggests IG may capture fine-grained signed input sensitivity, but ranked-pixel perturbations are less spatially coherent and can produce non-monotonic model responses.
- Practical interpretation rule:
  - strong deletion drop = method found pixels important to the model score;
  - strong insertion rise = method found pixels sufficient to restore the model score;
  - fluctuation = method ranking, sign mixture, baseline artifacts, or distributed model evidence causes non-monotonic response;
  - good faithfulness but poor mask overlap = model may be using clinically questionable evidence;
  - good mask overlap but weak faithfulness = visually plausible explanation may not actually control the model score.
- Current result reading:
  - `Grad-CAM` and `Grad-CAM++` having higher faithfulness AUCs around `0.62` suggests CAM-style maps are more aligned with `TorchXRayVision` score-controlling regions under the insertion protocol;
  - IG variants having lower AUCs around `0.55–0.57` suggests the current IG perturbation ranking behaves differently and is less favorable under hard pixel replacement;
  - this does not make IG useless, but indicates it gives a different, more pixel-level sensitivity view.
- Thesis-safe wording:

```text
Deletion and insertion curves were computed by re-evaluating the TorchXRayVision pneumothorax probability after perturbing the input image according to each attribution ranking. CAM-based methods produced a stronger and more coherent probability response, suggesting that their highlighted regions better captured score-controlling image regions for this model. Integrated Gradients variants showed more fluctuation, likely because pixel-level signed attributions are less spatially coherent and because magnitude-based IG can mix positive and negative contributions. Therefore, faithfulness curves should be interpreted as model-behavior diagnostics rather than direct evidence of clinical lesion localization.
```

## 2026-05-15 - Iteration 17: Faithfulness Plot Readability, GradientSHAP, and Occlusion Sensitivity

- Improved deletion/insertion faithfulness readability in `scripts/run_cxr_torchxray_smoke.py`:
  - original aggregate `faithfulness_curves.png` is still written;
  - added `faithfulness_curves_zoomed.png` with auto-scaled y-axis;
  - added family-specific plots such as `faithfulness_curves_cam_family.png`, `faithfulness_curves_integrated_gradients_family.png`, `faithfulness_curves_gradient_shap_family.png`, and `faithfulness_curves_occlusion_family.png` when corresponding rows exist;
  - added `faithfulness_summary.png` bar plot comparing insertion AUC and deletion-drop AUC.
- Added `GradientSHAP` via Captum:
  - `gradient_shap` = neutral/violet magnitude map;
  - `gradient_shap_positive` = red positive signed contribution;
  - `gradient_shap_negative` = blue negative signed contribution;
  - `gradient_shap_signed` = red positive map with blue negative diagnostic overlay.
- Added `Occlusion Sensitivity`:
  - `occlusion` = neutral/violet absolute score-change map;
  - `occlusion_positive` = red regions where occlusion decreases the pneumothorax logit, meaning the region supports the target score;
  - `occlusion_negative` = blue regions where occlusion increases the pneumothorax logit, meaning the region suppresses the target score.
- Added runtime controls:
  - `--gradshap-samples`;
  - `--gradshap-stdevs`;
  - `--occlusion-patch-size`;
  - `--occlusion-stride`.
- Integrated new methods into:
  - main evaluation metrics and overlays;
  - selected-threshold images;
  - deletion/insertion faithfulness curves;
  - top-fraction calibration in `scripts/calibrate_cxr_xai_thresholds.py`.
- Smoke outputs generated:
  - `outputs/iter_17_occlusion_gradshap_smoke`;
  - `outputs/iter_17_occlusion_gradshap_calibration_smoke`.
- Verification:
  - WSL `py_compile` passed for `src/explainai_thesis/xai.py`, `scripts/run_cxr_torchxray_smoke.py`, and `scripts/calibrate_cxr_xai_thresholds.py`;
  - CUDA main smoke with one positive test case generated all expected method rows, including `gradient_shap`, `gradient_shap_signed`, `occlusion`, `occlusion_positive`, and `occlusion_negative`;
  - verified new faithfulness plot artifacts exist, including zoomed and family-specific plots plus `faithfulness_summary.png`;
  - CUDA calibration smoke confirmed selected fractions are written for all new methods.

### Model/Weights Concern and Next Evaluation Direction

- Current clinical interpretation concern is important:
  - many inspected predictions/localizations look clinically random or anatomically implausible;
  - current overlap scores are very low, often only a few percent `IoU` at best;
  - the model can still achieve moderate classifier signal because some pneumothorax-correlated information may be inferred indirectly from lung roots, mediastinal configuration, costophrenic/deep sulcus changes, positioning, or dataset bias;
  - however, attribution on ribs, outside body regions, chin/neck artifacts, or other irrelevant anatomy is clinically suspicious.
- The currently used weight identifier is still `densenet121-res224-all` from TorchXRayVision:
  - it is not necessarily pneumothorax-specialized;
  - it is a broad pretrained multi-pathology baseline;
  - weak localization does not prove the weights are technically wrong, but it strongly suggests dataset/model mismatch and/or insufficient pneumothorax task specificity.
- Before running `1000+` cases, recommended sanity sequence:
  1. run a `100`-case sanity pass with the new methods and inspect runtime/artifacts;
  2. calibrate thresholds for the expanded method set;
  3. then run `1000` mixed-label cases for classifier outcome diversity and reporting.
- A second model should be added for comparison because the current TorchXRayVision baseline may be clinically weak for this SIIM/Kaggle pneumothorax task.
- Candidate direction:
  - CheXNet-style DenseNet trained on chest pathology can be a historical comparator;
  - preferably use a newer pneumothorax-specific or stronger CXR model if available with reproducible weights and license;
  - compare classification performance first, then run the same XAI/localization/faithfulness protocol on both models.
- Thesis-safe framing:

```text
The current TorchXRayVision DenseNet should be treated as an external pretrained baseline rather than a clinically adequate pneumothorax model. Its moderate classifier signal and weak lesion localization make it useful for demonstrating why XAI validation is necessary, but a second, stronger pneumothorax-oriented model is needed to show how explanation behavior changes when the underlying classifier is more clinically appropriate.
```

## 2026-05-15 - Iteration 18: Faithfulness Readability, Output Naming, and Recalibration Notes

- Follow-up issues addressed in the code:
  - main per-case PNGs in `case_XXX_<source_stem>` folders now include the source X-ray stem in the filename, e.g. `<source_stem>_grad_cam.png`, `<source_stem>_grad_cam_selected.png`, and `<source_stem>_faithfulness_curves.png`;
  - root-level faithfulness bar plots are now written under both `faithfulness_summary.png` and the clearer alias `faithfulness_auc_bars.png`;
  - family-specific faithfulness plots now use one shared zoomed y-axis range derived from the full run, so `CAM`, `IG`, `GradientSHAP`, and `Occlusion` family plots are visually comparable and do not exaggerate small changes independently.
- Calibration was extended for the expanded method set:
  - `calibration_metrics.csv` and `calibration_summary.csv` now include `negative_mask_overlap_fraction` and `negative_mask_avoidance_fraction` where applicable;
  - `selected_fractions_by_metric.csv` is written in addition to `selected_fractions.csv` so fractions can be inspected for `IoU`, `Dice`, `precision_at_fraction`, `pointing_hit`, and negative-evidence diagnostics;
  - `--selection-metric` can now directly choose `negative_mask_avoidance_fraction` or `negative_mask_overlap_fraction` when intentionally calibrating blue/suppressive evidence diagnostics.
- Important interpretation for recalibration:
  - positive/red methods should generally be calibrated by lesion-localization metrics such as `Dice` or `IoU`;
  - negative/blue methods should not be optimized for overlap with the lesion, because blue means evidence against pneumothorax; for blue maps, high `negative_mask_avoidance_fraction` is usually the healthier diagnostic;
  - faithfulness insertion/deletion is still a separate model-behavior evaluation and should not be treated as the same thing as mask threshold calibration.
- Why `100%` deletion can still return about `60%` pneumothorax probability:
  - deletion currently replaces removed pixels with the zero baseline in the normalized TorchXRayVision input space, not with a clinically realistic negative X-ray;
  - the pretrained model has a high pneumothorax baseline/poor threshold calibration on this dataset, so even a blank or heavily perturbed image can retain a high sigmoid output;
  - this means the absolute deletion probability floor is partly a model calibration/bias artifact, not evidence that the removed image still contains pneumothorax;
  - for thesis interpretation, deletion should emphasize probability change/drop and AUC differences rather than assuming the final probability after `100%` removal is clinically meaningful.
- New method explanation notes:
  - `GradientSHAP` estimates IG/SHAP-like attribution by sampling noisy baselines, so it tests whether pixel attribution is robust to baseline uncertainty;
  - `Occlusion Sensitivity` directly masks square patches and measures score change, so it is more causal/intuitive but coarser and slower;
  - occlusion may identify a clinically plausible positive square with good `IoU` while showing weak insertion/deletion degradation if the model also relies on broad context or remains highly biased toward pneumothorax on baseline images.
- Recommended next recalibration command pattern:

```bash
wsl.exe --cd /mnt/c/Users/Dmytro.Valantsevych/Downloads/master_thesis_draft_explainAI python3 scripts/calibrate_cxr_xai_thresholds.py --device auto --split train --max-positive 100 --ig-steps 16 --gradshap-samples 8 --occlusion-patch-size 32 --occlusion-stride 16 --fractions 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50 --selection-metric dice --output-dir outputs/iter_18_xai_threshold_calibration_train100_expanded_dice
```

- For negative-evidence-specific inspection, repeat with:

```bash
--selection-metric negative_mask_avoidance_fraction --output-dir outputs/iter_18_xai_threshold_calibration_train100_expanded_negative_avoidance
```

## 2026-05-15 - Iteration 19: First-Order Baseline Stabilization Started

- Started the first-order plan before larger XAI runs: recalibrate the classifier threshold and finish output naming/layout consistency.
- Updated `scripts/evaluate_cxr_torchxray_model.py` so classifier evaluation now writes `selected_thresholds.csv` with explicit candidate operating points:
  - `best_f1`;
  - `best_youden_j`;
  - `high_sensitivity` using `--high-sensitivity-min`, defaulting to `0.95`.
- Ran train-split calibration as the current calibration reference:

```bash
wsl.exe python3 scripts/evaluate_cxr_torchxray_model.py --device auto --split train --batch-size 64 --threshold 0.5 --high-sensitivity-min 0.95 --output-dir outputs/iter_19_classifier_threshold_calibration_train
```

- Train calibration result for `densenet121-res224-all`:

| Selection | Threshold | Sensitivity | Specificity | Precision | F1 | Youden J |
|---|---:|---:|---:|---:|---:|---:|
| `best_f1` | `0.62` | `0.893232` | `0.534595` | `0.354995` | `0.508069` | `0.427827` |
| `best_youden_j` | `0.62` | `0.893232` | `0.534595` | `0.354995` | `0.508069` | `0.427827` |
| `high_sensitivity` | `0.58` | `0.950399` | `0.441900` | `0.328109` | `0.487810` | `0.392299` |

- Interpretation:
  - the old exploratory `0.61` is close to the train-calibrated `best_f1`/`best_youden_j` threshold of `0.62`;
  - for a screening/high-sensitivity view, `0.58` is the current frozen candidate from the train split;
  - these are still thresholds for a weak external baseline and should be presented as calibration-derived operating points, not as proof of clinical adequacy.
- Refactored `scripts/visualize_cxr_threshold_selection.py` output naming:
  - now creates one `case_XXX_<source_stem>` folder;
  - PNGs are flat inside that folder;
  - every PNG filename includes the source X-ray stem, method name, and artifact type;
  - no nested per-method folders are created in the new layout.
- Smoke output generated:

```bash
wsl.exe python3 scripts/visualize_cxr_threshold_selection.py --device auto --split train --case-index 0 --ig-steps 2 --fractions 0.05,0.10 --output-dir outputs/iter_19_threshold_selection_grouped_flat_smoke
```

- Verified example case folder: `outputs/iter_19_threshold_selection_grouped_flat_smoke/case_000_2_train_1_`.
- Verified example source-stemmed files include:
  - `2_train_1__grad_cam_continuous_heatmap.png`;
  - `2_train_1__grad_cam_selected_top_05pct.png`;
  - `2_train_1__grad_cam_threshold_sweep_panel.png`;
  - `2_train_1__consensus_continuous_heatmap.png`.
- Verification:
  - WSL `py_compile` passed for `scripts/evaluate_cxr_torchxray_model.py`, `scripts/visualize_cxr_threshold_selection.py`, `scripts/run_cxr_torchxray_smoke.py`, and `scripts/visualize_cxr_classifier_outcome_thresholds.py`;
  - classifier calibration completed successfully on CUDA;
  - grouped-flat single-image threshold smoke completed successfully on CUDA.

## 2026-05-15 - Iteration 20: Threshold Smoke Updated for Expanded Metrics

- Follow-up issue: the Iteration 19 single-image threshold smoke confirmed the new folder/filename layout, but it did not include the newly added expanded XAI methods/metric-detail columns.
- Updated `scripts/visualize_cxr_threshold_selection.py` so the single-image threshold smoke now includes the full expanded method set:
  - `grad_cam`, `grad_cam_plus_plus`, and signed negative variants;
  - `integrated_gradients` magnitude/positive/negative/signed;
  - `gradient_shap` magnitude/positive/negative/signed;
  - `occlusion` magnitude/positive/negative;
  - `consensus` including positive, negative, and neutral evidence layers.
- Expanded `threshold_metrics.csv` in this smoke path with the newer clarity fields:
  - `metric_component`;
  - `top_fraction_percent`;
  - `selected_pixel_count`, `mask_pixel_count`, `intersection_pixel_count`, `union_pixel_count`;
  - `negative_mask_overlap_fraction` and `negative_mask_avoidance_fraction` where applicable.
- Verification command:

```bash
wsl.exe python3 scripts/visualize_cxr_threshold_selection.py --device auto --split train --case-index 0 --ig-steps 2 --gradshap-samples 2 --occlusion-patch-size 112 --occlusion-stride 112 --fractions 0.05,0.10 --output-dir outputs/iter_20_threshold_selection_new_metrics_smoke_v2
```

- Verification result:
  - WSL `py_compile` passed for `scripts/visualize_cxr_threshold_selection.py`;
  - smoke completed successfully on CUDA;
  - `threshold_metrics.csv` contains `32` rows = `16` methods × `2` fractions;
  - verified source-stemmed PNGs exist for `gradient_shap`, `gradient_shap_signed`, `occlusion`, `occlusion_negative`, and `consensus` panels.

## 2026-05-15 - Iteration 21: Stage 3/4 Follow-up, Expanded XAI Recalibration

- Completed the requested next plan stages after the first-order stabilization:
  - Stage 3: faithfulness plot readability was already implemented and re-verified by compile checks; the main evaluation now writes full-scale, zoomed, family-split, and AUC/bar faithfulness plots.
  - Stage 4: reran XAI threshold calibration with the expanded method set on `100` positive train cases.
- Compile verification:

```bash
wsl.exe python3 -m py_compile scripts/run_cxr_torchxray_smoke.py scripts/calibrate_cxr_xai_thresholds.py scripts/visualize_cxr_threshold_selection.py
```

- Positive-localization calibration command:

```bash
wsl.exe python3 scripts/calibrate_cxr_xai_thresholds.py --device auto --split train --max-positive 100 --ig-steps 16 --gradshap-samples 8 --occlusion-patch-size 56 --occlusion-stride 56 --fractions 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50 --selection-metric dice --output-dir outputs/iter_21_xai_calibration_train100_all_methods_dice
```

- Negative-evidence avoidance calibration command:

```bash
wsl.exe python3 scripts/calibrate_cxr_xai_thresholds.py --device auto --split train --max-positive 100 --ig-steps 16 --gradshap-samples 8 --occlusion-patch-size 56 --occlusion-stride 56 --fractions 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50 --selection-metric negative_mask_avoidance_fraction --output-dir outputs/iter_21_xai_calibration_train100_all_methods_negative_avoidance
```

- Dice-selected fractions on `100` positive train cases:

| Method | Selected fraction | Mean Dice |
|---|---:|---:|
| `grad_cam` | `0.10` | `0.048577` |
| `grad_cam_plus_plus` | `0.50` | `0.035571` |
| `consensus` | `0.25` | `0.036091` |
| `integrated_gradients` | `0.45` | `0.024766` |
| `integrated_gradients_positive` | `0.25` | `0.024002` |
| `integrated_gradients_negative` | `0.30` | `0.023064` |
| `gradient_shap` | `0.50` | `0.023444` |
| `gradient_shap_positive` | `0.50` | `0.023431` |
| `gradient_shap_negative` | `0.50` | `0.022702` |
| `occlusion` | `0.40` | `0.023441` |
| `occlusion_positive` | `0.45` | `0.025307` |
| `occlusion_negative` | `0.35` | `0.025893` |

- Negative-avoidance selected fractions for suppressive/signed evidence:

| Method | Selected fraction | Mean negative-mask avoidance |
|---|---:|---:|
| `grad_cam_negative` | `0.15` | `0.992465` |
| `grad_cam_plus_plus_negative` | `0.05` | `0.986044` |
| `integrated_gradients_negative` | `0.50` | `0.987988` |
| `integrated_gradients_signed` | `0.50` | `0.987869` |
| `gradient_shap_negative` | `0.15` | `0.988309` |
| `gradient_shap_signed` | `0.50` | `0.987805` |
| `occlusion_negative` | `0.10` | `0.990804` |
| `consensus` | `0.50` | `0.983692` |

- Artifacts generated:
  - `outputs/iter_21_xai_calibration_train100_all_methods_dice/calibration_metrics.csv`
  - `outputs/iter_21_xai_calibration_train100_all_methods_dice/calibration_summary.csv`
  - `outputs/iter_21_xai_calibration_train100_all_methods_dice/selected_fractions.csv`
  - `outputs/iter_21_xai_calibration_train100_all_methods_dice/selected_fractions_by_metric.csv`
  - `outputs/iter_21_xai_calibration_train100_all_methods_negative_avoidance/calibration_metrics.csv`
  - `outputs/iter_21_xai_calibration_train100_all_methods_negative_avoidance/calibration_summary.csv`
  - `outputs/iter_21_xai_calibration_train100_all_methods_negative_avoidance/selected_fractions.csv`
  - `outputs/iter_21_xai_calibration_train100_all_methods_negative_avoidance/selected_fractions_by_metric.csv`
- Interpretation:
  - `grad_cam` remains the best positive-localization method by mean Dice in this calibration pass, but the absolute Dice is still low.
  - Many expanded methods select larger top fractions (`0.35`-`0.50`) for their best Dice, suggesting their signal is diffuse or that the model evidence is not tightly localized to pneumothorax masks.
  - Negative-evidence calibration should be read separately from positive localization: high avoidance means blue/suppressive evidence mostly stays outside the true lesion mask, which is generally healthier than maximizing blue-mask overlap.
  - These calibration files are now ready to use as frozen threshold candidates for the next `100`- or `200`-case all-method sanity evaluation before any `1000+` case run.

## 2026-05-15 - Iteration 22: Random Train-100 XAI Calibration, 5%-95% Fractions

- Updated calibration sampling after reviewing the `0.62` classifier cutoff result:
  - `0.62` is now the preferred train-calibrated operating threshold for the current TorchXRayVision baseline, replacing the earlier exploratory `0.61` in future classifier-outcome runs unless another operating point is intentionally selected.
  - `scripts/calibrate_cxr_xai_thresholds.py` now supports `--random-sample` and `--seed` so calibration cases can be sampled randomly and reproducibly instead of taking the first manifest rows.
  - The default behavior remains sequential unless `--random-sample` is explicitly passed.
- Runtime fix:
  - batched `occlusion_sensitivity` in `src/explainai_thesis/xai.py` so all-method calibration with occlusion is feasible on larger fraction grids;
  - also fixed a missing `defaultdict` import in the calibration script that surfaced during the wider run.
- Negative avoidance reminder:
  - `negative_mask_avoidance_fraction = 1 - negative_mask_overlap_fraction`;
  - it measures how much selected suppressive/blue evidence stays outside the true pneumothorax mask;
  - for negative evidence, higher avoidance is usually better because blue evidence inside the lesion would mean the model is treating annotated pneumothorax area as evidence against pneumothorax.
- Random positive-localization calibration command:

```bash
wsl.exe python3 scripts/calibrate_cxr_xai_thresholds.py --device auto --split train --max-positive 100 --random-sample --seed 20260515 --ig-steps 16 --gradshap-samples 8 --occlusion-patch-size 56 --occlusion-stride 56 --fractions 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95 --selection-metric dice --output-dir outputs/iter_22_xai_calibration_train100_random_all_methods_dice
```

- Random negative-evidence avoidance calibration command:

```bash
wsl.exe python3 scripts/calibrate_cxr_xai_thresholds.py --device auto --split train --max-positive 100 --random-sample --seed 20260515 --ig-steps 16 --gradshap-samples 8 --occlusion-patch-size 56 --occlusion-stride 56 --fractions 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95 --selection-metric negative_mask_avoidance_fraction --output-dir outputs/iter_22_xai_calibration_train100_random_all_methods_negative_avoidance
```

- Dice-selected fractions on random `100` positive train cases:

| Method | Selected fraction | Mean Dice |
|---|---:|---:|
| `grad_cam` | `0.15` | `0.050771` |
| `grad_cam_plus_plus` | `0.40` | `0.043228` |
| `consensus` | `0.30` | `0.040034` |
| `occlusion` | `0.40` | `0.034777` |
| `occlusion_positive` | `0.40` | `0.034230` |
| `occlusion_negative` | `0.35` | `0.032562` |
| `integrated_gradients` | `0.45` | `0.032322` |
| `gradient_shap` | `0.55` | `0.031276` |

- Negative-avoidance selected fractions for suppressive/signed evidence:

| Method | Selected fraction | Mean negative-mask avoidance |
|---|---:|---:|
| `grad_cam_negative` | `0.05` | `0.991260` |
| `occlusion_negative` | `0.10` | `0.985373` |
| `gradient_shap_negative` | `0.50` | `0.984785` |
| `grad_cam_plus_plus_negative` | `0.70` | `0.984683` |
| `integrated_gradients_negative` | `0.50` | `0.984612` |
| `consensus` | `0.95` | `0.984111` |

- Artifacts generated:
  - `outputs/iter_22_xai_calibration_train100_random_all_methods_dice/calibration_metrics.csv`
  - `outputs/iter_22_xai_calibration_train100_random_all_methods_dice/calibration_summary.csv`
  - `outputs/iter_22_xai_calibration_train100_random_all_methods_dice/selected_fractions.csv`
  - `outputs/iter_22_xai_calibration_train100_random_all_methods_dice/selected_fractions_by_metric.csv`
  - `outputs/iter_22_xai_calibration_train100_random_all_methods_negative_avoidance/calibration_metrics.csv`
  - `outputs/iter_22_xai_calibration_train100_random_all_methods_negative_avoidance/calibration_summary.csv`
  - `outputs/iter_22_xai_calibration_train100_random_all_methods_negative_avoidance/selected_fractions.csv`
  - `outputs/iter_22_xai_calibration_train100_random_all_methods_negative_avoidance/selected_fractions_by_metric.csv`
- Verification:
  - WSL `py_compile` passed for `src/explainai_thesis/xai.py` and `scripts/calibrate_cxr_xai_thresholds.py`;
  - both CUDA calibration runs completed successfully;
  - verified all expected Iteration 22 CSV artifacts exist.

## 2026-05-15 - Iteration 23: Stage 7 TorchXRayVision Baseline / Blank-Image Diagnostic

- Stage 7 question investigated:
  - why `100%` deletion in faithfulness curves could still show about `60%` pneumothorax probability;
  - whether this was caused by model output interpretation, preprocessing, baseline choice, or model calibration.
- Important implementation finding:
  - current classifier and faithfulness paths correctly treat TorchXRayVision outputs as logits and apply `sigmoid` for multi-label pneumothorax probability;
  - the larger problem was baseline interpretation: `xrv.datasets.normalize(array, 255)` maps image pixels approximately to `[-1024, 1024]`;
  - therefore a faithfulness baseline tensor of all `0.0` is not a black image. It is approximately the normalized value of a mid-gray pixel (`~128`), and this current model assigns that baseline a high pneumothorax probability.
- Added diagnostic script:
  - `scripts/diagnose_cxr_torchxray_baselines.py`;
  - compares `original_image`, historical `current_faithfulness_zero_tensor`, normalized black pixel `0`, normalized mid-gray pixel `128`, normalized white pixel `255`, blurred original, and case-mean baseline;
  - writes per-case `baseline_diagnostics.csv` and aggregate `baseline_diagnostics_summary.csv`.
- Diagnostic command:

```bash
wsl.exe python3 scripts/diagnose_cxr_torchxray_baselines.py --device auto --split test --max-cases 20 --output-dir outputs/iter_23_torchxray_baseline_diagnostic_test20
```

- Diagnostic summary on first `20` test cases:

| Variant | Mean pneumothorax probability |
|---|---:|
| `current_faithfulness_zero_tensor` | `0.633799` |
| `mid_gray_pixel_128_normalized` | `0.633781` |
| `case_mean_pixel_normalized` | `0.630749` |
| `original_image` | `0.569075` |
| `black_pixel_0_normalized` | `0.532825` |
| `blurred_original_normalized` | `0.526610` |
| `white_pixel_255_normalized` | `0.500022` |

- Interpretation:
  - the previous `~60%` fully-deleted score was largely a baseline artifact: full deletion to a zero tensor means replacement with a normalized mid-gray baseline, not with black pixels;
  - however, even true normalized black and white baselines remain near `0.50`, so the model is also poorly calibrated around the pneumothorax decision boundary for blank/out-of-distribution images;
  - the model can assign higher pneumothorax probability to non-anatomical constant images than to some real cases, reinforcing that the current `densenet121-res224-all` baseline is not clinically reliable for this dataset.
- Main faithfulness workflow update:
  - `scripts/run_cxr_torchxray_smoke.py` now supports `--faithfulness-baseline` with choices `zero_tensor`, `black`, `white`, and `case_mean`;
  - `zero_tensor` preserves historical behavior for comparison;
  - `black` uses normalized black image-space replacement (`-1024`) and is the preferred immediate replacement baseline for clearer deletion/insertion interpretation;
  - `faithfulness_curves.csv` now records the baseline name in a `baseline` column.
- Smoke verification command:

```bash
wsl.exe python3 scripts/run_cxr_torchxray_smoke.py --device auto --split test --max-positive 1 --ig-steps 2 --gradshap-samples 2 --occlusion-patch-size 112 --occlusion-stride 112 --max-overlays 1 --faithfulness-fractions 0.0,0.5,1.0 --faithfulness-baseline black --output-dir outputs/iter_23_faithfulness_black_baseline_smoke
```

- Smoke result:
  - with `--faithfulness-baseline black`, insertion at fraction `0.0` starts at `0.532825` instead of the historical `~0.6338` zero-tensor baseline;
  - deletion at fraction `1.0` also ends at `0.532825`, confirming the baseline is now actually used in the faithfulness perturbation path.
- Thesis-safe wording:
  - `Deletion/insertion curves depend strongly on the replacement baseline. In the initial implementation, a zero tensor in TorchXRayVision-normalized space did not represent a black image, but rather an approximately mid-gray normalized input, which the pretrained model scored highly for pneumothorax. Subsequent diagnostics showed that normalized constant-image baselines are out-of-distribution and can still receive near-threshold pneumothorax probabilities. Therefore, deletion/insertion results should be interpreted as model-behavior diagnostics under a specified perturbation baseline, not as direct clinical probability estimates for realistic images.`
- Recommended next decision:
  - use `--faithfulness-baseline black` for the next sanity evaluation to avoid the misleading mid-gray zero-tensor baseline;
  - optionally compare `black` versus `blurred` later, because blurred original images may be a more natural medical-image baseline but require a separate implementation in the main faithfulness runner.
