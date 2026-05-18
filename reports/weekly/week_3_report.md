# Weekly Progress Report 2

Reporting period: 2026-05-15 to 2026-05-21

## Week 2 Update - Larger Real-Data XAI Evaluation

The next technical step was executed on the real Kaggle/SIIM-style pneumothorax dataset using TorchXRayVision and mask-based explanation metrics.

Command:

```bash
wsl.exe --cd /mnt/c/Users/Dmytro.Valantsevych/Downloads/master_thesis_draft_explainAI python3 scripts/run_cxr_torchxray_smoke.py --device auto --max-positive 50 --ig-steps 16 --max-overlays 12 --output-dir outputs/cxr_torchxray_week2_50
```

Completed result:
- evaluated 50 positive test cases with available pneumothorax masks;
- generated Grad-CAM, Integrated Gradients, and consensus heatmaps;
- exported overlays for the first 12 cases;
- wrote per-case metrics to `outputs/cxr_torchxray_week2_50/metrics.csv`;
- wrote method-level aggregate metrics to `outputs/cxr_torchxray_week2_50/metrics_summary.csv`.

Output examples included in this report are limited to the Week 2 pre-SHAP stage. Later classifier-outcome screening and high-stability diagnostics are intentionally reserved for the Week 3 report.

Aggregate localization results on the 50-case subset:

| Method | N | Mean IoU | Mean Dice | Pointing Hit Rate | Mean Precision-at-15% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grad-CAM | 50 | 0.0213 | 0.0400 | 0.0000 | 0.0234 |
| Integrated Gradients | 50 | 0.0147 | 0.0282 | 0.0200 | 0.0168 |
| Consensus | 50 | 0.0213 | 0.0400 | 0.0000 | 0.0234 |

Built-in visual examples from the same Week 2 run:

| Example | Grad-CAM | Integrated Gradients | Consensus |
| --- | --- | --- | --- |
| Sample 00 | ![Week 2 sample 00 Grad-CAM](../../outputs/cxr_torchxray_week2_50/sample_00_grad_cam.png) | ![Week 2 sample 00 Integrated Gradients](../../outputs/cxr_torchxray_week2_50/sample_00_integrated_gradients.png) | ![Week 2 sample 00 consensus](../../outputs/cxr_torchxray_week2_50/sample_00_consensus.png) |
| Sample 01 | ![Week 2 sample 01 Grad-CAM](../../outputs/cxr_torchxray_week2_50/sample_01_grad_cam.png) | ![Week 2 sample 01 Integrated Gradients](../../outputs/cxr_torchxray_week2_50/sample_01_integrated_gradients.png) | ![Week 2 sample 01 consensus](../../outputs/cxr_torchxray_week2_50/sample_01_consensus.png) |

Interpretation:
- the first larger run confirms that uncalibrated explanations from the pretrained TorchXRayVision model localize pneumothorax weakly on this subset;
- Grad-CAM and the simple consensus method performed similarly because consensus is currently dominated by the Grad-CAM spatial pattern;
- Integrated Gradients produced one pointing-game hit but lower average overlap;
- the result strengthens the thesis motivation: explanation maps should be evaluated quantitatively and clinically rather than accepted as visually plausible.

Next technical step:
- add one stronger CAM-family explanation method, preferably Grad-CAM++, while keeping heavier SHAP-style methods for a later iteration;
- run threshold calibration on a validation subset;
- compare calibrated explanation masks on held-out test cases;
- decide whether the classifier should be fine-tuned specifically for pneumothorax before final XAI comparison.

## Week 2 Update - Classifier Baseline Check

To interpret the weak XAI localization, the pretrained TorchXRayVision classifier was evaluated directly on the same pneumothorax manifest.

Test split command:

```bash
wsl.exe --cd /mnt/c/Users/Dmytro.Valantsevych/Downloads/master_thesis_draft_explainAI python3 scripts/evaluate_cxr_torchxray_model.py --device auto --split test --batch-size 64 --output-dir outputs/cxr_torchxray_model_eval_test
```

Test split result:
- N = 1,372, with 290 positive and 1,082 negative cases;
- ROC AUC = 0.7711;
- average precision = 0.4120;
- default threshold 0.5: accuracy 0.2114, sensitivity 1.0000, specificity 0.0000, F1 0.3490;
- best-F1 threshold in the sweep: 0.61, with accuracy 0.5940, sensitivity 0.8931, specificity 0.5139, F1 0.4819.

Interpretation:
- TorchXRayVision is not random on this dataset because AUC is about 0.77;
- the default sigmoid threshold is not calibrated for this dataset and predicts all test cases as positive;
- classification quality is moderate, not strong enough to treat as a final pneumothorax model without calibration or fine-tuning;
- weak XAI localization should therefore be interpreted as both a model-match/calibration problem and an explanation-localization problem.

Output selection note:
- The Week 2 final report should use `outputs/cxr_torchxray_week2_50` as the primary visual-output example because it contains the first real CXR masked explanations using Grad-CAM, Integrated Gradients, and consensus.
- The classifier baseline output `outputs/cxr_torchxray_model_eval_test` should be used as supporting evidence for the calibration and model-match discussion.
- Iterations 26, 27, and 28 should not be presented as Week 2 outputs because they belong to the later Week 3 story: broader classifier-outcome screening, representative case mining, and high-stability diagnostics.

## Background Carried Over From Week 1

## Completed Work

- Created the initial project structure and working documents.
- Created thesis skeleton, experiment protocol, supervisor one-pager, data handling note, and week-1 report template.
- Installed the real-data and XAI Python stack after sandbox network approval.
- Verified imports for PyTorch, torchvision, pandas, scikit-learn, matplotlib, OpenCV, pydicom, nibabel, Captum, SHAP, and pytorch-grad-cam.
- Implemented a dependency-light synthetic lesion smoke test.
- Ran the smoke test successfully: synthetic classification accuracy reached 1.000, and Grad-CAM, Integrated Gradients, and consensus heatmaps were exported.
- Created a generic PNG image/mask manifest builder for the first real pneumothorax dataset.

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

## Research Methodology / MVP

Initial MVP:
- a reproducible Python environment;
- synthetic smoke test proving that classification, heatmap generation, overlay export, and localization metrics work end to end;
- manifest builder for image/mask datasets;
- defined primary dataset and next model baseline.

Research methodology used at this stage:
- start with a controlled synthetic task to verify that the explanation pipeline behaves correctly when the true lesion location is known;
- move to a public chest X-ray pneumothorax dataset with image-level labels and segmentation masks, so explanation quality can be measured objectively;
- use a pretrained TorchXRayVision DenseNet baseline first, rather than training a new model immediately, to separate pipeline feasibility from later model-improvement work;
- compare explanation maps with masks using IoU, Dice, pointing-game hit rate, and precision-at-fraction;
- treat visual overlays as qualitative evidence only after the quantitative mask-based checks are available;
- keep the initial method set small: Grad-CAM, Integrated Gradients, and a simple consensus heatmap.

Next experimental MVP:
- run a TorchXRayVision model or DenseNet-style chest X-ray baseline on the Kaggle pneumothorax dataset;
- generate first real-data Grad-CAM, Integrated Gradients, and consensus outputs;
- compare heatmaps against pneumothorax masks.

## Initial MVP Flow and Smoke Test Results

The initial MVP validates the technical flow before real patient or Kaggle data are used. A synthetic lesion dataset is generated with binary labels and pixel masks. A small CNN is trained to classify whether a lesion is present. Then three explanation outputs are produced: Grad-CAM, Integrated Gradients, and a simple consensus heatmap. Each heatmap is normalized, overlaid on the image, thresholded, and compared with the known lesion mask using IoU, Dice, pointing-game hit rate, and precision-at-fraction.

Smoke test command:

```bash
python3 scripts/run_smoke_test.py --device auto
```

Smoke test result:
- synthetic classification accuracy: 1.000;
- all tested explanation methods localized the lesion peak inside the mask on the inspected positive samples;
- consensus heatmap gave the best mean localization metrics in this synthetic test: IoU 0.1628, Dice 0.2730, pointing hit 1.0000, precision-at-fraction 0.1628;
- generated outputs are stored in `outputs/smoke_test/`.

Sample smoke-test overlays are shown below. Red indicates model attribution; green indicates the known lesion mask.

| Grad-CAM | Integrated Gradients | Consensus |
| --- | --- | --- |
| ![Sample 00 Grad-CAM](../../outputs/smoke_test/sample_00_grad_cam.png) | ![Sample 00 Integrated Gradients](../../outputs/smoke_test/sample_00_integrated_gradients.png) | ![Sample 00 Consensus](../../outputs/smoke_test/sample_00_consensus.png) |
| ![Sample 04 Grad-CAM](../../outputs/smoke_test/sample_04_grad_cam.png) | ![Sample 04 Integrated Gradients](../../outputs/smoke_test/sample_04_integrated_gradients.png) | ![Sample 04 Consensus](../../outputs/smoke_test/sample_04_consensus.png) |

## First Real-Data TorchXRayVision Result

The primary Kaggle pneumothorax dataset was downloaded locally and converted into a manifest:

```bash
wsl.exe --cd /mnt/c/Users/Dmytro.Valantsevych/Downloads/master_thesis_draft_explainAI python3 scripts/build_manifest.py data_local/cxr_pneumothorax --output data/cxr_pneumothorax_manifest.csv
```

Manifest result:
- total rows: 12,047;
- rows with masks: 12,047;
- labels: 9,378 negative and 2,669 positive.

A first TorchXRayVision DenseNet pass was then run on 6 positive test cases:

```bash
wsl.exe --cd /mnt/c/Users/Dmytro.Valantsevych/Downloads/master_thesis_draft_explainAI python3 scripts/run_cxr_torchxray_smoke.py --device auto --max-positive 6 --ig-steps 16
```

Result:
- TorchXRayVision `densenet121-res224-all` ran on CUDA;
- Grad-CAM, Integrated Gradients, and consensus heatmaps were generated;
- metrics and overlays were written to `outputs/cxr_torchxray_smoke/`;
- initial uncalibrated localization was low on this tiny sample, with pointing-game hits equal to 0 for the first inspected cases.

Interpretation:
- this is an early feasibility result, not a final benchmark;
- the result supports the research problem because model pneumothorax scores can be produced while explanation localization remains weak or inconsistent;
- the next step is to run a larger subset, calibrate thresholds, and compare methods more rigorously.

## Supervisor Feedback and Open Questions

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

Actual answer after the Week 2 experiments:
- The first real-data outputs are Grad-CAM, Integrated Gradients, and consensus heatmaps for pneumothorax-positive chest X-rays.
- The outputs can be saved as overlays for visual inspection and as thresholded masks for quantitative comparison with pneumothorax masks.
- On the early real-data runs, the pretrained TorchXRayVision model produced valid pneumothorax scores and heatmaps, but the uncalibrated localization metrics were weak.
- This means the research question remains valid: the project should not assume that a plausible-looking heatmap is clinically meaningful.
- The next answer to test is whether calibration, a stronger CAM-family method, and larger positive/negative evaluation make the explanations more stable before adding heavier later methods.

## Hypotheses Before Testing

- H1: Different XAI methods will produce different localization quality on the same classifier and dataset.
- H2: High classification performance will not guarantee good explanation localization.
- H3: Mask-calibrated thresholding or consensus heatmaps will improve localization compared with at least one individual explanation method.
- H4: Quantitative localization metrics will correlate with radiologist usefulness scores, but not perfectly.
- H5: XAI method rankings may differ between X-ray pneumothorax and CT hemorrhage tasks.

## Artifacts

- Progress memory: `docs/progress.md`
- Experiment protocol: `docs/experiment_protocol.md`
- Thesis skeleton: `thesis/thesis_skeleton.md`
- Data handling note: `data/README.md`
- Dataset source plan: `docs/dataset_sources.md`
- Environment checker: `scripts/check_environment.py`
- Smoke test: `scripts/run_smoke_test.py`
- Manifest builder: `scripts/build_manifest.py`
- Real-data TorchXRayVision smoke test: `scripts/run_cxr_torchxray_smoke.py`
- CXR manifest: `data/cxr_pneumothorax_manifest.csv`
- Smoke-test outputs: `outputs/smoke_test/`
- Real-data CXR smoke-test outputs: `outputs/cxr_torchxray_smoke/`

## Risks and Challenges

- Dataset download/setup may take longer than expected; Kaggle credentials are not yet configured locally.
- CT pilot may require manual annotation time.
- Too many XAI methods may over-expand the scope.
- Local patient data use requires strict anonymization.

Why these risks matter:
- Without Kaggle data, real-data metrics cannot be produced.
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

- Run the TorchXRayVision evaluation beyond the first positive-only smoke tests, including a larger positive subset and later positive/negative classifier-outcome checks.
- Aggregate real-data explanation metrics by method and use them to identify representative good and poor localization cases.
- Add one stronger CAM-family method, preferably Grad-CAM++, before considering heavier SHAP-style methods in a later report.
- Calibrate heatmap thresholds on validation masks and compare calibrated masks on held-out cases.
- Evaluate whether the pretrained TorchXRayVision classifier is sufficiently calibrated for pneumothorax or whether thresholding/fine-tuning/model comparison is needed.
- Prepare the Week 3 report around the work that follows this stage: larger outcome-based evaluation, candidate case selection, higher-stability diagnostics, and the decision about whether weak localization is model-specific.
- Build the literature matrix.
- Draft Chapter 1 and start Chapter 2.
