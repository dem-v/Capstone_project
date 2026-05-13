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

Aggregate localization results on the 50-case subset:

| Method | N | Mean IoU | Mean Dice | Pointing Hit Rate | Mean Precision-at-15% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grad-CAM | 50 | 0.0213 | 0.0400 | 0.0000 | 0.0234 |
| Integrated Gradients | 50 | 0.0147 | 0.0282 | 0.0200 | 0.0168 |
| Consensus | 50 | 0.0213 | 0.0400 | 0.0000 | 0.0234 |

Interpretation:
- the first larger run confirms that uncalibrated explanations from the pretrained TorchXRayVision model localize pneumothorax weakly on this subset;
- Grad-CAM and the simple consensus method performed similarly because consensus is currently dominated by the Grad-CAM spatial pattern;
- Integrated Gradients produced one pointing-game hit but lower average overlap;
- the result strengthens the thesis motivation: explanation maps should be evaluated quantitatively and clinically rather than accepted as visually plausible.

Next technical step:
- add a stronger third explanation method, preferably Grad-CAM++ / Score-CAM or Captum GradientSHAP;
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

- Run the TorchXRayVision evaluation on a larger positive and negative subset.
- Aggregate real-data explanation metrics by method.
- Add one more explanation method, preferably Grad-CAM++ / Score-CAM or GradientSHAP.
- Calibrate heatmap thresholds on validation masks and test on held-out masks.
- Decide whether to fine-tune a pneumothorax-specific classifier or keep TorchXRayVision as the pretrained baseline.
- Build literature matrix.
- Draft Chapter 1 and start Chapter 2.
