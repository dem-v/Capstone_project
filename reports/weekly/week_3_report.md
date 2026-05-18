# Weekly Progress Report 3

Reporting period: 2026-05-15 to 2026-05-22

## Week 3 Update - Expanded XAI Method Set: GradientSHAP and Occlusion Sensitivity

Following the Week 2 finding that uncalibrated Grad-CAM and Integrated Gradients localized SIIM pneumothorax weakly, Week 3 expanded the XAI method set with two additional families and broadened the polarity model to make positive, negative, and magnitude evidence first-class outputs.

Iteration 17 added:
- `GradientSHAP` via Captum, exposed as `gradient_shap` (magnitude, violet), `gradient_shap_positive` (red), `gradient_shap_negative` (blue), and `gradient_shap_signed` (red positive with blue negative diagnostic overlay).
- `Occlusion Sensitivity` with patch sweep, exposed as `occlusion` (magnitude, violet), `occlusion_positive` (red, regions whose occlusion reduces the pneumothorax logit), and `occlusion_negative` (blue, regions whose occlusion increases the pneumothorax logit).
- Grad-CAM++ with positive/negative polarity variants.
- Runtime controls `--gradshap-samples`, `--gradshap-stdevs`, `--occlusion-patch-size`, `--occlusion-stride`.
- Faithfulness plot family-splits and zoomed y-axis variants (`faithfulness_curves_cam_family.png`, `..._integrated_gradients_family.png`, `..._gradient_shap_family.png`, `..._occlusion_family.png`, `faithfulness_auc_bars.png`).
- Negative-evidence diagnostics `negative_mask_overlap_fraction` and `negative_mask_avoidance_fraction`, separated from positive localization metrics on the rationale that suppressive evidence should not be evaluated as if mask overlap were good.

Output: `outputs/iter_17_occlusion_gradshap_smoke/`, `outputs/iter_17_occlusion_gradshap_calibration_smoke/`.

**Figure 1.** Full XAI method-set inventory on iteration-17 smoke case `0_test_1`. Red indicates positive (supporting) evidence, blue indicates negative (suppressive) evidence, violet indicates magnitude (impact regardless of direction). Signed views combine red and blue diagnostic overlays on the same canvas.

| Family | Positive | Negative | Plus-plus / signed / magnitude |
| --- | --- | --- | --- |
| Grad-CAM | ![Grad-CAM positive](../../outputs/iter_17_occlusion_gradshap_smoke/case_000_0_test_1/grad_cam.png) | ![Grad-CAM negative](../../outputs/iter_17_occlusion_gradshap_smoke/case_000_0_test_1/grad_cam_negative.png) | ![Grad-CAM++ positive](../../outputs/iter_17_occlusion_gradshap_smoke/case_000_0_test_1/grad_cam_plus_plus.png) ![Grad-CAM++ negative](../../outputs/iter_17_occlusion_gradshap_smoke/case_000_0_test_1/grad_cam_plus_plus_negative.png) |
| Integrated Gradients | ![IG positive](../../outputs/iter_17_occlusion_gradshap_smoke/case_000_0_test_1/integrated_gradients_positive.png) | ![IG negative](../../outputs/iter_17_occlusion_gradshap_smoke/case_000_0_test_1/integrated_gradients_negative.png) | ![IG magnitude](../../outputs/iter_17_occlusion_gradshap_smoke/case_000_0_test_1/integrated_gradients.png) ![IG signed](../../outputs/iter_17_occlusion_gradshap_smoke/case_000_0_test_1/integrated_gradients_signed.png) |
| GradientSHAP | ![GradientSHAP positive](../../outputs/iter_17_occlusion_gradshap_smoke/case_000_0_test_1/gradient_shap_positive.png) | ![GradientSHAP negative](../../outputs/iter_17_occlusion_gradshap_smoke/case_000_0_test_1/gradient_shap_negative.png) | ![GradientSHAP magnitude](../../outputs/iter_17_occlusion_gradshap_smoke/case_000_0_test_1/gradient_shap.png) ![GradientSHAP signed](../../outputs/iter_17_occlusion_gradshap_smoke/case_000_0_test_1/gradient_shap_signed.png) |
| Occlusion Sensitivity | ![Occlusion positive](../../outputs/iter_17_occlusion_gradshap_smoke/case_000_0_test_1/occlusion_positive.png) | ![Occlusion negative](../../outputs/iter_17_occlusion_gradshap_smoke/case_000_0_test_1/occlusion_negative.png) | ![Occlusion magnitude](../../outputs/iter_17_occlusion_gradshap_smoke/case_000_0_test_1/occlusion.png) ![Consensus](../../outputs/iter_17_occlusion_gradshap_smoke/case_000_0_test_1/consensus.png) |

Interpretation:
- The expanded method set surfaced visible disagreement between Grad-CAM (gradient-weighted activation), Integrated Gradients (path-integrated input attribution), GradientSHAP (sampled-baseline gradient attribution), and Occlusion Sensitivity (causal patch perturbation), confirming Week 2 H1 in a stronger form.
- The clinical-implausibility concern carried from Week 2 sharpened: many positive-evidence maps show attribution on ribs, mediastinum, costophrenic regions, or even outside the body, which is consistent with a baseline model that produces moderate classifier signal from broad cues rather than localized pneumothorax features.

## Week 3 Update - Recalibrated Top-Fractions for the Expanded Method Set

Iteration 18 reworked calibration so that selected thresholds are no longer one global fraction but per-method per-metric:
- `calibration_metrics.csv` and `calibration_summary.csv` now include negative-evidence diagnostics.
- `selected_fractions_by_metric.csv` is written in addition to `selected_fractions.csv`, so calibrated fractions can be inspected separately for `Dice`, `IoU`, `precision_at_fraction`, `pointing_hit`, and the new negative-evidence diagnostics.
- `--selection-metric` accepts `negative_mask_avoidance_fraction` and `negative_mask_overlap_fraction` for intentional blue/suppressive-evidence calibration.
- Output naming convention updated so every per-case PNG includes the source X-ray stem.

Iteration 22 ran a Random Train-100 XAI calibration across the 0.05-0.95 fraction sweep, producing the first thesis-grade per-method calibrated fractions used by all subsequent runs.

**Table 1.** Per-method optimal top-fraction selected by the IoU criterion on the Random Train-100 calibration set. Source: `outputs/iter_22_xai_calibration_train100_random_all_methods_dice/selected_fractions_by_metric.csv`. The 0.10-0.85 spread is direct evidence that a single global fraction would mis-threshold roughly half the method set.

| Method | Optimal fraction (IoU) | Mean IoU at optimum |
| --- | ---: | ---: |
| `grad_cam` | 0.15 | 0.0274 |
| `grad_cam_negative` | 0.85 | 0.0156 |
| `grad_cam_plus_plus` | 0.40 | 0.0230 |
| `grad_cam_plus_plus_negative` | 0.10 | 0.0197 |
| `integrated_gradients` | 0.45 | 0.0168 |
| `integrated_gradients_positive` | 0.25 | 0.0160 |
| `integrated_gradients_negative` | 0.30 | 0.0156 |
| `integrated_gradients_signed` | 0.25 | 0.0160 |
| `gradient_shap` | 0.55 | 0.0163 |
| `gradient_shap_positive` | 0.35 | 0.0157 |
| `gradient_shap_negative` | 0.55 | 0.0154 |
| `gradient_shap_signed` | 0.35 | 0.0157 |
| `occlusion` | 0.40 | 0.0184 |
| `occlusion_positive` | 0.40 | 0.0180 |
| `occlusion_negative` | 0.35 | 0.0170 |
| `consensus` | 0.30 | 0.0211 |

Interpretation:
- Positive/red methods are calibrated against lesion-localization metrics (`Dice`, `IoU`).
- Negative/blue methods are calibrated against avoidance, not overlap, because blue evidence overlapping the lesion is a concerning behavior, not a useful one.
- Faithfulness insertion/deletion remains a separate model-behavior evaluation and is not used as a calibration target.
- The 8-fold spread between `grad_cam_plus_plus_negative` at 0.10 and `grad_cam_negative` at 0.85 directly supports the new H10 hypothesis that one global fraction is inappropriate for the current method panel.

## Week 3 Update - Faithfulness Baseline Diagnostic

Iteration 23 ran a Stage-7 TorchXRayVision baseline diagnostic on blank and blurred inputs to surface a faithfulness-curve interpretation hazard: the previous `zero_tensor` deletion baseline was not a clinically meaningful black image in the normalized TorchXRayVision input space, and even a blank input could score around 60% pneumothorax. The diagnostic informed the decision to default future faithfulness runs to `--faithfulness-baseline black` and treat `zero_tensor` only as a historical option.

**Table 2.** Mean pneumothorax probability of TorchXRayVision `densenet121-res224-all` on 20 perturbed-input variants. Source: `outputs/iter_23_torchxray_baseline_diagnostic_test20/baseline_diagnostics_summary.csv`. The historical `current_faithfulness_zero_tensor` baseline scores `0.634` on a uniform input, meaning the deletion-curve floor was a baseline artifact rather than preserved pathology signal.

| Variant | n | Mean probability | Std |
| --- | ---: | ---: | ---: |
| `original_image` | 20 | 0.569 | 0.048 |
| `black_pixel_0_normalized` | 20 | 0.533 | 0.000 |
| `white_pixel_255_normalized` | 20 | 0.500 | 0.000 |
| `blurred_original_normalized` | 20 | 0.527 | 0.033 |
| `case_mean_pixel_normalized` | 20 | 0.631 | 0.004 |
| `mid_gray_pixel_128_normalized` | 20 | 0.634 | 0.000 |
| `current_faithfulness_zero_tensor` (historical) | 20 | **0.634** | 0.000 |

Interpretation:
- Properly normalized `black_pixel_0_normalized` and `white_pixel_255_normalized` reach 0.50-0.53 on a uniform input, close to the model's prior.
- `zero_tensor` and `mid_gray_pixel_128_normalized` sit at 0.634, which is roughly where the model predicts pneumothorax for many real cases. Deletion curves using `zero_tensor` therefore cannot drop below this artificial floor, distorting AUC comparisons.
- All faithfulness runs after iteration 23 default to `--faithfulness-baseline black`. The `zero_tensor` option remains in the script for reproducing historical results but is annotated as not recommended.

## Week 3 Update - GradientSHAP Stability

Iteration 25 inspected `case_019`, which had earlier appeared contradictory under `--gradshap-samples 8`. At `--gradshap-samples 64`, the GradientSHAP map stabilized and no longer disagreed with `Grad-CAM++` in the way that the low-sample run had suggested. The finding established a stability rule: `--gradshap-samples 8` is acceptable only for broad exploratory screening, and thesis-worthy or contradictory cases must be rerun at 64 (or 128 if 64 still drifts).

## Week 3 Update - Baseline Consolidation Screening

Iteration 26 consolidated the TorchXRayVision baseline before committing to a stronger second model. The pretrained `densenet121-res224-all` classifier was screened on a 5000-case random sample of the manifest using the current calibrated cutoff `0.62`.

Commands:

```bash
wsl.exe python3 scripts/evaluate_cxr_torchxray_model.py --device auto --split test --batch-size 128 --threshold 0.62 --max-cases 5000 --random-sample --seed 20260517 --output-dir outputs/iter_26_torchxray_test5000_random_classifier_screen

wsl.exe python3 scripts/evaluate_cxr_torchxray_model.py --device auto --split any --batch-size 128 --threshold 0.62 --max-cases 5000 --random-sample --seed 20260517 --output-dir outputs/iter_26_torchxray_any5000_random_classifier_screen
```

Classifier screening results:

| Run | Rows | Positives | Negatives | ROC AUC | Average precision | Outcome counts at 0.62 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| iter_26 test 5000 | 1372 | 290 | 1082 | 0.7711 | 0.4120 | TP=250, FP=508, TN=574, FN=40 |
| iter_26 any 5000 | 5000 | 1132 | 3868 | 0.7773 | 0.4433 | TP=1016, FP=1781, TN=2087, FN=116 |

Interpretation:
- AUC ≈ 0.77 confirms the baseline classifier is informative on this dataset, but moderate rather than strong.
- The test split contains only 1372 rows in total, so `--max-cases 5000 --split test` evaluates the full test set.
- At threshold 0.62, the test split provides only 40 false negatives, making 100-per-outcome balanced runs impossible on test-only. `--split any` provides FN=116 and was selected for the consolidated balanced run, labelled as exploratory rather than final held-out reporting.

## Week 3 Update - Long Balanced Classifier-Outcome Run With Resume Support

Iteration 27 launched the consolidated balanced run with the full all-method XAI set, including GradientSHAP and Occlusion Sensitivity, on up to 100 cases per `tp`/`fp`/`tn`/`fn` outcome. The script was extended with three durability features required by the run length:
- Six-line live progress display with scan, selection, outcome counts, recent throughput, and ETA estimated from completed selected cases rather than scanned candidates.
- Checkpoint writes after every selected case to `cases.csv`, `threshold_metrics.csv`, and `progress.json`.
- `--resume` reads the existing checkpoint, reconstructs the per-outcome counts, replays the deterministic candidate order with the same `--random-sample --seed`, skips images already in `cases.csv`, and continues from the next `sample_index` without duplicating completed rows.

Reference command (current canonical long-run pattern):

```bash
wsl.exe python3 scripts/visualize_cxr_classifier_outcome_thresholds.py --device auto --split any --max-cases 5000 --random-sample --seed 20260517 --threshold 0.62 --max-per-outcome 100 --ig-steps 8 --gradshap-samples 8 --occlusion-patch-size 56 --occlusion-stride 56 --fractions 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50 --progress-every 10 --checkpoint-every 1 --output-dir outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods
```

For restart after a stop, append `--resume`.

Output structure:
- `cases.csv` and `threshold_metrics.csv` at the run root.
- One folder per source X-ray under `tp/`, `fp/`, `tn/`, `fn/`.
- Every per-case PNG includes the source X-ray stem in the filename so a copied PNG remains traceable without its parent folder.

Final outcome counts at threshold `0.62` from `outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/outcome_summary.csv`: `n=5000`, `tp=100`, `fp=100`, `tn=100`, `fn=100`. The balanced 100-per-outcome target was reached on the `--split any` candidate pool.

**Figure 2.** One representative case per classifier outcome from the balanced run, showing Grad-CAM (red positive evidence) and `integrated_gradients_signed` (red positive plus blue negative overlay) side by side. Same XAI method panel on cases the model classified correctly (TP, TN) and incorrectly (FP, FN) makes visible how XAI behavior covaries with classifier outcome.

| Outcome | Source | Grad-CAM | IG signed |
| --- | --- | --- | --- |
| TP | `585_test_1` | ![TP Grad-CAM](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tp/case_001_tp_585_test_1/585_test_1_grad_cam.png) | ![TP IG signed](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tp/case_001_tp_585_test_1/585_test_1_integrated_gradients_signed.png) |
| FP | `9009_train_0` | ![FP Grad-CAM](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fp/case_000_fp_9009_train_0/9009_train_0_grad_cam.png) | ![FP IG signed](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fp/case_000_fp_9009_train_0/9009_train_0_integrated_gradients_signed.png) |
| TN | `8080_train_0` | ![TN Grad-CAM](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tn/case_004_tn_8080_train_0/8080_train_0_grad_cam.png) | ![TN IG signed](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tn/case_004_tn_8080_train_0/8080_train_0_integrated_gradients_signed.png) |
| FN | `152_test_1` | ![FN Grad-CAM](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fn/case_089_fn_152_test_1/152_test_1_grad_cam.png) | ![FN IG signed](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fn/case_089_fn_152_test_1/152_test_1_integrated_gradients_signed.png) |

**Figure 3.** Faithfulness AUC summary across the full XAI method set on the Stage-5 Random Test-100 evaluation with `--faithfulness-baseline black`. Insertion AUC (probability gained as top-attributed pixels are restored) and deletion-drop AUC (probability lost as top-attributed pixels are removed) per method. Source: `outputs/iter_24_stage5_test100_random_all_methods_black_baseline/faithfulness_auc_bars.png`.

![Faithfulness AUC summary](../../outputs/iter_24_stage5_test100_random_all_methods_black_baseline/faithfulness_auc_bars.png)

Companion family-split curves for the same run are available at:
- `outputs/iter_24_stage5_test100_random_all_methods_black_baseline/faithfulness_curves_cam_family.png`
- `outputs/iter_24_stage5_test100_random_all_methods_black_baseline/faithfulness_curves_integrated_gradients_family.png`
- `outputs/iter_24_stage5_test100_random_all_methods_black_baseline/faithfulness_curves_gradient_shap_family.png`
- `outputs/iter_24_stage5_test100_random_all_methods_black_baseline/faithfulness_curves_occlusion_family.png`

The bar chart is the direct visual for hypothesis H8: cases where a method has high faithfulness AUC but low mask-localization scores (or vice versa) demonstrate that the two evaluations measure different things and should be reported separately.

## Week 3 Update - Review Candidate Mining

Iteration 28 added `scripts/select_cxr_review_candidates.py` to mine representative case lists from the consolidated balanced run. It reads `cases.csv` and `threshold_metrics.csv` from iteration 27 and writes ranked CSVs by category, plus a `run_selected_high_stability_diagnostics.ps1` shell script with one rerun command per selected case at the stability settings `--ig-steps 16 --gradshap-samples 64 --occlusion-patch-size 32 --occlusion-stride 12`.

Selected manual-review categories:
- best `tp` cases by `Dice`/`IoU`,
- suspicious `tp` cases with low localization despite positive prediction,
- `fp` cases with high classifier score (proxy for strong false-positive evidence; negatives have no mask-localization metrics),
- `fn` cases with relatively good localization but classifier score below threshold,
- positive-label cases with unusually high negative evidence inside the mask.

Output: `outputs/iter_28_review_candidate_selection/`. The 10 selected high-stability diagnostic reruns are pending manual execution.

**Table 3.** Top-10 selected manual-review cases mined from the iteration-27 balanced run. Source: `outputs/iter_28_review_candidate_selection/selected_manual_review_cases.csv`.

| Rank | Category | File | Outcome | Score | Best method | Best Dice | Best IoU | Negative method | Max negative overlap |
| ---: | --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: |
| 1 | best TP | `4052_train_1_.png` | tp | 0.6322 | `grad_cam` | 0.4667 | 0.3044 | `grad_cam_plus_plus_negative` | 0.2078 |
| 2 | best TP | `2541_train_1_.png` | tp | 0.6238 | `occlusion` | 0.3784 | 0.2334 | `occlusion_negative` | 0.2755 |
| 3 | suspicious TP | `1556_train_1_.png` | tp | 0.6229 | `integrated_gradients_positive` | 0.0031 | 0.0016 | `grad_cam_negative` | 0.0024 |
| 4 | suspicious TP | `335_train_1_.png` | tp | 0.6225 | `grad_cam` | 0.0068 | 0.0034 | `grad_cam_plus_plus_negative` | 0.0011 |
| 5 | high-score FP | `3027_train_0_.png` | fp | 0.6343 | n/a | n/a | n/a | n/a | n/a |
| 6 | high-score FP | `271_train_0_.png` | fp | 0.6323 | n/a | n/a | n/a | n/a | n/a |
| 7 | good-localization FN | `101_test_1_.png` | fn | 0.6122 | `grad_cam_plus_plus` | 0.5683 | 0.3969 | `gradient_shap_negative` | 0.2002 |
| 8 | good-localization FN | `3671_train_1_.png` | fn | 0.5830 | `grad_cam` | 0.4536 | 0.2934 | `occlusion_negative` | 0.3753 |
| 9 | high negative-in-mask | `2349_train_1_.png` | tp | 0.6276 | `grad_cam_plus_plus` | 0.3721 | 0.2286 | `grad_cam_negative` | 0.4880 |
| 10 | high negative-in-mask | `4228_train_1_.png` | fn | 0.6105 | `grad_cam_plus_plus` | 0.4032 | 0.2525 | `occlusion_negative` | 0.4104 |

### Best TP cases (Rank 1-2)

These cases anchor the upper bound of what the current TorchXRayVision baseline plus the current XAI panel can achieve on SIIM pneumothorax. Best Grad-CAM Dice on the full balanced run is 0.4667, which is order-of-magnitude better than the random-Test-100 mean reported in Week 2 but still well short of segmentation-grade.

**Figure 4a.** Rank 1, best TP, `4052_train_1`.

| Grad-CAM | Grad-CAM negative | GradientSHAP | Consensus |
| --- | --- | --- | --- |
| ![Rank 1 Grad-CAM](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tp/case_132_tp_4052_train_1/4052_train_1_grad_cam.png) | ![Rank 1 Grad-CAM neg](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tp/case_132_tp_4052_train_1/4052_train_1_grad_cam_negative.png) | ![Rank 1 GradientSHAP](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tp/case_132_tp_4052_train_1/4052_train_1_gradient_shap.png) | ![Rank 1 Consensus](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tp/case_132_tp_4052_train_1/4052_train_1_consensus.png) |

**Figure 4b.** Rank 2, best TP, `2541_train_1`.

| Grad-CAM | Occlusion | Occlusion negative | Consensus |
| --- | --- | --- | --- |
| ![Rank 2 Grad-CAM](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tp/case_292_tp_2541_train_1/2541_train_1_grad_cam.png) | ![Rank 2 Occlusion](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tp/case_292_tp_2541_train_1/2541_train_1_occlusion.png) | ![Rank 2 Occlusion neg](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tp/case_292_tp_2541_train_1/2541_train_1_occlusion_negative.png) | ![Rank 2 Consensus](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tp/case_292_tp_2541_train_1/2541_train_1_consensus.png) |

### Suspicious TP cases (Rank 3-4)

The model classifies these as pneumothorax-positive with confidence comparable to the best TP cases (scores 0.6225-0.6229 vs 0.6238-0.6322) but localizes essentially nowhere on the lesion (Dice 0.003-0.007 vs 0.38-0.47). Strong direct evidence for hypothesis H2.

**Figure 4c.** Rank 3, suspicious TP, `1556_train_1`.

| Grad-CAM | IG positive | IG signed | Consensus |
| --- | --- | --- | --- |
| ![Rank 3 Grad-CAM](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tp/case_066_tp_1556_train_1/1556_train_1_grad_cam.png) | ![Rank 3 IG positive](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tp/case_066_tp_1556_train_1/1556_train_1_integrated_gradients_positive.png) | ![Rank 3 IG signed](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tp/case_066_tp_1556_train_1/1556_train_1_integrated_gradients_signed.png) | ![Rank 3 Consensus](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tp/case_066_tp_1556_train_1/1556_train_1_consensus.png) |

**Figure 4d.** Rank 4, suspicious TP, `335_train_1`.

| Grad-CAM | Grad-CAM++ negative | GradientSHAP signed | Consensus |
| --- | --- | --- | --- |
| ![Rank 4 Grad-CAM](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tp/case_208_tp_335_train_1/335_train_1_grad_cam.png) | ![Rank 4 GC++ neg](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tp/case_208_tp_335_train_1/335_train_1_grad_cam_plus_plus_negative.png) | ![Rank 4 SHAP signed](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tp/case_208_tp_335_train_1/335_train_1_gradient_shap_signed.png) | ![Rank 4 Consensus](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tp/case_208_tp_335_train_1/335_train_1_consensus.png) |

### High-score FP cases (Rank 5-6)

False positives at the upper tail of classifier confidence. No ground-truth mask is available because the cases are label-negative, so quantitative localization metrics do not apply; the overlays surface what the model is responding to when it confidently calls a non-pneumothorax case positive.

**Figure 4e.** Rank 5, high-score FP, `3027_train_0`.

| Grad-CAM | IG signed | GradientSHAP | Occlusion |
| --- | --- | --- | --- |
| ![Rank 5 Grad-CAM](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fp/case_098_fp_3027_train_0/3027_train_0_grad_cam.png) | ![Rank 5 IG signed](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fp/case_098_fp_3027_train_0/3027_train_0_integrated_gradients_signed.png) | ![Rank 5 GradientSHAP](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fp/case_098_fp_3027_train_0/3027_train_0_gradient_shap.png) | ![Rank 5 Occlusion](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fp/case_098_fp_3027_train_0/3027_train_0_occlusion.png) |

**Figure 4f.** Rank 6, high-score FP, `271_train_0`.

| Grad-CAM | IG signed | GradientSHAP | Occlusion |
| --- | --- | --- | --- |
| ![Rank 6 Grad-CAM](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fp/case_128_fp_271_train_0/271_train_0_grad_cam.png) | ![Rank 6 IG signed](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fp/case_128_fp_271_train_0/271_train_0_integrated_gradients_signed.png) | ![Rank 6 GradientSHAP](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fp/case_128_fp_271_train_0/271_train_0_gradient_shap.png) | ![Rank 6 Occlusion](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fp/case_128_fp_271_train_0/271_train_0_occlusion.png) |

### Good-localization FN cases (Rank 7-8)

These cases score just below the 0.62 classifier threshold (`fn` outcome at this cutoff) yet produce stronger localization than the best TP cases. They flag cases where the classifier is the failure mode, not the XAI methods.

**Figure 4g.** Rank 7, good-localization FN, `101_test_1`.

| Grad-CAM++ | GradientSHAP negative | IG signed | Consensus |
| --- | --- | --- | --- |
| ![Rank 7 GC++](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fn/case_343_fn_101_test_1/101_test_1_grad_cam_plus_plus.png) | ![Rank 7 SHAP neg](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fn/case_343_fn_101_test_1/101_test_1_gradient_shap_negative.png) | ![Rank 7 IG signed](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fn/case_343_fn_101_test_1/101_test_1_integrated_gradients_signed.png) | ![Rank 7 Consensus](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fn/case_343_fn_101_test_1/101_test_1_consensus.png) |

**Figure 4h.** Rank 8, good-localization FN, `3671_train_1`.

| Grad-CAM | Occlusion | Occlusion negative | Consensus |
| --- | --- | --- | --- |
| ![Rank 8 Grad-CAM](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fn/case_285_fn_3671_train_1/3671_train_1_grad_cam.png) | ![Rank 8 Occlusion](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fn/case_285_fn_3671_train_1/3671_train_1_occlusion.png) | ![Rank 8 Occlusion neg](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fn/case_285_fn_3671_train_1/3671_train_1_occlusion_negative.png) | ![Rank 8 Consensus](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fn/case_285_fn_3671_train_1/3671_train_1_consensus.png) |

### High negative-evidence-inside-mask cases (Rank 9-10)

Negative attribution overlapping the ground-truth lesion mask is a concerning behavior: the model is producing suppressive evidence within the lesion itself. These cases probe hypothesis H6 from the inverse direction.

**Figure 4i.** Rank 9, high negative-in-mask, `2349_train_1`.

| Grad-CAM++ | Grad-CAM negative | IG negative | Consensus |
| --- | --- | --- | --- |
| ![Rank 9 GC++](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tp/case_295_tp_2349_train_1/2349_train_1_grad_cam_plus_plus.png) | ![Rank 9 Grad-CAM neg](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tp/case_295_tp_2349_train_1/2349_train_1_grad_cam_negative.png) | ![Rank 9 IG neg](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tp/case_295_tp_2349_train_1/2349_train_1_integrated_gradients_negative.png) | ![Rank 9 Consensus](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/tp/case_295_tp_2349_train_1/2349_train_1_consensus.png) |

**Figure 4j.** Rank 10, high negative-in-mask, `4228_train_1`.

| Grad-CAM++ | Occlusion negative | IG negative | Consensus |
| --- | --- | --- | --- |
| ![Rank 10 GC++](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fn/case_154_fn_4228_train_1/4228_train_1_grad_cam_plus_plus.png) | ![Rank 10 Occlusion neg](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fn/case_154_fn_4228_train_1/4228_train_1_occlusion_negative.png) | ![Rank 10 IG neg](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fn/case_154_fn_4228_train_1/4228_train_1_integrated_gradients_negative.png) | ![Rank 10 Consensus](../../outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/fn/case_154_fn_4228_train_1/4228_train_1_consensus.png) |

## Week 3 Update - Planning Pass: Code Quality Review, Pre-Mortem, and Refactor Plan

A code-quality, AI-smell, and optimization-assurance review (`docs/progress.md` 2026-05-18 entries) produced a phased refactor and protocol-completion plan now tracked in `docs/refactor_plan.md`. Key decisions documented in `AGENT.md`:
- Introduce a `SignedAttribution` four-view contract (`positive`, `negative`, `magnitude`, `signed`) for every polarity-supporting method. Each XAI method runs once per case; views are derived in microseconds. Expected per-case speedup roughly 2× for Grad-CAM, 3× for GradientSHAP, 3× for Occlusion, and 24-45× for IG combined with batched-step refactor.
- Add an orange/teal diverging color pair for signed scalar maps (`positive - negative`); preserves the documented red/blue/violet/green/yellow/cyan semantics already in use.
- Fix a likely `grad_cam_plus_plus` polarity double-flip bug. Negative Grad-CAM++ overlays are expected to materially differ after the fix; before/after comparison will be communicated to the supervisor.
- Add `Eigen-CAM` and `Score-CAM` to close the protocol gap; both inherit the four-view contract.
- Promote a diagnostic A/B (`Phase 1.7`) **before** further protocol-completion scope, to disambiguate whether the weak localization is caused by the baseline model or by the methods themselves. Stage A pairs five `torchxrayvision` weights with one out-of-family external model on the calibration cases.
- Versioned calibration (`Phase 1.2.5`): the SignedAttribution refactor invalidates pre-refactor calibrated top-fractions. A `v2` calibration is produced and stored separately from `v1`; only `v2` is used for held-out evaluation.
- CT pilot uses an off-the-shelf pretrained classifier and a small student-annotated subset, with a hour-1 model-availability fallback to qualitative external validation if no usable public CT model is found.
- Radiologist review tooling is a hybrid static HTML index plus CSV scoring template (`scripts/build_review_workbook.py`, planned), with binding rubric-clarity rules so that 100 cases at the rater's typical 1-3 minutes per case fit comfortably under 0.5 day of focused scoring.
- LIME and Captum infidelity/sensitivity are both kept on the menu as conditional add-ons rather than dropped: LIME (Phase 5.7) activates only if the rest of Phase 5 lands by 2026-06-01 and the buffer holds; Captum infidelity/sensitivity (Phase 5.6) activates either when Phase 5.5 is skipped or as a parallel add-on if Phase 5 finishes early.

## Hypotheses for Week 3 (Revised From the Week 2 Set)

Week 2 stated H1-H5 before SHAP and Occlusion were available, before the consolidated balanced run produced 100 cases per outcome, and before the planned signed-attribution refactor and diagnostic A/B. The Week 3 revision keeps all five carried-forward hypotheses, sharpens H1, H3, and H4, scopes H5 conditionally, and adds four new hypotheses (H6-H9) that the now-available data and the upcoming Phase 1.7 / 5.1 / 5.3 work make directly testable.

### Cross-method behavior

- **H1 (revised, sharpened)**: Different XAI methods will produce both different localization quality and different spatial patterns on the same classifier, quantifiable via per-case `agreement_score` (cosine similarity between signed maps from different methods). The presence of cases where two methods give high IoU on the same lesion but disagree spatially is itself a finding.
- **H6 (new)**: Negative attribution will systematically avoid the lesion mask more reliably than positive attribution overlaps it. Direct test on the consolidated balanced run via `negative_mask_avoidance_fraction` vs positive `IoU`/`Dice`/`precision_at_fraction` across all polarity-supporting methods.
- **H7 (new)**: Cross-method agreement (signed cosine similarity) will correlate with mask-based localization quality, enabling agreement to act as a mask-free reliability indicator. If the correlation holds, this is a methodological contribution; if not, it is a publishable negative result about reliability indicators in clinical XAI.

### Model and explanation alignment

- **H2 (kept)**: High classification performance does not guarantee good explanation localization. Already strongly supported on the consolidated run; `Phase 1.7` extends the supporting evidence from one classifier to multiple training distributions.
- **H8 (new)**: Faithfulness AUC (insertion or deletion) will correlate weakly or not at all with mask-based localization metrics. `AGENT.md` already states this informally as the "faithful to the model but clinically poorly localized" interpretation rule; the consolidated run has both sets of numbers and supports a direct paired-correlation test.
- **H9 (new)**: The poor localization on SIIM is cross-distribution-stable across training distributions in the TorchXRayVision DenseNet-121 family, and is not eliminated by one out-of-family external classifier. `Phase 1.7` Stage A is the direct test. Both outcomes are publishable: a positive result is a strong negative finding about pretrained large-CXR-multitask classifiers as a class; a negative result localizes the problem to training-distribution mismatch.

### Improvement and clinical assessment

- **H3a (revised, stronger thesis claim)**: Consensus heatmaps will outperform the best individual XAI method on held-out IoU/Dice in the formal improvement experiment. Falsifiable; pre-mortem already prepared a fallback Discussion narrative for the negative case.
- **H3b (revised, methodologically safer)**: Calibrated thresholding will produce a different per-method ranking on held-out cases than uncalibrated thresholding does, demonstrating that calibration is methodologically necessary even if no single method or consensus wins outright on every metric.
- **H4 (revised)**: Quantitative localization metrics will correlate moderately, not strongly, with radiologist usefulness scores, because mask-based metrics penalize lesion-size and lesion-shape mismatches that a radiologist may still rate as partially useful, and because radiologists incorporate anatomical context (apex region for pneumothorax) that pixel-level metrics ignore.

### Cross-modality (conditional)

- **H5 (kept, conditional)**: XAI method rankings may differ between X-ray pneumothorax and CT hemorrhage tasks. The Week-3 plan adds an hour-1 CT model-availability check in `Phase 5.4`; if no off-the-shelf CT pneumothorax-relevant classifier is found, the CT pilot falls back to qualitative external validation and H5 is downgraded to discussion-only with no original quantitative test.

## Artifacts

- Progress memory: `docs/progress.md`
- Agent guide: `AGENT.md`
- Refactor and protocol-completion plan: `docs/refactor_plan.md`
- Experiment protocol: `docs/experiment_protocol.md`
- Thesis skeleton: `thesis/thesis_skeleton.md`
- Expanded XAI methods: `src/explainai_thesis/xai.py` (`GradCAM`, `integrated_gradients`, `gradient_shap`, `occlusion_sensitivity`, `consensus_heatmap`)
- Main CXR XAI script: `scripts/run_cxr_torchxray_smoke.py`
- Classifier evaluation/threshold sweep: `scripts/evaluate_cxr_torchxray_model.py`
- Long balanced classifier-outcome run with resume: `scripts/visualize_cxr_classifier_outcome_thresholds.py`
- Review candidate mining: `scripts/select_cxr_review_candidates.py`
- XAI threshold calibration: `scripts/calibrate_cxr_xai_thresholds.py`
- Consolidated balanced output: `outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods/`
- Review candidate selection: `outputs/iter_28_review_candidate_selection/`

## Risks and Challenges (Updated)

Carried from Week 2:
- CT pilot may require manual annotation and a usable off-the-shelf classifier.
- Scope creep across too many XAI methods, datasets, or modeling directions.
- Local patient data anonymization discipline.

New in Week 3:
- The baseline TorchXRayVision classifier looks clinically weak for SIIM pneumothorax localization despite moderate ranking performance, which raises the risk that protocol-completion work decorates a broken foundation. Mitigation: the `Phase 1.7` diagnostic A/B runs before further protocol work and includes one out-of-family external model from the start.
- The `SignedAttribution` refactor changes underlying signed maps for Grad-CAM, IG, GradientSHAP, and Occlusion, invalidating pre-refactor calibrated top-fractions. Mitigation: versioned `v2` calibration step is mandatory immediately after the refactor; `v1` artifacts are preserved.
- The improvement experiment may show consensus is no better than the best individual method, undermining the thesis's "low-risk improvement" claim. Mitigation: both Discussion narratives (consensus improves vs. consensus does not improve) are drafted before the held-out evaluation runs.
- Long classifier-outcome runs may stop mid-run on the development machine. Mitigation: `--resume` was added in iteration 27 and verified against a partial run before the consolidated 100-per-outcome run was launched.
- GradientSHAP can produce unstable, contradictory maps at low sample counts. Mitigation: low-sample (8) screening is allowed for broad exploration only; thesis-worthy cases are rerun at 64 samples (or 128 if 64 drifts), and the rule is encoded in `AGENT.md`.

External-coordination risks:
- Earlier figures shown to the supervisor may have used buggy `grad_cam_plus_plus_negative` overlays. After the polarity fix lands in `Phase 1.1`, a before/after comparison will be sent proactively, framed as instrument calibration.
- Institutional disclosure policy for AI tooling (GPT-5.5, Codex, Claude Sonnet 4.6, Claude Opus 4.7, Junie) needs explicit confirmation with the supervisor this week so the methodology paragraph can be drafted before `2026-05-22`.

## Plan for Next Week

Execution of the refactor and protocol-completion plan from `docs/refactor_plan.md`, in deadline-anchored order:

- Phase 0 foundation: `pyproject.toml`, `pip install -e .`, remove `sys.path.insert`, `pytest` + `ruff` + `mypy` dev requirements, golden-output snapshot test.
- Phase 1 correctness: metric unit tests, manifest label-inference fix, `grad_cam_plus_plus` polarity fix, signed-attribution four-view contract for Grad-CAM, Grad-CAM++, IG, GradientSHAP, Occlusion, and Consensus.
- Phase 1.2.5 versioned calibration: regenerate calibrated top-fractions as `v2` against the refactored signed maps; preserve `v1`.
- Compressed Phase 2: `MethodSpec` registry, `src/explainai_thesis/cxr/io.py`, `load_classifier(name)` seam.
- Phase 3 performance: batched IG, vectorized Occlusion, dropped wasted `normalize_map` calls, vectorized mask-contour.
- Phase 1.7 Stage A diagnostic A/B: five in-family TorchXRayVision weights plus one out-of-family external model, on the calibration cases. Stage B decision rule writes back into `AGENT.md`.
- Phase 5.1 add `Eigen-CAM` and `Score-CAM` via the new registry.
- Phase 5.2 improvement-experiment script with pre-drafted Discussion narratives for both outcomes.
- Phase 5.4 CT pilot: hour-1 model-availability check; scaffold or qualitative fallback.
- Phase 5.3 radiologist review workbook with binding rubric-clarity rules; 100-case scoring pass on the consolidated CXR run.
- Phase 4 minimum: `run_meta.json` stamping in every output, `load_classifier` seam audit.
- Parallel thesis writing track starting 2026-05-19: methodology chapter as soon as Phase 1.2 lands, results chapter populates from CSVs as each phase finishes, Discussion and Conclusions wait for the improvement experiment.

Deadline anchors:
- Phase 0 + Phase 1 by 2026-05-21.
- Phase 1.2.5 + compressed Phase 2 by 2026-05-23.
- Phase 3 performance by 2026-05-24.
- Phase 1.7 Stage A diagnostic A/B and Stage B decision by 2026-05-27.
- Phase 5.1 + 5.2 by 2026-05-28.
- Phase 5.4 CT pilot or qualitative fallback by 2026-05-31.
- Phase 5.3 radiologist review by 2026-06-02.
- Optional Phase 5.5 full second-model protocol (conditional on Phase 1.7 outcome) by 2026-06-03.
- Final figures, tables, and thesis-draft completion by 2026-06-04.
