# Weekly Progress Report 5

Reporting period: 2026-05-29 to 2026-06-03

## Week 5 Summary

Week 5 focused on finishing the final experiment track and converting it into thesis-ready evidence. The CXR method panel was expanded with `Eigen-CAM` and `Score-CAM`, fresh v3 calibration artifacts were generated, held-out improvement experiments were completed for the selected DenseNet-chex and ResNet-50 CXR baselines, and the CT pilot moved from a conditional plan to a completed cross-modality experiment on real PhysioNet `ct-ich` data. The original DenseNet-all artifacts remain historical continuity evidence, but DenseNet-chex supersedes DenseNet-all as the stronger DenseNet baseline.

The main final interpretation is deliberately cautious: there is no general law that `consensus` improves localization. On CXR, consensus is approximately as good as its strongest individual constituents and behaves differently across DenseNet and ResNet. The cleanest consensus advantage appears in the CT pilot for pointing-game peak localization, but that result is modality- and method-set-specific.

## 1. Completed Project Updates

### 1.1 Expanded XAI method panel

Week 5 completed the Phase 5.1 method expansion:

- `eigen_cam` was added as a first-class signed-attribution method.
- `score_cam` was added with `--score-cam-channels-cap` so broad runs can cap activation channels while thesis-quality reruns can use all channels when feasible.
- The frozen CXR consensus definition was preserved: `consensus` averages the original four constituents (`grad_cam`, `integrated_gradients`, `gradient_shap`, `occlusion`) and does **not** absorb `eigen_cam` or `score_cam`. This protects cross-iteration comparability.
- `scripts/calibrate_cxr_xai_thresholds.py` now supports `--calibration-version v3`, producing `calibrated_thresholds_v3.csv` for the expanded method set.

### 1.2 CXR v3 calibration and held-out improvement experiments

Fresh calibration artifacts were produced:

- DenseNet-chex: `outputs/iter_57_chex_calibration_v3/calibrated_thresholds_v3.csv`.
- ResNet-50: `outputs/iter_50_resnet_calibration_v3/calibrated_thresholds_v3.csv`.

Held-out improvement experiments were then completed:

- DenseNet-chex: `outputs/iter_58_chex_improvement_v3/`.
- ResNet-50: `outputs/iter_52_resnet_improvement_v3/`.

Each CXR improvement run contains:
- `200` held-out positive masked test cases.
- `1,600` per-case metric rows (`200` cases × `8` positive-view methods).
- Paired Wilcoxon signed-rank tests against `consensus`.
- Holm-Bonferroni correction.
- 10,000-bootstrap confidence intervals for paired median differences.
- Chapter-ready plots and paired-comparison tables.

**Figure 1. DenseNet-chex held-out improvement experiment plots.** Source: `outputs/iter_58_chex_improvement_v3/`.

| Aggregate boxplots | Paired differences |
| --- | --- |
| ![DenseNet-chex improvement boxplots](../../outputs/iter_58_chex_improvement_v3/improvement_experiment_boxplots.png) | ![DenseNet-chex paired differences](../../outputs/iter_58_chex_improvement_v3/improvement_experiment_paired_diff.png) |

**Figure 2. ResNet-50 held-out improvement experiment plots.** Source: `outputs/iter_52_resnet_improvement_v3/`.

| Aggregate boxplots | Paired differences |
| --- | --- |
| ![ResNet improvement boxplots](../../outputs/iter_52_resnet_improvement_v3/improvement_experiment_boxplots.png) | ![ResNet paired differences](../../outputs/iter_52_resnet_improvement_v3/improvement_experiment_paired_diff.png) |

### 1.3 CT pilot completed

The CT branch passed its hour-1 gate and was completed on real data:

- Dataset: PhysioNet `ct-ich` v1.3.1, NIfTI volumes with masks.
- Model: `DifeiT/rsna-intracranial-hemorrhage-detection` ViT.
- Attribution target: binary hemorrhage score defined as `1 - P(normal)`.
- Brain window: WL=40, WW=80.
- Dataset count verified: `2814` total slices, `318` hemorrhage-positive slices, `2496` normal slices.

CT outputs:
- Smoke run: `outputs/iter_53_ct_smoke_test/`.
- Calibrated held-out improvement run: `outputs/iter_54_ct_improvement_test/`.

**Figure 3. CT held-out improvement experiment plots.** Source: `outputs/iter_54_ct_improvement_test/`.

| Aggregate boxplots | Paired differences |
| --- | --- |
| ![CT improvement boxplots](../../outputs/iter_54_ct_improvement_test/ct_improvement_experiment_boxplots.png) | ![CT paired differences](../../outputs/iter_54_ct_improvement_test/ct_improvement_experiment_paired_diff.png) |

**Figure 4. CT smoke-run faithfulness curves.** Deletion/insertion faithfulness curves by method family on the CT hemorrhage slices, confirming the input-space methods transfer end-to-end to the ViT with the brain-window baseline. Source: `outputs/iter_53_ct_smoke_test/faithfulness_curves.png`.

![CT faithfulness curves](../../outputs/iter_53_ct_smoke_test/faithfulness_curves.png)

### 1.4 Thesis consolidation

The thesis skeleton was updated with final result wording:

- Chapter 3 now contains the implemented methodology: method panel, calibration discipline, CXR/CT preprocessing, faithfulness baselines, and paired statistical testing.
- Chapter 4 now includes CXR classifier metrics, Stage A model comparison, improvement-experiment results, balanced review findings, and CT pilot findings.
- Chapter 5 now uses the corrected conclusion: consensus is stabilizing / competitive with the best constituent, not reliably superior.

Submitted for review in Week 5:

- 📝 `Chapter 3. Methodology` — first version in `thesis/thesis_skeleton.md`.
- 📝 `Chapter 4. Results and Discussion` — first version in `thesis/thesis_skeleton.md`.

## 2. Testing Performed and Results

### 2.1 Code-level tests

The following verification checks were run during the Week 5 implementation path:

| Area | Verification result |
| --- | --- |
| Signed attribution and method-view behavior | `tests/test_signed_attribution.py` → `21 passed` |
| Statistical helpers and improvement-experiment contracts | `tests/test_stats.py` → `21 passed`; `tests/test_improvement_experiment.py` → `3 passed` (`24` combined, after the Week-5 statistics hardening) |
| CT I/O preprocessing | `tests/test_ct_io.py` → `12 passed` (windowing, clipping, resizing, NIfTI round trip) |
| CT classifier wrapper | `tests/test_ct_models.py` added for the `1 - P(normal)` head on a stub backbone |
| Documentation-only thesis updates | No code tests required; files were manually reviewed for stale CT/consensus wording |

**Full suite at end of Week 5:** `wsl.exe python3 -m pytest -q` → **`107 passed`** (8 cosmetic `torchxrayvision` source-change warnings, no failures).

### 2.2 Experiment-output validation

| Experiment | Validation result |
| --- | --- |
| DenseNet-chex v3 calibration | `144` rows, `32` method/view entries, `200` train positive masked cases per selected fraction |
| ResNet v3 calibration | Same expected v3 schema and method coverage as DenseNet |
| DenseNet-chex improvement | `1,600` metric rows, paired table, summary, plots, `run_meta.json` present |
| ResNet improvement | `1,600` metric rows, paired table, summary, plots, `run_meta.json` present |
| CT smoke | `105` positive masked test slices evaluated end-to-end |
| CT improvement | Train calibration + held-out test, paired Wilcoxon/Holm results and plots generated |
| DenseNet-chex classifier test screen | `1372` test rows, expected CSVs present in `outputs/iter_56_chex_classifier_eval_test/` |
| ResNet classifier test screen | `1372` test rows, expected CSVs present in `outputs/iter_55_resnet_classifier_eval_test/` |

## 3. Main Results and Interpretation

### 3.1 CXR classifier performance

**Table 1. Final CXR classifier performance on the SIIM test split.**

| Model | Threshold | AUC | AP | Accuracy | Sensitivity | Specificity | F1 | TP / FP / TN / FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `densenet121-res224-chex` | 0.565 | 0.7460 | 0.4224 | 0.6822 | 0.7138 | 0.6738 | 0.4871 | 207 / 353 / 729 / 83 |
| `resnet50-res512-all` | 0.525 | 0.9163 | 0.7617 | 0.8601 | 0.7103 | 0.9002 | 0.6821 | 206 / 108 / 974 / 84 |

**Figure 5. CXR classifier performance — selected DenseNet (CheX) vs ResNet-50.** ResNet-50 is the stronger classifier on every metric except sensitivity (where the two are within `0.004`). This visual underlines the thesis point that classifier ranking and localization quality are separate axes: ResNet-50's much higher AUC/AP does not translate into clinically strong lesion overlap (see the improvement experiments below). Source: `outputs/iter_55_resnet_classifier_eval_test/`, `outputs/iter_56_chex_classifier_eval_test/`.

![CXR classifier comparison](../../outputs/iter_59_report_figures/cxr_classifier_comparison.png)

Interpretation:
- DenseNet-chex replaces DenseNet-all as the selected DenseNet baseline because Stage A showed better localization than the original DenseNet-all checkpoint.
- ResNet-50 remains the strongest tested TorchXRayVision classifier and the stronger co-primary CXR follow-up baseline overall.
- Classification performance alone is insufficient; localization and review evidence remain necessary. DenseNet-all is now historical/original evidence only.

### 3.2 Consensus-vs-individual XAI results

**Table 2. Cross-modality consensus summary after correction.**

| Setting | Reference method | Main result | Correct interpretation |
| --- | --- | --- | --- |
| CXR DenseNet-chex | Frozen 4-method `consensus` | Significant paired gains over `score_cam` for Dice/IoU/precision; pointing-hit differences are significant vs several methods but median effects are zero | Consensus is not universally superior; occlusion and Grad-CAM++ have slightly higher aggregate mean Dice, while consensus mainly improves over Score-CAM in paired overlap tests |
| CXR ResNet-50 | Frozen 4-method `consensus` | Small significant IoU/Dice/precision gains over weaker methods, but not over `grad_cam` or `score_cam` | Consensus can stabilize relative to weaker methods but remains comparable to the strongest individuals |
| CT ViT | `consensus_input3` over IG / GradientSHAP / Occlusion | Holm-significant pointing-hit gains over all three individuals; overlap metrics not significant | Cleanest consensus advantage is peak localization on CT, not broad overlap |

Final thesis-safe statement:

> There is no general rule that consensus improves localization. Consensus is usually about as good as its best constituent and can stabilize weaker methods in some settings. The strongest clean consensus advantage observed in this project is the CT pointing-game result, while CXR overlap localization remains weak and model-dependent.

## 4. Submitted Chapters 3 and 4

Submitted content in `thesis/thesis_skeleton.md`:

- 📝 **Chapter 3. Methodology**
  - Dataset and preprocessing design for CXR and CT.
  - Off-the-shelf model framing.
  - XAI method families and the `SignedAttribution` four-view contract.
  - Calibration methodology.
  - Localization, faithfulness, agreement, and review metrics.
  - Paired Wilcoxon / Holm-Bonferroni / bootstrap statistics.
  - Ethical and validity considerations.

- 📝 **Chapter 4. Results and Discussion**
  - Classifier performance table.
  - Stage A model comparison.
  - CXR improvement experiment results.
  - Balanced 40-case radiologist-review summary.
  - CT pilot and calibrated CT improvement result.
  - Cross-modality synthesis and limitations.

## 5. Problem Analysis and Resolution Paths

| Problem | Impact | Resolution / current status |
| --- | --- | --- |
| v3 calibration required after adding Eigen-CAM / Score-CAM | Old v2 fractions would be stale | Produced selected-baseline `iter_57` DenseNet-chex and `iter_50` ResNet v3 calibration artifacts |
| Score-CAM and occlusion runtime | Full held-out runs can exceed short agent sessions | Long CUDA runs executed manually; commands and `run_meta.json` preserved |
| DenseNet baseline correction | DenseNet-all was historical, while Stage A selected DenseNet-chex as the stronger DenseNet checkpoint | Re-ran CheX classifier, v3 calibration, and held-out improvement artifacts in `iter_56`, `iter_57`, and `iter_58` |
| CT model target mismatch | Planned `any` output did not exist in DifeiT checkpoint | Redefined binary target as `1 - P(normal)` after gate verification |
| CT faithfulness baseline semantics | Brain-window midpoint differs from CXR black baseline | For this ViT processor, brain-window center normalizes to `zero_tensor`; documented explicitly |
| CT calibration floor at 0.05 | Tiny hemorrhage masks may prefer fractions below sweep floor | Caveat added; pointing-hit emphasized because it is fraction-independent |
| Initial overstatement of consensus result | Could produce an inaccurate thesis conclusion | Cross-modality wording corrected: no universal consensus-improvement claim |
| Final thesis formatting and bibliography | Still needed for mentor/supervisor submission | Left as remaining non-experimental work after results consolidation |

## 6. Plan for Final Demo Preparation

1. Prepare a concise demo narrative:
   - Problem: XAI heatmaps need validation, especially in medical imaging.
   - Method: CXR + CT, localization + faithfulness + agreement + human review.
   - Result: pretrained models can classify while localizing weakly; consensus is stabilizing but not universally superior.
2. Select final demo figures:
   - One CXR good / partial / misleading example from the balanced review workbook.
   - DenseNet-chex vs ResNet classifier table.
   - CXR improvement boxplots / paired differences.
   - CT pointing-hit result and CT caveats.
3. Finalize thesis document formatting:
   - Table / figure / graph / chart lists.
   - Bibliography in one consistent style.
   - Final page numbers and captions.
   - Student ID, signature dates, and supervisor-owned fields.
4. Freeze experiment narrative:
   - Do not start new major experiments before the final demo unless explicitly requested.
   - Keep `LIME`, transformer-specific CT CAMs, Captum infidelity/sensitivity, and external CXR model search as future work.
5. Package reproducibility appendix:
   - Commands, output folders, random seeds, calibration files, thresholds, and environment snapshot.

## 7. Artifact Links

### Code

- `src/explainai_thesis/xai.py` — Eigen-CAM, Score-CAM, signed attribution families.
- `src/explainai_thesis/stats.py` — Wilcoxon, Holm-Bonferroni, bootstrap helpers.
- `src/explainai_thesis/ct/io.py` — CT NIfTI loading and brain-window preprocessing.
- `src/explainai_thesis/ct/models.py` — CT model wrapper and `1 - P(normal)` target.
- `scripts/run_improvement_experiment.py` — CXR held-out improvement experiment.
- `scripts/run_ct_smoke.py` — CT smoke / localization / faithfulness run.
- `scripts/run_ct_improvement_experiment.py` — CT calibrated improvement experiment.
- `scripts/tabulate_improvement_paired.py` — paired-result table renderer.

### Documents

- `thesis/thesis_skeleton.md` — submitted Chapters 3–4 and final result synthesis.
- `docs/progress.md` — detailed chronological record of Week 5 decisions and results.
- `docs/refactor_plan.md` — Phase 5 plan and completed/pending markers.
- `docs/references.md` — references for methods, statistics, and datasets.
- `docs/thesis-notes.md` — thesis-safe interpretation notes.

### Experiment artifacts

- `outputs/iter_56_chex_classifier_eval_test/` — DenseNet-chex classifier test screen.
- `outputs/iter_57_chex_calibration_v3/calibrated_thresholds_v3.csv` — DenseNet-chex v3 calibration.
- `outputs/iter_50_resnet_calibration_v3/calibrated_thresholds_v3.csv` — ResNet v3 calibration.
- `outputs/iter_58_chex_improvement_v3/` — DenseNet-chex held-out improvement run.
- `outputs/iter_52_resnet_improvement_v3/` — ResNet held-out improvement run.
- `outputs/iter_53_ct_smoke_test/` — CT smoke run.
- `outputs/iter_54_ct_improvement_test/` — CT calibrated improvement run.
- `outputs/iter_55_resnet_classifier_eval_test/` — final ResNet classifier test screen.

### Visual artifacts / screenshots

This is a research-pipeline project with no interactive end-user application, so there is no screen-recording demo. The mentor-facing visual evidence is the set of generated figures embedded in this report (all also embedded inline in the self-contained `week_5_report_final.md` / PDF):

- **Figure 1** — DenseNet-chex held-out improvement plots: `outputs/iter_58_chex_improvement_v3/improvement_experiment_{boxplots,paired_diff}.png`.
- **Figure 2** — ResNet-50 held-out improvement plots: `outputs/iter_52_resnet_improvement_v3/improvement_experiment_{boxplots,paired_diff}.png`.
- **Figure 3** — CT held-out improvement plots: `outputs/iter_54_ct_improvement_test/ct_improvement_experiment_{boxplots,paired_diff}.png`.
- **Figure 4** — CT faithfulness curves: `outputs/iter_53_ct_smoke_test/faithfulness_curves.png`.
- **Figure 5** — CXR classifier comparison: `outputs/iter_59_report_figures/cxr_classifier_comparison.png`.
- Interactive CXR review workbook (open locally in a browser): `outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/review/index.html`.

## 8. Hypotheses (Final Status)

Week 5 closed every experiment track, so the carried-forward hypotheses (H1–H11) can be given final verdicts. The headline correction is on H3a: the strong "consensus wins" form is **not** supported, and the pre-mortem fallback narrative is now the actual thesis position.

### Cross-method behavior

- **H1 (methods differ in localization and spatial pattern): supported (final).** Confirmed across both CXR backbones and the CT pilot; method disagreement is a stable, reported feature.
- **H6 (negative attribution avoids the lesion more reliably than positive overlaps it): observed, not elevated to a universal claim.** Negative evidence behaves as a distinct construct and is kept separate in both calibration and review, but it is reported as a diagnostic, not as a formal cross-panel paired law.
- **H7 (cross-method agreement is a mask-free reliability indicator): not supported (cautious negative result).** Score–metric and agreement–localization associations stayed modest (`|rho| <= 0.42`); agreement is reported as suggestive, not as a substitute for mask-based evaluation. This is a publishable negative result about reliability indicators in clinical XAI.

### Model and explanation alignment

- **H2 (high classification ≠ good localization): strongly supported — central finding.** ResNet-50 reaches AUC `0.9163` / AP `0.7617` while consensus overlap stays at Dice ≈ `0.05`; classifier quality and localization quality are decisively separate axes.
- **H8 (faithfulness AUC correlates weakly/not with localization): supported.** The two are reported as separate constructs; high faithfulness does not imply clinically aligned localization.
- **H9 (weak localization is cross-distribution-stable, not fixed by an external model): supported.** All seven DenseNet-121 checkpoints localize weakly; the CT pilot reinforces the broader point that low overlap is not a single-method artifact.

### Improvement and clinical assessment

- **H3a (consensus outperforms the best individual on held-out IoU/Dice): NOT supported as a general law — corrected.** DenseNet-chex consensus is Holm-significantly better only than `score_cam` for overlap and is not the best by aggregate mean Dice (`occlusion`, `grad_cam_plus_plus` are higher). ResNet-50 consensus beats the weaker methods but ties `grad_cam` and `score_cam`. The only clean, substantial consensus advantage anywhere is the **CT pointing-game** (`0.343` vs `0.038`–`0.229`, Holm-significant vs all three input-space methods). An earlier over-clean "consensus never loses / wins peak localization across modalities" reading was withdrawn after a symmetric win/tie/loss decomposition. Net: consensus is *about as good as its best constituent and can stabilize weaker methods*, not reliably superior.
- **H3b (calibration changes the per-method ranking): supported — methodologically necessary.** Per-method v3 fractions differ across methods and models; calibrated thresholding is required for a fair comparison even though no single method or consensus wins outright. This is the robust methodological contribution that survives the H3a correction.
- **H4 (localization metrics correlate moderately, not strongly, with radiologist usefulness): supported.** Confirmed at `|rho| <= 0.42`, justifying the separate `usefulness_score` / `localization_score` reporting.

### Cross-modality

- **H5 (XAI rankings differ between CXR pneumothorax and CT hemorrhage): supported, with a concrete instance.** The consensus signature differs by modality: on CXR consensus does not win peak localization, while on CT it wins the pointing-game cleanly. The conditional clause is resolved — the CT pilot produced a real quantitative test, not only qualitative validation.

### Hypotheses carried from Week 4

- **H10 (a better-localizing classifier is still not clinically sufficient): supported.** ResNet-50 is the strongest classifier and the strongest aggregate localizer, yet absolute overlap remains weak.
- **H11 (usefulness and localization are separable axes): supported.** Useful + potentially-useful (`25/40`) exceeds localization-correct (`11/40`).

### New hypotheses arising in Week 5

- **H12 (new): consensus's benefit is metric- and modality-specific, not a uniform overlap gain.** Its single clean advantage is CT peak localization; CXR overlap gains are small, mixed-sign, and model/metric-dependent. Directly supported by the corrected cross-modality synthesis.
- **H13 (new): input-space attribution methods transfer across architectures (CNN → ViT) with byte-identical code, whereas CAM-family methods do not.** The CT pilot runs Integrated Gradients, GradientSHAP, and Occlusion unchanged on a Vision Transformer; CAM-family methods would need an architecture-specific token→grid adapter and are deliberately excluded to keep the explanation algorithm constant across modalities (documented as future work with citations).

## 9. Risks and Challenges (Final Update)

### Week 3/4 risks, resolved or materialized

- **"Improvement experiment may show consensus is no better than the best individual" — materialized, and the mitigation worked.** Consensus is not universally superior; because both Discussion narratives were pre-drafted, the result needed no thesis-claim rewrite, only an honest framing.
- **Initial overstatement of the consensus result — caught and corrected.** A symmetric pointing-hit win/tie/loss decomposition exposed an asymmetric reading; the over-claims ("never significantly loses", "improves peak localization across modalities") were withdrawn and the corrected synthesis recorded.
- **CT pilot viability — resolved.** Branch A delivered end-to-end on real PhysioNet data; the hour-1 gate passed with a corrected attribution target (`1 - P(normal)`, after the planned `any` head proved absent). CAM-family explanations on the ViT are deferred to future work to avoid an implementation confound.
- **Baseline weak-localization / external-validity** — reframed as the thesis's finding (H2/H9); the unresolved out-of-family external model is documented as a limitation, not a gap that invalidates the comparison.

### New / remaining in Week 5

- **CT calibration floor at top-fraction `0.05`.** Tiny hemorrhage masks may prefer smaller fractions; the overlap comparison is treated as boundary-calibrated and the fraction-independent pointing-game result is emphasized.
- **ResNet-50 runtime at 512×512.** Managed by running long CUDA jobs deliberately and preserving commands + `run_meta.json`; outputs are numerically unaffected.
- **Final non-experimental work outstanding.** Front-matter personal data, template conversion/pagination, table/figure renumbering, and a single-style bibliography remain before the formatted draft — these are finalization tasks, not analysis.

## 10. Review of Week 4 Plan Commitments and Path to Final Demo

This section reviews the Week 4 plan in the deadline-anchor style and complements the final-demo preparation in §6.

### Week 4 commitments reviewed (as of 2026-06-03)

| Week 4 commitment | Status at end of Week 5 |
| --- | --- |
| Phase 5.1 Eigen-CAM + Score-CAM | ✅ Done — added as individual methods; frozen consensus unchanged |
| v3 calibration for the expanded panel | ✅ Done — DenseNet-chex `iter_57`, ResNet `iter_50` |
| Held-out improvement experiments (DenseNet + ResNet) | ✅ Done — `iter_58`, `iter_52`; paired Wilcoxon/Holm/bootstrap |
| CT pilot (hour-1 gate) | ✅ Done — Branch A; gate passed; `iter_53`/`iter_54` |
| Thesis Chapters 3–4 first version | ✅ Done — submitted with this report |

All five Week-4 priorities landed; the DenseNet baseline was additionally corrected to `densenet121-res224-chex` (`iter_56`/`57`/`58`) after Stage A localization, with DenseNet-all retained as historical continuity evidence.

### Path to final demo and draft submission

1. **Demo narrative** (per §6): problem → multi-layer validation method (CXR + CT, localization + faithfulness + agreement + human review) → result (classify-well-but-localize-weakly; consensus stabilizing, not universally superior; CT peak-localization the one clean consensus win).
2. **Final figures** selected from the embedded set above plus the review good/misleading panel.
3. **Finalization** (non-analytical): front-matter personal data, ToC + table/figure/graph/chart lists with page numbers, single-style bibliography, template conversion.
4. **Experiment freeze**: no new major experiments before the demo; LIME, transformer-specific CT CAMs, Captum infidelity/sensitivity, and external CXR model search remain explicitly future work.

### Deadline anchors

- Chapters 3–4 first version submitted with this Week 5 report: **2026-06-03** (done).
- Full thesis draft (content-complete, pending formatting/front-matter): **2026-06-04**.
- Final corrections, formatting, and defense preparation window: **2026-06-05 → 2026-06-21**.
