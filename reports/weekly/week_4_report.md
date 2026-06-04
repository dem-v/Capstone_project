# Weekly Progress Report 4

Reporting period: 2026-05-22 to 2026-05-29

## Week 4 Summary

Week 4 converted the exploratory XAI pipeline from a collection of ad-hoc method variants into a more thesis-ready validation workflow. The main work was: completing the `SignedAttribution` refactor, verifying the expanded CXR model comparison, preparing radiologist-review artifacts, and drafting the first version of thesis Chapters 1 and 2 for supervisor feedback.

The central technical conclusion after the second demo is cautious but useful: XAI maps should be validated as model-behavior evidence, not treated as direct segmentations. The reviewed CXR examples and Stage A model sweep support the thesis framing that off-the-shelf CXR classifiers can be moderately predictive while still producing weak or clinically questionable lesion localization.

## 1. Progress Analysis: Implemented Updates and Changes After Testing

### 1.1 Signed-attribution refactor and method-view contract

The XAI pipeline was refactored around a single `SignedAttribution` object per method family. Instead of recomputing separate positive, negative, magnitude, and signed maps, each method now produces one signed tensor and derives four views from it:

| View | Meaning | Color convention |
| --- | --- | --- |
| `positive` | Evidence supporting the target output | Red |
| `negative` | Evidence suppressing the target output | Blue |
| `magnitude` | Impact regardless of direction | Violet |
| `signed` | Positive-minus-negative tug-of-war view | Orange / teal |

Implemented or stabilized components:
- `src/explainai_thesis/xai.py`: `SignedAttribution`, signed Grad-CAM, Grad-CAM++, Integrated Gradients, GradientSHAP, Occlusion, and signed consensus.
- `src/explainai_thesis/visualization.py`: signed diverging overlays and shared method-view color selection.
- CXR scripts now emit explicit `view`, `family`, `signed_positive_fraction`, and `signed_prediction_alignment` metadata.
- Legacy output folders were preserved; no historical results were renamed.

Testing-driven updates:
- Fixed a Grad-CAM++ negative-polarity double sign flip that made positive and negative Grad-CAM++ maps too similar.
- Replaced fragile filename label inference with token-aware regular expressions.
- Changed the CXR smoke faithfulness default from `zero_tensor` to `black`, because the old baseline could still produce a high pneumothorax probability.
- Added regression coverage for signed-map algebra, overlay colors, localization metrics, manifest inference, faithfulness sanity, and classifier loading.

Verification milestones during Week 4 included:
- `37 passed` after metric / manifest / faithfulness tests were added.
- `43 passed` after the classifier-loading seam landed.
- `50 passed` after method-view and correlation tooling stabilization.
- `52 passed` after CXR I/O extraction and package-level helpers were added.

### 1.2 Model comparison and second-baseline decision

The diagnostic Stage A sweep compared multiple TorchXRayVision checkpoints to test whether weak localization was specific to the original DenseNet baseline.

**Table 1. Stage A model comparison summary.** Source: `outputs/iter_33_stage_a_diagnostic_ab/weights_ab_summary.csv`.

| Model | Mean Dice | Mean IoU | Mean precision at fraction | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `resnet50-res512-all` | **0.0397** | **0.0221** | **0.0296** | Best tested TorchXRayVision candidate, but still weak absolute localization |
| `densenet121-res224-chex` | 0.0284 | 0.0160 | n/a | Better than original DenseNet-all but not strong |
| `densenet121-res224-all` | 0.0237 | 0.0130 | n/a | Original baseline; clinically weak localization persists |

**Figure 1. Stage A localization sweep across all seven TorchXRayVision checkpoints.** The two annotated bars are the selected baselines (`resnet50-res512-all` as the strongest follow-up, `densenet121-res224-chex` as the best DenseNet-121 branch). Absolute localization is low for every checkpoint, which is the point: model choice changes the *relative* ranking but does not, on its own, produce clinically strong pneumothorax localization.

![Stage A model comparison](../../outputs/iter_59_report_figures/stage_a_model_comparison.png)

Decision after testing:
- `resnet50-res512-all` was promoted to the co-primary CXR baseline for follow-up.
- The thesis wording remains cautious: ResNet is relatively better, not clinically strong.
- The out-of-family MONAI CXR option was checked and not integrated because the available MONAI CXR bundle was generative rather than a pneumothorax classifier with a usable target head.

### 1.3 Radiologist-review tooling and balanced review workflow

The review workflow was built around a static HTML workbook instead of an interactive app. This keeps the artifact reproducible and easy to send to the mentor/supervisor.

Implemented components:
- `scripts/build_review_workbook.py`: builds `index.html`, `INSTRUCTIONS.md`, `scores_template.csv`, and local PNG assets.
- `scripts/analyze_review_scores.py`: validates scores, joins review labels to metrics, summarizes qualitative flags, and computes exploratory correlations.
- `scripts/select_cxr_review_candidates.py`: selects representative `tp` / `fp` / `tn` / `fn` cases and generates diagnostic commands.

Balanced 40-case review result:
- `10` true positives, `10` false positives, `10` true negatives, `10` false negatives.
- Localization scores: `11` correct, `15` partial, `14` incorrect.
- Usefulness scores: `12` useful, `13` potentially useful, `14` misleading, `1` not useful.
- Dominant failure category: non-pathological high-contrast regions (`13/40`).
- Devices or tubes were flagged in `19/40` cases.

**Figure 2. Balanced 40-case review outcome distributions.** Localization, usefulness, and failure-taxonomy counts over the outcome-stratified review set. The dominant failure mode is attention on non-pathological high-contrast structure (`13/40`), and useful + potentially-useful (`25/40`) outweighs misleading + not-useful (`15/40`) — but only modestly. Source: `outputs/iter_48_resnet_review_analysis_balanced40_smoothed_faithfulness/`.

![Review outcome distributions](../../outputs/iter_59_report_figures/review_distributions.png)

**Figure 3. Representative review cases — clinically aligned vs misleading.** Each row shows the source X-ray, the ground-truth pneumothorax mask, the Grad-CAM positive view, and the consensus positive view. These heatmaps are class-specific attribution maps, not anatomical segmentations. The top case (`case_07`) was scored *correct / useful*; the bottom case (`case_05`) was scored *incorrect / misleading*, with attribution drawn to non-pathological high-contrast structure away from the lesion — the project's most common failure mode.

| Case (score) | Source X-ray | Ground-truth mask | Grad-CAM positive | Consensus positive |
| --- | --- | --- | --- | --- |
| `case_07` — correct / useful | ![src7](../../outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/review/assets/case_07/source_16_train_1_.png) | ![mask7](../../outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/review/assets/case_07/mask_16_train_1_.png) | ![gc7](../../outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/review/assets/case_07/16_train_1__grad_cam_continuous_heatmap.png) | ![cons7](../../outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/review/assets/case_07/16_train_1__consensus_continuous_heatmap.png) |
| `case_05` — incorrect / misleading | ![src5](../../outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/review/assets/case_05/source_2518_train_1_.png) | ![mask5](../../outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/review/assets/case_05/mask_2518_train_1_.png) | ![gc5](../../outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/review/assets/case_05/2518_train_1__grad_cam_continuous_heatmap.png) | ![cons5](../../outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/review/assets/case_05/2518_train_1__consensus_continuous_heatmap.png) |

## 2. Conclusions After the Second Demo and Recommendations Received

The second demo clarified the direction of the project:

1. **Do not present heatmaps as clinical segmentations.** The maps must be described as class-specific attribution evidence for the model output.
2. **Separate classifier performance from localization quality.** Moderate classifier ranking can coexist with poor mask overlap or misleading saliency.
3. **Keep positive and negative evidence separate.** Blue/suppressive evidence overlapping a lesion is not automatically a good localization result.
4. **Use reviewer feedback and qualitative failure categories.** The mentor-facing narrative should combine metrics, review flags, and representative cases.
5. **Continue with a second CXR baseline.** ResNet-50 was selected because it was the best tested TorchXRayVision model, while still preserving the weak-localization thesis framing.
6. **Move writing forward in parallel with experiments.** Chapters 1 and 2 were prepared first so the supervisor can comment on framing and literature coverage before final results are frozen.

## 3. Initial Thesis-Writing Work Submitted for Feedback

Submitted for supervisor / mentor review:

- 📝 `Chapter 1. Introduction` — first version in `thesis/thesis_skeleton.md`.
- 📝 `Chapter 2. Literature Review` — first version in `thesis/thesis_skeleton.md`.

Content prepared:
- Research context and relevance of XAI validation in medical imaging.
- Problem statement: XAI maps require validation before they can support clinical interpretation.
- Aim, research objectives, and research questions.
- Literature review on deep learning in radiology, saliency/CAM methods, explanation validation, shortcut learning, and clinical review considerations.
- Required chapter conclusions for Chapters 1 and 2 according to the Neoversity template.

## 4. Problem Analysis and Resolution Paths

| Problem | Impact | Resolution / current status |
| --- | --- | --- |
| Grad-CAM++ negative-polarity bug | Negative evidence could be misrepresented | Fixed with regression tests in `tests/test_gradcam_polarity.py` |
| Old `zero_tensor` faithfulness baseline | Deletion/insertion curves could be misleading | Default switched to `black`; old option kept for historical reproduction |
| DenseNet-only loading assumptions | ResNet and multi-model comparisons failed or needed special cases | Added `load_classifier(name)` seam under `src/explainai_thesis/cxr/classifier.py` |
| ResNet `GradientSHAP` CUDA OOM | High-stability review diagnostics could fail at 512x512 | Added `--gradshap-internal-batch-size`, defaulting to a safer small batch |
| Review workbook fragile external image links | HTML could show missing images outside its original path | Workbook now copies PNGs into local `assets/` folders |
| Weak absolute localization | Could undermine a simplistic “XAI improves trust” thesis claim | Reframed thesis around validation, limitations, and model-behavior diagnostics |

## 5. Updated Plan for the Next Week

Priority order for Week 5:

1. Complete Phase 5.1 method expansion with `Eigen-CAM` and `Score-CAM`.
2. Produce fresh v3 CXR calibration artifacts for the expanded method panel.
3. Run held-out improvement experiments for DenseNet-all and ResNet-50.
4. Implement the CT pilot only if the hour-1 gate passes: model load, license, real slice, mask alignment, and target definition.
5. Update Chapter 3 `Methodology` and Chapter 4 `Results and Discussion` with completed experiments.
6. Finalize result tables, figures, and thesis-safe interpretation.
7. Avoid new optional add-ons such as LIME unless all required experiments and writing are already complete.

## 6. Artifact Links

### Code

- `src/explainai_thesis/xai.py` — signed-attribution method implementations.
- `src/explainai_thesis/visualization.py` — overlay rendering and color semantics.
- `src/explainai_thesis/cxr/classifier.py` — multi-weight classifier-loading seam.
- `src/explainai_thesis/faithfulness.py` — extracted faithfulness helpers.
- `scripts/run_stage_a_diagnostic_ab.ps1` — Stage A model comparison orchestrator.
- `scripts/build_review_workbook.py` — static review workbook generator.
- `scripts/analyze_review_scores.py` — review scoring analysis.

### Documents

- `docs/progress.md` — chronological implementation and experiment log.
- `docs/refactor_plan.md` — Phase 0–5 implementation and risk plan.
- `docs/thesis-notes.md` — thesis prose notes and interpretation anchors.
- `docs/references.md` — working bibliography.
- `thesis/thesis_skeleton.md` — submitted Chapters 1–2 draft and thesis scaffold.

### Experiment and review artifacts

- `outputs/iter_33_stage_a_diagnostic_ab/weights_ab_summary.csv` — Stage A model comparison summary.
- `outputs/iter_35_metric_correlations_iter33_stage_a_all_models/` — all-model metric correlation analysis.
- `outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/review/index.html` — balanced 40-case review workbook.
- `outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/review/scores.csv` — completed review scores.
- `outputs/iter_48_resnet_review_analysis_balanced40_smoothed_faithfulness/` — review analysis output.

### Visual artifacts / screenshots

This is a research-pipeline project with no interactive end-user application, so there is no screen-recording demo. The mentor-facing visual evidence is the set of generated figures and outputs embedded in this report:

- **Figure 1** — Stage A model-comparison chart (`outputs/iter_59_report_figures/stage_a_model_comparison.png`).
- **Figure 2** — balanced 40-case review distributions (`outputs/iter_59_report_figures/review_distributions.png`).
- **Figure 3** — representative clinically-aligned vs misleading review cases (from `outputs/iter_48_.../review/assets/`).
- Interactive static review workbook (open locally in a browser): `outputs/iter_48_resnet_review_workbook_balanced40_smoothed_faithfulness/review/index.html`.
- The self-contained `week_4_report_final.md` / PDF embed every figure above inline, so the submission renders without access to the repository.

## 7. Hypotheses (Status Update from the Week 3 Set)

Week 3 carried forward H1–H5 and added H6–H9. Week 4 was the first iteration to run the diagnostic Stage A sweep and complete the balanced radiologist review, so several hypotheses move from "planned test" to "evidenced". The improvement experiment and the CT pilot are scheduled for Week 5, so the hypotheses that depend on them (H3a, H3b, H5) remain open at the end of this reporting period.

### Cross-method behavior

- **H1 (different methods → different localization quality and spatial pattern): supported and strengthened.** The Stage A sweep and the 40-case review both show visible cross-method disagreement; `method_disagreement` was flagged in `8/40` review cases, and `agreement_score` is now computed per case. The existence of cases where two methods overlap the same lesion but differ spatially is confirmed qualitatively.
- **H6 (negative attribution avoids the lesion more reliably than positive attribution overlaps it): preliminary, not yet formally tested.** The review keeps negative evidence separate and the `negative_mask_avoidance_fraction` / `negative_mask_overlap_fraction` diagnostics exist, but a paired statistical test against positive `IoU`/`Dice` across the full method panel is deferred to the Week 5 held-out evaluation.
- **H7 (cross-method agreement correlates with localization quality → mask-free reliability indicator): weakly supported / leaning negative.** The all-model metric-correlation analysis (`outputs/iter_35_metric_correlations_iter33_stage_a_all_models/`) and the exploratory review correlations show only modest associations (`|rho| <= 0.42`). Agreement does not yet look like a strong mask-free reliability proxy; this is trending toward a cautious/negative result.

### Model and explanation alignment

- **H2 (high classification performance does not guarantee good localization): strongly supported — Week 4 headline.** Stage A's best-localizing checkpoint still has mean Dice `0.0397`, and the review found `14/40` localization-incorrect and `14/40` misleading maps despite the classifier's moderate ranking. This is now a core, well-evidenced thesis claim rather than a single-model observation.
- **H8 (faithfulness AUC correlates weakly or not at all with mask localization): preliminarily supported.** Faithfulness curves and localization metrics are produced from the same runs and behave as separate constructs; a direct paired-correlation test is folded into the Week 5 evaluation.
- **H9 (poor localization is cross-distribution-stable across the DenseNet-121 family and is not eliminated by one out-of-family external model): supported — Week 4 headline.** All seven TorchXRayVision checkpoints localize weakly (mean Dice `0.013`–`0.040`, Figure 1), and the only checked out-of-family CXR option (MONAI) was generative rather than a pneumothorax classifier and could not be integrated. The weak-localization finding therefore generalizes across training distributions; this is the strong-negative-finding branch the hypothesis anticipated.

### Improvement and clinical assessment

- **H3a (consensus outperforms the best individual method on held-out IoU/Dice): deferred — not yet tested.** The held-out improvement experiment runs in Week 5 once v3 calibration for the expanded method panel lands. Both Discussion narratives (consensus improves / consensus does not improve) remain pre-drafted per the Week 3 pre-mortem.
- **H3b (calibration changes the per-method ranking versus uncalibrated thresholding): in progress.** v2 calibration is in place; the v3 regeneration required by the expanded panel and the formal calibrated-vs-uncalibrated comparison are Week 5 work.
- **H4 (localization metrics correlate moderately, not strongly, with radiologist usefulness): supported — Week 4 headline.** The completed balanced 40-case review yields modest score–metric associations (`|rho| <= 0.42`), exactly the "moderate, not strong" pattern predicted, and motivates reporting `usefulness_score` separately from `localization_score`.

### Cross-modality (conditional)

- **H5 (XAI rankings may differ between CXR pneumothorax and CT hemorrhage): deferred, conditional.** The CT pilot has not started; its hour-1 model-availability gate is Week 5. H5 remains discussion-only until that gate passes.

### New hypotheses arising in Week 4

- **H10 (new): a relatively better-localizing classifier is still not clinically sufficient.** ResNet-50 is the strongest tested checkpoint by aggregate localization yet remains far below segmentation-grade overlap, so model selection is a *relative* improvement, not a fix for localization. Directly supported by Figure 1.
- **H11 (new): radiologist usefulness and pixel-level localization are separable axes.** In the review, useful + potentially-useful (`25/40`) exceeds the count of localization-correct cases (`11/40`); low-overlap maps can still be audit-useful, and visually plausible maps can still be misleading. Supports a two-axis review rubric.

## 8. Risks and Challenges (Updated from Week 3)

### Carried from Week 2/3

- **CT pilot needs a usable off-the-shelf classifier and masks.** Still open at end of Week 4; mitigated by the planned Week 5 hour-1 gate (model load, license, real slice, mask alignment, target definition) with a qualitative-fallback branch.
- **Scope creep across methods/datasets/models.** Controlled this week: Stage A used existing weights through the `load_classifier` seam; no new methods were added (Eigen-CAM/Score-CAM held for Week 5).
- **Local patient-data anonymization discipline.** Not exercised this week (no local clinical data handled).

### New-in-Week-3 risks, now updated

- **"Protocol-completion work decorates a broken foundation."** Resolved into a *finding* rather than a threat: Stage A shows the weak localization is cross-model (H9), so it is genuine evidence about pretrained CXR classifiers, and the thesis is framed around validation and limitations accordingly.
- **SignedAttribution refactor invalidates calibrated fractions.** Addressed: v2 calibration is in place and versioned; v3 for the expanded panel is queued for Week 5.
- **Improvement experiment may show consensus is no better than the best individual.** Still open (experiment not yet run); mitigation holds — both Discussion narratives are pre-drafted, so a null/negative result is publishable without rewriting the thesis claim.
- **Long runs may stop mid-run.** Addressed: `--resume` verified earlier and used for the balanced run.
- **GradientSHAP instability at low sample counts.** Addressed: low-sample screening only; thesis-worthy cases rerun at higher samples (rule encoded in AGENTS.md).

### External-coordination risks

- **Buggy `grad_cam_plus_plus_negative` overlays previously shown to the supervisor.** Resolved: the polarity double-flip is fixed with regression tests (`tests/test_gradcam_polarity.py`); a before/after comparison can be shown as instrument calibration.
- **AI-tooling disclosure policy.** The methodology framing is drafted; the explicit tool-disclosure list still needs confirmation with the supervisor before the methodology chapter is finalized.

### New in Week 4

- **ResNet-50 at 512×512 is markedly slower than DenseNet at 224×224** (occlusion windows and per-forward cost scale together, ≈37–40× more compute). Risk to the Week 5 held-out improvement-run timeline; mitigated by scheduling long CUDA runs deliberately and tuning occlusion/batch settings (numerically identical output).
- **Out-of-family external model still unresolved.** The MONAI CXR candidate was generative; no pneumothorax-specific external classifier is integrated. Documented as an external-validity limitation rather than a blocker; the in-family DenseNet sweep plus ResNet-50 stands as the model comparison.

## 9. Review of Week 3 Plan Commitments and Detailed Plan for Week 5

This section expands the priority list in §5 in the Week 3 deadline-anchor style: first a verdict on what Week 3 committed to, then the Week 5 iteration plan and the parallel writing track.

### Week 3 commitments reviewed (as of 2026-05-29)

| Week 3 commitment (anchor date) | Status at end of Week 4 |
| --- | --- |
| Stage A diagnostic sweep + Stage B decision (05-24) | ✅ Done — `iter_33`; ResNet-50 promoted to co-primary baseline |
| Metric–metric correlation heatmap (05-22) | ✅ Done — `iter_35` |
| Review workbook + scoring pass (05-28) | ✅ Done — rescoped to a balanced 40-case set (10 per outcome), completed 05-25 |
| Phase 2 structural refactor / SignedAttribution contract (05-26) | ✅ Done — four-view contract across all methods |
| Phase 5.1 Eigen-CAM + Score-CAM (05-28) | ⏳ Slipped to Week 5 |
| `consensus_attention` (05-30) | ⤴ Deferred to thesis future work (Chapter 5.3) |
| Held-out improvement experiment (06-02) | ⏳ Week 5 (pending v3 calibration) |
| CT pilot or qualitative fallback (05-31) | ⏳ Week 5 (hour-1 gate) |
| Phase 5.5 full second-model protocol (06-03) | 🟡 Partly — ResNet Stage A + review done; full held-out protocol in Week 5 |
| Final figures/tables/draft (06-04) | ⏳ On track for Week 5 |

### Committed Week 5 iteration

1. **Phase 5.1** — add `Eigen-CAM` and `Score-CAM` as first-class signed-attribution methods (kept as individuals; the frozen four-method consensus is unchanged for cross-iteration comparability).
2. **v3 calibration** — regenerate per-method top-fractions for the expanded panel for the selected CXR baselines.
3. **Held-out improvement experiments** — DenseNet and ResNet-50, calibrate→freeze→test with paired Wilcoxon + Holm-Bonferroni + 10k bootstrap CIs (directly tests H3a/H3b).
4. **CT pilot** — run only if the hour-1 gate passes (model load, license, real slice, mask alignment, target definition); otherwise fall back to qualitative external validation (tests H5).
5. **Thesis Chapters 3–4** — populate Methodology and Results from the completed CSVs.

### Deadline anchors (revised against Week 4 actuals)

- Phase 5.1 + v3 calibration: **2026-05-31**.
- Held-out improvement experiments (DenseNet + ResNet): **2026-06-02**.
- CT pilot hour-1 gate decision, then Branch A or fallback: **2026-06-02**.
- Chapters 3–4 first version submitted with Week 5 report: **2026-06-03**.
- Final figures, tables, and draft completion: **2026-06-04**.

### Parallel thesis-writing track

- Methodology (Chapter 3): drafted now — the seam, SignedAttribution contract, calibration discipline, and faithfulness baseline are stable.
- Results (Chapter 4): the Stage A table and the balanced-review distributions are already populatable; the improvement-experiment tables fill in as Week 5 runs complete.
- Discussion / Conclusions: still gated on the improvement-experiment outcome; both pre-drafted narratives remain in place so neither result is overstated.
