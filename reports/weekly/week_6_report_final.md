# Weekly Progress Report 6

Reporting period: 2026-06-04 to 2026-06-07

## Week 6 Summary

Week 6 was a finalization and consolidation window rather than a new-experiment week. With the experiment narrative already frozen at the end of Week 5, the agreed working agreement for this period was **non-dramatic** work only: completing the thesis draft, generating the outstanding result charts, adding a reusable visualization component, hardening and bundling the codebase, and one bounded negative side-probe. No new major experiment tracks, model swaps, or scope changes were undertaken, and the frozen consensus definition, locked statistics, and corrected cross-modality synthesis from Week 5 were left unchanged.

The headline outcomes are: (1) an AI-assisted full-thesis draft pass over Chapters 1–4 with standalone export files for PDF review; (2) the two remaining Chapter 4 charts (radiologist-review score distributions and the failure taxonomy) generated from the frozen balanced 40-case review counts; (3) a reusable CT attribution/visualization component that produces CXR-like per-case CT artifacts; (4) a documented **negative** side-probe confirming that an off-the-shelf transformer CXR classifier plugs into the input-space pipeline but does not improve on ResNet-50; and (5) an output-preserving consolidation of the loose scripts into a single bundled `explainai-thesis` command-line interface.

## 1. Completed Project Updates

### 1.1 Thesis draft consolidation (Chapters 1–4)

- Completed an AI-assisted draft pass over `thesis/thesis_skeleton.md` Chapters 1–4 for supervisor/student review, with an explicit draft-status note at the top of the document.
- Normalized wording so CT is described consistently as a completed but limited pilot (not a conditional future branch), and so `densenet121-res224-chex` is the selected DenseNet baseline while `densenet121-res224-all` remains historical context only.
- Added traceable visual/graph references for the final artifacts (CheX and ResNet held-out improvement plots, CT pilot plots, and candidate review-workbook figures) and converted remaining open items into explicit `TODO (human review)` markers for final figure selection, citation/access-date verification, template formatting, and final captions/page numbers.
- Generated two standalone Markdown exports for PDF review: `thesis/chapters_1_2_final_AI_draft.md` (Chapters 1–2 with a built-in conceptual schematic) and `thesis/chapters_3_4_final_AI_draft.md` (Chapters 3–4 with all local images embedded as inline data URIs so the file renders without external files). No remaining broken relative image links were found.

### 1.2 Outstanding Chapter 4 charts generated

- Generated the two outstanding Chapter 4 charts referenced in the skeleton from the **frozen** canonical balanced 40-case ResNet-50 review counts via a new charts-only script `scripts/make_thesis_charts.py`:
  - **Chart 4.2 (review score distributions):** localization and usefulness counts — `outputs/iter_61_thesis_charts/chart_4_2_review_scores.png`.
  - **Chart 4.3 (failure taxonomy):** failure-category and qualitative-flag counts — `outputs/iter_61_thesis_charts/chart_4_3_failure_taxonomy.png`.
- Both PNGs were wired into Section 4.4 of the skeleton. No experiment was re-run and no source review CSV was modified; the charts only visualize the already-frozen counts (localization: `11` correct / `15` partial / `14` incorrect; usefulness: `12` useful / `13` potentially useful / `14` misleading / `1` not useful; dominant failure mode: non-pathological high-contrast structure, `13/40`).

### 1.3 Reusable CT visualization component

- Added a reusable CT attribution dispatch in `src/explainai_thesis/ct/methods.py` (stable method names `integrated_gradients`, `gradient_shap`, `occlusion`, `consensus_input3`, three-method consensus dependency expansion, and CLI-independent settings) and reusable CT visualization helpers in `src/explainai_thesis/ct/visualization.py` (HU-window display conversion, CT slice export without per-image renormalization, per-view overlays, selected top-fraction masks, and positive/signed contact sheets).
- Added `scripts/visualize_ct_xai_maps.py` to produce CXR-like per-case CT artifacts under a stable output folder; a tiny end-to-end validation run on CUDA produced `outputs/iter_59_ct_visual_smoke/` (CT slice PNG, mask-contour PNG, IG positive/negative/magnitude/signed overlays, selected masks, and contact sheets).
- Thesis-safe interpretation is preserved: these are model-attribution maps for the `1 - P(normal)` hemorrhage score, not hemorrhage segmentations; thesis-quality CT figures are to be regenerated with more stable settings.

### 1.4 Negative side-probe: drop-in transformer CXR classifier

- Explored whether an off-the-shelf transformer CXR classifier with a native pathology head could plug into the pipeline and possibly beat ResNet-50. The cleanest structural match found was `taheera/vit-in1k-chestxray14` (a `vit_base_patch16_224` with a 14-label NIH ChestX-ray14 head, Pneumothorax at index 7).
- **Structure verdict — positive:** all three input-space methods (Integrated Gradients, GradientSHAP, Occlusion) ran through the existing `xai.py`/`metrics.py` on the ViT with only a thin loader, producing the same per-case panels and localization metrics as the CNN pipeline.
- **Quality verdict — negative:** on 10 SIIM positive masked cases the checkpoint was a weak detector (mean `P(pneumothorax)=0.299`, only `2/10` positive at `0.5` vs ResNet-50 sensitivity `0.71`) and a weak/off-lesion localizer (best Dice `0.0393` for Occlusion; pointing-hit `0.000` for every method on every case).
- Logged as a documented negative finding (artifacts under `outputs/iter_60_taheera_vit_*`, explicitly isolated and **not** thesis artifacts): a transformer CXR classifier with a native head is a clean structural drop-in for the input-space XAI pipeline, but this checkpoint would not improve classification or localization over ResNet-50, and being a ViT it still breaks CAM-family comparability. A controlled transformer comparison with a token→grid CAM adapter remains explicit future work.

### 1.5 Codebase consolidation (output-preserving)

- Bundled the loose scripts into a single installable command-line interface: `explainai-thesis <command>` (and `python -m explainai_thesis <command>`), auto-discovering all 28 commands. Both the bundled entry point and the legacy `python scripts/<x>.py ...` invocations continue to work; outputs (CSV schema, PNG layout, folder names, numeric results) are byte-for-byte identical.
- Migrated each command's logic verbatim into importable modules under `src/explainai_thesis/cli/commands/` (now the single source of truth), leaving each `scripts/<x>.py` as a thin backward-compatible shim. Verified that no script relied on `__file__`/path-relative resolution, so the move cannot change any path or output.
- Deduplicated the two genuinely repeated argparse flags (`--device`, `--split`) through shared helpers while preserving each command's exact `--help` text and defaults.
- Verified Table 4.1 classifier rows against their frozen sources (CheX threshold `0.565`; ResNet `0.525`) as a finalization check; no experiment was rerun.

## 2. Testing Performed and Results

### 2.1 Code-level tests

| Area | Verification result |
| --- | --- |
| CT attribution dispatch and visualization helpers | Targeted CT method/visualization tests → `10 passed`; full suite `117 passed` after the component landed |
| Bundled CLI dispatcher | New `tests/test_cli_dispatcher.py` → `5 passed`; `explainai-thesis --list` lists all `28` commands; `--help` pass-through confirmed |
| Standalone-module migration (logic moved into the package) | Full suite `122 passed`; `make-thesis-charts` reproduced byte-identical PNGs (md5 match) via both the shim and the bundled command |
| Argparse dedup (`--device`, `--split`) | Every command's `--help` captured before/after — all argparse commands byte-identical; full suite `122 passed` |
| Thesis draft / chart wiring | Documentation/charts-only; all referenced artifact paths exist; `py_compile` clean; pytest not required for doc-only changes |

**Full suite at end of Week 6:** `wsl.exe python3 -m pytest -q` → **`122 passed`** (no failures), up from `107` at the end of Week 5.

### 2.2 Output validation

| Item | Validation result |
| --- | --- |
| Chapter 4 charts | Both PNGs generated from the frozen review-count CSVs; no source CSV modified |
| CT visualization smoke | `outputs/iter_59_ct_visual_smoke/` produced end-to-end on CUDA with `run_meta.json` and all expected overlays |
| Transformer side-probe | `outputs/iter_60_taheera_vit_eval/` with `metrics_per_case.csv` / `metrics_summary.csv` (uncalibrated, n=10, isolated, gitignored) |
| Table 4.1 thresholds | CheX (`0.565`) and ResNet (`0.525`) rows match the frozen `outputs/iter_56_*` / `iter_55_*` classification metrics byte-for-value |

## 3. Main Results and Interpretation

Week 6 did not change any frozen experimental conclusion; its analytical contribution is one additional **negative** datapoint and the visualization of already-frozen results.

- **Transformer drop-in is structurally clean but clinically no better.** The ViT side-probe reinforces the central thesis findings rather than overturning them: a stronger or different-architecture classifier does not, by itself, produce lesion-aligned saliency (weak/off-lesion localization, pointing-hit `0.000`), and CAM-family methods still require an architecture-specific adapter to transfer.
- **The review evidence is now chart-ready.** The frozen balanced 40-case findings — useful + potentially-useful (`25/40`) modestly outweighing misleading + not-useful (`15/40`), with non-pathological high-contrast structure the dominant failure mode (`13/40`) — are now presented as Chart 4.2 / Chart 4.3 without re-deriving any number.

Final thesis-safe statement (unchanged from Week 5):

> There is no general rule that consensus improves localization. Consensus is usually about as good as its best constituent and can stabilize weaker methods in some settings. The strongest clean consensus advantage observed in this project is the CT pointing-game result, while CXR overlap localization remains weak and model-dependent.

## 4. Thesis Chapters Status (Sections 1–5)

By the end of Week 6 the thesis content is complete across all five chapters in `thesis/thesis_skeleton.md`, pending human review and formatting:

- 📝 **Chapter 1. Introduction** — research context, problem statement, aim/objectives, novelty, structure.
- 📝 **Chapter 2. Literature Review** — XAI methods, medical-imaging datasets/models, validation, and the shortcut/generalization gap.
- 📝 **Chapter 3. Methodology** — datasets and preprocessing (CXR + CT), off-the-shelf model framing, the four-view signed-attribution contract, calibration discipline, localization/faithfulness/agreement/review metrics, and paired Wilcoxon / Holm-Bonferroni / bootstrap statistics.
- 📝 **Chapter 4. Results and Discussion** — classifier performance, Stage A model comparison, CXR improvement experiments, the balanced 40-case review summary (now with Chart 4.2 / Chart 4.3), the CT pilot, and the cross-modality synthesis with limitations.
- 📝 **Chapter 5. Conclusions and Recommendations** — the corrected conclusion that consensus is stabilizing / competitive with the best constituent rather than reliably superior, plus practical recommendations and future work (LIME, transformer-specific CT CAMs, infidelity/sensitivity, and an external CXR model search).

The remaining open items are explicit `TODO (human review)` markers only: final figure selection, citation/access-date verification, template conversion, and front-matter personal fields. These are finalization tasks, not analysis.

## 5. Problem Analysis and Resolution Paths

| Problem | Impact | Resolution / current status |
| --- | --- | --- |
| Two Chapter 4 charts still unrendered | Review evidence could not be presented graphically | Generated Chart 4.2 / Chart 4.3 from the frozen review counts via a charts-only script; no experiment rerun |
| CT figures lacked a CXR-like per-case visualization path | Cross-modality figures were not directly comparable to CXR overlays | Added a reusable CT attribution/visualization component and a smoke run to validate it end-to-end |
| Open question: would a transformer classifier improve results? | Risk of an unverified assumption entering the thesis | Bounded side-probe answered it negatively and was documented as a non-thesis negative finding; future work scoped |
| Loose, hard-to-discover scripts | Reproducibility/usability friction before submission | Bundled into one `explainai-thesis` CLI and migrated logic into the package, with verified byte-identical outputs and a backward-compatible shim layer |
| Risk of scope creep in the polish window | Could destabilize frozen results/conclusions | Enforced the non-dramatic working agreement: additive, reversible changes only; consensus/statistics/synthesis untouched |
| Final formatting and front-matter still outstanding | Needed for the formatted submission | Left as explicit human-review TODOs after content consolidation |

## 6. Plan for Final Demo and Draft Submission

1. **Demo narrative** (unchanged): problem → multi-layer validation method (CXR + CT; localization + faithfulness + agreement + human review) → result (classify-well-but-localize-weakly; consensus stabilizing, not universally superior; CT peak-localization the one clean consensus win).
2. **Final figures**: the held-out improvement plots, the CT pilot plots and faithfulness curves, the classifier comparison, and now the review charts (Chart 4.2 / Chart 4.3) plus a good/misleading review panel.
3. **Finalization (non-analytical)**: template conversion and pagination, table/figure/graph/chart lists with page numbers, single-style bibliography, captions, and front-matter personal fields.
4. **Experiment freeze maintained**: no new major experiments; LIME, transformer-specific CT CAMs, Captum infidelity/sensitivity, and an external CXR model search remain explicit future work.

## 7. Artifact Links

### Code

- `src/explainai_thesis/ct/methods.py` — CT attribution dispatch and three-method consensus expansion.
- `src/explainai_thesis/ct/visualization.py` — CT HU-window display, overlays, selected masks, contact sheets.
- `src/explainai_thesis/cli/` — bundled `explainai-thesis` command-line interface (registry, dispatcher) and migrated command modules under `cli/commands/`.
- `scripts/visualize_ct_xai_maps.py` — CXR-like per-case CT visualization.
- `scripts/make_thesis_charts.py` — Chapter 4 chart generation from frozen review counts.

### Documents

- `thesis/thesis_skeleton.md` — content-complete Chapters 1–5 (pending review/formatting).
- `thesis/chapters_1_2_final_AI_draft.md`, `thesis/chapters_3_4_final_AI_draft.md` — standalone PDF-review exports.
- `docs/progress.md` — chronological record of the 2026-06-04 → 2026-06-07 work.
- `docs/cli.md` — bundled-CLI command index and output-preservation guarantee.
- `docs/refactor_plan_standalone_module.md` — the executed standalone-module bundling plan.

### Experiment / figure artifacts

- `outputs/iter_61_thesis_charts/chart_4_2_review_scores.png`, `chart_4_3_failure_taxonomy.png` — Chapter 4 charts.
- `outputs/iter_59_ct_visual_smoke/` — CT CXR-like visualization smoke run.
- `outputs/iter_60_taheera_vit_eval/` — isolated transformer side-probe (negative finding; not a thesis artifact).

### Visual artifacts

This is a research-pipeline project with no interactive end-user application, so there is no screen-recording demo. The mentor-facing visual evidence for Week 6 is:

- **Chart 4.2** — radiologist-review score distributions: `outputs/iter_61_thesis_charts/chart_4_2_review_scores.png`.
- **Chart 4.3** — failure taxonomy and qualitative flags: `outputs/iter_61_thesis_charts/chart_4_3_failure_taxonomy.png`.
- **CT visualization** — per-case CXR-like CT attribution artifacts: `outputs/iter_59_ct_visual_smoke/`.

## 8. Hypotheses (Status After Week 6)

Week 6 ran no new confirmatory experiment, so the final H1–H13 verdicts from Week 5 stand. The one new datapoint, the transformer side-probe, reinforces three of them:

- **H2 (high classification ≠ good localization): reinforced.** A different-architecture transformer classifier still produced off-lesion saliency (pointing-hit `0.000`), so classifier choice does not buy localization.
- **H9 (weak localization is cross-distribution-stable): reinforced.** Weak/off-lesion overlap now also holds for a ViT trained on a different CXR distribution (NIH ChestX-ray14), not only the TorchXRayVision CNN family.
- **H13 (input-space methods transfer across architectures; CAM-family methods do not): reinforced.** IG, GradientSHAP, and Occlusion ran unchanged on the ViT, while CAM-family transfer would still require a token→grid adapter.

No new hypotheses were opened; the experiment narrative remains frozen.

## 9. Risks and Challenges

### Carried from Week 5, still managed

- **Final non-experimental work outstanding.** Front-matter personal data, template conversion/pagination, table/figure renumbering, and a single-style bibliography remain — finalization tasks, not analysis.
- **CT calibration floor and ResNet-50 runtime** — unchanged from Week 5; no new runs were launched in this window.

### New / specific to Week 6

- **Refactor-without-regression risk.** Bundling and migrating the scripts could have silently changed outputs; mitigated by verifying byte-identical CSV/PNG outputs (md5 match), identical `--help` text, and a passing `122`-test suite, plus a backward-compatible shim layer so no documented invocation path broke.
- **Scope discipline in the polish window.** The soft 2026-06-07 cutoff allowed only non-dramatic work; larger ideas (transformer CXR probe via CheXzero/ELIXR, `consensus_attention`, LIME, transformer-specific CT CAMs) were explicitly kept as future work.

## 10. Review of Week 5 Plan Commitments

| Week 5 commitment | Status at end of Week 6 |
| --- | --- |
| Full thesis draft (content-complete) | ✅ Done — AI-assisted Chapters 1–4 pass + standalone export files; Chapter 5 conclusion synchronized |
| Generate the remaining Chapter 4 charts | ✅ Done — Chart 4.2 / Chart 4.3 from the frozen review counts (`iter_61`) |
| Keep the experiment narrative frozen | ✅ Held — no new major experiments; only a bounded negative side-probe and additive tooling |
| Package reproducibility (commands, environment, tooling) | ✅ Advanced — bundled `explainai-thesis` CLI with verified output preservation |
| Final formatting / front-matter | ⏳ Pending — left as explicit human-review tasks |

### Deadline anchors

- Full thesis draft (content-complete, pending formatting/front-matter): **2026-06-04** (done).
- Non-dramatic polish/consolidation soft cutoff: **2026-06-07** (this report's window).
- Final corrections, formatting, and defense preparation window: **2026-06-07 → 2026-06-21**.
