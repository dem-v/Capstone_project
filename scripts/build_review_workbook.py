from __future__ import annotations

import argparse
import csv
import html
import os
import re
from pathlib import Path


REVIEW_FIELDS = [
    "case_id",
    "filename",
    "localization_score",
    "usefulness_score",
    "failure_category",
    "artifact_note",
    "comment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a static radiologist-review workbook from selected CXR diagnostic outputs."
    )
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=Path("outputs/iter_28_review_candidate_selection/selected_manual_review_cases.csv"),
        help="CSV produced by select_cxr_review_candidates.py.",
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=Path("outputs/iter_28_review_diagnostics"),
        help="Directory containing per-case high-stability diagnostic output folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/iter_28_review_workbook/review"),
        help="Directory where index.html, scores_template.csv, and INSTRUCTIONS.md are written.",
    )
    parser.add_argument(
        "--methods",
        default="grad_cam,grad_cam_plus_plus,integrated_gradients,gradient_shap,occlusion,consensus",
        help="Comma-separated continuous heatmap methods to show in each card.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "case"


def diagnostic_dir_for(row: dict[str, str], diagnostics_dir: Path) -> Path:
    rank = int(row["review_rank"])
    category = safe_name(row["category"])
    source_stem = safe_name(Path(row["filename"]).stem)
    return diagnostics_dir / f"case_{rank:02d}_{category}_{source_stem}"


def first_case_image_dir(run_dir: Path) -> Path | None:
    for path in sorted(run_dir.iterdir() if run_dir.exists() else []):
        if path.is_dir():
            return path
    return None


def relative_path(path: Path, base: Path) -> str:
    rel = os.path.relpath(path.resolve(), start=base.resolve())
    return html.escape(Path(rel).as_posix())


def image_cell(path: Path | None, base: Path, caption: str) -> str:
    if path is None or not path.exists():
        return f'<figure class="missing"><div>missing</div><figcaption>{html.escape(caption)}</figcaption></figure>'
    rel = relative_path(path, base)
    return f'<figure><img src="{rel}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def metric_table(row: dict[str, str]) -> str:
    fields = [
        ("Best method", "best_method"),
        ("Best top fraction", "best_top_fraction"),
        ("Best Dice", "best_dice"),
        ("Best IoU", "best_iou"),
        ("Best precision@fraction", "best_precision_at_fraction"),
        ("Best pointing hit", "best_pointing_hit"),
        ("Negative method", "negative_method"),
        ("Negative overlap in mask", "max_negative_mask_overlap_fraction"),
        ("Negative avoidance", "negative_mask_avoidance_fraction"),
        ("Signed method", "signed_method"),
        ("Signed positive fraction", "max_signed_positive_fraction"),
        ("Signed prediction alignment", "signed_prediction_alignment"),
    ]
    rows = []
    for label, key in fields:
        value = row.get(key, "")
        if value not in (None, ""):
            rows.append(f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(value))}</td></tr>")
    if not rows:
        return ""
    return f'<table class="metrics"><tbody>{"".join(rows)}</tbody></table>'


def find_image(case_dir: Path | None, filename: str) -> Path | None:
    if case_dir is None:
        return None
    matches = sorted(case_dir.glob(filename))
    return matches[0] if matches else None


def build_html(rows: list[dict[str, str]], args: argparse.Namespace, methods: list[str]) -> str:
    cards: list[str] = []
    for row in rows:
        run_dir = diagnostic_dir_for(row, args.diagnostics_dir)
        case_dir = first_case_image_dir(run_dir)
        source_stem = Path(row["filename"]).stem
        case_id = f"case_{int(row['review_rank']):02d}"
        figures = []
        figures.append(image_cell(find_image(case_dir, f"{source_stem}*consensus_threshold_sweep_panel.png"), args.output_dir, "consensus threshold sweep"))
        for method in methods:
            figures.append(image_cell(find_image(case_dir, f"{source_stem}*{method}_continuous_heatmap.png"), args.output_dir, method))
        meta = " | ".join(
            [
                f"category={row.get('category', '')}",
                f"weights={row.get('weights', '')}",
                f"image_size={row.get('image_size', '')}",
                f"outcome={row.get('classifier_outcome', '')}",
                f"label={row.get('label', '')}",
                f"prediction={row.get('prediction', '')}",
                f"prob={row.get('xrv_pneumothorax_sigmoid', '')}",
                f"best={row.get('best_method', '')}@{row.get('best_top_fraction', '')}",
            ]
        )
        cards.append(
            f"""
<section class="card" id="{case_id}">
  <h2>{case_id}: {html.escape(row['filename'])}</h2>
  <p class="meta">{html.escape(meta)}</p>
  {metric_table(row)}
  <div class="grid">{''.join(figures)}</div>
</section>
"""
        )
    rubric = """
<aside class="rubric">
  <h2>Scoring rubric</h2>
  <div class="rubric-grid">
    <div><b>localization_score</b><ul><li><code>correct</code>: main positive evidence is inside or tightly follows the pneumothorax/mask region.</li><li><code>partial</code>: some relevant lesion/pleural evidence is present, but substantial signal is missing or off-target.</li><li><code>incorrect</code>: dominant evidence is outside the clinically relevant region.</li><li><code>none</code>: no interpretable positive localization is visible.</li></ul></div>
    <div><b>usefulness_score</b><ul><li><code>useful</code>: would help explain or audit the classifier decision.</li><li><code>potentially_useful</code>: contains some plausible signal but needs caution.</li><li><code>misleading</code>: visually persuasive but clinically points to the wrong reason.</li><li><code>not_useful</code>: too diffuse, noisy, absent, or artifact-driven.</li></ul></div>
    <div><b>failure_category</b><ul><li><code>correct</code>: no major failure.</li><li><code>partial</code>: mixed lesion and non-lesion evidence.</li><li><code>anatomically_related</code>: plausible nearby anatomy, not the lesion itself.</li><li><code>devices_text_artifacts</code>: tubes, labels, borders, text, or markers dominate.</li><li><code>non_pathological_high_contrast</code>: ribs, diaphragm, edges, or contrast structures dominate.</li><li><code>diffuse_non_specific</code>: broad nonspecific signal.</li><li><code>clinically_misleading</code>: explanation supports an unsafe or wrong clinical story.</li></ul></div>
  </div>
  <p>Color semantics: red/orange = positive evidence for pneumothorax score; blue/teal = negative evidence against that score; violet = magnitude/impact; green/yellow/cyan mark mask/selection intersections where present. Heatmaps are model-behavior diagnostics, not pathology segmentations.</p>
</aside>
"""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CXR XAI review workbook</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 0; background: #f6f7f9; color: #1f2933; }}
header, .rubric {{ position: sticky; top: 0; z-index: 2; background: #fff; border-bottom: 1px solid #d8dee9; padding: 10px 18px; }}
.card {{ margin: 18px; padding: 16px; background: #fff; border: 1px solid #d8dee9; border-radius: 8px; }}
.meta {{ font-size: 13px; color: #4b5563; }}
.rubric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; font-size: 13px; }}
.rubric ul {{ margin: 6px 0 0 18px; padding: 0; }}
.metrics {{ border-collapse: collapse; margin: 8px 0 12px; font-size: 13px; }}
.metrics th, .metrics td {{ border: 1px solid #e5e7eb; padding: 4px 8px; text-align: left; }}
.metrics th {{ background: #f3f4f6; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
figure {{ margin: 0; border: 1px solid #e5e7eb; padding: 6px; background: #fafafa; }}
img {{ width: 100%; height: auto; display: block; }}
figcaption {{ font-size: 12px; margin-top: 4px; }}
.missing div {{ min-height: 160px; display: grid; place-items: center; color: #9ca3af; }}
</style>
</head>
<body>
<header><h1>CXR XAI review workbook</h1><p>Fill <code>scores.csv</code> from <code>scores_template.csv</code>. Use browser find or anchors <code>#case_01</code>, <code>#case_02</code>, ... for navigation.</p></header>
{rubric}
{''.join(cards)}
</body>
</html>
"""


def write_instructions(path: Path, case_count: int) -> None:
    path.write_text(
        f"""# CXR XAI review workbook instructions

1. Open `index.html` in a browser.
2. Copy `scores_template.csv` to `scores.csv` in the same folder.
3. For each of the {case_count} cases, fill `localization_score`, `usefulness_score`, `failure_category`, and optional notes.
4. Use the first three cases as warmup anchors before continuing the full scoring pass.
5. Do not edit `scores_template.csv`; keep it as the reproducible blank template.

Allowed values:
- `localization_score`: `correct` = main positive evidence matches the pneumothorax region; `partial` = some lesion/pleural evidence but incomplete/off-target; `incorrect` = dominant evidence outside the relevant region; `none` = no interpretable positive localization.
- `usefulness_score`: `useful` = helps audit/explain the decision; `potentially_useful` = plausible but caution needed; `misleading` = visually plausible but clinically wrong reason; `not_useful` = diffuse/noisy/absent/artifact-driven.
- `failure_category`: `correct`, `partial`, `anatomically_related`, `devices_text_artifacts`, `non_pathological_high_contrast`, `diffuse_non_specific`, `clinically_misleading`.

Metric hints:
- `Dice`, `IoU`, and `precision_at_fraction` summarize positive-mask overlap for the best available method/fraction.
- `pointing_hit` asks whether the single strongest point lands inside the mask.
- `negative_mask_overlap_fraction` and `negative_mask_avoidance_fraction` describe whether suppressive evidence falls inside or avoids the lesion.
- `signed_prediction_alignment` is a report-only diagnostic: whether signed evidence direction agrees with the classifier prediction at the frozen threshold.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    rows = read_csv(args.selection_csv)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    template_rows = [
        {
            "case_id": f"case_{int(row['review_rank']):02d}",
            "filename": row["filename"],
            "localization_score": "",
            "usefulness_score": "",
            "failure_category": "",
            "artifact_note": "",
            "comment": "",
        }
        for row in rows
    ]
    write_csv(args.output_dir / "scores_template.csv", template_rows)
    write_instructions(args.output_dir / "INSTRUCTIONS.md", len(rows))
    (args.output_dir / "index.html").write_text(build_html(rows, args, methods), encoding="utf-8")

    print(f"Wrote review workbook for {len(rows)} cases to {args.output_dir}")


if __name__ == "__main__":
    main()