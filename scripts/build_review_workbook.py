from __future__ import annotations

import argparse
import csv
import html
import os
import re
import shutil
from pathlib import Path

from explainai_thesis.run_metadata import write_run_metadata


REVIEW_FIELDS = [
    "case_id",
    "filename",
    "localization_score",
    "usefulness_score",
    "failure_category",
    "flag_devices_or_tubes",
    "flag_subcutaneous_emphysema",
    "flag_mask_quality_issue",
    "flag_indirect_evidence",
    "flag_method_disagreement",
    "flag_weak_pixel_attribution",
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


def copy_asset(path: Path | None, output_dir: Path, case_id: str) -> Path | None:
    if path is None or not path.exists():
        return None
    assets_dir = output_dir / "assets" / case_id
    assets_dir.mkdir(parents=True, exist_ok=True)
    destination = assets_dir / path.name
    if path.resolve() != destination.resolve():
        shutil.copy2(path, destination)
    return destination


def image_cell(path: Path | None, base: Path, caption: str, css_class: str = "") -> str:
    if path is None or not path.exists():
        return f'<figure class="missing"><div>missing</div><figcaption>{html.escape(caption)}</figcaption></figure>'
    rel = relative_path(path, base)
    class_attr = f' class="{html.escape(css_class)}"' if css_class else ""
    escaped_caption = html.escape(caption)
    return f'<figure{class_attr}><a href="{rel}" target="_blank"><img src="{rel}" alt="{escaped_caption}"></a><figcaption>{escaped_caption}</figcaption></figure>'


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


def asset_image_cell(
    source_path: Path | None,
    output_dir: Path,
    case_id: str,
    caption: str,
    css_class: str = "",
) -> str:
    return image_cell(copy_asset(source_path, output_dir, case_id), output_dir, caption, css_class)


def format_float(value: str | float, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def faithfulness_summary_table(summary_rows: list[dict[str, str]]) -> str:
    if not summary_rows:
        return ""
    columns = [
        ("Method", "method"),
        ("View", "view"),
        ("Insertion AUC", "faithfulness_insertion_auc"),
        ("Deletion AUC", "faithfulness_deletion_auc"),
        ("Deletion drop", "faithfulness_deletion_drop"),
        ("Insertion gain", "faithfulness_insertion_gain"),
    ]
    body = []
    for row in sorted(
        summary_rows,
        key=lambda item: (
            item.get("method", ""),
            item.get("view", ""),
        ),
    ):
        cells = []
        for _, key in columns:
            value = row.get(key, "")
            if key.startswith("faithfulness_"):
                value = format_float(value)
            else:
                value = html.escape(str(value))
            cells.append(f"<td>{value}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    header = "".join(f"<th>{html.escape(label)}</th>" for label, _ in columns)
    return f'<table class="metrics faithfulness-table"><thead><tr>{header}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def faithfulness_curve_svg(curve_rows: list[dict[str, str]]) -> str:
    if not curve_rows:
        return ""
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in curve_rows:
        grouped.setdefault((row.get("method", ""), row.get("view", "")), []).append(row)
    selected_groups = [
        item for item in grouped.items() if item[0][1] in {"positive", "signed", "magnitude"}
    ][:8]
    if not selected_groups:
        selected_groups = list(grouped.items())[:8]
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4b5563", "#be123c"]
    width = 760
    height = 300
    left = 48
    right = 16
    top = 18
    bottom = 42
    plot_width = width - left - right
    plot_height = height - top - bottom

    def point(row: dict[str, str], key: str) -> tuple[float, float]:
        x = left + float(row["fraction"]) * plot_width
        y = top + (1.0 - float(row[key])) * plot_height
        return x, y

    elements = [
        f'<svg class="faithfulness-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Faithfulness deletion and insertion curves">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#9ca3af"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#9ca3af"/>',
        f'<text x="{left}" y="{height - 12}" font-size="11">fraction restored/removed</text>',
        f'<text x="5" y="14" font-size="11">probability</text>',
    ]
    legend_y = 22
    for index, ((method, view), rows) in enumerate(selected_groups):
        rows = sorted(rows, key=lambda item: float(item["fraction"]))
        color = colors[index % len(colors)]
        insertion_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(row, "insertion_probability") for row in rows))
        deletion_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(row, "deletion_probability") for row in rows))
        label = html.escape(f"{method} {view}")
        elements.append(f'<polyline points="{insertion_points}" fill="none" stroke="{color}" stroke-width="2"/>')
        elements.append(f'<polyline points="{deletion_points}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="5 4"/>')
        elements.append(f'<line x1="585" y1="{legend_y}" x2="615" y2="{legend_y}" stroke="{color}" stroke-width="2"/>')
        elements.append(f'<line x1="585" y1="{legend_y + 10}" x2="615" y2="{legend_y + 10}" stroke="{color}" stroke-width="2" stroke-dasharray="5 4"/>')
        elements.append(f'<text x="620" y="{legend_y + 4}" font-size="10">{label}</text>')
        legend_y += 28
    elements.append('<text x="585" y="270" font-size="10">solid=insertion, dashed=deletion</text>')
    elements.append("</svg>")
    return "".join(elements)


def faithfulness_summary_svg(
    summary_rows: list[dict[str, str]],
    *,
    zoomed: bool = False,
) -> str:
    if not summary_rows:
        return ""
    rows = sorted(summary_rows, key=lambda item: (item.get("method", ""), item.get("view", "")))
    values: list[tuple[str, float, float]] = []
    for row in rows:
        try:
            insertion = float(row["faithfulness_insertion_auc"])
            deletion_drop = float(row["faithfulness_deletion_drop"])
        except (KeyError, TypeError, ValueError):
            continue
        label = f"{row.get('method', '')} {row.get('view', '')}".strip()
        values.append((label, insertion, deletion_drop))
    if not values:
        return ""

    all_values = [value for _, insertion, deletion_drop in values for value in (insertion, deletion_drop)]
    if zoomed:
        low = max(0.0, min(all_values) - 0.05)
        high = min(1.0, max(all_values) + 0.05)
        if high - low < 0.10:
            center = (high + low) / 2.0
            low = max(0.0, center - 0.05)
            high = min(1.0, center + 0.05)
    else:
        low = 0.0
        high = 1.0
    if high <= low:
        high = low + 1.0

    width = max(900, len(values) * 54)
    height = 360
    left = 58
    right = 20
    top = 32
    bottom = 112
    plot_width = width - left - right
    plot_height = height - top - bottom
    group_width = plot_width / len(values)
    bar_width = min(16, group_width * 0.32)

    def y_pos(value: float) -> float:
        return top + (high - value) / (high - low) * plot_height

    title = "Faithfulness AUC summary, zoomed" if zoomed else "Faithfulness AUC summary"
    elements = [
        f'<svg class="faithfulness-svg faithfulness-summary-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="20" font-size="14" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#9ca3af"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#9ca3af"/>',
        f'<text x="{left + plot_width - 230}" y="20" font-size="11"><tspan fill="#2563eb">■</tspan> insertion AUC  <tspan fill="#dc2626">■</tspan> deletion drop</text>',
    ]
    for tick in range(5):
        value = low + (high - low) * tick / 4
        y = y_pos(value)
        elements.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        elements.append(f'<text x="8" y="{y + 4:.1f}" font-size="10">{value:.2f}</text>')
    for index, (label, insertion, deletion_drop) in enumerate(values):
        center = left + group_width * (index + 0.5)
        for x, value, color in (
            (center - bar_width, insertion, "#2563eb"),
            (center + 2, deletion_drop, "#dc2626"),
        ):
            y = y_pos(value)
            bar_height = max(1.0, top + plot_height - y)
            elements.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}"/>')
        escaped_label = html.escape(label)
        elements.append(f'<text transform="translate({center - 6:.1f},{height - 10}) rotate(-55)" font-size="9">{escaped_label}</text>')
    elements.append("</svg>")
    return "".join(elements)


def faithfulness_section(run_dir: Path, output_dir: Path, case_id: str) -> str:
    summary_path = run_dir / "faithfulness_summary.csv"
    curves_path = run_dir / "faithfulness_curves.csv"
    summary_rows = read_csv(summary_path) if summary_path.exists() else []
    curve_rows = read_csv(curves_path) if curves_path.exists() else []
    summary_png = copy_asset(run_dir / "faithfulness_summary.png", output_dir, case_id)
    zoomed_png = copy_asset(run_dir / "faithfulness_summary_zoomed.png", output_dir, case_id)
    if not summary_rows and not curve_rows and summary_png is None and zoomed_png is None:
        return '<section class="faithfulness"><h3>Faithfulness</h3><p class="meta">No faithfulness outputs found for this case.</p></section>'
    summary_figure = (
        image_cell(summary_png, output_dir, "faithfulness summary plot", "wide")
        if summary_png is not None
        else f'<figure class="wide"><figcaption>faithfulness summary plot</figcaption>{faithfulness_summary_svg(summary_rows)}</figure>'
    )
    zoomed_figure = (
        image_cell(zoomed_png, output_dir, "faithfulness summary plot, zoomed", "wide")
        if zoomed_png is not None
        else f'<figure class="wide"><figcaption>faithfulness summary plot, zoomed</figcaption>{faithfulness_summary_svg(summary_rows, zoomed=True)}</figure>'
    )
    figures = [summary_figure, zoomed_figure]
    return f"""
  <section class="faithfulness">
    <h3>Faithfulness</h3>
    <p class="meta">Insertion starts from the selected baseline and restores high-attribution pixels; deletion starts from the original image and replaces high-attribution pixels with the baseline. Higher insertion AUC/gain and larger deletion drop indicate stronger model-behavior faithfulness, not clinical correctness.</p>
    <div class="faithfulness-grid">{''.join(figures)}{faithfulness_curve_svg(curve_rows)}</div>
    {faithfulness_summary_table(summary_rows)}
  </section>
"""


def method_view_name(method: str, view: str) -> str:
    return method if view == "positive" else f"{method}_{view}"


def add_method_section(
    figures: list[str],
    case_dir: Path | None,
    output_dir: Path,
    case_id: str,
    source_stem: str,
    method: str,
) -> None:
    views = [
        ("positive", "positive / red"),
        ("negative", "negative / blue"),
        ("magnitude", "magnitude / violet"),
        ("signed", "signed / orange-teal"),
    ]
    figures.append(f'<h3 class="method-title">{html.escape(method)}</h3>')
    for view, label in views:
        name = method_view_name(method, view)
        continuous = find_image(case_dir, f"{source_stem}*{name}_continuous_heatmap.png")
        sweep = find_image(case_dir, f"{source_stem}*{name}_threshold_sweep_panel.png")
        if continuous is None and sweep is None:
            continue
        figures.append(
            asset_image_cell(
                continuous,
                output_dir,
                case_id,
                f"{method} {label} continuous",
            )
        )
        figures.append(
            asset_image_cell(
                sweep,
                output_dir,
                case_id,
                f"{method} {label} threshold sweep",
                "wide",
            )
        )


def build_html(rows: list[dict[str, str]], args: argparse.Namespace, methods: list[str]) -> str:
    cards: list[str] = []
    for row in rows:
        run_dir = diagnostic_dir_for(row, args.diagnostics_dir)
        case_dir = first_case_image_dir(run_dir)
        source_stem = Path(row["filename"]).stem
        case_id = f"case_{int(row['review_rank']):02d}"
        figures = []
        figures.append(
            asset_image_cell(
                Path(row["image_path"]),
                args.output_dir,
                case_id,
                "native source image",
                "native",
            )
        )
        figures.append(
            asset_image_cell(
                Path(row["mask_path"]),
                args.output_dir,
                case_id,
                "native ground-truth mask",
                "native",
            )
        )
        for method in methods:
            add_method_section(figures, case_dir, args.output_dir, case_id, source_stem, method)
        faithfulness_html = faithfulness_section(run_dir, args.output_dir, case_id)
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
  {faithfulness_html}
  <div class="review-grid">{''.join(figures)}</div>
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
    <div><b>binary flags</b><ul><li><code>flag_devices_or_tubes</code>: tubes, wires, drains, text, or device artifacts are relevant.</li><li><code>flag_subcutaneous_emphysema</code>: subcutaneous emphysema affects interpretation.</li><li><code>flag_mask_quality_issue</code>: mask/label is missing, incomplete, or questionable.</li><li><code>flag_indirect_evidence</code>: clinically related but indirect signs dominate.</li><li><code>flag_method_disagreement</code>: methods clearly highlight different regions.</li><li><code>flag_weak_pixel_attribution</code>: IG/GradientSHAP remain weak/noisy or strongly disagree with the other methods.</li></ul></div>
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
.faithfulness {{ margin: 12px 0 18px; padding: 10px; border: 1px solid #dbeafe; background: #eff6ff; border-radius: 6px; }}
.faithfulness h3 {{ margin: 0 0 6px; }}
.faithfulness-grid {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 12px; align-items: start; }}
.faithfulness-svg {{ width: 100%; min-height: 260px; border: 1px solid #e5e7eb; background: #fff; }}
.faithfulness-table {{ background: #fff; }}
.review-grid {{ display: grid; grid-template-columns: repeat(4, minmax(240px, 1fr)); gap: 12px; align-items: start; }}
.method-title {{ grid-column: 1 / -1; margin: 18px 0 0; padding: 8px 10px; background: #eef2ff; border-left: 4px solid #6366f1; font-size: 17px; }}
figure {{ margin: 0; border: 1px solid #e5e7eb; padding: 6px; background: #fafafa; }}
img {{ width: 100%; height: auto; display: block; }}
.native img {{ image-rendering: auto; }}
.wide {{ grid-column: span 3; }}
.wide img {{ width: 100%; }}
figcaption {{ font-size: 12px; margin-top: 4px; }}
.missing div {{ min-height: 160px; display: grid; place-items: center; color: #9ca3af; }}
@media (max-width: 1200px) {{ .review-grid {{ grid-template-columns: repeat(2, minmax(240px, 1fr)); }} .wide {{ grid-column: span 2; }} }}
@media (max-width: 720px) {{ .review-grid {{ grid-template-columns: 1fr; }} .wide {{ grid-column: span 1; }} }}
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
3. For each of the {case_count} cases, fill `localization_score`, `usefulness_score`, `failure_category`, binary qualitative flags, and optional notes.
4. Use the first three cases as warmup anchors before continuing the full scoring pass.
5. Do not edit `scores_template.csv`; keep it as the reproducible blank template.

Workbook layout:
- Native source image and native ground-truth mask are shown first. Click any image to open the full-size file in a new tab.
- If faithfulness was computed, each case shows deletion/insertion curves and a compact summary table before the heatmap grid. Solid lines are insertion; dashed lines are deletion.
- Each method block shows available positive, negative, magnitude, and signed views. The paired threshold-sweep panel follows each continuous heatmap when it exists.

Allowed values:
- `localization_score`: `correct` = main positive evidence matches the pneumothorax region; `partial` = some lesion/pleural evidence but incomplete/off-target; `incorrect` = dominant evidence outside the relevant region; `none` = no interpretable positive localization.
- `usefulness_score`: `useful` = helps audit/explain the decision; `potentially_useful` = plausible but caution needed; `misleading` = visually plausible but clinically wrong reason; `not_useful` = diffuse/noisy/absent/artifact-driven.
- `failure_category`: `correct`, `partial`, `anatomically_related`, `devices_text_artifacts`, `non_pathological_high_contrast`, `diffuse_non_specific`, `clinically_misleading`.
- Binary flags use `0` = absent/not relevant and `1` = present/relevant: `flag_devices_or_tubes`, `flag_subcutaneous_emphysema`, `flag_mask_quality_issue`, `flag_indirect_evidence`, `flag_method_disagreement`, `flag_weak_pixel_attribution`.
- `flag_weak_pixel_attribution` covers IG/GradientSHAP that remain weak/noisy after smoothing or disagree strongly with the other methods.

Metric hints:
- `Dice`, `IoU`, and `precision_at_fraction` summarize positive-mask overlap for the best available method/fraction.
- `pointing_hit` asks whether the single strongest point lands inside the mask.
- `negative_mask_overlap_fraction` and `negative_mask_avoidance_fraction` describe whether suppressive evidence falls inside or avoids the lesion.
- `signed_prediction_alignment` is a report-only diagnostic: whether signed evidence direction agrees with the classifier prediction at the frozen threshold.
- Faithfulness metrics evaluate model behavior under perturbation, not clinical correctness: higher insertion AUC/gain and larger deletion drop mean the attribution ranking changes the model probability more strongly.
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
            field: (
                f"case_{int(row['review_rank']):02d}"
                if field == "case_id"
                else row["filename"]
                if field == "filename"
                else ""
            )
            for field in REVIEW_FIELDS
        }
        for row in rows
    ]
    write_csv(args.output_dir / "scores_template.csv", template_rows)
    write_instructions(args.output_dir / "INSTRUCTIONS.md", len(rows))
    (args.output_dir / "index.html").write_text(build_html(rows, args, methods), encoding="utf-8")

    run_meta_path = write_run_metadata(args.output_dir, args, methods=methods, case_count=len(rows))

    print(f"Wrote review workbook for {len(rows)} cases to {args.output_dir}")
    print(f"Run metadata written to: {run_meta_path}")


if __name__ == "__main__":
    main()