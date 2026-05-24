from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


LOCALIZATION_ORDER = {
    "none": 0,
    "incorrect": 1,
    "partial": 2,
    "correct": 3,
}
USEFULNESS_ORDER = {
    "not_useful": 0,
    "misleading": 1,
    "potentially_useful": 2,
    "useful": 3,
}
FAILURE_CATEGORIES = {
    "correct",
    "partial",
    "anatomically_related",
    "devices_text_artifacts",
    "non_pathological_high_contrast",
    "diffuse_non_specific",
    "clinically_misleading",
}
FLAG_COLUMNS = [
    "flag_devices_or_tubes",
    "flag_subcutaneous_emphysema",
    "flag_mask_quality_issue",
    "flag_indirect_evidence",
    "flag_method_disagreement",
    "flag_weak_pixel_attribution",
]
PATTERN_METHODS = ["integrated_gradients", "gradient_shap"]
PATTERN_VIEWS = ["positive", "negative", "magnitude", "signed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate review scores and join them with per-case XAI metrics."
    )
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/review_score_analysis"),
    )
    parser.add_argument(
        "--reference-fraction",
        type=float,
        default=0.10,
        help="Top-fraction used for method/view mean metrics.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Optional[str]) -> float:
    if value in (None, ""):
        return math.nan
    try:
        out = float(value)
    except ValueError:
        return math.nan
    return out if math.isfinite(out) else math.nan


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return math.nan
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom == 0.0:
        return math.nan
    return float(np.sum(x * y) / denom)


def spearman(x_values: list[float], y_values: list[float]) -> tuple[float, int]:
    pairs = [
        (x, y)
        for x, y in zip(x_values, y_values, strict=True)
        if math.isfinite(x) and math.isfinite(y)
    ]
    if len(pairs) < 3:
        return math.nan, len(pairs)
    x = np.asarray([pair[0] for pair in pairs], dtype=float)
    y = np.asarray([pair[1] for pair in pairs], dtype=float)
    return pearson(rankdata(x), rankdata(y)), len(pairs)


def validate_scores(
    scores: list[dict[str, str]], selection_rows: list[dict[str, str]]
) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    required = [
        "case_id",
        "filename",
        "localization_score",
        "usefulness_score",
        "failure_category",
    ]
    selected_filenames = {row["filename"] for row in selection_rows}
    seen_case_ids: Counter[str] = Counter(row.get("case_id", "") for row in scores)
    seen_filenames: Counter[str] = Counter(row.get("filename", "") for row in scores)

    for row_number, row in enumerate(scores, start=2):
        for column in required:
            if column not in row:
                issues.append({"row": row_number, "column": column, "issue": "missing_column"})
            elif row[column] == "":
                issues.append({"row": row_number, "column": column, "issue": "empty_value"})

        case_id = row.get("case_id", "")
        filename = row.get("filename", "")
        if case_id and seen_case_ids[case_id] > 1:
            issues.append({"row": row_number, "column": "case_id", "issue": "duplicate"})
        if filename and seen_filenames[filename] > 1:
            issues.append({"row": row_number, "column": "filename", "issue": "duplicate"})
        if filename and filename not in selected_filenames:
            issues.append({"row": row_number, "column": "filename", "issue": "not_in_selection"})
        if row.get("localization_score") not in LOCALIZATION_ORDER:
            issues.append({"row": row_number, "column": "localization_score", "issue": "invalid_value"})
        if row.get("usefulness_score") not in USEFULNESS_ORDER:
            issues.append({"row": row_number, "column": "usefulness_score", "issue": "invalid_value"})
        if row.get("failure_category") not in FAILURE_CATEGORIES:
            issues.append({"row": row_number, "column": "failure_category", "issue": "invalid_value"})
        for column in FLAG_COLUMNS:
            if column in row and row[column] not in {"", "0", "1"}:
                issues.append({"row": row_number, "column": column, "issue": "invalid_flag_value"})

    scored_filenames = {row.get("filename", "") for row in scores}
    for filename in sorted(selected_filenames - scored_filenames):
        issues.append({"row": "", "column": "filename", "issue": "selection_case_not_scored", "value": filename})
    return issues


def summarize_case_metrics(case_dir: Path, reference_fraction: float) -> dict[str, object]:
    rows = read_csv(case_dir / "threshold_metrics.csv")
    positives = [row for row in rows if row.get("metric_component") == "positive"]
    negatives = [row for row in rows if row.get("metric_component") == "negative"]
    reference_positive = [
        row for row in positives if math.isclose(as_float(row.get("top_fraction")), reference_fraction)
    ]
    if not reference_positive:
        reference_positive = positives

    def max_metric(metric: str, source_rows: list[dict[str, str]]) -> float:
        values = [as_float(row.get(metric)) for row in source_rows]
        finite = [value for value in values if math.isfinite(value)]
        return max(finite) if finite else math.nan

    def mean_metric(metric: str, source_rows: list[dict[str, str]]) -> float:
        values = [as_float(row.get(metric)) for row in source_rows]
        finite = [value for value in values if math.isfinite(value)]
        return float(np.mean(finite)) if finite else math.nan

    return {
        "case_folder": case_dir.name,
        "best_positive_dice": round(max_metric("dice", positives), 6),
        "best_positive_iou": round(max_metric("iou", positives), 6),
        "best_precision_at_fraction": round(max_metric("precision_at_fraction", positives), 6),
        "any_positive_pointing_hit": round(max_metric("pointing_hit", positives), 6),
        f"mean_positive_dice_at_{int(reference_fraction * 100):02d}pct": round(
            mean_metric("dice", reference_positive), 6
        ),
        f"mean_positive_iou_at_{int(reference_fraction * 100):02d}pct": round(
            mean_metric("iou", reference_positive), 6
        ),
        "max_negative_mask_overlap_fraction": round(
            max_metric("negative_mask_overlap_fraction", negatives), 6
        ),
        "mean_negative_mask_avoidance_fraction": round(
            mean_metric("negative_mask_avoidance_fraction", negatives), 6
        ),
        "threshold_metric_rows": len(rows),
    }


def summarize_case_faithfulness(case_dir: Path) -> dict[str, object]:
    path = case_dir / "faithfulness_summary.csv"
    if not path.exists():
        return {"faithfulness_summary_rows": 0}
    rows = read_csv(path)

    def values_for(metric: str, view: str | None = None) -> list[float]:
        values: list[float] = []
        for row in rows:
            if view is not None and row.get("view") != view:
                continue
            value = as_float(row.get(metric))
            if math.isfinite(value):
                values.append(value)
        return values

    def mean_value(metric: str, view: str | None = None) -> float:
        values = values_for(metric, view)
        return round(float(np.mean(values)), 6) if values else math.nan

    def max_value(metric: str, view: str | None = None) -> float:
        values = values_for(metric, view)
        return round(max(values), 6) if values else math.nan

    def min_value(metric: str, view: str | None = None) -> float:
        values = values_for(metric, view)
        return round(min(values), 6) if values else math.nan

    return {
        "faithfulness_summary_rows": len(rows),
        "mean_faithfulness_insertion_auc": mean_value("faithfulness_insertion_auc"),
        "mean_faithfulness_deletion_auc": mean_value("faithfulness_deletion_auc"),
        "mean_faithfulness_deletion_drop": mean_value("faithfulness_deletion_drop"),
        "mean_faithfulness_insertion_gain": mean_value("faithfulness_insertion_gain"),
        "best_positive_faithfulness_insertion_auc": max_value(
            "faithfulness_insertion_auc", "positive"
        ),
        "best_positive_faithfulness_deletion_drop": max_value(
            "faithfulness_deletion_drop", "positive"
        ),
        "lowest_positive_faithfulness_deletion_auc": min_value(
            "faithfulness_deletion_auc", "positive"
        ),
        "mean_signed_faithfulness_insertion_auc": mean_value(
            "faithfulness_insertion_auc", "signed"
        ),
        "mean_signed_faithfulness_deletion_drop": mean_value(
            "faithfulness_deletion_drop", "signed"
        ),
    }


def score_counts(rows: list[dict[str, object]], column: str) -> list[dict[str, object]]:
    counts = Counter(str(row[column]) for row in rows)
    return [{"field": column, "value": key, "n": counts[key]} for key in sorted(counts)]


def flag_counts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for column in FLAG_COLUMNS:
        if not any(column in row for row in rows):
            continue
        present = sum(1 for row in rows if str(row.get(column, "")) == "1")
        absent = sum(1 for row in rows if str(row.get(column, "")) == "0")
        missing = sum(1 for row in rows if str(row.get(column, "")) == "")
        total = present + absent + missing
        output.append(
            {
                "flag": column,
                "present": present,
                "absent": absent,
                "missing": missing,
                "n": total,
                "present_fraction": round(present / total, 6) if total else "",
            }
        )
    return output


def image_vector(path: Path, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    image = Image.open(path).convert("L").resize(size)
    values = np.asarray(image, dtype=np.float32).reshape(-1) / 255.0
    values = values - float(values.mean())
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        return values
    return values / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


def continuous_heatmap_path(case_dir: Path, filename: str, method: str, view: str) -> Path:
    stem = Path(filename).stem
    suffix = "" if view == "positive" else f"_{view}"
    return case_dir / f"{stem}_{method}{suffix}_continuous_heatmap.png"


def pattern_similarity_rows(
    case_dirs: dict[str, Path], rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    vectors: dict[tuple[str, str, str], np.ndarray] = {}
    filenames: dict[str, str] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        filename = str(row.get("filename", ""))
        case_dir = case_dirs.get(case_id)
        if not case_id or not filename or case_dir is None:
            continue
        filenames[case_id] = filename
        for method in PATTERN_METHODS:
            for view in PATTERN_VIEWS:
                path = continuous_heatmap_path(case_dir, filename, method, view)
                if path.exists():
                    vectors[(case_id, method, view)] = image_vector(path)

    output: list[dict[str, object]] = []
    case_ids = [str(row.get("case_id", "")) for row in rows]
    for method in PATTERN_METHODS:
        for view in PATTERN_VIEWS:
            for left_index, left_case in enumerate(case_ids):
                left_vector = vectors.get((left_case, method, view))
                if left_vector is None:
                    continue
                for right_case in case_ids[left_index + 1:]:
                    right_vector = vectors.get((right_case, method, view))
                    if right_vector is None:
                        continue
                    output.append(
                        {
                            "method": method,
                            "view": view,
                            "case_id_a": left_case,
                            "filename_a": filenames.get(left_case, ""),
                            "case_id_b": right_case,
                            "filename_b": filenames.get(right_case, ""),
                            "visual_cosine_similarity": round(
                                cosine_similarity(left_vector, right_vector), 6
                            ),
                        }
                    )
    return output


def pattern_similarity_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["view"]))
        grouped.setdefault(key, []).append(float(row["visual_cosine_similarity"]))

    output: list[dict[str, object]] = []
    for (method, view), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=float)
        output.append(
            {
                "method": method,
                "view": view,
                "n_pairs": len(values),
                "mean_visual_cosine_similarity": round(float(array.mean()), 6),
                "median_visual_cosine_similarity": round(float(np.median(array)), 6),
                "max_visual_cosine_similarity": round(float(array.max()), 6),
                "min_visual_cosine_similarity": round(float(array.min()), 6),
            }
        )
    return output


def pattern_similarity_case_rows(
    pair_rows: list[dict[str, object]], joined_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in pair_rows:
        method = str(row["method"])
        view = str(row["view"])
        similarity = float(row["visual_cosine_similarity"])
        for case_column in ("case_id_a", "case_id_b"):
            key = (str(row[case_column]), method, view)
            grouped.setdefault(key, []).append(similarity)

    score_by_case = {str(row["case_id"]): row for row in joined_rows}
    output: list[dict[str, object]] = []
    for (case_id, method, view), values in sorted(grouped.items()):
        score_row = score_by_case.get(case_id, {})
        output.append(
            {
                "case_id": case_id,
                "filename": score_row.get("filename", ""),
                "method": method,
                "view": view,
                "mean_similarity_to_other_cases": round(float(np.mean(values)), 6),
                "max_similarity_to_other_case": round(float(np.max(values)), 6),
                "localization_score": score_row.get("localization_score", ""),
                "usefulness_score": score_row.get("usefulness_score", ""),
                "failure_category": score_row.get("failure_category", ""),
                "flag_weak_pixel_attribution": score_row.get("flag_weak_pixel_attribution", ""),
                "flag_method_disagreement": score_row.get("flag_method_disagreement", ""),
            }
        )
    return output


def pattern_similarity_correlation_rows(
    case_rows: list[dict[str, object]], score_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    score_by_case = {str(row["case_id"]): row for row in score_rows}
    output: list[dict[str, object]] = []
    targets = [
        "localization_score_numeric",
        "usefulness_score_numeric",
        "flag_weak_pixel_attribution",
        "flag_method_disagreement",
    ]
    for method in PATTERN_METHODS:
        for view in PATTERN_VIEWS:
            subset = [
                row for row in case_rows
                if row.get("method") == method and row.get("view") == view
            ]
            x_values = [as_float(str(row.get("mean_similarity_to_other_cases", ""))) for row in subset]
            for target in targets:
                y_values = []
                for row in subset:
                    score_row = score_by_case.get(str(row.get("case_id", "")), {})
                    y_values.append(as_float(str(score_row.get(target, ""))))
                rho, n = spearman(x_values, y_values)
                output.append(
                    {
                        "method": method,
                        "view": view,
                        "similarity_metric": "mean_similarity_to_other_cases",
                        "target": target,
                        "spearman_rho": round(rho, 6) if math.isfinite(rho) else "",
                        "n": n,
                    }
                )
    return output


def main() -> None:
    args = parse_args()
    scores = read_csv(args.scores_csv)
    selection_rows = read_csv(args.selection_csv)
    issues = validate_scores(scores, selection_rows)
    write_csv(args.output_dir / "validation_issues.csv", issues)

    case_dirs = sorted(path for path in args.diagnostics_dir.iterdir() if path.is_dir())
    case_by_id = {f"case_{index:02d}": path for index, path in enumerate(case_dirs, start=1)}

    joined_rows: list[dict[str, object]] = []
    for row in scores:
        case_id = row["case_id"]
        joined: dict[str, object] = dict(row)
        joined["localization_score_numeric"] = LOCALIZATION_ORDER.get(row["localization_score"], "")
        joined["usefulness_score_numeric"] = USEFULNESS_ORDER.get(row["usefulness_score"], "")
        case_dir = case_by_id.get(case_id)
        if case_dir is not None:
            joined.update(summarize_case_metrics(case_dir, args.reference_fraction))
            joined.update(summarize_case_faithfulness(case_dir))
        joined_rows.append(joined)
    write_csv(args.output_dir / "review_scores_with_metrics.csv", joined_rows)

    aggregate_rows: list[dict[str, object]] = []
    for column in ("localization_score", "usefulness_score", "failure_category"):
        aggregate_rows.extend(score_counts(joined_rows, column))
    write_csv(args.output_dir / "review_score_counts.csv", aggregate_rows)
    write_csv(args.output_dir / "review_flag_counts.csv", flag_counts(joined_rows))

    metric_columns = [
        "best_positive_dice",
        "best_positive_iou",
        "best_precision_at_fraction",
        "any_positive_pointing_hit",
        f"mean_positive_dice_at_{int(args.reference_fraction * 100):02d}pct",
        f"mean_positive_iou_at_{int(args.reference_fraction * 100):02d}pct",
        "max_negative_mask_overlap_fraction",
        "mean_negative_mask_avoidance_fraction",
        "mean_faithfulness_insertion_auc",
        "mean_faithfulness_deletion_auc",
        "mean_faithfulness_deletion_drop",
        "mean_faithfulness_insertion_gain",
        "best_positive_faithfulness_insertion_auc",
        "best_positive_faithfulness_deletion_drop",
        "lowest_positive_faithfulness_deletion_auc",
        "mean_signed_faithfulness_insertion_auc",
        "mean_signed_faithfulness_deletion_drop",
    ]
    correlation_rows: list[dict[str, object]] = []
    for score_column in ("localization_score_numeric", "usefulness_score_numeric"):
        y_values = [as_float(str(row.get(score_column, ""))) for row in joined_rows]
        for metric_column in metric_columns:
            x_values = [as_float(str(row.get(metric_column, ""))) for row in joined_rows]
            rho, n = spearman(x_values, y_values)
            correlation_rows.append(
                {
                    "score": score_column,
                    "metric": metric_column,
                    "spearman_rho": round(rho, 6) if math.isfinite(rho) else "",
                    "n": n,
                }
            )
    write_csv(args.output_dir / "score_metric_spearman.csv", correlation_rows)

    asset_case_dirs = {
        case_dir.name: case_dir
        for case_dir in (args.scores_csv.parent / "assets").iterdir()
        if case_dir.is_dir()
    } if (args.scores_csv.parent / "assets").is_dir() else {}
    if asset_case_dirs:
        pair_rows = pattern_similarity_rows(asset_case_dirs, joined_rows)
        write_csv(args.output_dir / "pixel_attribution_pattern_similarity_pairs.csv", pair_rows)
        write_csv(
            args.output_dir / "pixel_attribution_pattern_similarity_summary.csv",
            pattern_similarity_summary(pair_rows),
        )
        case_pattern_rows = pattern_similarity_case_rows(pair_rows, joined_rows)
        write_csv(
            args.output_dir / "pixel_attribution_pattern_similarity_by_case.csv",
            case_pattern_rows,
        )
        write_csv(
            args.output_dir / "pixel_attribution_pattern_similarity_correlations.csv",
            pattern_similarity_correlation_rows(case_pattern_rows, joined_rows),
        )

    summary_lines = [
        "# Review score analysis",
        "",
        f"Scores CSV: `{args.scores_csv}`",
        f"Diagnostics directory: `{args.diagnostics_dir}`",
        f"Rows scored: {len(scores)}",
        f"Validation issues: {len(issues)}",
        "",
        "## Outputs",
        "",
        "- `validation_issues.csv`",
        "- `review_scores_with_metrics.csv`",
        "- `review_score_counts.csv`",
        "- `review_flag_counts.csv`",
        "- `score_metric_spearman.csv`",
        "- `pixel_attribution_pattern_similarity_pairs.csv` if workbook assets are available",
        "- `pixel_attribution_pattern_similarity_summary.csv` if workbook assets are available",
        "- `pixel_attribution_pattern_similarity_by_case.csv` if workbook assets are available",
        "- `pixel_attribution_pattern_similarity_correlations.csv` if workbook assets are available",
    ]
    (args.output_dir / "README.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()