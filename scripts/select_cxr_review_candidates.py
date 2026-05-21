from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


POSITIVE_METHODS = {
    "grad_cam",
    "grad_cam_plus_plus",
    "integrated_gradients",
    "integrated_gradients_signed",
    "gradient_shap",
    "gradient_shap_signed",
    "occlusion",
    "consensus",
}

NEGATIVE_METHODS = {
    "grad_cam_negative",
    "grad_cam_plus_plus_negative",
    "integrated_gradients_negative",
    "integrated_gradients_signed",
    "gradient_shap_negative",
    "gradient_shap_signed",
    "occlusion_negative",
    "consensus",
}


def is_positive_metric_row(row: dict[str, str]) -> bool:
    view = row.get("view")
    if view:
        return view in {"positive", "signed"}
    return row.get("method") in POSITIVE_METHODS


def is_negative_metric_row(row: dict[str, str]) -> bool:
    view = row.get("view")
    if view:
        return view in {"negative", "signed"}
    return row.get("method") in NEGATIVE_METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select representative CXR classifier-outcome cases for manual XAI review."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/iter_27_classifier_outcome_any5000_balanced100_all_methods"),
        help="Classifier-outcome output directory containing cases.csv and threshold_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/iter_28_review_candidate_selection"),
        help="Directory where ranked candidate CSVs and rerun commands will be written.",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Rows to keep per ranked list.")
    parser.add_argument(
        "--selected-per-category",
        type=int,
        default=2,
        help="Candidate cases to select from each category for the short manual-review list.",
    )
    parser.add_argument("--ig-steps", type=int, default=16, help="IG steps for generated diagnostic commands.")
    parser.add_argument(
        "--gradshap-samples",
        type=int,
        default=64,
        help="GradientSHAP samples for generated diagnostic commands.",
    )
    parser.add_argument("--occlusion-patch-size", type=int, default=32)
    parser.add_argument("--occlusion-stride", type=int, default=12)
    parser.add_argument(
        "--fractions",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
        help="Top-fraction sweep for generated diagnostic commands.",
    )
    parser.add_argument(
        "--diagnostic-root",
        default="outputs/iter_28_review_diagnostics",
        help="Root prefix for generated high-stability diagnostic output folders.",
    )
    parser.add_argument(
        "--weights",
        default="densenet121-res224-all",
        help="Classifier weights to pass to generated diagnostic commands.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image size to pass to generated diagnostic commands; use 512 for resnet50-res512-all.",
    )
    parser.add_argument(
        "--max-selected",
        type=int,
        default=10,
        help="Maximum number of manual-review cases selected across categories.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str | None, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "case"


def case_key(row: dict[str, str]) -> str:
    return row["sample_index"]


def best_positive_rows(metrics: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for row in metrics:
        if row.get("positive_localization_applicable") != "1":
            continue
        if not is_positive_metric_row(row):
            continue
        key = case_key(row)
        dice = as_float(row.get("dice"))
        iou = as_float(row.get("iou"))
        precision = as_float(row.get("precision_at_fraction"))
        current = best.get(key)
        if current is None or (dice, iou, precision) > (
            float(current["best_dice"]),
            float(current["best_iou"]),
            float(current["best_precision_at_fraction"]),
        ):
            best[key] = {
                "best_method": row["method"],
                "best_top_fraction": row["top_fraction"],
                "best_dice": dice,
                "best_iou": iou,
                "best_precision_at_fraction": precision,
                "best_pointing_hit": as_float(row.get("pointing_hit")),
            }
    return best


def strongest_negative_rows(metrics: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    strongest: dict[str, dict[str, object]] = {}
    for row in metrics:
        if row.get("label") != "1":
            continue
        if not is_negative_metric_row(row):
            continue
        if row.get("negative_mask_overlap_fraction") in (None, ""):
            continue
        key = case_key(row)
        overlap = as_float(row.get("negative_mask_overlap_fraction"))
        current = strongest.get(key)
        if current is None or overlap > float(current["max_negative_mask_overlap_fraction"]):
            strongest[key] = {
                "negative_method": row["method"],
                "negative_top_fraction": row["top_fraction"],
                "max_negative_mask_overlap_fraction": overlap,
                "negative_mask_avoidance_fraction": as_float(row.get("negative_mask_avoidance_fraction")),
            }
    return strongest


def signed_diagnostic_rows(metrics: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    diagnostics: dict[str, dict[str, object]] = {}
    for row in metrics:
        method = row.get("method", "")
        view = row.get("view", "")
        if view != "signed" and not method.endswith("_signed"):
            continue
        key = case_key(row)
        signed_positive_fraction = as_float(row.get("signed_positive_fraction"), default=-1.0)
        current = diagnostics.get(key)
        if current is None or signed_positive_fraction > float(current["max_signed_positive_fraction"]):
            diagnostics[key] = {
                "signed_method": method,
                "signed_top_fraction": row.get("top_fraction", ""),
                "max_signed_positive_fraction": signed_positive_fraction,
                "signed_prediction_alignment": as_float(row.get("signed_prediction_alignment"), default=0.0),
            }
    return diagnostics


def enrich_case(
    case: dict[str, str],
    category: str,
    positive: dict[str, object] | None,
    negative: dict[str, object] | None,
    signed: dict[str, object] | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "category": category,
        "sample_index": case["sample_index"],
        "candidate_index": case.get("candidate_index", ""),
        "filename": case["filename"],
        "split": case["split"],
        "label": case["label"],
        "prediction": case["prediction"],
        "classifier_outcome": case["classifier_outcome"],
        "xrv_pneumothorax_sigmoid": as_float(case.get("xrv_pneumothorax_sigmoid")),
        "classifier_threshold": as_float(case.get("classifier_threshold")),
        "image_path": case.get("image_path", ""),
        "mask_path": case.get("mask_path", ""),
        "weights": case.get("weights", ""),
        "image_size": case.get("image_size", ""),
    }
    if positive:
        row.update(positive)
    if negative:
        row.update(negative)
    if signed:
        row.update(signed)
    return row


def select_unique(rows: list[dict[str, object]], limit: int, used: set[str]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in rows:
        key = str(row["sample_index"])
        if key in used:
            continue
        selected.append(row)
        used.add(key)
        if len(selected) >= limit:
            break
    return selected


def diagnostic_command(row: dict[str, object], args: argparse.Namespace, rank: int) -> str:
    filename = str(row["filename"])
    source_stem = safe_name(Path(filename).stem)
    category = safe_name(str(row["category"]))
    output_dir = f"{args.diagnostic_root}/case_{rank:02d}_{category}_{source_stem}"
    return (
        "wsl.exe python3 scripts/visualize_cxr_threshold_selection.py "
        "--device auto "
        "--split any "
        f"--weights {args.weights} "
        f"--image-size {args.image_size} "
        f"--case-filename {filename} "
        f"--ig-steps {args.ig_steps} "
        f"--gradshap-samples {args.gradshap_samples} "
        f"--occlusion-patch-size {args.occlusion_patch_size} "
        f"--occlusion-stride {args.occlusion_stride} "
        f"--fractions {args.fractions} "
        f"--output-dir {output_dir}"
    )


def main() -> None:
    args = parse_args()
    cases_path = args.input_dir / "cases.csv"
    metrics_path = args.input_dir / "threshold_metrics.csv"
    cases = read_csv(cases_path)
    metrics = read_csv(metrics_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases_by_key = {case_key(row): row for row in cases}
    positive_by_case = best_positive_rows(metrics)
    negative_by_case = strongest_negative_rows(metrics)
    signed_by_case = signed_diagnostic_rows(metrics)

    enriched: list[dict[str, object]] = []
    for key, case in cases_by_key.items():
        enriched.append(
            enrich_case(
                case,
                "all_cases",
                positive_by_case.get(key),
                negative_by_case.get(key),
                signed_by_case.get(key),
            )
        )

    tp = [row for row in enriched if row["classifier_outcome"] == "tp"]
    fp = [row for row in enriched if row["classifier_outcome"] == "fp"]
    fn = [row for row in enriched if row["classifier_outcome"] == "fn"]

    best_tp = sorted(tp, key=lambda row: (float(row.get("best_dice", 0.0)), float(row.get("best_iou", 0.0))), reverse=True)
    suspicious_tp = sorted(tp, key=lambda row: (float(row.get("best_dice", 0.0)), -float(row["xrv_pneumothorax_sigmoid"])))
    strong_fp = sorted(fp, key=lambda row: float(row["xrv_pneumothorax_sigmoid"]), reverse=True)
    good_fn = sorted(fn, key=lambda row: (float(row.get("best_dice", 0.0)), -float(row["xrv_pneumothorax_sigmoid"])), reverse=True)
    high_negative_overlap = sorted(
        [row for row in enriched if "max_negative_mask_overlap_fraction" in row],
        key=lambda row: float(row["max_negative_mask_overlap_fraction"]),
        reverse=True,
    )
    signed_model_aligned = sorted(
        [row for row in enriched if "max_signed_positive_fraction" in row],
        key=lambda row: (float(row["signed_prediction_alignment"]), float(row["max_signed_positive_fraction"])),
        reverse=True,
    )

    ranked_lists = {
        "best_tp_by_dice_iou": best_tp,
        "suspicious_tp_low_dice_positive_prediction": suspicious_tp,
        "fp_high_classifier_score_strong_positive_evidence_proxy": strong_fp,
        "fn_good_localization_low_classifier_score": good_fn,
        "high_negative_evidence_inside_mask": high_negative_overlap,
        "signed_evidence_model_aligned": signed_model_aligned,
    }
    for name, rows in ranked_lists.items():
        write_csv(args.output_dir / f"{name}.csv", rows[: args.top_n])

    used: set[str] = set()
    selected: list[dict[str, object]] = []
    for name in [
        "best_tp_by_dice_iou",
        "suspicious_tp_low_dice_positive_prediction",
        "fp_high_classifier_score_strong_positive_evidence_proxy",
        "fn_good_localization_low_classifier_score",
        "high_negative_evidence_inside_mask",
        "signed_evidence_model_aligned",
    ]:
        rows = []
        for row in ranked_lists[name]:
            copied = dict(row)
            copied["category"] = name
            rows.append(copied)
        selected.extend(select_unique(rows, args.selected_per_category, used))

    selected = selected[: args.max_selected]
    for rank, row in enumerate(selected, start=1):
        row["review_rank"] = rank
        row["diagnostic_command"] = diagnostic_command(row, args, rank)

    write_csv(args.output_dir / "selected_manual_review_cases.csv", selected)
    with (args.output_dir / "run_selected_high_stability_diagnostics.ps1").open("w", encoding="utf-8") as handle:
        handle.write("# Run from the repository root in PowerShell.\n")
        handle.write("# These commands rerun selected cases with high-stability GradientSHAP and finer occlusion settings.\n\n")
        for row in selected:
            handle.write(f"# {row['review_rank']}. {row['category']} | {row['filename']}\n")
            handle.write(str(row["diagnostic_command"]) + "\n\n")

    print(f"Read {len(cases)} cases and {len(metrics)} metric rows from {args.input_dir}")
    print(f"Wrote ranked candidate lists to {args.output_dir}")
    print(f"Selected {len(selected)} manual-review cases")
    print(f"Run commands: {args.output_dir / 'run_selected_high_stability_diagnostics.ps1'}")


if __name__ == "__main__":
    main()