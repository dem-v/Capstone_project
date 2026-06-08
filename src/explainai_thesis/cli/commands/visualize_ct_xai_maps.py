#!/usr/bin/env python3
"""Generate CXR-like per-case XAI map panels for the CT pilot."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from PIL import Image

from explainai_thesis.cli.common import add_device_arg, add_split_arg, resolve_device
from explainai_thesis.cli.progress import RollingLogDisplay
from explainai_thesis.ct.io import BRAIN_WINDOW_LEVEL, extract_slice, load_nifti_volume
from explainai_thesis.ct.methods import (
    CTAttributionSettings,
    CT_VISUAL_METHODS,
    compute_ct_signed_attributions,
    normalize_ct_method_names,
)
from explainai_thesis.ct.models import load_ct_classifier
from explainai_thesis.ct.visualization import (
    CTVisualizationConfig,
    ct_display_tensor,
    render_ct_case_visuals,
)
from explainai_thesis.run_metadata import write_run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/ct_hemorrhage_manifest.csv")
    parser.add_argument("--output-dir", required=True)
    add_split_arg(parser, choices=("test", "train", "any"), help=None)
    parser.add_argument("--max-positive", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--methods", default=",".join(CT_VISUAL_METHODS))
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--gradshap-samples", type=int, default=32)
    parser.add_argument("--gradshap-stdevs", type=float, default=0.02)
    parser.add_argument("--occlusion-patch-size", type=int, default=32)
    parser.add_argument("--occlusion-stride", type=int, default=16)
    parser.add_argument("--smoothing-kernel", type=int, default=5)
    parser.add_argument("--no-selected", action="store_true")
    parser.add_argument("--no-signed", action="store_true")
    parser.add_argument("--no-contact-sheets", action="store_true")
    add_device_arg(parser, help=None)
    parser.add_argument("--window-level", type=float, default=BRAIN_WINDOW_LEVEL)
    parser.add_argument("--window-width", type=float, default=80.0)
    return parser.parse_args()


def read_positive_masked_ct_rows(manifest_path: Path, split: str, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["label"]) != 1:
                continue
            if split != "any" and row.get("split") != split:
                continue
            if not row.get("mask_path"):
                continue
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def load_mask_slice(mask_path: Path, slice_index: int, image_size: int) -> torch.Tensor:
    mask_vol = load_nifti_volume(mask_path)
    mask_2d = extract_slice(mask_vol, slice_index, axis=2) > 0
    resized = Image.fromarray((mask_2d * 255).astype(np.uint8)).resize(
        (image_size, image_size), Image.NEAREST
    )
    return torch.from_numpy(np.asarray(resized) > 0)


def parse_methods(raw: str) -> tuple[str, ...]:
    return normalize_ct_method_names([value.strip() for value in raw.split(",")])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_start = time.perf_counter()
    progress = RollingLogDisplay(line_count=12)
    progress.log("started CT XAI map visualization", run_start)

    methods = parse_methods(args.methods)
    device = resolve_device(args.device)
    rows = read_positive_masked_ct_rows(Path(args.manifest), args.split, args.max_positive)
    if not rows:
        raise RuntimeError(f"No positive masked CT rows in {args.manifest} for split={args.split}.")

    progress.log(f"loading CT classifier on {device}", run_start)
    bundle = load_ct_classifier(device=device, window_width=args.window_width)
    attribution_settings = CTAttributionSettings(
        ig_steps=args.ig_steps,
        gradshap_samples=args.gradshap_samples,
        gradshap_stdevs=args.gradshap_stdevs,
        occlusion_patch_size=args.occlusion_patch_size,
        occlusion_stride=args.occlusion_stride,
    )
    visualization_config = CTVisualizationConfig(
        top_fraction=args.top_fraction,
        smoothing_kernel=args.smoothing_kernel,
        save_selected=not args.no_selected,
        save_signed=not args.no_signed,
        make_contact_sheets=not args.no_contact_sheets,
    )

    summary_rows: list[dict[str, str | int | float]] = []
    for sample_idx, row in enumerate(rows):
        slice_index = int(row["slice_index"])
        progress.log(f"case {sample_idx + 1}/{len(rows)} | {row['filename']}", run_start)
        volume = load_nifti_volume(Path(row["image_path"]))
        slice_hu = extract_slice(volume, slice_index, axis=2)
        model_input = bundle.preprocess(slice_hu).to(device)
        display_image = ct_display_tensor(
            slice_hu,
            image_size=args.image_size,
            window_level=args.window_level,
            window_width=args.window_width,
        )
        mask = load_mask_slice(Path(row["mask_path"]), slice_index, args.image_size)

        with torch.inference_mode():
            hemorrhage_prob = float(torch.sigmoid(bundle.model(model_input)[0, bundle.class_idx]))

        signed_attributions = compute_ct_signed_attributions(
            bundle.model,
            model_input,
            class_idx=bundle.class_idx,
            methods=methods,
            settings=attribution_settings,
        )
        result = render_ct_case_visuals(
            display_image=display_image,
            mask=mask,
            signed_attributions=signed_attributions,
            output_dir=output_dir,
            row=row,
            sample_idx=sample_idx,
            config=visualization_config,
        )
        summary_rows.append({
            "sample_id": sample_idx,
            "filename": row.get("filename", ""),
            "split": row.get("split", ""),
            "subtype": row.get("subtype", ""),
            "slice_index": slice_index,
            "hemorrhage_prob": round(hemorrhage_prob, 6),
            "case_dir": str(result.case_dir),
            "positive_contact_sheet": str(result.contact_sheet_paths[0]) if result.contact_sheet_paths else "",
            "signed_contact_sheet": str(result.contact_sheet_paths[1]) if len(result.contact_sheet_paths) > 1 else "",
        })

    summary_path = output_dir / "ct_xai_visualization_cases.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id", "filename", "split", "subtype", "slice_index",
                "hemorrhage_prob", "case_dir", "positive_contact_sheet", "signed_contact_sheet",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    run_meta_path = write_run_metadata(
        output_dir,
        args,
        model_id="DifeiT/rsna-intracranial-hemorrhage-detection",
        attribution_target="1 - P(normal)",
        split=args.split,
        modality="ct",
        methods=methods,
        output_type="ct_xai_maps",
    )

    progress.log("CT XAI map visualization complete", run_start)
    progress.finish()
    print(f"CT XAI map visualization complete on {device}.")
    print(f"Cases: {len(rows)}")
    print(f"Summary: {summary_path}")
    print(f"Run metadata: {run_meta_path}")


if __name__ == "__main__":
    main()