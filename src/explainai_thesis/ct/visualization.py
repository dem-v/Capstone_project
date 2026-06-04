"""Reusable CT XAI visualization helpers.

The CLI layer should only load data/model state and call these functions.  The
output naming, CT window display conversion, overlay rendering, and contact
sheet assembly live here so future modality/systematization work has a stable
component to reuse.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import torch
from PIL import Image

from explainai_thesis.metrics import threshold_top_fraction
from explainai_thesis.visualization import (
    make_contact_sheet,
    readable_heatmap_for_method,
    save_binary_selection,
    save_overlay,
    signed_diverging_overlay,
)
from explainai_thesis.xai import SignedAttribution, iter_method_views

from .io import BRAIN_WINDOW_LEVEL, BRAIN_WINDOW_WIDTH, resize_display, window_hu


@dataclass(frozen=True)
class CTVisualizationConfig:
    """Configuration for per-case CT XAI rendering."""

    top_fraction: float = 0.05
    smoothing_kernel: int = 5
    save_selected: bool = True
    save_signed: bool = True
    make_contact_sheets: bool = True


@dataclass(frozen=True)
class CTCaseVisualizationResult:
    """Paths emitted for one visualized CT slice."""

    case_dir: Path
    source_stem: str
    slice_path: Path
    mask_path: Path
    overlay_paths: tuple[Path, ...]
    selected_paths: tuple[Path, ...]
    contact_sheet_paths: tuple[Path, ...]


def safe_ct_source_stem(row: dict[str, str]) -> str:
    """Return a filesystem-safe source stem for a CT manifest row."""

    stem = Path(row.get("filename") or row["image_path"]).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return safe_stem or "ct_slice"


def safe_ct_case_name(sample_idx: int, row: dict[str, str]) -> str:
    """Return a stable per-slice output directory name."""

    slice_index = int(row.get("slice_index", 0))
    return f"case_{sample_idx:03d}_{safe_ct_source_stem(row)}_slice_{slice_index:03d}"


def ct_display_tensor(
    slice_hu: np.ndarray,
    *,
    image_size: int = 224,
    window_level: float = BRAIN_WINDOW_LEVEL,
    window_width: float = BRAIN_WINDOW_WIDTH,
) -> torch.Tensor:
    """Convert an HU slice into a single-channel display tensor in [0, 1]."""

    display = window_hu(slice_hu, window_level=window_level, window_width=window_width)
    resized = resize_display(display, image_size)
    return torch.from_numpy(resized.copy()).unsqueeze(0).float()


def save_ct_slice_image(image: torch.Tensor, output_path: str | Path) -> None:
    """Save a CT display tensor without per-image min-max renormalization."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = image.detach().cpu().float()
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(
                f"expected a 2-D or single-channel image tensor, got shape {tuple(image.shape)}"
            )
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D or single-channel image tensor, got shape {tuple(image.shape)}")
    pixels = (array.clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(output_path)


def _validate_visualization_config(config: CTVisualizationConfig) -> None:
    if not 0 < config.top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1].")
    if config.smoothing_kernel < 1:
        raise ValueError("smoothing_kernel must be at least 1.")
    if config.smoothing_kernel > 1 and config.smoothing_kernel % 2 == 0:
        raise ValueError("smoothing_kernel must be odd or 1.")


def _overlay_color(view: str) -> str:
    if view == "negative":
        return "blue"
    if view == "magnitude":
        return "neutral"
    return "red"


def render_ct_case_visuals(
    *,
    display_image: torch.Tensor,
    mask: torch.Tensor,
    signed_attributions: dict[str, SignedAttribution],
    output_dir: Path,
    row: dict[str, str],
    sample_idx: int,
    config: CTVisualizationConfig | None = None,
) -> CTCaseVisualizationResult:
    """Render CXR-like CT overlays for one case.

    Generated filenames always include the source CT stem and slice index so a
    copied image remains traceable without its parent directory.
    """

    config = config or CTVisualizationConfig()
    _validate_visualization_config(config)

    case_dir = output_dir / safe_ct_case_name(sample_idx, row)
    case_dir.mkdir(parents=True, exist_ok=True)
    source_stem = f"{safe_ct_source_stem(row)}_slice_{int(row.get('slice_index', 0)):03d}"

    slice_path = case_dir / f"{source_stem}_ct_slice.png"
    mask_path = case_dir / f"{source_stem}_mask_contour.png"
    save_ct_slice_image(display_image, slice_path)
    save_overlay(
        display_image,
        torch.zeros_like(display_image[0] if display_image.ndim == 3 else display_image),
        mask,
        mask_path,
        alpha=0.0,
        heatmap_color="red",
    )

    overlay_paths: list[Path] = []
    selected_paths: list[Path] = []
    positive_contact_paths = [slice_path, mask_path]
    positive_captions = ["CT slice", "Mask contour"]
    signed_contact_paths = [slice_path, mask_path]
    signed_captions = ["CT slice", "Mask contour"]

    for method_view in iter_method_views(signed_attributions):
        if method_view.view == "signed" and not config.save_signed:
            continue

        overlay_path = case_dir / f"{source_stem}_{method_view.method}.png"
        if method_view.view == "signed":
            signed_diverging_overlay(
                display_image,
                signed_attributions[method_view.family].signed,
                mask,
                overlay_path,
            )
            signed_contact_paths.append(overlay_path)
            signed_captions.append(method_view.method)
        else:
            heatmap = readable_heatmap_for_method(
                method_view.heatmap,
                method_view.family,
                config.smoothing_kernel,
            )
            save_overlay(
                display_image,
                heatmap,
                mask,
                overlay_path,
                heatmap_color=_overlay_color(method_view.view),
            )
            if method_view.view == "positive":
                positive_contact_paths.append(overlay_path)
                positive_captions.append(method_view.method)
            if config.save_selected:
                selected = threshold_top_fraction(heatmap, fraction=config.top_fraction)
                selected_path = case_dir / f"{source_stem}_{method_view.method}_selected.png"
                save_binary_selection(
                    display_image,
                    selected,
                    mask,
                    selected_path,
                    negative_style=(method_view.view == "negative"),
                    neutral_style=(method_view.view == "magnitude"),
                )
                selected_paths.append(selected_path)
        overlay_paths.append(overlay_path)

    contact_sheet_paths: list[Path] = []
    if config.make_contact_sheets:
        positive_sheet = case_dir / f"{source_stem}_positive_overview_contact_sheet.png"
        make_contact_sheet(positive_contact_paths, positive_captions, positive_sheet)
        contact_sheet_paths.append(positive_sheet)
        if config.save_signed and len(signed_contact_paths) > 2:
            signed_sheet = case_dir / f"{source_stem}_signed_overview_contact_sheet.png"
            make_contact_sheet(signed_contact_paths, signed_captions, signed_sheet)
            contact_sheet_paths.append(signed_sheet)

    return CTCaseVisualizationResult(
        case_dir=case_dir,
        source_stem=source_stem,
        slice_path=slice_path,
        mask_path=mask_path,
        overlay_paths=tuple(overlay_paths),
        selected_paths=tuple(selected_paths),
        contact_sheet_paths=tuple(contact_sheet_paths),
    )