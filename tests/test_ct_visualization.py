from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from explainai_thesis.ct.visualization import (
    CTVisualizationConfig,
    ct_display_tensor,
    render_ct_case_visuals,
    safe_ct_case_name,
    safe_ct_source_stem,
    save_ct_slice_image,
)
from explainai_thesis.xai import SignedAttribution


def test_ct_display_tensor_preserves_window_scale() -> None:
    slice_hu = np.array([[0.0, 40.0], [80.0, 120.0]], dtype=np.float32)
    display = ct_display_tensor(slice_hu, image_size=2)

    assert display.shape == (1, 2, 2)
    torch.testing.assert_close(display[0], torch.tensor([[0.0, 0.5], [1.0, 1.0]]))


def test_safe_ct_case_name_sanitizes_filename_and_includes_slice() -> None:
    row = {
        "filename": "patient 01/scan.nii.gz",
        "image_path": "ignored.nii.gz",
        "slice_index": "7",
    }

    assert safe_ct_source_stem(row) == "scan.nii"
    assert safe_ct_case_name(3, row) == "case_003_scan.nii_slice_007"


def test_save_ct_slice_image_rejects_invalid_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected a 2-D"):
        save_ct_slice_image(torch.zeros(2, 3, 4), tmp_path / "bad.png")


def test_render_ct_case_visuals_writes_overlays_and_contact_sheets(tmp_path: Path) -> None:
    display = torch.linspace(0.0, 1.0, steps=64, dtype=torch.float32).reshape(1, 8, 8)
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[2:5, 3:6] = True
    raw = torch.zeros(8, 8, dtype=torch.float32)
    raw[2:5, 3:6] = 1.0
    raw[0:2, 0:2] = -0.5
    row = {
        "filename": "ct case 01.nii.gz",
        "image_path": "ct case 01.nii.gz",
        "slice_index": "4",
    }

    result = render_ct_case_visuals(
        display_image=display,
        mask=mask,
        signed_attributions={"integrated_gradients": SignedAttribution(raw)},
        output_dir=tmp_path,
        row=row,
        sample_idx=0,
        config=CTVisualizationConfig(top_fraction=0.25, smoothing_kernel=1),
    )

    assert result.slice_path.exists()
    assert result.mask_path.exists()
    assert len(result.overlay_paths) == 4
    assert len(result.selected_paths) == 3
    assert len(result.contact_sheet_paths) == 2
    for path in (*result.overlay_paths, *result.selected_paths, *result.contact_sheet_paths):
        assert path.exists()
    assert Image.open(result.contact_sheet_paths[0]).size[0] > 0


def test_render_ct_case_visuals_rejects_invalid_top_fraction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="top_fraction"):
        render_ct_case_visuals(
            display_image=torch.zeros(1, 4, 4),
            mask=torch.zeros(4, 4, dtype=torch.bool),
            signed_attributions={"integrated_gradients": SignedAttribution(torch.zeros(4, 4))},
            output_dir=tmp_path,
            row={"filename": "case.nii.gz", "image_path": "case.nii.gz", "slice_index": "0"},
            sample_idx=0,
            config=CTVisualizationConfig(top_fraction=0.0),
        )