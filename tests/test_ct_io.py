from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from explainai_thesis.ct.io import (
    BRAIN_WINDOW_LEVEL,
    BRAIN_WINDOW_WIDTH,
    extract_slice,
    load_nifti_volume,
    preprocess_ct_slice,
    resize_display,
    to_three_channel,
    window_hu,
)


def test_window_hu_midpoint_and_bounds() -> None:
    # Brain window WL=40, WW=80 → display range HU 0..80.
    hu = np.array([0.0, 20.0, 40.0, 60.0, 80.0], dtype=np.float32)
    display = window_hu(hu)
    np.testing.assert_allclose(display, [0.0, 0.25, 0.5, 0.75, 1.0], atol=1e-6)


def test_window_hu_clips_outside_window() -> None:
    hu = np.array([-1024.0, -1.0, 81.0, 3000.0], dtype=np.float32)
    display = window_hu(hu)
    # Below the window floor clips to 0.0; above the ceiling clips to 1.0.
    np.testing.assert_allclose(display, [0.0, 0.0, 1.0, 1.0], atol=1e-6)


def test_window_hu_midpoint_is_baseline_fill() -> None:
    # The brain_window_center faithfulness baseline (display space) is the
    # window midpoint mapping to exactly 0.5.
    midpoint = window_hu(np.array([BRAIN_WINDOW_LEVEL], dtype=np.float32))
    assert midpoint[0] == pytest.approx(0.5, abs=1e-6)


def test_window_hu_custom_width_matches_formula() -> None:
    # Hssayeni secondary window WW=120 centered at 40 → range HU -20..100.
    # Midpoint HU 40 → 0.5; HU 70 → (70+20)/120 = 0.75; HU 100 → ceiling 1.0.
    display = window_hu(
        np.array([40.0, 70.0, 100.0], dtype=np.float32),
        window_level=BRAIN_WINDOW_LEVEL,
        window_width=120.0,
    )
    np.testing.assert_allclose(display, [0.5, 0.75, 1.0], atol=1e-6)


def test_window_hu_rejects_nonpositive_width() -> None:
    with pytest.raises(ValueError, match="window_width"):
        window_hu(np.zeros((2, 2), dtype=np.float32), window_width=0.0)


def test_to_three_channel_shape_and_replication() -> None:
    display = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    rgb = to_three_channel(display)
    assert rgb.shape == (2, 2, 3)
    np.testing.assert_array_equal(rgb[..., 0], rgb[..., 1])
    np.testing.assert_array_equal(rgb[..., 1], rgb[..., 2])


def test_to_three_channel_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        to_three_channel(np.zeros((2, 2, 3), dtype=np.float32))


def test_resize_display_shape_and_range() -> None:
    display = np.linspace(0.0, 1.0, num=64, dtype=np.float32).reshape(8, 8)
    resized = resize_display(display, 224)
    assert resized.shape == (224, 224)
    assert resized.min() >= 0.0 and resized.max() <= 1.0


def test_extract_slice_axial() -> None:
    volume = np.stack(
        [np.full((4, 4), fill_value=float(z), dtype=np.float32) for z in range(5)],
        axis=2,
    )
    assert volume.shape == (4, 4, 5)
    slice_2 = extract_slice(volume, 2, axis=2)
    assert slice_2.shape == (4, 4)
    np.testing.assert_array_equal(slice_2, np.full((4, 4), 2.0, dtype=np.float32))


def test_extract_slice_rejects_non_3d() -> None:
    with pytest.raises(ValueError, match="3-D"):
        extract_slice(np.zeros((4, 4), dtype=np.float32), 0)


def test_preprocess_ct_slice_end_to_end() -> None:
    # Synthetic HU slice with a bright hemorrhage-like blob.
    slice_hu = np.full((128, 160), fill_value=-20.0, dtype=np.float32)
    slice_hu[40:80, 50:90] = 70.0
    processed = preprocess_ct_slice(slice_hu, size=224)
    assert processed.shape == (224, 224, 3)
    assert processed.min() >= 0.0 and processed.max() <= 1.0
    # The blob (HU 70 → 0.875) should produce values well above the background
    # (HU -20 → 0.0) somewhere in the image.
    assert processed.max() > 0.5


def test_load_nifti_volume_roundtrip(tmp_path: Path) -> None:
    nibabel = pytest.importorskip("nibabel")
    volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    nifti_path = tmp_path / "synthetic.nii.gz"
    nibabel.save(nibabel.Nifti1Image(volume, affine=np.eye(4)), str(nifti_path))

    loaded = load_nifti_volume(nifti_path)
    assert loaded.shape == (2, 3, 4)
    np.testing.assert_allclose(loaded, volume, atol=1e-6)
