"""CT slice I/O and HU windowing for the head-CT hemorrhage pilot.

All transforms here are locked per `docs/refactor_plan.md` § "Phase 5.4 —
HU windowing — locked choices": the thesis brain window is WL=40, WW=80
(clinical standard for hemorrhage evaluation). The windowing math is pure
and CPU-cheap, so this module imports without `transformers`, a GPU, or any
network access. NIfTI reading is the only optional dependency and is
imported lazily so the windowing/slice helpers work even where `nibabel`
is absent.

The model-specific normalization (ImageNet mean/std for the ViT candidate)
is intentionally NOT applied here — that belongs to the deferred
`ct/models.py` bundle and its `AutoImageProcessor`. This module produces a
display-normalized [0, 1] 3-channel array; downstream preprocessing
finishes the job.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


# Locked brain window for the thesis runs (clinical hemorrhage standard).
BRAIN_WINDOW_LEVEL = 40.0
BRAIN_WINDOW_WIDTH = 80.0

# Secondary window used by the Hssayeni dataset paper (WL=40, WW=120). Kept
# for provenance only; thesis runs use the brain window above.
HSSAYENI_WINDOW_WIDTH = 120.0


def window_hu(
    slice_hu: np.ndarray,
    *,
    window_level: float = BRAIN_WINDOW_LEVEL,
    window_width: float = BRAIN_WINDOW_WIDTH,
) -> np.ndarray:
    """Map raw Hounsfield-Unit values into a [0, 1] display range.

    Implements the locked formula:
        lower = WL - WW/2; upper = WL + WW/2
        display = clip(slice_hu, lower, upper) shifted to [0, 1] over [lower, upper].
    The window midpoint (HU = WL) maps to 0.5, which is exactly the
    `brain_window_center` faithfulness-baseline fill value (in display space,
    before any model normalization).
    """
    if window_width <= 0:
        raise ValueError("window_width must be positive")
    lower = window_level - window_width / 2.0
    upper = window_level + window_width / 2.0
    clipped = np.clip(np.asarray(slice_hu, dtype=np.float32), lower, upper)
    display = (clipped - lower) / window_width
    return display.astype(np.float32)


def to_three_channel(display: np.ndarray) -> np.ndarray:
    """Replicate a 2-D display image into a 3-channel (H, W, 3) array.

    The ViT-base candidate expects 3-channel RGB input; brain-window CT is
    single-channel, so the canonical handling is channel replication.
    """
    array = np.asarray(display, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D display image, got shape {array.shape}")
    return np.stack([array, array, array], axis=-1)


def resize_display(display: np.ndarray, size: int) -> np.ndarray:
    """Resize a 2-D [0, 1] display image to (size, size), preserving floats.

    Uses bilinear interpolation in PIL's 32-bit float mode so display
    precision survives the resize (no uint8 quantization).
    """
    array = np.asarray(display, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D display image, got shape {array.shape}")
    if size <= 0:
        raise ValueError("size must be positive")
    image = Image.fromarray(array, mode="F").resize((size, size), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def extract_slice(volume: np.ndarray, index: int, *, axis: int = 2) -> np.ndarray:
    """Extract a single 2-D slice from a 3-D volume along `axis` (default
    axial / last axis, matching NIfTI (X, Y, Z) convention)."""
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"expected a 3-D volume, got shape {array.shape}")
    return np.take(array, index, axis=axis).astype(np.float32)


def preprocess_ct_slice(
    slice_hu: np.ndarray,
    *,
    size: int = 224,
    window_level: float = BRAIN_WINDOW_LEVEL,
    window_width: float = BRAIN_WINDOW_WIDTH,
) -> np.ndarray:
    """Compose the locked CT preprocessing: brain-window → resize → 3-channel.

    Returns a (size, size, 3) float32 array in [0, 1], ready to hand to the
    model's `AutoImageProcessor` (which applies the model-specific
    normalization). Kept deliberately free of model normalization so this
    stays a pure, testable transform.
    """
    display = window_hu(slice_hu, window_level=window_level, window_width=window_width)
    resized = resize_display(display, size)
    return to_three_channel(resized)


def load_nifti_volume(path: Path) -> np.ndarray:
    """Load a NIfTI volume as a float32 numpy array (HU values preserved).

    `nibabel` is imported lazily so this module is usable for windowing/slice
    work even where the NIfTI reader is not installed; the import error is
    surfaced only when a NIfTI file is actually read.
    """
    try:
        import nibabel  # noqa: PLC0415  (lazy by design)
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "Reading NIfTI volumes requires `nibabel`; add it to "
            "requirements-dev.txt and `pip install nibabel`."
        ) from exc
    return np.asarray(nibabel.load(str(path)).get_fdata(), dtype=np.float32)
