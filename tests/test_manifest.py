"""Phase 1.5 — manifest label-inference robustness.

The pre-1.5 substring markers (``_1_``, ``_0_``) would also match
``_10_``, ``_11_``, ``_100_``, ``_01_``. The regex variant in
``infer_label_from_name`` now requires the digit/keyword to appear as a
standalone token bounded by ``_`` / ``-`` / ``.`` (or string start/end).
"""

from __future__ import annotations

from pathlib import Path

from explainai_thesis.manifest import infer_label_from_name


def test_exact_positive_token_returns_one() -> None:
    assert infer_label_from_name(Path("study_1_seg.png")) == 1
    assert infer_label_from_name(Path("image_pneumo.png")) == 1
    assert infer_label_from_name(Path("foo-positive-bar.png")) == 1


def test_exact_negative_token_returns_zero() -> None:
    assert infer_label_from_name(Path("study_0_seg.png")) == 0
    assert infer_label_from_name(Path("image_normal.png")) == 0
    assert infer_label_from_name(Path("foo-negative-bar.png")) == 0


def test_adversarial_multidigit_does_not_match_one() -> None:
    """``_10_`` / ``_11_`` / ``_100_`` must NOT be inferred as label=1."""
    assert infer_label_from_name(Path("case_10_chest.png")) is None
    assert infer_label_from_name(Path("case_11_chest.png")) is None
    assert infer_label_from_name(Path("case_100_chest.png")) is None


def test_adversarial_multidigit_does_not_match_zero() -> None:
    """``_01_`` / ``_02_`` must NOT be inferred as label=0.

    ``image_01_pneumothorax.png`` does match label=1 via the standalone
    ``pneumothorax`` token — that is intentional and unrelated to the
    ``_01_`` adversarial case; the test below isolates the digit-only
    pathway.
    """
    assert infer_label_from_name(Path("scan_01_chest.png")) is None
    assert infer_label_from_name(Path("scan_02_chest.png")) is None
    assert infer_label_from_name(Path("image_01_pneumothorax.png")) == 1


def test_pneumothorax_keyword_is_positive() -> None:
    # Standalone keyword token, with or without surrounding underscores.
    assert infer_label_from_name(Path("study_pneumothorax.png")) == 1
    assert infer_label_from_name(Path("study-pneumothorax-2.png")) == 1


def test_no_marker_returns_none() -> None:
    assert infer_label_from_name(Path("random_image.png")) is None
    assert infer_label_from_name(Path("chestXray.png")) is None
