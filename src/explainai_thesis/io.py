"""Single source of truth for output CSV column lists and field orders.

Importing these constants instead of repeating fieldname lists prevents
drift between writers and downstream readers. AGENT.md freezes the
output schema, so any change here must be deliberate and reflected in
the chronological progress log.
"""
from __future__ import annotations

from typing import Final


METRICS_FIELDS: Final[tuple[str, ...]] = (
    "sample_id",
    "filename",
    "split",
    "xrv_pneumothorax_score",
    "xrv_pneumothorax_sigmoid",
    "method",
    "view",
    "family",
    "top_fraction",
    "iou",
    "dice",
    "pointing_hit",
    "precision_at_fraction",
    "negative_mask_overlap_fraction",
    "negative_mask_avoidance_fraction",
    "signed_positive_fraction",
    "signed_prediction_alignment",
)


AGREEMENT_FIELDS: Final[tuple[str, ...]] = (
    "sample_id",
    "filename",
    "split",
    "method_a",
    "method_b",
    "agreement_score",
)


FAITHFULNESS_CURVE_FIELDS: Final[tuple[str, ...]] = (
    "sample_id",
    "filename",
    "split",
    "method",
    "baseline",
    "fraction",
    "insertion_probability",
    "deletion_probability",
)


FAITHFULNESS_SUMMARY_FIELDS: Final[tuple[str, ...]] = (
    "method",
    "case_count",
    "insertion_auc_mean",
    "deletion_auc_mean",
    "deletion_drop_auc_mean",
)
