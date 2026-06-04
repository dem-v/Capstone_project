from __future__ import annotations

import pytest

from explainai_thesis.ct.methods import (
    CT_CONSENSUS_NAME,
    CT_INPUT_SPACE_METHODS,
    normalize_ct_method_names,
    required_input_methods,
)


def test_normalize_ct_method_names_deduplicates_and_preserves_order() -> None:
    methods = normalize_ct_method_names([
        "gradient_shap",
        "integrated_gradients",
        "gradient_shap",
    ])

    assert methods == ("gradient_shap", "integrated_gradients")


def test_normalize_ct_method_names_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unsupported CT XAI method"):
        normalize_ct_method_names(["lime"])


def test_normalize_ct_method_names_requires_one_method() -> None:
    with pytest.raises(ValueError, match="At least one"):
        normalize_ct_method_names(["", "  "])


def test_required_input_methods_expands_consensus_dependencies() -> None:
    required = required_input_methods([CT_CONSENSUS_NAME])

    assert required == CT_INPUT_SPACE_METHODS


def test_required_input_methods_keeps_non_consensus_subset() -> None:
    required = required_input_methods(["occlusion", "integrated_gradients"])

    assert required == ("occlusion", "integrated_gradients")