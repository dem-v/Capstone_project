from __future__ import annotations

import pytest

from explainai_thesis.stats import (
    bootstrap_paired_diff_ci,
    holm_bonferroni,
    paired_finite_values,
    wilcoxon_paired,
)


def test_wilcoxon_paired_known_inputs() -> None:
    reference = [0.3, 0.4, 0.6, 0.8, 1.0]
    alternative = [0.1, 0.5, 0.5, 0.7, 0.9]

    result = wilcoxon_paired(
        reference,
        {"method_a": alternative},
        n_resamples=100,
        seed=0,
    )["method_a"]

    assert result["n_pairs"] == 5
    assert result["wilcoxon_stat"] == pytest.approx(2.0, abs=1e-12)
    assert result["p_raw"] == pytest.approx(0.25, abs=1e-12)
    assert result["median_diff"] == pytest.approx(0.1, abs=1e-12)


def test_holm_bonferroni_known_pvalues() -> None:
    result = holm_bonferroni([0.001, 0.01, 0.03, 0.04, 0.2], alpha=0.05)

    assert [row["holm_significant"] for row in result] == [True, True, False, False, False]
    assert [row["p_holm_threshold"] for row in result] == pytest.approx(
        [0.01, 0.0125, 0.05 / 3.0, 0.025, 0.05],
        abs=1e-12,
    )
    assert [row["p_holm_adjusted"] for row in result] == pytest.approx(
        [0.005, 0.04, 0.09, 0.09, 0.2],
        abs=1e-12,
    )


def test_bootstrap_paired_diff_ci_deterministic() -> None:
    first = bootstrap_paired_diff_ci([1, 2, 3, 4], [0, 1, 3, 5], n_resamples=100, seed=0)
    second = bootstrap_paired_diff_ci([1, 2, 3, 4], [0, 1, 3, 5], n_resamples=100, seed=0)

    assert first == second
    assert first[0] <= first[1]
    assert first == pytest.approx((-1.0, 1.0), abs=1e-12)


def test_paired_finite_values_drops_nan_pairs() -> None:
    reference, alternative = paired_finite_values([1.0, float("nan"), 3.0], [1.5, 2.0, 2.5])

    assert reference.tolist() == [1.0, 3.0]
    assert alternative.tolist() == [1.5, 2.5]


def test_paired_finite_values_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        paired_finite_values([1.0, 2.0], [1.0])