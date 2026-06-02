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


def test_wilcoxon_paired_all_zero_differences() -> None:
    # The pointing_hit collapse: reference and method tie on every case.
    # scipy.wilcoxon would raise on an all-zero vector, so the helper must
    # short-circuit to a degenerate stat=0.0, p=1.0 with a zero-width CI.
    values = [0.0, 1.0, 1.0, 0.0, 1.0]
    result = wilcoxon_paired(values, {"tied": list(values)}, n_resamples=50, seed=0)["tied"]

    assert result["n_pairs"] == 5
    assert result["wilcoxon_stat"] == pytest.approx(0.0, abs=1e-12)
    assert result["p_raw"] == pytest.approx(1.0, abs=1e-12)
    assert result["median_diff"] == pytest.approx(0.0, abs=1e-12)
    assert result["bootstrap_ci_low"] == pytest.approx(0.0, abs=1e-12)
    assert result["bootstrap_ci_high"] == pytest.approx(0.0, abs=1e-12)
    assert result["holm_significant_bool"] is False


def test_wilcoxon_paired_empty_after_nan_filter() -> None:
    nan = float("nan")
    result = wilcoxon_paired([nan, nan], {"empty": [1.0, 2.0]}, n_resamples=50, seed=0)["empty"]

    assert result["n_pairs"] == 0
    assert result["p_raw"] == pytest.approx(1.0, abs=1e-12)
    assert result["holm_significant_bool"] is False
    for key in ("median_diff", "bootstrap_ci_low", "bootstrap_ci_high", "wilcoxon_stat"):
        assert result[key] != result[key]  # NaN is not equal to itself


def test_wilcoxon_paired_holm_correction_across_alternatives() -> None:
    # Three alternatives evaluated together must share one Holm family: the
    # adjusted p-values are non-decreasing in raw-p rank and each carries its
    # own step-down threshold.
    reference = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    alternatives = {
        "far": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "near": [0.85, 0.92, 0.88, 0.91, 0.87, 0.90],
        "mid": [0.5, 0.55, 0.45, 0.6, 0.5, 0.52],
    }
    result = wilcoxon_paired(reference, alternatives, alpha=0.05, n_resamples=200, seed=0)

    assert set(result) == {"far", "near", "mid"}
    ordered = sorted(result.values(), key=lambda row: row["p_raw"])
    adjusted = [row["p_holm_adjusted"] for row in ordered]
    assert adjusted == sorted(adjusted)  # monotonic non-decreasing by rank
    for row in result.values():
        assert 0.0 <= row["p_holm_adjusted"] <= 1.0


def test_holm_bonferroni_all_significant() -> None:
    result = holm_bonferroni([0.001, 0.002, 0.003], alpha=0.05)
    assert [row["holm_significant"] for row in result] == [True, True, True]


def test_holm_bonferroni_none_significant() -> None:
    result = holm_bonferroni([0.5, 0.6, 0.9], alpha=0.05)
    assert [row["holm_significant"] for row in result] == [False, False, False]


def test_holm_bonferroni_preserves_input_order() -> None:
    # Output order matches input order even when raw p-values are unsorted.
    result = holm_bonferroni([0.2, 0.001, 0.04], alpha=0.05)
    assert [row["p_raw"] for row in result] == [0.2, 0.001, 0.04]
    assert [row["holm_significant"] for row in result] == [False, True, False]


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
def test_holm_bonferroni_rejects_invalid_alpha(bad_alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        holm_bonferroni([0.01, 0.02], alpha=bad_alpha)


@pytest.mark.parametrize("bad_p", [-0.01, 1.01, float("nan"), float("inf")])
def test_holm_bonferroni_rejects_invalid_pvalues(bad_p: float) -> None:
    with pytest.raises(ValueError, match="p-values"):
        holm_bonferroni([0.01, bad_p], alpha=0.05)


def test_bootstrap_paired_diff_ci_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_paired_diff_ci([1, 2], [0, 1], n_resamples=0)
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_paired_diff_ci([1, 2], [0, 1], confidence=1.0)


def test_bootstrap_paired_diff_ci_empty_after_nan_filter() -> None:
    low, high = bootstrap_paired_diff_ci([float("nan")], [1.0], n_resamples=50, seed=0)
    assert low != low and high != high  # both NaN