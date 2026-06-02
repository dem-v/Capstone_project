from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite

import numpy as np
from scipy.stats import wilcoxon


def paired_finite_values(
    reference: Sequence[float] | np.ndarray,
    alternative: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference_array = np.asarray(reference, dtype=float)
    alternative_array = np.asarray(alternative, dtype=float)
    if reference_array.shape != alternative_array.shape:
        raise ValueError(
            f"reference and alternative must have the same shape; "
            f"got {reference_array.shape} and {alternative_array.shape}"
        )
    finite = np.isfinite(reference_array) & np.isfinite(alternative_array)
    return reference_array[finite], alternative_array[finite]


def bootstrap_paired_diff_ci(
    reference: Sequence[float] | np.ndarray,
    alternative: Sequence[float] | np.ndarray,
    *,
    n_resamples: int = 10_000,
    seed: int = 20260515,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    reference_array, alternative_array = paired_finite_values(reference, alternative)
    if reference_array.size == 0:
        return float("nan"), float("nan")

    differences = reference_array - alternative_array
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, differences.size, size=(n_resamples, differences.size))
    sample_medians = np.median(differences[sample_indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(sample_medians, [tail, 1.0 - tail])
    return float(low), float(high)


def holm_bonferroni(
    p_raw: Sequence[float],
    *,
    alpha: float = 0.05,
) -> list[dict[str, float | bool]]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    p_values = [float(p_value) for p_value in p_raw]
    if any(not isfinite(p_value) or p_value < 0.0 or p_value > 1.0 for p_value in p_values):
        raise ValueError("p-values must be finite values in [0, 1]")

    result: list[dict[str, float | bool]] = [
        {"p_raw": p_value, "p_holm_adjusted": 1.0, "p_holm_threshold": alpha, "holm_significant": False}
        for p_value in p_values
    ]
    sorted_indices = sorted(range(len(p_values)), key=p_values.__getitem__)
    running_adjusted = 0.0
    still_rejecting = True
    m = len(sorted_indices)
    for rank, original_index in enumerate(sorted_indices):
        p_value = p_values[original_index]
        threshold = alpha / (m - rank)
        running_adjusted = max(running_adjusted, min(1.0, p_value * (m - rank)))
        rejected = still_rejecting and p_value <= threshold
        if not rejected:
            still_rejecting = False
        result[original_index] = {
            "p_raw": p_value,
            "p_holm_adjusted": running_adjusted,
            "p_holm_threshold": threshold,
            "holm_significant": rejected,
        }
    return result


def wilcoxon_paired(
    reference: Sequence[float] | np.ndarray,
    alternatives: Mapping[str, Sequence[float] | np.ndarray],
    *,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    seed: int = 20260515,
) -> dict[str, dict[str, float | int | bool]]:
    raw_results: dict[str, dict[str, float | int | bool]] = {}
    p_values: list[float] = []
    method_names: list[str] = []

    for offset, (method_name, alternative) in enumerate(alternatives.items()):
        reference_array, alternative_array = paired_finite_values(reference, alternative)
        if reference_array.size == 0:
            statistic = float("nan")
            p_value = 1.0
            median_diff = float("nan")
            ci_low = float("nan")
            ci_high = float("nan")
        else:
            differences = reference_array - alternative_array
            median_diff = float(np.median(differences))
            ci_low, ci_high = bootstrap_paired_diff_ci(
                reference_array,
                alternative_array,
                n_resamples=n_resamples,
                seed=seed + offset,
            )
            if np.allclose(differences, 0.0):
                statistic = 0.0
                p_value = 1.0
            else:
                test_result = wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
                statistic = float(test_result.statistic)
                p_value = float(test_result.pvalue)

        raw_results[method_name] = {
            "n_pairs": int(reference_array.size),
            "median_diff": median_diff,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "wilcoxon_stat": statistic,
            "p_raw": p_value,
        }
        method_names.append(method_name)
        p_values.append(p_value)

    holm_results = holm_bonferroni(p_values, alpha=alpha)
    for method_name, holm_result in zip(method_names, holm_results):
        raw_results[method_name].update(
            {
                "p_holm_adjusted": float(holm_result["p_holm_adjusted"]),
                "p_holm_threshold": float(holm_result["p_holm_threshold"]),
                "holm_significant_bool": bool(holm_result["holm_significant"]),
            }
        )
    return raw_results