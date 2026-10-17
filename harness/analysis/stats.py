"""Scenario-balanced factorial contrasts, block bootstrap CIs, exact sign-flip tests, Holm.

Ported unchanged in method from the paper notebook, including the seeding
scheme, so the paper's intervals are reproduced exactly.

The unit of inference is the scenario, not the run: repetitions are averaged
within each (scenario, condition) cell, contrasts are linear combinations of
the 12 condition means, and the bootstrap resamples whole scenarios.
"""
from __future__ import annotations

import hashlib
import itertools
import warnings

import numpy as np
import pandas as pd

from harness.analysis.design import CONDITION_COLUMNS, CONDITIONS

SEED = 42
N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95
EPSILON = 1e-12


def _language(condition) -> float:
    return 1.0 if condition[0] == "ko" else -1.0


def _urgency(condition) -> float:
    return 1.0 if condition[1] == "strong" else -1.0


def _persona(condition, target: str) -> float:
    if condition[2] == target:
        return 1.0
    return -1.0 if condition[2] == "neutral" else 0.0


def _weights(fn) -> np.ndarray:
    return np.array([fn(condition) for condition in CONDITIONS])


# Weights are scaled so each contrast reads as a difference of cell means
# (e.g. Korean minus English averages 6 cells on each side).
CONTRASTS = {
    "language_ko_minus_en": _weights(lambda c: _language(c) / 6),
    "urgency_strong_minus_none": _weights(lambda c: _urgency(c) / 6),
    "persona_cooperative_minus_neutral": _weights(lambda c: _persona(c, "cooperative") / 4),
    "persona_headstrong_minus_neutral": _weights(lambda c: _persona(c, "headstrong") / 4),
    "language_x_urgency": _weights(lambda c: _language(c) * _urgency(c) / 3),
    "language_x_cooperative_vs_neutral": _weights(lambda c: _language(c) * _persona(c, "cooperative") / 2),
    "language_x_headstrong_vs_neutral": _weights(lambda c: _language(c) * _persona(c, "headstrong") / 2),
    "urgency_x_cooperative_vs_neutral": _weights(lambda c: _urgency(c) * _persona(c, "cooperative") / 2),
    "urgency_x_headstrong_vs_neutral": _weights(lambda c: _urgency(c) * _persona(c, "headstrong") / 2),
    "language_x_urgency_x_cooperative_vs_neutral": _weights(
        lambda c: _language(c) * _urgency(c) * _persona(c, "cooperative")
    ),
    "language_x_urgency_x_headstrong_vs_neutral": _weights(
        lambda c: _language(c) * _urgency(c) * _persona(c, "headstrong")
    ),
}

# The four prespecified agreement tests that share one Holm family in the paper.
HEADLINE_AGREEMENT_CONTRASTS = [
    "language_ko_minus_en",
    "language_x_urgency",
    "language_x_cooperative_vs_neutral",
    "language_x_headstrong_vs_neutral",
]

FACTORIAL_METRICS = [
    "agreed",
    "n_turns",
    "normalized_final_price",
    "buyer_first_offer_norm",
    "buyer_net_concession_norm",
    "seller_net_concession_norm",
    "terminal_gap_norm",
]


def stable_seed(*parts) -> int:
    """Seed derived from a label, so each estimate's draws don't depend on run order."""
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return (SEED + int.from_bytes(digest[:4], "little")) % (2**32 - 1)


def scenario_cell_matrix(frame: pd.DataFrame, metric: str, scenario_ids: list[str]) -> np.ndarray:
    """(n_scenarios, 12) matrix of cell means; NaN where a cell has no finite value."""
    grouped = frame.groupby(["scenario_id", *CONDITION_COLUMNS], observed=True)[metric].mean()
    matrix = np.full((len(scenario_ids), len(CONDITIONS)), np.nan)
    rows = {scenario_id: index for index, scenario_id in enumerate(scenario_ids)}
    cols = {condition: index for index, condition in enumerate(CONDITIONS)}
    for (scenario_id, *condition), value in grouped.items():
        matrix[rows[scenario_id], cols[tuple(condition)]] = float(value)
    return matrix


def bootstrap_contrast(matrix: np.ndarray, weights: np.ndarray, *seed_parts):
    """Point estimate and percentile CI from resampling scenarios with replacement.

    Conditional metrics (e.g. agreed price) leave some cells empty, so a draw is
    kept only if every cell the contrast needs has at least one sampled value.
    """
    required = np.flatnonzero(np.abs(weights) > EPSILON)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        observed = np.nanmean(matrix, axis=0)
    if not np.isfinite(observed[required]).all():
        raise ValueError(f"A required condition has no observations: {seed_parts}")
    estimate = float(observed @ weights)

    rng = np.random.default_rng(stable_seed(*seed_parts))
    collected, n_collected, n_attempted = [], 0, 0
    while n_collected < N_BOOTSTRAP:
        batch = min(2_000, max(250, 2 * (N_BOOTSTRAP - n_collected)))
        indices = rng.integers(0, matrix.shape[0], size=(batch, matrix.shape[0]))
        n_attempted += batch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sampled = np.nanmean(matrix[indices, :], axis=1)
        valid = np.isfinite(sampled[:, required]).all(axis=1)
        effects = sampled[valid] @ weights
        if len(effects):
            collected.append(effects)
            n_collected += len(effects)
        if n_attempted > 100 * N_BOOTSTRAP:
            raise RuntimeError(f"Could not collect enough valid bootstrap draws: {seed_parts}")

    draws = np.concatenate(collected)[:N_BOOTSTRAP]
    alpha = (1 - CI_LEVEL) / 2
    low, high = np.quantile(draws, [alpha, 1 - alpha])
    complete = int(np.isfinite(matrix[:, required]).all(axis=1).sum())
    return estimate, float(low), float(high), complete


def exact_sign_flip_pvalue(scenario_effects: np.ndarray) -> float:
    """Two-sided p over all 2^n sign flips of per-scenario effects (n=12 -> 4096)."""
    effects = np.asarray(scenario_effects, dtype=float)
    if not np.isfinite(effects).all():
        raise ValueError("Exact sign-flip test needs a complete effect for every scenario")
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=len(effects))))
    permuted = (signs * effects).mean(axis=1)
    return float(np.mean(np.abs(permuted) >= abs(effects.mean()) - EPSILON))


def holm_adjust(p_values: pd.Series) -> pd.Series:
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    ordered = p_values.dropna().sort_values()
    running_max, m = 0.0, len(ordered)
    for rank, (index, p_value) in enumerate(ordered.items()):
        running_max = max(running_max, min(1.0, (m - rank) * float(p_value)))
        adjusted.loc[index] = running_max
    return adjusted


def factorial_contrasts(
    transcripts: pd.DataFrame,
    scenario_ids: list[str],
    metrics: list[str] = FACTORIAL_METRICS,
    contrasts: dict[str, np.ndarray] = CONTRASTS,
) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        matrix = scenario_cell_matrix(transcripts, metric, scenario_ids)
        for name, weights in contrasts.items():
            try:
                estimate, low, high, complete = bootstrap_contrast(matrix, weights, "factorial", metric, name)
            except ValueError:
                # A needed cell has no data, e.g. no agreements at all in some
                # condition, so a conditional-price contrast is undefined.
                estimate = low = high = np.nan
                complete = 0
            headline = metric == "agreed" and name in HEADLINE_AGREEMENT_CONTRASTS
            rows.append(
                {
                    "metric": metric,
                    "contrast": name,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_observations": int(transcripts[metric].notna().sum()),
                    "n_complete_scenarios": complete,
                    "exact_p": (
                        exact_sign_flip_pvalue(matrix @ weights) if headline and complete == len(scenario_ids) else np.nan
                    ),
                    "headline_test": headline,
                }
            )
    result = pd.DataFrame(rows)
    result["holm_p"] = np.nan
    mask = result["headline_test"]
    result.loc[mask, "holm_p"] = holm_adjust(result.loc[mask, "exact_p"])
    return result
