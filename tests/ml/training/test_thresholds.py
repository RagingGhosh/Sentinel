"""The shared threshold primitive (D31): linear-interpolated p75, <100 fallback.

This file covers only the pure primitive Task 11 needs to make
``category_breach_rate`` well defined. Task 13's frozen fitting and applying
API is not implemented here; Task 13 reuses this primitive and extends this
file.

Contract, from D31:

* p75 is linear interpolation between adjacent order statistics,
  ``x[floor(h)] + frac(h) * (x[ceil(h)] - x[floor(h)])`` with
  ``h = (n - 1) * 0.75``;
* a category with fewer than 100 *eligible* observations (``resolution_hours``
  not ``None``) falls back to the global p75 of the eligible observations;
* open requests contribute to nothing; with no eligible observation the
  threshold is undefined (``NaN``);
* a breach is resolution strictly longer than the threshold (plan line 301,
  addendum line 806).
"""

import ast
import dataclasses
import math
import random
from pathlib import Path

import pytest

from ml.training.thresholds import (
    MIN_ELIGIBLE_OBSERVATIONS,
    THRESHOLD_QUANTILE,
    CategoryThresholds,
    fit_category_thresholds,
    is_breach,
    linear_percentile,
)


def reference_percentile(values, quantile):
    """D31's formula, written independently of the implementation."""
    x = sorted(values)
    h = (len(x) - 1) * quantile
    low, high = math.floor(h), math.ceil(h)
    return x[low] + (h - low) * (x[high] - x[low])


# --- constants ---------------------------------------------------------------------


def test_the_threshold_is_the_seventy_fifth_percentile():
    assert THRESHOLD_QUANTILE == 0.75


def test_the_fallback_count_is_one_hundred():
    assert MIN_ELIGIBLE_OBSERVATIONS == 100


# --- linear_percentile -------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([5.0], 5.0),
        ([10.0, 20.0], 17.5),
        ([1.0, 2.0, 3.0, 4.0], 3.25),
        ([0.0, 100.0, 50.0, 25.0], 62.5),
        ([10.0, 20.0, 30.0, 40.0, 50.0], 40.0),
        ([7.0, 7.0, 7.0], 7.0),
    ],
    ids=["one", "two", "four", "unsorted-four", "exact-order-statistic", "constant"],
)
def test_p75_matches_hand_computed_values(values, expected):
    assert linear_percentile(values, 0.75) == expected


def test_percentile_matches_the_d31_formula_on_random_samples():
    rng = random.Random(31)
    for _ in range(500):
        values = [round(rng.uniform(0, 10_000), 3) for _ in range(rng.randint(1, 60))]
        for quantile in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert linear_percentile(values, quantile) == pytest.approx(
                reference_percentile(values, quantile), rel=0, abs=1e-9
            )


def test_percentile_does_not_depend_on_input_order():
    values = [float(v) for v in range(1, 42)]
    shuffled = values[:]
    random.Random(1).shuffle(shuffled)
    assert linear_percentile(values, 0.75) == linear_percentile(shuffled, 0.75)


def test_percentile_does_not_mutate_its_input():
    values = [3.0, 1.0, 2.0]
    linear_percentile(values, 0.75)
    assert values == [3.0, 1.0, 2.0]


def test_percentile_of_no_values_raises():
    with pytest.raises(ValueError):
        linear_percentile([], 0.75)


@pytest.mark.parametrize("quantile", [-0.01, 1.01, math.nan])
def test_percentile_rejects_a_quantile_outside_zero_to_one(quantile):
    with pytest.raises(ValueError):
        linear_percentile([1.0, 2.0], quantile)


# --- fit_category_thresholds -------------------------------------------------------


def observations(category, count, start=1.0):
    return [(category, start + i) for i in range(count)]


def test_a_category_with_exactly_one_hundred_eligible_observations_gets_its_own_p75():
    obs = observations("A", 100) + observations("B", 30, start=1000.0)
    fitted = fit_category_thresholds(obs)

    own = reference_percentile([h for c, h in obs if c == "A"], 0.75)
    assert fitted.per_category["A"] == own
    assert fitted.threshold_for("A") == own
    assert "A" not in fitted.fallback_categories


def test_a_category_with_ninety_nine_eligible_observations_uses_the_global_p75():
    obs = observations("A", 99) + observations("B", 30, start=1000.0)
    fitted = fit_category_thresholds(obs)

    global_p75 = reference_percentile([h for _, h in obs], 0.75)
    assert "A" not in fitted.per_category
    assert fitted.global_fallback == global_p75
    assert fitted.threshold_for("A") == global_p75
    assert "A" in fitted.fallback_categories


def test_open_requests_do_not_count_toward_the_one_hundred():
    """T2a: 99 closed plus 50 open is still 99 eligible observations."""
    obs = observations("A", 99) + [("A", None)] * 50 + observations("B", 30, start=1000.0)
    fitted = fit_category_thresholds(obs)

    assert "A" not in fitted.per_category
    assert "A" in fitted.fallback_categories


def test_open_requests_do_not_enter_any_percentile():
    closed_a = observations("A", 100)
    closed_b = observations("B", 20, start=500.0)
    obs = closed_a + [("A", None)] * 40 + closed_b + [("B", None)] * 40
    fitted = fit_category_thresholds(obs)

    assert fitted.per_category["A"] == reference_percentile([h for _, h in closed_a], 0.75)
    assert fitted.global_fallback == reference_percentile([h for _, h in closed_a + closed_b], 0.75)


def test_an_unseen_category_uses_the_global_p75():
    fitted = fit_category_thresholds(observations("A", 10))
    assert fitted.threshold_for("never-seen") == fitted.global_fallback


def test_a_category_with_only_open_requests_is_not_seen():
    """E1: open requests never make a category known, however many there are."""
    obs = observations("A", 30) + [("D", None)] * 200
    fitted = fit_category_thresholds(obs)

    assert "D" not in fitted.per_category
    assert fitted.threshold_for("D") == fitted.global_fallback
    assert "D" not in fitted.fallback_categories


def test_fallback_categories_are_exactly_those_with_one_to_ninety_nine_eligible():
    """Metadata only: 150 eligible is not a fallback, 99 and 1 are, and a
    category with no eligible observation is not seen at all."""
    obs = (
        observations("A", 150)
        + observations("B", 99, start=500.0)
        + [("B", None)] * 10
        + observations("C", 1, start=900.0)
        + [("D", None)] * 120
    )
    fitted = fit_category_thresholds(obs)

    assert fitted.fallback_categories == frozenset({"B", "C"})
    assert set(fitted.per_category) == {"A"}


def test_no_eligible_observation_leaves_every_threshold_undefined():
    fitted = fit_category_thresholds([("A", None), ("B", None)])

    assert math.isnan(fitted.global_fallback)
    assert dict(fitted.per_category) == {}
    assert math.isnan(fitted.threshold_for("A"))


def test_no_observations_at_all_leave_every_threshold_undefined():
    fitted = fit_category_thresholds([])
    assert math.isnan(fitted.global_fallback)
    assert dict(fitted.per_category) == {}


def test_thresholds_do_not_depend_on_observation_order():
    obs = observations("A", 120) + observations("B", 40, start=900.0) + [("A", None)] * 5
    shuffled = obs[:]
    random.Random(7).shuffle(shuffled)
    first, second = fit_category_thresholds(obs), fit_category_thresholds(shuffled)

    assert dict(first.per_category) == dict(second.per_category)
    assert first.global_fallback == second.global_fallback
    assert first.fallback_categories == second.fallback_categories


def test_fitted_thresholds_are_immutable():
    fitted = fit_category_thresholds(observations("A", 100))

    assert isinstance(fitted, CategoryThresholds)
    with pytest.raises(dataclasses.FrozenInstanceError):
        fitted.global_fallback = 0.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        fitted.per_category["A"] = 0.0  # type: ignore[index]


# --- is_breach -----------------------------------------------------------------------


def test_a_breach_is_resolution_strictly_longer_than_the_threshold():
    assert is_breach(10.000001, 10.0) is True
    assert is_breach(10.0, 10.0) is False, "equal to the p75 is not slower than it"
    assert is_breach(9.0, 10.0) is False


# --- dependencies ----------------------------------------------------------------------


def test_the_threshold_module_imports_neither_ingest_nor_numpy():
    tree = ast.parse(Path("ml/training/thresholds.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = {"ingest", "numpy", "pandas", "scipy", "sklearn", "django", "random"}
    assert not {m.split(".")[0] for m in imported} & forbidden, imported
