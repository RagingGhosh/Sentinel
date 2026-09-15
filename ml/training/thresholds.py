"""The shared 311 SLA threshold primitive (addendum §7, plan §K, D31).

``nyc311_sla_breach`` is resolution slower than a complaint type's p75 of
resolution hours. D31 makes this module the one definition of that threshold,
used by Task 11's out-of-fold breach rates today and by Task 13's frozen label
thresholds later — so the per-fold thresholds and the label thresholds can never
diverge in definition. Nothing here fits a whole training period or freezes a
result for serving; that is Task 13.

* **p75 is linear interpolation** between adjacent order statistics: with the
  values sorted as ``x[0] … x[n - 1]`` and ``h = (n - 1) * 0.75``, the value is
  ``x[floor(h)] + frac(h) * (x[ceil(h)] - x[floor(h)])``.
* **Only eligible observations count** — those whose ``resolution_hours`` is not
  ``None``. An open request has no resolution time, so it contributes to no
  category p75, no global p75, and no count toward the minimum.
* **A category is seen only through eligible observations.** One with fewer
  than 100 of them, including one with none, uses the global p75.
* **With no eligible observation at all** the threshold is undefined: ``NaN``.
* **A breach is strictly longer** than the threshold (plan line 301, addendum
  line 806); a resolution exactly at the p75 is not slower than it.

Pure and standard-library only: no ``ingest``, no ``numpy``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

THRESHOLD_QUANTILE = 0.75
"""Addendum §7: each complaint type's own p75 of resolution hours."""

MIN_ELIGIBLE_OBSERVATIONS = 100
"""Addendum §7 and D31: fewer eligible observations than this fall back to the
global p75."""


def linear_percentile(values: Sequence[float], quantile: float) -> float:
    """Linear interpolation between adjacent order statistics (D31).

    Raises ``ValueError`` for no values or a quantile outside ``[0, 1]``.
    """
    if isinstance(quantile, bool) or not isinstance(quantile, int | float):
        raise ValueError(f"quantile must be a number, got {quantile!r}")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be within [0, 1], got {quantile!r}")
    if len(values) == 0:
        raise ValueError("cannot take a percentile of no values")

    ordered = sorted(values)
    h = (len(ordered) - 1) * quantile
    low, high = math.floor(h), math.ceil(h)
    return ordered[low] + (h - low) * (ordered[high] - ordered[low])


@dataclass(frozen=True)
class CategoryThresholds:
    """Per-category p75 thresholds with the global fallback (D31).

    ``per_category`` holds only categories with at least
    ``MIN_ELIGIBLE_OBSERVATIONS`` eligible observations; every other category —
    a fallback one, one seen only through open requests, or one never seen —
    uses ``global_fallback``.
    """

    per_category: Mapping[str, float]
    """Read-only. Categories with enough eligible observations for their own p75."""
    global_fallback: float
    """The p75 of every eligible observation, or ``NaN`` when there is none."""
    fallback_categories: frozenset[str]
    """Categories with 1 to 99 eligible observations. Metadata only: it never
    alters a threshold or a Task 11 feature value."""

    def threshold_for(self, category: str) -> float:
        return self.per_category.get(category, self.global_fallback)


def fit_category_thresholds(
    observations: Iterable[tuple[str, float | None]],
    min_eligible: int = MIN_ELIGIBLE_OBSERVATIONS,
) -> CategoryThresholds:
    """Fit thresholds from ``(category, resolution_hours)`` observations.

    The caller decides which observations are in scope — for Task 11, exactly
    one fold's fit block. Observations whose hours are ``None`` are skipped
    entirely.
    """
    eligible_by_category: dict[str, list[float]] = {}
    for category, hours in observations:
        if hours is None:
            continue
        eligible_by_category.setdefault(category, []).append(hours)

    everything = [h for values in eligible_by_category.values() for h in values]
    global_fallback = linear_percentile(everything, THRESHOLD_QUANTILE) if everything else math.nan

    per_category = {
        category: linear_percentile(values, THRESHOLD_QUANTILE)
        for category, values in sorted(eligible_by_category.items())
        if len(values) >= min_eligible
    }
    fallback_categories = frozenset(
        category for category, values in eligible_by_category.items() if len(values) < min_eligible
    )
    return CategoryThresholds(
        per_category=MappingProxyType(per_category),
        global_fallback=global_fallback,
        fallback_categories=fallback_categories,
    )


def is_breach(resolution_hours: float, threshold: float) -> bool:
    """Resolution strictly longer than the threshold."""
    return resolution_hours > threshold
