"""Out-of-fold target-derived category aggregates (plan §J, Task 11, §6.3, D11, D31).

Two ``RiskFeaturesV1`` features are computed from the label itself:
``category_mean_resolution_hours`` and ``category_breach_rate``. Built naively
over the training period, a row's own outcome would shape its own feature. The
construction here prevents that, and every rule below is D31's.

**Training rows are out-of-fold** over Task 10's forward-chaining folds. For
each fold, everything is fitted on that fold's fit block alone and written to
its apply block:

  (A) the observations averaged for the category mean and counted for the rate;
  (B) the p75 thresholds that turn ``resolution_hours`` into breach labels.

A fit block holds only records strictly earlier than its apply block, so a row's
own outcome, a same-timestamp partner's, anything in its own apply block, and
anything later can influence neither side. Warm-up rows receive ``NaN``: nothing
earlier exists to fit on, and the training global mean would contain their own
outcomes.

**Validation and test rows** use ``FrozenAggregates`` fitted once on the whole
training period and applied as constants; no validation or test outcome can
reach them because none is ever passed in.

**Which observations count.** Only eligible ones — ``resolution_hours`` not
``None``. A category is seen only through its eligible observations, so a
category absent from the fitted rows, or present only as open requests, receives
the fitted rows' global aggregate. A statistic with no eligible observation at
all is ``NaN``.

**Arithmetic.** Plain record-weighted means and rates, no smoothing, no minimum
count; sums use ``math.fsum``, so a result does not depend on the order records
arrive in. Standard library only.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from ingest.schema import CorpusRecord, NYC311Outcome
from ml.training.splits import Fold
from ml.training.thresholds import fit_category_thresholds, is_breach


@dataclass(frozen=True)
class AggregateColumns:
    """The two features, aligned by position to the records they were built for."""

    category_mean_resolution_hours: tuple[float, ...]
    category_breach_rate: tuple[float, ...]


@dataclass(frozen=True)
class FrozenAggregates:
    """Aggregates fitted on one set of rows, applied unchanged elsewhere.

    Immutable, with read-only mappings, so validation and test can only look
    values up — never refit them.
    """

    mean_resolution_hours_by_category: Mapping[str, float]
    breach_rate_by_category: Mapping[str, float]
    global_mean_resolution_hours: float
    """Over every eligible observation, or ``NaN`` when there is none."""
    global_breach_rate: float
    """Over every eligible observation, or ``NaN`` when there is none."""


def _validated_pairs(records: Sequence[CorpusRecord], outcomes: Sequence[object]) -> None:
    """Positional alignment with consistent external IDs and 311 outcomes only."""
    if len(records) != len(outcomes):
        raise ValueError(f"{len(records)} records but {len(outcomes)} outcomes")
    for position, (record, outcome) in enumerate(zip(records, outcomes, strict=True)):
        if not isinstance(outcome, NYC311Outcome):
            raise ValueError(
                f"outcome at position {position} is {type(outcome).__name__}, not NYC311Outcome"
            )
        if record.external_id != outcome.external_id:
            raise ValueError(
                f"position {position}: record {record.external_id!r} "
                f"does not match outcome {outcome.external_id!r}"
            )


def _validated_folds(folds: Sequence[Fold], row_count: int) -> None:
    """Folds must partition the rows, and each fit block must be exactly the
    warm-up plus every preceding apply block (D30, D31)."""
    if len(folds) == 0:
        raise ValueError("a fold collection must contain at least one fold")

    warmup = folds[0].fit_indices
    if len(set(warmup)) != len(warmup):
        raise ValueError("the warm-up contains a position more than once")

    preceding = set(warmup)
    covered = list(warmup)
    for number, fold in enumerate(folds):
        fit, apply = fold.fit_indices, fold.apply_indices
        if len(set(fit)) != len(fit) or set(fit) != preceding:
            raise ValueError(
                f"fold {number}'s fit block is not exactly the warm-up plus every preceding "
                "apply block"
            )
        if len(set(apply)) != len(apply) or set(apply) & preceding:
            raise ValueError(
                f"fold {number}'s apply block repeats a position or overlaps its fit block"
            )
        preceding |= set(apply)
        covered.extend(apply)

    if sorted(covered) != list(range(row_count)):
        raise ValueError(f"the folds do not partition the {row_count} row positions exactly")


def _fit(
    records: Sequence[CorpusRecord],
    outcomes: Sequence[NYC311Outcome],
    positions: Iterable[int],
) -> FrozenAggregates:
    """Aggregates and their breach thresholds, both from ``positions`` alone."""
    observations = [(records[i].label, outcomes[i].resolution_hours) for i in positions]
    thresholds = fit_category_thresholds(observations)

    hours_by_category: dict[str, list[float]] = {}
    breaches_by_category: dict[str, int] = {}
    for category, hours in observations:
        if hours is None:
            continue
        hours_by_category.setdefault(category, []).append(hours)
        breaches_by_category[category] = breaches_by_category.get(category, 0) + is_breach(
            hours, thresholds.threshold_for(category)
        )

    means = {c: math.fsum(v) / len(v) for c, v in sorted(hours_by_category.items())}
    rates = {c: breaches_by_category[c] / len(v) for c, v in sorted(hours_by_category.items())}

    eligible = [h for values in hours_by_category.values() for h in values]
    if eligible:
        global_mean = math.fsum(eligible) / len(eligible)
        global_rate = sum(breaches_by_category.values()) / len(eligible)
    else:
        global_mean = global_rate = math.nan

    return FrozenAggregates(
        mean_resolution_hours_by_category=MappingProxyType(means),
        breach_rate_by_category=MappingProxyType(rates),
        global_mean_resolution_hours=global_mean,
        global_breach_rate=global_rate,
    )


def _lookup(frozen: FrozenAggregates, category: str) -> tuple[float, float]:
    """A category's values, or the fitted rows' globals when it was not seen."""
    return (
        frozen.mean_resolution_hours_by_category.get(category, frozen.global_mean_resolution_hours),
        frozen.breach_rate_by_category.get(category, frozen.global_breach_rate),
    )


def oof_category_aggregates(
    records: Sequence[CorpusRecord],
    outcomes: Sequence[NYC311Outcome],
    folds: Sequence[Fold],
) -> AggregateColumns:
    """Out-of-fold aggregates for the training period (plan Task 11, D31).

    ``records``, ``outcomes`` and the folds' indices align by position; build the
    folds with ``forward_chaining_folds([r.submitted_at for r in records])``.
    Warm-up rows are ``NaN``; every apply row's values come from its fold's fit
    block alone.

    Raises ``ValueError`` for misaligned inputs, a non-311 outcome, or folds that
    are not a valid expanding-window partition of the rows.
    """
    _validated_pairs(records, outcomes)
    _validated_folds(folds, len(records))

    means = [math.nan] * len(records)
    rates = [math.nan] * len(records)
    for fold in folds:
        if not fold.apply_indices:
            continue
        fitted = _fit(records, outcomes, fold.fit_indices)
        for position in fold.apply_indices:
            means[position], rates[position] = _lookup(fitted, records[position].label)

    return AggregateColumns(
        category_mean_resolution_hours=tuple(means),
        category_breach_rate=tuple(rates),
    )


def fit_category_aggregates(
    train_records: Sequence[CorpusRecord],
    train_outcomes: Sequence[NYC311Outcome],
) -> FrozenAggregates:
    """Fit aggregates on the whole training period, for validation and test.

    Raises ``ValueError`` for misaligned inputs or a non-311 outcome.
    """
    _validated_pairs(train_records, train_outcomes)
    return _fit(train_records, train_outcomes, range(len(train_records)))


def apply_category_aggregates(
    frozen: FrozenAggregates,
    records: Sequence[CorpusRecord],
) -> AggregateColumns:
    """Apply frozen training aggregates unchanged; unseen categories get the
    training globals."""
    pairs = [_lookup(frozen, record.label) for record in records]
    return AggregateColumns(
        category_mean_resolution_hours=tuple(mean for mean, _ in pairs),
        category_breach_rate=tuple(rate for _, rate in pairs),
    )
