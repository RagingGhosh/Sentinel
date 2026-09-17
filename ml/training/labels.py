"""Frozen 311 SLA thresholds and the labels they produce (plan Task 13, §7, §K, D33).

`nyc311_sla_breach` is resolution slower than a complaint type's own p75 of
resolution hours, fitted on the **training period only** and then frozen:

```
TRAIN      fit per-type p75 of resolution_hours  ->  freeze
VALIDATION apply frozen thresholds
TEST       apply frozen thresholds
```

`fit_thresholds` returns a `FrozenThresholds` and `apply_thresholds` accepts
one, so recomputing a threshold on validation or test data is a type error
rather than a matter of discipline (§K). **This threshold defines the label
only and is never a training feature** (§7, D15).

**The arithmetic is not here.** `ml.training.thresholds` owns the percentile,
the p75 interpolation, the minimum-count rule, the global fallback and the
breach comparison, and Task 11's out-of-fold aggregates fit their per-fold
thresholds through the same primitive. This module turns corpus records and 311
outcomes into the calls that primitive expects, and wraps its result in Task
13's own vocabulary. That is what keeps the per-fold thresholds and the label
thresholds from ever diverging in definition (D31), and it is why the primitive
imports neither `ingest` nor `numpy` while this module imports both (D33).

**Eligibility.** Only observations whose `resolution_hours` is not `None` count
— toward a type's p75, toward the global p75, and toward the hundred. The rule
counts eligible observations, never raw records: a type with 150 requests of
which 40 are resolved falls back (D31, D33).

**Two refusals, both deliberate** (D33). Applying a threshold to a request with
no `resolution_hours` raises rather than labelling it `False`, which would
manufacture a "not breached" label for a request that was never resolved, and
rather than dropping it, which would break alignment with `records`. Applying an
undefined (`NaN`) threshold raises too: every comparison with `NaN` is false, so
an undefined threshold would otherwise report a spotlessly clean period. Whether
open requests remain in the model-training population is the risk-model task's
decision, not this module's (D31).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ingest.schema import CorpusRecord, NYC311Outcome
from ml.training.thresholds import MIN_ELIGIBLE_OBSERVATIONS, fit_category_thresholds, is_breach


@dataclass(frozen=True)
class FrozenThresholds:
    """Per-type thresholds, fitted once on the training period and never refitted.

    Immutable, with a read-only mapping, so validation and test code can look a
    threshold up and nothing else.
    """

    per_type: Mapping[str, float]
    """Read-only. Only types with at least `min_eligible` eligible observations;
    every other type resolves to `global_fallback`."""
    global_fallback: float
    """The p75 of every eligible training observation, or `NaN` when there is
    none. Written to the artifact's metadata beside the per-type values (§7)."""
    fallback_type_count: int
    """How many training types hit the global fallback — published per §7.

    Deliberately **not** the same quantity as the primitive's
    `CategoryThresholds.fallback_categories`, which is the types with 1 to 99
    eligible observations. This count also includes a type seen only through
    open requests, which has zero eligible observations and so appears in no
    fallback category set, yet still resolves to the global (D33).
    """

    def threshold_for(self, complaint_type: str) -> float:
        """This type's threshold, or the global fallback when it has none.

        A lookup and nothing more: an unseen type receives the training-period
        global, per §6.3.
        """
        return self.per_type.get(complaint_type, self.global_fallback)


def _validated_pairs(records: Sequence[CorpusRecord], outcomes: Sequence[object]) -> None:
    """Positional alignment, consistent external IDs, and 311 outcomes only."""
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


def fit_thresholds(
    train_records: Sequence[CorpusRecord],
    train_outcomes: Sequence[NYC311Outcome],
    min_eligible: int = MIN_ELIGIBLE_OBSERVATIONS,
) -> FrozenThresholds:
    """Fit and freeze per-type thresholds from the training period alone.

    Only the rows passed here can influence the result, which is what makes the
    freeze meaningful: a validation or test outcome never reaches this function.

    Raises ``ValueError`` for misaligned inputs, a non-311 outcome, or a
    ``min_eligible`` below 1 — a minimum of zero or less would make the
    fewer-than-100 rule vacuous rather than lenient.
    """
    if min_eligible < 1:
        raise ValueError(f"min_eligible must be at least 1; got {min_eligible}")
    _validated_pairs(train_records, train_outcomes)

    observations = (
        (record.label, outcome.resolution_hours)
        for record, outcome in zip(train_records, train_outcomes, strict=True)
    )
    fitted = fit_category_thresholds(observations, min_eligible=min_eligible)

    types_seen = {record.label for record in train_records}
    return FrozenThresholds(
        per_type=MappingProxyType(dict(fitted.per_category)),
        global_fallback=fitted.global_fallback,
        fallback_type_count=len(types_seen - set(fitted.per_category)),
    )


def apply_thresholds(
    frozen: FrozenThresholds,
    records: Sequence[CorpusRecord],
    outcomes: Sequence[NYC311Outcome],
) -> np.ndarray:
    """Label each record `nyc311_sla_breach` against the frozen thresholds.

    Returns a boolean array aligned by position to ``records``. ``frozen`` is
    only read from; applying never refits or mutates it.

    Raises ``ValueError`` for misaligned inputs, a non-311 outcome, an outcome
    with no ``resolution_hours``, or an applicable threshold that is undefined
    (D33). Neither refusal has a silent form: both would otherwise emit a label
    of ``False`` that means "we could not tell" rather than "resolved in time".
    """
    _validated_pairs(records, outcomes)

    labels = np.empty(len(records), dtype=bool)
    for position, (record, outcome) in enumerate(zip(records, outcomes, strict=True)):
        hours = outcome.resolution_hours
        if hours is None:
            raise ValueError(
                f"position {position}: {record.external_id!r} has no resolution_hours; "
                "an unresolved request has no breach label, and labelling it False "
                "would report a request that was never resolved as resolved in time"
            )
        threshold = frozen.threshold_for(record.label)
        if math.isnan(threshold):
            raise ValueError(
                f"position {position}: the threshold for {record.label!r} is undefined "
                "(NaN); no eligible training observation defined it, and every "
                "comparison with NaN is false, so applying it would label the record "
                "not breached for the wrong reason"
            )
        labels[position] = is_breach(hours, threshold)
    return labels


def breach_rate(labels: Sequence[bool] | np.ndarray) -> float:
    """The share of labels that are breaches, for one period (§7, §K).

    Reported per period beside the thresholds that produced it. An empty period
    has no rate, which is `NaN` rather than zero — zero would read as "nothing
    breached" (D31's convention for an undefined statistic).
    """
    total = len(labels)
    if total == 0:
        return math.nan
    return float(np.count_nonzero(np.asarray(labels, dtype=bool)) / total)
