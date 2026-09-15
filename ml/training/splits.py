"""Date-cut temporal splitting (plan §H, addendum §6.1).

Every model in this project is evaluated on a time-ordered split. Records are
cut into three contiguous, disjoint periods — **train (earliest), validation,
test (latest)** — by record count, and the cuts fall on *timestamps*, never on
row positions. Both sources carry duplicate timestamps, and a boundary drawn
between two rows sharing one would put identical instants in two periods.

**The boundary rule is literal.** Sort the unique timestamps ascending and take
the cumulative record count at each. For each boundary, choose the largest
timestamp whose cumulative count does not exceed that boundary's target — the
train target is ``train_fraction * n``, the validation target is
``(train_fraction + validation_fraction) * n`` — and every record at the chosen
timestamp belongs to the earlier period.

**The fallback, and why it exists.** That rule has no answer when even the
earliest eligible group of tied records is larger than the target: a single
timestamp holding all *n* records never has a cumulative count at or below
``0.70 * n``. The plan nevertheless requires a single-timestamp input to put
everything in train. The contract is resolved (interpretation A) by one
explicit fallback, and nothing else:

* when no timestamp fits under the **train** target, train ends at the earliest
  timestamp, so that whole group is train;
* when no timestamp beyond train's boundary fits under the **validation**
  target, validation ends where train ends, and is empty.

Empty periods are valid and reported, never raised: an empty validation period
is visible as ``val_end == train_end`` and a zero count. Outside the fallback,
the literal rule means train never exceeds its requested share and test never
falls below its own.

The rejected alternative (interpretation B) assigns the group whose cumulative
count *crosses* a target to the earlier period. It contradicts "does not exceed",
and under heavy ties it can empty the test period entirely — 5000 / 3000 / 2000
tied records would split 80 / 20 / 0 instead of 50 / 30 / 20. A test pins that B
is not what runs.

**Forward-chaining folds (plan §J, D30)** divide the *training* period again, for
out-of-fold aggregates. The warm-up is realised by the same rule and fallback as
a Task 9 boundary, over ``warmup_fraction * n``. With W the realised warm-up
count, the remaining records form exactly ``n_folds`` apply blocks whose targets
are ``W + (n - W) * k / n_folds``, each cut by the same rule, so no timestamp
group is split and a target no timestamp satisfies leaves that fold empty. The
windows expand: a fold fits on everything strictly before its apply block, never
on a later record. Empty folds are reported, never raised, and there are always
exactly ``n_folds`` of them.

Deterministic and order-independent: the result depends only on the multiset of
timestamps, so no randomness exists anywhere to seed. Standard library only, and
Django-independent like the rest of ``ml.training``.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from itertools import accumulate

DEFAULT_FRACTIONS: tuple[float, float, float] = (0.70, 0.15, 0.15)
"""Addendum §6.1: 70 / 15 / 15 by record count within the window."""

FRACTION_TOLERANCE = 1e-9
"""How far three fractions may sum from 1, and how much float error a target may
carry. Relative to the record count when applied to a target, so a cumulative
count sitting exactly on ``0.85 * n`` is not lost to binary rounding while the
slack stays far below one record at any realistic corpus size."""


class Period(Enum):
    """The three periods, in chronological order."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class TemporalSplit:
    """The outcome of a date-cut split: where the cuts fell and what they held.

    Both the requested and the achieved fractions are kept, because ties make
    them differ and addendum §6.1 requires the actual cut dates and counts to be
    recorded whatever was asked for.
    """

    requested_fractions: dict[Period, float]
    train_end: datetime
    """The last timestamp in train, inclusive."""
    val_end: datetime
    """The last timestamp in validation, inclusive. Equal to ``train_end`` when
    validation is empty."""
    counts: dict[Period, int]
    achieved_fractions: dict[Period, float]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def period_of(self, ts: datetime) -> Period:
        """The period a timestamp belongs to.

        Defined for every instant, not only the inputs, because the cuts
        partition time: anything at or before ``train_end`` is train, anything
        after ``val_end`` is test.
        """
        if ts <= self.train_end:
            return Period.TRAIN
        if ts <= self.val_end:
            return Period.VALIDATION
        return Period.TEST


def _validated_fractions(fractions: Sequence[float]) -> tuple[float, float, float]:
    """Exactly three finite, non-negative numbers summing to 1, train positive."""
    try:
        values = tuple(fractions)
    except TypeError as exc:
        raise ValueError(
            f"fractions must be a sequence of three numbers, got {fractions!r}"
        ) from exc

    if len(values) != 3:
        raise ValueError(f"fractions must have exactly three values, got {len(values)}")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"fractions must be numbers, got {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"fractions must be finite, got {value!r}")
        if value < 0:
            raise ValueError(f"fractions must be non-negative, got {value!r}")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=FRACTION_TOLERANCE):
        raise ValueError(
            f"fractions must sum to 1 (within {FRACTION_TOLERANCE}), got {sum(values)!r}"
        )
    if values[0] <= 0:
        raise ValueError("the train fraction must be positive")

    return (float(values[0]), float(values[1]), float(values[2]))


def _boundary(cumulative: list[int], target: float, total: int, floor: int) -> int:
    """Index of the boundary timestamp: the literal rule, then the fallback.

    The largest index whose cumulative count does not exceed ``target``; when
    none does, or the one found lies before ``floor``, the boundary stays at
    ``floor`` and the later period is empty.
    """
    limit = target + FRACTION_TOLERANCE * total
    fits = bisect_right(cumulative, limit) - 1
    return max(fits, floor)


def temporal_split(
    timestamps: Sequence[datetime],
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
) -> TemporalSplit:
    """Cut a time-ordered dataset into train/val/test at *date* boundaries.

    Records sharing a timestamp always land in the same period. Achieved
    fractions therefore differ from requested ones; both are recorded. See the
    module docstring for the boundary rule and its explicit fallback.

    Raises ``ValueError`` for an empty input or invalid fractions.
    """
    train, validation, test = _validated_fractions(fractions)
    if len(timestamps) == 0:
        raise ValueError("cannot split an empty set of timestamps")

    per_timestamp = Counter(timestamps)
    ordered = sorted(per_timestamp)
    cumulative = list(accumulate(per_timestamp[ts] for ts in ordered))
    total = cumulative[-1]

    train_index = _boundary(cumulative, train * total, total, floor=0)
    val_index = _boundary(cumulative, (train + validation) * total, total, floor=train_index)

    train_count = cumulative[train_index]
    val_count = cumulative[val_index] - train_count
    counts = {
        Period.TRAIN: train_count,
        Period.VALIDATION: val_count,
        Period.TEST: total - train_count - val_count,
    }

    return TemporalSplit(
        requested_fractions={
            Period.TRAIN: train,
            Period.VALIDATION: validation,
            Period.TEST: test,
        },
        train_end=ordered[train_index],
        val_end=ordered[val_index],
        counts=counts,
        achieved_fractions={period: count / total for period, count in counts.items()},
    )


# --- forward-chaining folds (plan §J, addendum §6.3, D11, D30) -----------------

DEFAULT_N_FOLDS = 5
"""Plan §J: five folds, so each apply block holds ~16% of the training period."""

DEFAULT_WARMUP_FRACTION = 0.20
"""Plan §J: the first 20% of the training period by record count is warm-up."""


@dataclass(frozen=True)
class Fold:
    """One expanding-window fold over the training period (D30).

    ``fit_indices`` is every record strictly before the apply block — the
    warm-up plus every earlier apply block — and ``apply_indices`` is the block
    itself. Both hold original input positions in ascending chronological order,
    records sharing a timestamp in ascending input position.
    """

    fit_indices: tuple[int, ...]
    apply_indices: tuple[int, ...]
    fit_end: datetime
    """The previous boundary: the last timestamp in the fit block, inclusive."""
    apply_start: datetime | None
    """The earliest timestamp in the apply block, or ``None`` when it is empty."""
    apply_end: datetime
    """This fold's boundary, inclusive. Equal to ``fit_end`` when the fold is empty."""


def _validated_n_folds(n_folds: int) -> int:
    """An integer of at least 1; ``bool`` is not accepted (D30)."""
    if isinstance(n_folds, bool) or not isinstance(n_folds, int):
        raise ValueError(f"n_folds must be an integer, got {n_folds!r}")
    if n_folds < 1:
        raise ValueError(f"n_folds must be at least 1, got {n_folds}")
    return n_folds


def _validated_warmup_fraction(warmup_fraction: float) -> float:
    """Finite and strictly between 0 and 1 (D30).

    Zero is rejected: the first fold would have nothing to fit on. One is
    rejected: no record would ever receive an out-of-fold value.
    """
    if isinstance(warmup_fraction, bool) or not isinstance(warmup_fraction, int | float):
        raise ValueError(f"warmup_fraction must be a number, got {warmup_fraction!r}")
    if not math.isfinite(warmup_fraction) or not 0 < warmup_fraction < 1:
        raise ValueError(
            f"warmup_fraction must be finite and strictly between 0 and 1, got {warmup_fraction!r}"
        )
    return float(warmup_fraction)


def forward_chaining_folds(
    timestamps: Sequence[datetime],
    n_folds: int = DEFAULT_N_FOLDS,
    warmup_fraction: float = DEFAULT_WARMUP_FRACTION,
) -> list[Fold]:
    """Expanding-window folds over the TRAIN period, cut at date boundaries.

    Fold i's ``fit`` block is every training record strictly before the fold's
    start date; its ``apply`` block is the records inside the fold. Records in
    the warm-up prefix appear in no fold's apply block.

    Pass the training period's timestamps only — for example the records
    ``temporal_split`` assigns to ``Period.TRAIN``. See the module docstring and
    D30 for the boundary rule, its fallback, and degenerate inputs. The warm-up
    size is ``len(folds[0].fit_indices)``.

    Raises ``ValueError`` for empty timestamps, an invalid ``n_folds``, or an
    invalid ``warmup_fraction``.
    """
    fold_count = _validated_n_folds(n_folds)
    warmup = _validated_warmup_fraction(warmup_fraction)
    if len(timestamps) == 0:
        raise ValueError("cannot build folds from an empty set of timestamps")

    per_timestamp = Counter(timestamps)
    unique_timestamps = sorted(per_timestamp)
    cumulative = list(accumulate(per_timestamp[ts] for ts in unique_timestamps))
    total = cumulative[-1]

    # The warm-up, realised exactly as Task 9 realises a boundary (D29 fallback
    # to the earliest timestamp included).
    warmup_index = _boundary(cumulative, warmup * total, total, floor=0)
    warmup_count = cumulative[warmup_index]
    remaining = total - warmup_count

    # D30: targets divide what actually remains after the realised warm-up.
    boundary_indices: list[int] = []
    previous = warmup_index
    for k in range(1, fold_count + 1):
        target = warmup_count + remaining * k / fold_count
        previous = _boundary(cumulative, target, total, floor=previous)
        boundary_indices.append(previous)

    rank = {ts: position for position, ts in enumerate(unique_timestamps)}
    chronological = sorted(range(len(timestamps)), key=lambda i: (rank[timestamps[i]], i))
    ranks = [rank[timestamps[i]] for i in chronological]

    folds: list[Fold] = []
    fit_stop = bisect_right(ranks, warmup_index)
    previous = warmup_index
    for boundary in boundary_indices:
        apply_stop = bisect_right(ranks, boundary)
        folds.append(
            Fold(
                fit_indices=tuple(chronological[:fit_stop]),
                apply_indices=tuple(chronological[fit_stop:apply_stop]),
                fit_end=unique_timestamps[previous],
                apply_start=unique_timestamps[previous + 1] if boundary > previous else None,
                apply_end=unique_timestamps[boundary],
            )
        )
        fit_stop = apply_stop
        previous = boundary
    return folds
