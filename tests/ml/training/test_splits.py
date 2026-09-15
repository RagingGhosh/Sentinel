"""Temporal split: date cuts by record count, never straddling a timestamp.

Plan §H and Task 9, addendum §6.1. The rule is literal: each boundary is the
largest unique timestamp whose cumulative record count does not exceed that
boundary's target, and every record at the boundary timestamp belongs to the
earlier period.

That rule cannot place a boundary when even the earliest eligible group of tied
records is larger than the target — the plan's own single-timestamp case is the
simplest instance. The resolution (interpretation A) is an explicit fallback:
the boundary stays at the earliest timestamp for train, or at the previous
boundary for validation, and the later period is reported empty. Several tests
below pin that fallback, and one pins that the rejected alternative — assigning
the group that crosses a target to the earlier period — is not what runs.

Pure standard library, so this module runs in the application job with no
training dependencies installed.
"""

import ast
import dataclasses
import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ml.training.splits import (
    DEFAULT_FRACTIONS,
    FRACTION_TOLERANCE,
    Period,
    TemporalSplit,
    temporal_split,
)

BASE = datetime(2024, 1, 1, tzinfo=UTC)


def at(minute: int) -> datetime:
    return BASE + timedelta(minutes=minute)


def unique(n: int) -> list[datetime]:
    return [at(i) for i in range(n)]


def tied(*group_sizes: int) -> list[datetime]:
    """`group_sizes[k]` records at minute k: heavy ties, one timestamp per group."""
    return [at(k) for k, size in enumerate(group_sizes) for _ in range(size)]


def by_period(split: TemporalSplit, timestamps: list[datetime]) -> dict[Period, list[datetime]]:
    grouped: dict[Period, list[datetime]] = {period: [] for period in Period}
    for ts in timestamps:
        grouped[split.period_of(ts)].append(ts)
    return grouped


def counts(split: TemporalSplit) -> tuple[int, int, int]:
    return (
        split.counts[Period.TRAIN],
        split.counts[Period.VALIDATION],
        split.counts[Period.TEST],
    )


def assert_consistent(split: TemporalSplit, timestamps: list[datetime]) -> None:
    """The invariants every split must satisfy, whatever its shape."""
    grouped = by_period(split, timestamps)

    # Reported counts are the counts period_of actually produces.
    assert {p: len(ts) for p, ts in grouped.items()} == dict(split.counts)
    assert sum(split.counts.values()) == len(timestamps)

    # Contiguous and disjoint in time: every earlier period ends strictly
    # before the next non-empty one begins. This is the plan's acceptance
    # criterion, generalised to both boundaries.
    non_empty = [grouped[p] for p in Period if grouped[p]]
    for earlier, later in zip(non_empty, non_empty[1:], strict=False):
        assert max(earlier) < min(later)

    # No timestamp appears in two periods.
    seen: dict[datetime, Period] = {}
    for period, members in grouped.items():
        for ts in members:
            assert seen.setdefault(ts, period) == period, f"{ts} straddles a boundary"

    assert split.train_end <= split.val_end
    for period in Period:
        assert split.achieved_fractions[period] == pytest.approx(
            split.counts[period] / len(timestamps)
        )


# --- the public shape ---------------------------------------------------------


def test_period_is_an_enum_of_exactly_three_ordered_members():
    assert [p.name for p in Period] == ["TRAIN", "VALIDATION", "TEST"]


def test_the_default_fractions_are_seventy_fifteen_fifteen():
    assert DEFAULT_FRACTIONS == (0.70, 0.15, 0.15)


def test_the_split_records_requested_fractions_boundaries_counts_and_achieved_fractions():
    split = temporal_split(unique(10))
    assert split.requested_fractions == {
        Period.TRAIN: 0.70,
        Period.VALIDATION: 0.15,
        Period.TEST: 0.15,
    }
    assert isinstance(split.train_end, datetime)
    assert isinstance(split.val_end, datetime)
    assert set(split.counts) == set(Period)
    assert set(split.achieved_fractions) == set(Period)
    assert split.total == 10


def test_a_split_is_immutable():
    split = temporal_split(unique(10))
    with pytest.raises(dataclasses.FrozenInstanceError):
        split.train_end = at(0)  # type: ignore[misc]


# --- normal chronological splitting -------------------------------------------


@pytest.mark.parametrize("n", [10, 20, 99, 100, 101, 997, 1000])
def test_with_unique_timestamps_achieved_counts_are_within_one_record(n):
    timestamps = unique(n)
    split = temporal_split(timestamps)

    for period, fraction in zip(Period, DEFAULT_FRACTIONS, strict=True):
        assert abs(split.counts[period] - fraction * n) <= 1, (period, split.counts)
    assert_consistent(split, timestamps)


def test_exact_boundaries_for_ten_unique_timestamps():
    """Train target 7, validation target 8.5.

    Largest cumulative count <= 7 is 7, so train ends at the seventh record.
    Largest cumulative count <= 8.5 is 8, so validation holds one record and
    test the remaining two: 7 / 1 / 2.
    """
    timestamps = unique(10)
    split = temporal_split(timestamps)

    assert counts(split) == (7, 1, 2)
    assert split.train_end == timestamps[6]
    assert split.val_end == timestamps[7]


def test_train_never_exceeds_and_test_never_falls_below_the_request():
    """A property of the literal rule outside the fallback case."""
    for n in range(3, 200):
        split = temporal_split(unique(n))
        assert split.counts[Period.TRAIN] <= 0.70 * n + FRACTION_TOLERANCE * n
        assert split.counts[Period.TEST] >= 0.15 * n - FRACTION_TOLERANCE * n


def test_a_cumulative_count_exactly_on_the_target_is_not_lost_to_float_error():
    """0.70 is not exact in binary. For 90 records the train target should be 63,
    but ``0.70 * 90`` evaluates to 62.999…, so without a tolerance the cumulative
    count of 63 would be judged to exceed it and train would lose a record.

    The first assertion guards the guard: if the float product were exact, this
    test could not tell a tolerant implementation from an intolerant one.
    """
    assert 0.70 * 90 < 63, "the float shortfall this test depends on"

    split = temporal_split(unique(90))
    assert counts(split) == (63, 13, 14)

    split = temporal_split(unique(100))
    assert counts(split) == (70, 15, 15)


# --- exact boundary behaviour --------------------------------------------------


def test_the_boundary_timestamp_belongs_to_the_earlier_period():
    timestamps = unique(10)
    split = temporal_split(timestamps)

    assert split.period_of(split.train_end) is Period.TRAIN
    assert split.period_of(split.val_end) is Period.VALIDATION


def test_period_of_is_defined_for_timestamps_outside_and_between_the_input():
    timestamps = unique(10)
    split = temporal_split(timestamps)

    assert split.period_of(timestamps[0] - timedelta(days=365)) is Period.TRAIN
    assert split.period_of(split.train_end + timedelta(seconds=1)) is Period.VALIDATION
    assert split.period_of(split.val_end + timedelta(seconds=1)) is Period.TEST
    assert split.period_of(timestamps[-1] + timedelta(days=365)) is Period.TEST


# --- ties -----------------------------------------------------------------------


def test_heavy_ties_never_place_one_timestamp_in_two_periods():
    """10,000 records across 3 timestamps (plan Task 9).

    Cumulative counts 5000, 8000, 10000 against targets 7000 and 8500: train
    ends at the first timestamp and validation at the second. The achieved
    fractions are reported as they fell, not forced toward 70/15/15.
    """
    timestamps = tied(5000, 3000, 2000)
    split = temporal_split(timestamps)

    assert counts(split) == (5000, 3000, 2000)
    assert split.achieved_fractions == {
        Period.TRAIN: 0.5,
        Period.VALIDATION: 0.3,
        Period.TEST: 0.2,
    }
    assert split.requested_fractions[Period.TRAIN] == 0.70, "the request is kept beside the result"
    assert_consistent(split, timestamps)


def test_the_rejected_crossing_interpretation_is_not_what_runs():
    """Interpretation B — the group whose count crosses a target joins the
    earlier period — would give 8000 / 2000 / 0 here and empty the test period.
    The literal rule gives 5000 / 3000 / 2000."""
    split = temporal_split(tied(5000, 3000, 2000))
    assert counts(split) != (8000, 2000, 0)
    assert split.counts[Period.TEST] == 2000


@pytest.mark.parametrize("group", [1, 2, 3, 7])
def test_no_boundary_ever_straddles_a_tie(group):
    """Acceptance: never max(train) == min(val), across many tie patterns."""
    for n in range(1, 120):
        timestamps = [at(i // group) for i in range(n)]
        split = temporal_split(timestamps)
        assert_consistent(split, timestamps)


# --- the fallback (interpretation A) -------------------------------------------


def test_an_oversized_first_group_falls_back_into_train():
    """No timestamp's cumulative count (9000, 9500, 10000) fits under the train
    target of 7000, nor under the validation target of 8500.

    Fallback: train ends at the earliest timestamp, validation's boundary stays
    at train's, so validation is reported empty and test keeps the rest.
    """
    timestamps = tied(9000, 500, 500)
    split = temporal_split(timestamps)

    assert counts(split) == (9000, 0, 1000)
    assert split.train_end == at(0)
    assert split.val_end == split.train_end, "an empty validation period ends where train does"
    assert split.achieved_fractions[Period.VALIDATION] == 0.0
    assert_consistent(split, timestamps)


def test_validation_can_fall_back_while_train_does_not():
    """Cumulative 60, 90, 100: train fits at 60 (<= 70); nothing fits under
    85 beyond it, so validation's boundary stays at train's and is empty."""
    timestamps = tied(60, 30, 10)
    split = temporal_split(timestamps)

    assert counts(split) == (60, 0, 40)
    assert split.val_end == split.train_end
    assert_consistent(split, timestamps)


@pytest.mark.parametrize("n", [1, 2, 5, 1000])
def test_a_single_timestamp_puts_everything_in_train_and_reports_it(n):
    timestamps = [at(0)] * n
    split = temporal_split(timestamps)

    assert counts(split) == (n, 0, 0)
    assert split.train_end == split.val_end == at(0)
    assert split.achieved_fractions == {
        Period.TRAIN: 1.0,
        Period.VALIDATION: 0.0,
        Period.TEST: 0.0,
    }
    assert split.period_of(at(0)) is Period.TRAIN
    assert_consistent(split, timestamps)


# --- adjustable fractions --------------------------------------------------------


def test_fractions_are_adjustable_and_an_empty_period_is_reported():
    timestamps = unique(10)

    split = temporal_split(timestamps, fractions=(0.8, 0.2, 0.0))
    assert counts(split) == (8, 2, 0)
    assert split.val_end == timestamps[-1]
    assert_consistent(split, timestamps)

    split = temporal_split(timestamps, fractions=(1.0, 0.0, 0.0))
    assert counts(split) == (10, 0, 0)
    assert_consistent(split, timestamps)


def test_fractions_within_the_tolerance_of_one_are_accepted():
    drift = FRACTION_TOLERANCE / 10
    split = temporal_split(unique(10), fractions=(0.70, 0.15, 0.15 + drift))
    assert sum(split.counts.values()) == 10


# --- invalid inputs --------------------------------------------------------------


def test_an_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        temporal_split([])


@pytest.mark.parametrize(
    "fractions",
    [
        pytest.param((0.7, 0.3), id="two-values"),
        pytest.param((0.5, 0.3, 0.1, 0.1), id="four-values"),
        pytest.param((0.7, 0.2, 0.2), id="sums-above-one"),
        pytest.param((0.6, 0.15, 0.15), id="sums-below-one"),
        pytest.param((0.7, 0.15, 0.15 + 1e-6), id="outside-tolerance"),
        pytest.param((-0.1, 0.6, 0.5), id="negative"),
        pytest.param((0.0, 0.5, 0.5), id="zero-train"),
        pytest.param((math.nan, 0.5, 0.5), id="nan"),
        pytest.param((math.inf, 0.0, 0.0), id="infinite"),
        pytest.param(("0.7", "0.15", "0.15"), id="strings"),
        pytest.param(None, id="not-a-sequence"),
    ],
)
def test_invalid_fractions_raise_value_error(fractions):
    with pytest.raises(ValueError):
        temporal_split(unique(10), fractions=fractions)


def test_an_empty_input_with_invalid_fractions_still_raises():
    """Either check may fire first; neither invalid input may pass."""
    with pytest.raises(ValueError):
        temporal_split([], fractions=(0.5, 0.5, 0.5))


# --- determinism, ordering, and the absence of randomness --------------------------


def test_the_split_is_deterministic_across_repeated_calls():
    timestamps = [at(i // 3) for i in range(300)]
    assert temporal_split(timestamps) == temporal_split(timestamps)


def test_the_split_is_independent_of_input_order():
    ordered = [at(i // 4) for i in range(400)]
    reference = temporal_split(ordered)

    shuffled = ordered[:]
    random.Random(20240901).shuffle(shuffled)
    for permutation in (list(reversed(ordered)), ordered[200:] + ordered[:200], shuffled):
        assert temporal_split(permutation) == reference


def test_global_random_state_cannot_influence_the_split():
    """Regression: a random split would change with the global seed."""
    timestamps = [at(i // 2) for i in range(500)]
    random.seed(1)
    first = temporal_split(timestamps)
    random.seed(987654321)
    second = temporal_split(timestamps)
    assert first == second


def test_every_training_record_precedes_every_evaluation_record():
    """Addendum §6.4: no validation or test record precedes the training cut.
    A random split fails this almost surely on 1,000 distinct timestamps."""
    timestamps = unique(1000)
    split = temporal_split(timestamps)
    grouped = by_period(split, timestamps)

    assert max(grouped[Period.TRAIN]) < min(grouped[Period.VALIDATION])
    assert max(grouped[Period.VALIDATION]) < min(grouped[Period.TEST])
    assert grouped[Period.TRAIN] == timestamps[: split.counts[Period.TRAIN]]


def test_the_split_module_imports_no_source_of_randomness_or_index_splitter():
    tree = ast.parse(Path("ml/training/splits.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = ("random", "secrets", "numpy", "sklearn", "pandas", "django")
    for module in imported:
        assert module.split(".")[0] not in forbidden, module
