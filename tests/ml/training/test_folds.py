"""Forward-chaining folds: expanding windows over train, cut at timestamps.

Plan §J and Task 10, addendum §6.3, D11, and D30, which resolves Task 10's
contradictions. Contract in brief:

* the warm-up is realised by Task 9's date-cut rule (largest timestamp whose
  cumulative count does not exceed ``warmup_fraction * n``, D29 fallback);
* with W the realised warm-up count, the remaining ``n - W`` records form
  exactly ``n_folds`` apply blocks with targets ``W + (n - W) * k / n_folds``;
* every boundary uses the Task 9 / D29 rule, so no timestamp group is split and
  a target no timestamp satisfies leaves that fold empty;
* windows expand: a fold's fit block is everything strictly before its apply
  block, so no fold ever fits on a later record.

Standard library only, so these tests run in the application job. The
``TimeSeriesSplit`` comparison simulates positional splitting rather than
importing scikit-learn.
"""

import ast
import dataclasses
import inspect
import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ml.training.splits import Fold, Period, forward_chaining_folds, temporal_split

BASE = datetime(2024, 1, 1, tzinfo=UTC)


def at(minute: int) -> datetime:
    return BASE + timedelta(minutes=minute)


def unique(n: int) -> list[datetime]:
    return [at(i) for i in range(n)]


def tied(*group_sizes: int) -> list[datetime]:
    """`group_sizes[k]` records at minute k."""
    return [at(k) for k, size in enumerate(group_sizes) for _ in range(size)]


def sizes(folds: list[Fold]) -> tuple[int, ...]:
    return tuple(len(fold.apply_indices) for fold in folds)


def warmup_size(folds: list[Fold]) -> int:
    """D30: the warm-up size is reported as the first fold's fit block."""
    return len(folds[0].fit_indices)


def chronological(timestamps: list[datetime]) -> list[int]:
    return sorted(range(len(timestamps)), key=lambda i: (timestamps[i], i))


def assert_fold_invariants(folds: list[Fold], timestamps: list[datetime], n_folds: int) -> None:
    """Every guarantee D30 makes, checked against the input it was built from."""
    n = len(timestamps)
    order = chronological(timestamps)
    assert len(folds) == n_folds, "exactly n_folds folds, empty or not"

    warmup = folds[0].fit_indices
    assert len(warmup) >= 1, "the warm-up is never empty"
    assert warmup == tuple(order[: len(warmup)]), "the warm-up is a chronological prefix"

    covered: list[int] = []
    previous_apply_end = folds[0].fit_end
    for i, fold in enumerate(folds):
        # Expanding window: fit is exactly everything before this apply block.
        assert fold.fit_indices == tuple(order[: len(warmup) + len(covered)]), f"fold {i} fit"
        # Index tuples are in ascending chronological order.
        assert list(fold.fit_indices) == sorted(fold.fit_indices, key=lambda j: (timestamps[j], j))
        assert list(fold.apply_indices) == sorted(
            fold.apply_indices, key=lambda j: (timestamps[j], j)
        )

        # Contiguous boundaries: each fold's fit ends where the previous apply ended.
        assert fold.fit_end == previous_apply_end
        assert fold.fit_end == max(timestamps[j] for j in fold.fit_indices)

        if fold.apply_indices:
            apply_ts = [timestamps[j] for j in fold.apply_indices]
            fit_ts = [timestamps[j] for j in fold.fit_indices]
            # No future leakage: every fit record strictly precedes every apply record.
            assert max(fit_ts) < min(apply_ts)
            assert fold.apply_start == min(apply_ts)
            assert fold.apply_end == max(apply_ts)
            assert fold.fit_end < fold.apply_start <= fold.apply_end
        else:
            assert fold.apply_start is None, "an empty fold has no start"
            assert fold.apply_end == fold.fit_end, "an empty fold ends where its fit ends"

        covered.extend(fold.apply_indices)
        previous_apply_end = fold.apply_end

    # Apply blocks are disjoint and together cover everything after the warm-up.
    assert len(covered) == len(set(covered))
    assert set(covered) == set(range(n)) - set(warmup)
    assert tuple(covered) == tuple(order[len(warmup) :])

    # No timestamp group is split across the warm-up or any two folds.
    block_of: dict[datetime, int] = {}
    for j in warmup:
        assert block_of.setdefault(timestamps[j], -1) == -1
    for i, fold in enumerate(folds):
        for j in fold.apply_indices:
            assert block_of.setdefault(timestamps[j], i) == i, f"{timestamps[j]} straddles"


# --- API shape ---------------------------------------------------------------------


def test_the_signature_and_defaults_match_plan_section_j():
    params = inspect.signature(forward_chaining_folds).parameters
    assert list(params) == ["timestamps", "n_folds", "warmup_fraction"]
    assert params["n_folds"].default == 5
    assert params["warmup_fraction"].default == 0.20


def test_the_default_produces_exactly_five_folds():
    folds = forward_chaining_folds(unique(100))
    assert isinstance(folds, list)
    assert len(folds) == 5
    assert all(isinstance(fold, Fold) for fold in folds)


def test_a_fold_exposes_the_contract_fields_and_is_immutable():
    fold = forward_chaining_folds(unique(100))[0]
    names = [field.name for field in dataclasses.fields(Fold)]
    assert names == ["fit_indices", "apply_indices", "fit_end", "apply_start", "apply_end"]
    assert isinstance(fold.fit_indices, tuple) and isinstance(fold.apply_indices, tuple)
    assert all(isinstance(i, int) for i in fold.fit_indices + fold.apply_indices)
    with pytest.raises(dataclasses.FrozenInstanceError):
        fold.fit_end = at(0)  # type: ignore[misc]


# --- normal construction and exact boundaries ---------------------------------------


def test_one_hundred_unique_records_give_a_twenty_warmup_and_five_blocks_of_sixteen():
    """W = 20; targets 36, 52, 68, 84, 100; every block holds exactly 16."""
    timestamps = unique(100)
    folds = forward_chaining_folds(timestamps)

    assert warmup_size(folds) == 20
    assert sizes(folds) == (16, 16, 16, 16, 16)
    for k, fold in enumerate(folds):
        start = 20 + 16 * k
        assert fold.fit_indices == tuple(range(start))
        assert fold.apply_indices == tuple(range(start, start + 16))
        assert fold.fit_end == at(start - 1)
        assert fold.apply_start == at(start)
        assert fold.apply_end == at(start + 15)
    assert_fold_invariants(folds, timestamps, 5)


def test_exact_boundaries_for_ten_unique_records():
    """Warm-up target 2 gives W = 2. Remaining 8; targets 3.6, 5.2, 6.8, 8.4 and
    10. The largest cumulative counts not exceeding them are 3, 5, 6, 8 and 10,
    so the apply blocks hold 1, 2, 1, 2 and 2 records."""
    timestamps = unique(10)
    folds = forward_chaining_folds(timestamps)

    assert warmup_size(folds) == 2
    assert sizes(folds) == (1, 2, 1, 2, 2)
    assert [fold.apply_end for fold in folds] == [at(2), at(4), at(5), at(7), at(9)]
    assert_fold_invariants(folds, timestamps, 5)


@pytest.mark.parametrize("n", [6, 7, 10, 11, 25, 99, 100, 101, 257, 1000])
def test_apply_blocks_are_equal_within_one_record_for_unique_timestamps(n):
    timestamps = unique(n)
    folds = forward_chaining_folds(timestamps)
    remaining = n - warmup_size(folds)

    for size in sizes(folds):
        assert abs(size - remaining / 5) <= 1, (n, sizes(folds))
    assert_fold_invariants(folds, timestamps, 5)


def test_the_windows_expand_rather_than_roll():
    """Each fold fits on the warm-up and every earlier apply block, never on a
    fixed-length trailing window."""
    timestamps = unique(100)
    folds = forward_chaining_folds(timestamps)

    for earlier, later in zip(folds, folds[1:], strict=False):
        assert later.fit_indices == earlier.fit_indices + earlier.apply_indices
        assert len(later.fit_indices) > len(earlier.fit_indices)
    assert all(fold.fit_indices[0] == 0 for fold in folds), (
        "every fit block starts at the beginning"
    )


def test_no_fold_fits_on_its_own_or_any_later_apply_block():
    timestamps = unique(200)
    folds = forward_chaining_folds(timestamps)

    for i, fold in enumerate(folds):
        fit = set(fold.fit_indices)
        for later in folds[i:]:
            assert not fit & set(later.apply_indices)


# --- D30: the warm-up is realised by Task 9's semantics -----------------------------


@pytest.mark.parametrize(
    "timestamps",
    [unique(100), unique(13), tied(30, *[1] * 70), tied(5, 5, 5, 5), tied(3, 9, 2, 7, 1, 8)],
    ids=["unique-100", "unique-13", "oversized-first", "equal-groups", "mixed-ties"],
)
def test_the_warmup_boundary_is_task_nines_date_cut(timestamps):
    """The warm-up equals Task 9's train period for fractions (w, 1 - w, 0)."""
    folds = forward_chaining_folds(timestamps, warmup_fraction=0.20)
    split = temporal_split(timestamps, fractions=(0.20, 0.80, 0.0))

    assert folds[0].fit_end == split.train_end
    assert warmup_size(folds) == split.counts[Period.TRAIN]


def test_fold_cuts_are_measured_against_the_realised_remainder():
    """D30, B1. A first timestamp holding 30 of 100 records forces the warm-up
    to 30. The remaining 70 split into five blocks of 14.

    Measuring cuts as fixed shares of the whole period (the rejected B2) would
    give 6, 16, 16, 16 and 16 instead.
    """
    timestamps = tied(30, *[1] * 70)
    folds = forward_chaining_folds(timestamps)

    assert warmup_size(folds) == 30
    assert sizes(folds) == (14, 14, 14, 14, 14)
    assert sizes(folds) != (6, 16, 16, 16, 16)
    assert_fold_invariants(folds, timestamps, 5)


# --- ties --------------------------------------------------------------------------


@pytest.mark.parametrize("group", [1, 2, 3, 4, 7])
def test_no_timestamp_group_ever_straddles_a_boundary(group):
    for n in range(1, 130):
        timestamps = [at(i // group) for i in range(n)]
        folds = forward_chaining_folds(timestamps)
        assert_fold_invariants(folds, timestamps, 5)


def test_a_target_no_timestamp_satisfies_leaves_that_fold_empty_and_reported():
    """Groups of 20 and 80. W = 20; targets 36, 52, 68 and 84 all fall inside the
    80-record group, so folds one to four are empty and fold five takes all 80."""
    timestamps = tied(20, 80)
    folds = forward_chaining_folds(timestamps)

    assert len(folds) == 5
    assert warmup_size(folds) == 20
    assert sizes(folds) == (0, 0, 0, 0, 80)
    for fold in folds[:4]:
        assert fold.apply_start is None
        assert fold.apply_end == fold.fit_end == at(0)
    assert folds[4].apply_start == folds[4].apply_end == at(1)
    assert_fold_invariants(folds, timestamps, 5)


@pytest.mark.parametrize("n", [1, 2, 5, 1000])
def test_a_single_timestamp_puts_everything_in_warmup_and_returns_empty_folds(n):
    timestamps = [at(0)] * n
    folds = forward_chaining_folds(timestamps)

    assert len(folds) == 5
    assert warmup_size(folds) == n
    assert sizes(folds) == (0, 0, 0, 0, 0)
    for fold in folds:
        assert fold.fit_indices == tuple(range(n))
        assert fold.apply_start is None
        assert fold.apply_end == fold.fit_end == at(0)
    assert_fold_invariants(folds, timestamps, 5)


def test_index_based_splitting_straddles_a_tie_where_date_cuts_do_not():
    """Why ``TimeSeriesSplit`` was rejected (plan §J).

    ``TimeSeriesSplit(n_splits=5)`` on 20 records uses test blocks of
    ``20 // 6 = 3`` starting at positions 5, 8, 11, 14 and 17. With four records
    per timestamp, position 5 sits inside the group at positions 4 to 7, so the
    first fit block and its test block share a timestamp. Simulated here so the
    test needs no scikit-learn.
    """
    timestamps = tied(4, 4, 4, 4, 4)
    n, n_splits = len(timestamps), 5
    test_size = n // (n_splits + 1)
    test_starts = range(n - n_splits * test_size, n, test_size)

    straddles = [s for s in test_starts if timestamps[s - 1] == timestamps[s]]
    assert straddles, "index-based splitting must straddle here, or this test proves nothing"

    folds = forward_chaining_folds(timestamps)
    for fold in folds:
        if fold.apply_indices:
            assert max(timestamps[j] for j in fold.fit_indices) < min(
                timestamps[j] for j in fold.apply_indices
            )
    assert_fold_invariants(folds, timestamps, 5)


# --- n_folds and warmup_fraction ----------------------------------------------------


def test_one_fold_holds_everything_after_the_warmup():
    timestamps = unique(10)
    folds = forward_chaining_folds(timestamps, n_folds=1)

    assert len(folds) == 1
    assert warmup_size(folds) == 2
    assert folds[0].apply_indices == tuple(range(2, 10))
    assert folds[0].apply_end == at(9)
    assert_fold_invariants(folds, timestamps, 1)


@pytest.mark.parametrize("n_folds", [2, 3, 7, 12])
def test_other_fold_counts_are_honoured_exactly(n_folds):
    timestamps = [at(i // 2) for i in range(240)]
    folds = forward_chaining_folds(timestamps, n_folds=n_folds)
    assert len(folds) == n_folds
    assert_fold_invariants(folds, timestamps, n_folds)


def test_more_folds_than_records_still_returns_every_fold():
    timestamps = unique(4)
    folds = forward_chaining_folds(timestamps, n_folds=10)

    assert len(folds) == 10
    assert sum(sizes(folds)) == 4 - warmup_size(folds)
    assert_fold_invariants(folds, timestamps, 10)


def test_the_warmup_fraction_is_honoured():
    timestamps = unique(10)
    folds = forward_chaining_folds(timestamps, warmup_fraction=0.5)
    assert warmup_size(folds) == 5
    assert sizes(folds) == (1, 1, 1, 1, 1)
    assert_fold_invariants(folds, timestamps, 5)


# --- invalid inputs ----------------------------------------------------------------


def test_empty_timestamps_raise():
    with pytest.raises(ValueError, match="empty"):
        forward_chaining_folds([])


@pytest.mark.parametrize(
    "n_folds",
    [0, -1, 1.5, 5.0, "5", True, False, None],
    ids=["zero", "negative", "fractional", "float-five", "string", "true", "false", "none"],
)
def test_invalid_fold_counts_raise_value_error(n_folds):
    with pytest.raises(ValueError):
        forward_chaining_folds(unique(10), n_folds=n_folds)


@pytest.mark.parametrize(
    "warmup_fraction",
    [0.0, 1.0, -0.1, 1.5, math.nan, math.inf, "0.2", True, None],
    ids=["zero", "one", "negative", "above-one", "nan", "infinite", "string", "bool", "none"],
)
def test_invalid_warmup_fractions_raise_value_error(warmup_fraction):
    with pytest.raises(ValueError):
        forward_chaining_folds(unique(10), warmup_fraction=warmup_fraction)


# --- determinism, ordering, and the absence of randomness ---------------------------


def boundaries(folds: list[Fold]) -> list[tuple[datetime, datetime | None, datetime]]:
    return [(fold.fit_end, fold.apply_start, fold.apply_end) for fold in folds]


def timestamps_per_block(folds: list[Fold], timestamps: list[datetime]) -> list[list[datetime]]:
    blocks = [sorted(timestamps[j] for j in folds[0].fit_indices)]
    blocks += [sorted(timestamps[j] for j in fold.apply_indices) for fold in folds]
    return blocks


def test_folds_are_deterministic_across_repeated_calls():
    timestamps = [at(i // 3) for i in range(300)]
    assert forward_chaining_folds(timestamps) == forward_chaining_folds(timestamps)


def test_folds_are_independent_of_input_order():
    """Boundaries and block contents are identical for any permutation; the
    indices differ only because they name positions in the permuted input."""
    ordered = [at(i // 4) for i in range(400)]
    reference = forward_chaining_folds(ordered)

    shuffled = ordered[:]
    random.Random(20240910).shuffle(shuffled)
    for permutation in (list(reversed(ordered)), ordered[150:] + ordered[:150], shuffled):
        folds = forward_chaining_folds(permutation)
        assert boundaries(folds) == boundaries(reference)
        assert timestamps_per_block(folds, permutation) == timestamps_per_block(reference, ordered)
        assert_fold_invariants(folds, permutation, 5)


def test_indices_are_original_positions_in_chronological_order_with_ties_by_position():
    timestamps = [at(3), at(1), at(1), at(0), at(2), at(1), at(4), at(0), at(2), at(3)]
    folds = forward_chaining_folds(timestamps, n_folds=2, warmup_fraction=0.2)

    everything = folds[-1].fit_indices + folds[-1].apply_indices
    assert list(everything) == chronological(timestamps)
    assert chronological(timestamps)[:2] == [3, 7], "two records at minute 0, by position"


def test_global_random_state_cannot_influence_the_folds():
    timestamps = [at(i // 2) for i in range(500)]
    random.seed(1)
    first = forward_chaining_folds(timestamps)
    random.seed(424242)
    second = forward_chaining_folds(timestamps)
    assert first == second


def test_the_split_module_imports_no_index_splitter_or_randomness():
    """Plan Task 10 acceptance: no ``sklearn.model_selection`` in ``splits.py``."""
    tree = ast.parse(Path("ml/training/splits.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not any(m == "sklearn.model_selection" or m.startswith("sklearn") for m in imported)
    for module in imported:
        assert module.split(".")[0] not in ("random", "secrets", "numpy", "pandas", "django")


# --- interaction with Task 9 --------------------------------------------------------


def test_folds_run_over_the_train_period_of_a_temporal_split():
    """Folds are built from train timestamps only; no fold reaches validation
    or test, and warm-up plus folds account for every training record."""
    everything = [at(i // 3) for i in range(900)]
    split = temporal_split(everything)
    train = [ts for ts in everything if split.period_of(ts) is Period.TRAIN]

    folds = forward_chaining_folds(train)

    assert warmup_size(folds) + sum(sizes(folds)) == split.counts[Period.TRAIN]
    for fold in folds:
        assert fold.apply_end <= split.train_end
        assert all(split.period_of(train[j]) is Period.TRAIN for j in fold.apply_indices)
    assert_fold_invariants(folds, train, 5)


def test_task_nine_behaviour_is_unchanged_by_the_folds_module():
    """A pinned Task 9 case from D29, re-asserted beside the new code."""
    split = temporal_split(tied(9000, 500, 500))
    assert (
        split.counts[Period.TRAIN],
        split.counts[Period.VALIDATION],
        split.counts[Period.TEST],
    ) == (9000, 0, 1000)
    assert split.val_end == split.train_end == at(0)
