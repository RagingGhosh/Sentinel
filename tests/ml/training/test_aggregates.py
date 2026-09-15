"""Out-of-fold category aggregates: the leakage-critical module (plan Task 11, D31).

Every out-of-fold value is built from its fold's fit block alone — both the
observations averaged (A) and the p75 threshold that turns ``resolution_hours``
into a breach label (B). Validation and test rows receive values fitted once on
the whole training period. The tests below make leakage *visible*: small
hand-computed fixtures pin exact values, and an exhaustive perturbation test
changes every target a row must not see, one at a time, to extreme values and
to ``None``, and requires that row's features to stay bit-for-bit identical.

Expected values are computed by hand or with an independent reference
percentile, never with the module under test.
"""

import ast
import dataclasses
import inspect
import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ingest.schema import CFPBOutcome, CorpusRecord, NYC311Outcome
from ml.training.aggregates import (
    AggregateColumns,
    FrozenAggregates,
    apply_category_aggregates,
    fit_category_aggregates,
    oof_category_aggregates,
)
from ml.training.splits import Period, forward_chaining_folds, temporal_split

BASE = datetime(2024, 1, 1, tzinfo=UTC)


def at(minute: int) -> datetime:
    return BASE + timedelta(minutes=minute)


def build(spec):
    """``spec`` is a list of ``(minute, category, resolution_hours)``."""
    records, outcomes = [], []
    for i, (minute, category, hours) in enumerate(spec):
        submitted = at(minute)
        records.append(
            CorpusRecord(
                source="nyc311",
                external_id=f"r{i}",
                text=f"descriptor {i}",
                label=category,
                submitted_at=submitted,
            )
        )
        outcomes.append(outcome(f"r{i}", submitted, hours))
    return records, outcomes


def outcome(external_id, submitted, hours):
    closed = None if hours is None else submitted + timedelta(hours=hours)
    return NYC311Outcome(external_id=external_id, closed_at=closed, resolution_hours=hours)


def folds_for(records, **kwargs):
    return forward_chaining_folds([r.submitted_at for r in records], **kwargs)


def with_hours(outcomes, records, index, hours):
    changed = list(outcomes)
    changed[index] = outcome(outcomes[index].external_id, records[index].submitted_at, hours)
    return changed


def same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


def row(columns: AggregateColumns, i: int) -> tuple[float, float]:
    return columns.category_mean_resolution_hours[i], columns.category_breach_rate[i]


def rows_same(x: tuple[float, float], y: tuple[float, float]) -> bool:
    return same(x[0], y[0]) and same(x[1], y[1])


def columns_same(a: AggregateColumns, b: AggregateColumns) -> bool:
    """NaN-aware equality: warm-up rows hold NaN, and NaN never equals itself."""
    n = len(a.category_mean_resolution_hours)
    if n != len(b.category_mean_resolution_hours) or len(a.category_breach_rate) != len(
        b.category_breach_rate
    ):
        return False
    return all(rows_same(row(a, i), row(b, i)) for i in range(n))


def reference_p75(values):
    x = sorted(values)
    h = (len(x) - 1) * 0.75
    low, high = math.floor(h), math.ceil(h)
    return x[low] + (h - low) * (x[high] - x[low])


# Ten unique minutes. Default folds: warm-up rows 0-1, then apply blocks
# [2], [3, 4], [5], [6, 7], [8, 9] with fit blocks 0-1, 0-2, 0-4, 0-5, 0-7.
SIMPLE = [(i, "A", float(10 * (i + 1))) for i in range(10)]

MULTI = [
    (0, "A", 10.0),
    (1, "B", 100.0),
    (2, "A", 20.0),
    (3, "C", 1000.0),
    (4, "A", 30.0),
    (5, "C", 2000.0),
    (6, "B", 200.0),
    (7, "A", 40.0),
    (8, "D", 5.0),
    (9, "C", 3000.0),
]

# Pairs of records share each minute. Folds: warm-up rows 0-1, an empty fold,
# then apply blocks [2, 3], [4, 5], [6, 7], [8, 9].
TIES = [(i // 2, "A", float(10 * (i + 1))) for i in range(10)]

OPEN = [
    (0, "A", None),
    (1, "A", None),
    (2, "A", 20.0),
    (3, "A", 30.0),
    (4, "A", None),
    (5, "A", 50.0),
    (6, "A", 60.0),
    (7, "A", None),
    (8, "A", 80.0),
    (9, "A", 90.0),
]

# Category B appears in the first two fit blocks only as open requests (E1).
ALL_OPEN = [
    (0, "A", 10.0),
    (1, "B", None),
    (2, "B", None),
    (3, "A", 20.0),
    (4, "B", 40.0),
    (5, "B", None),
    (6, "A", 60.0),
    (7, "A", 70.0),
    (8, "B", 80.0),
    (9, "A", 90.0),
]


# --- API shape -------------------------------------------------------------------------


def test_the_public_signatures_match_plan_task_eleven():
    assert list(inspect.signature(oof_category_aggregates).parameters) == [
        "records",
        "outcomes",
        "folds",
    ]
    assert list(inspect.signature(fit_category_aggregates).parameters) == [
        "train_records",
        "train_outcomes",
    ]
    assert list(inspect.signature(apply_category_aggregates).parameters) == ["frozen", "records"]


def test_aggregate_columns_hold_exactly_the_two_features_in_order_and_are_frozen():
    names = [f.name for f in dataclasses.fields(AggregateColumns)]
    assert names == ["category_mean_resolution_hours", "category_breach_rate"]

    records, outcomes = build(SIMPLE)
    columns = oof_category_aggregates(records, outcomes, folds_for(records))
    assert isinstance(columns, AggregateColumns)
    assert isinstance(columns.category_mean_resolution_hours, tuple)
    assert isinstance(columns.category_breach_rate, tuple)
    assert len(columns.category_mean_resolution_hours) == len(columns.category_breach_rate) == 10
    with pytest.raises(dataclasses.FrozenInstanceError):
        columns.category_breach_rate = ()  # type: ignore[misc]


def test_frozen_aggregates_reject_mutation():
    records, outcomes = build(MULTI)
    frozen = fit_category_aggregates(records, outcomes)

    assert isinstance(frozen, FrozenAggregates)
    for field in dataclasses.fields(FrozenAggregates):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(frozen, field.name, None)
    with pytest.raises(TypeError):
        frozen.mean_resolution_hours_by_category["A"] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        frozen.breach_rate_by_category["A"] = 0.0  # type: ignore[index]


# --- exact out-of-fold values ------------------------------------------------------------


def test_simple_out_of_fold_values_are_exact():
    """Category A throughout, so each breach threshold is the fit block's global
    p75 (A has fewer than 100 eligible observations everywhere).

    fold 0, fit 10,20:            mean 15;   p75 17.5 -> breaches 20       -> 1/2
    fold 1, fit 10..30:           mean 20;   p75 25   -> breaches 30       -> 1/3
    fold 2, fit 10..50:           mean 30;   p75 40   -> breaches 50       -> 1/5
    fold 3, fit 10..60:           mean 35;   p75 47.5 -> breaches 50, 60   -> 2/6
    fold 4, fit 10..80:           mean 45;   p75 62.5 -> breaches 70, 80   -> 2/8
    """
    records, outcomes = build(SIMPLE)
    columns = oof_category_aggregates(records, outcomes, folds_for(records))

    expected = {
        2: (15.0, 1 / 2),
        3: (20.0, 1 / 3),
        4: (20.0, 1 / 3),
        5: (30.0, 1 / 5),
        6: (35.0, 2 / 6),
        7: (35.0, 2 / 6),
        8: (45.0, 2 / 8),
        9: (45.0, 2 / 8),
    }
    for i, (mean, rate) in expected.items():
        assert row(columns, i) == pytest.approx((mean, rate), rel=0, abs=1e-12), i


def test_warmup_rows_are_nan_not_the_global_mean():
    records, outcomes = build(SIMPLE)
    folds = folds_for(records)
    columns = oof_category_aggregates(records, outcomes, folds)

    warmup = folds[0].fit_indices
    assert warmup == (0, 1)
    for i in warmup:
        mean, rate = row(columns, i)
        assert math.isnan(mean) and math.isnan(rate), i


def test_multi_category_values_are_exact():
    """Row 2 (A, fit 0-1): A mean 10; global p75 of {10, 100} is 77.5, A's 10 is
    not a breach -> 0.
    Row 5 (C, fit 0-4): C mean 1000; global p75 of {10,20,30,100,1000} is 100,
    C's 1000 breaches -> 1.
    Row 6 (B, fit 0-5): B mean 100; global p75 of {..,2000} is 775 -> 0.
    Row 9 (C, fit 0-7): C mean 1500; global p75 of 8 values is 400 -> 1.
    """
    records, outcomes = build(MULTI)
    columns = oof_category_aggregates(records, outcomes, folds_for(records))

    assert row(columns, 2) == (10.0, 0.0)
    assert row(columns, 5) == (1000.0, 1.0)
    assert row(columns, 6) == (100.0, 0.0)
    assert row(columns, 9) == (1500.0, 1.0)


# --- D31 Q2a: a category absent from the fit block -----------------------------------------


def test_a_category_absent_from_the_fit_block_gets_that_fit_blocks_global_aggregate():
    """Row 3 is category C's first appearance. Its fit block (rows 0-2) holds
    10, 100 and 20: global mean 130/3, global p75 60, one breach (100) -> 1/3.
    The whole-training-period global mean would be 640.5 instead."""
    records, outcomes = build(MULTI)
    columns = oof_category_aggregates(records, outcomes, folds_for(records))

    mean, rate = row(columns, 3)
    assert mean == pytest.approx(130 / 3, rel=0, abs=1e-12)
    assert rate == pytest.approx(1 / 3, rel=0, abs=1e-12)
    assert mean != pytest.approx(640.5), "never the whole-training-period global"


def test_a_category_appearing_only_in_the_current_apply_block_gets_the_fit_block_global():
    """Row 8 is the only D record. Fit block rows 0-7: global mean 3400/8 = 425;
    global p75 of {10,20,30,40,100,200,1000,2000} is 400 -> 1000 and 2000 breach
    -> 2/8."""
    records, outcomes = build(MULTI)
    columns = oof_category_aggregates(records, outcomes, folds_for(records))

    assert row(columns, 8) == pytest.approx((425.0, 2 / 8), rel=0, abs=1e-12)


# --- the defining property, made exhaustive ------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [SIMPLE, MULTI, TIES, OPEN, ALL_OPEN],
    ids=["simple", "multi", "ties", "open", "all-open"],
)
def test_no_target_outside_a_rows_fit_block_can_change_that_rows_features(spec):
    """For every row and every target the row must not see — its own, a
    same-timestamp partner's, anything in its own apply block or a later one —
    changing that single target to 0, to 1e6, or to an open request leaves the
    row's two features bit-for-bit identical. Warm-up rows stay NaN throughout.

    A threshold fitted on anything wider than the fit block, a mean that
    includes the row, or a cumulative history within the apply block each
    changes some value here.
    """
    records, outcomes = build(spec)
    folds = folds_for(records)
    baseline = oof_category_aggregates(records, outcomes, folds)
    warmup = set(folds[0].fit_indices)

    visible = {i: set() for i in range(len(records))}
    for fold in folds:
        for i in fold.apply_indices:
            visible[i] = set(fold.fit_indices)

    for j in range(len(records)):
        for hours in (0.0, 1e6, None):
            perturbed = oof_category_aggregates(
                records, with_hours(outcomes, records, j, hours), folds
            )
            for i in range(len(records)):
                if i in warmup:
                    assert all(math.isnan(v) for v in row(perturbed, i)), (i, j, hours)
                elif j not in visible[i]:
                    assert rows_same(row(perturbed, i), row(baseline, i)), (
                        f"row {i} changed when target {j} (outside its fit block) became {hours}"
                    )


def test_changing_a_rows_own_outcome_does_not_change_its_own_aggregate():
    """Plan Task 11's defining property, stated directly."""
    records, outcomes = build(MULTI)
    folds = folds_for(records)
    baseline = oof_category_aggregates(records, outcomes, folds)

    for i in range(len(records)):
        for hours in (0.0, 7_777.0, 1e6, None):
            changed = oof_category_aggregates(
                records, with_hours(outcomes, records, i, hours), folds
            )
            assert rows_same(row(changed, i), row(baseline, i)), (i, hours)


def test_changing_a_rows_outcome_does_change_a_later_same_category_aggregate():
    """The aggregate is live: row 3 (C) feeds row 5 (C) in a later fold."""
    records, outcomes = build(MULTI)
    folds = folds_for(records)
    before = oof_category_aggregates(records, outcomes, folds)
    after = oof_category_aggregates(records, with_hours(outcomes, records, 3, 5000.0), folds)

    assert after.category_mean_resolution_hours[5] != before.category_mean_resolution_hours[5]
    assert after.category_mean_resolution_hours[5] == 5000.0


def test_a_target_leaking_in_from_a_later_fold_would_be_visible():
    """Adversarial: one later outcome made enormous. If thresholds were fitted
    on the whole training period, the fold-0 threshold would rise and row 2's
    breach rate would fall from 1/2 to 0. It must not move."""
    records, outcomes = build(SIMPLE)
    folds = folds_for(records)
    leaked = with_hours(outcomes, records, 9, 1e6)

    assert oof_category_aggregates(records, leaked, folds).category_breach_rate[2] == 1 / 2


def test_a_same_timestamp_partner_never_contributes():
    """Rows 2 and 3 share a minute and an apply block; so do rows 4 and 5."""
    records, outcomes = build(TIES)
    folds = folds_for(records)
    baseline = oof_category_aggregates(records, outcomes, folds)

    for i, partner in ((2, 3), (3, 2), (4, 5), (5, 4)):
        changed = oof_category_aggregates(
            records, with_hours(outcomes, records, partner, 1e6), folds
        )
        assert rows_same(row(changed, i), row(baseline, i)), (i, partner)


# --- the breach threshold comes from the fit block, including a category's own p75 ----------


def big_fixture(open_a_rows=()):
    """Rows 0-49: category B, hours 1000-1049. Rows 50-299: category A, hours 1-250.
    Folds: warm-up rows 0-59, apply blocks 60-107, 108-155, 156-203, 204-251, 252-299.
    Fold 1's fit block holds 58 A rows (fallback); fold 2's holds 106 (own p75)."""
    spec = [(i, "B", 1000.0 + i) for i in range(50)]
    spec += [(50 + i, "A", float(i + 1)) for i in range(250)]
    for i in open_a_rows:
        minute, category, _ = spec[i]
        spec[i] = (minute, category, None)
    return build(spec)


def test_the_breach_threshold_switches_to_a_categorys_own_p75_at_one_hundred_eligible():
    records, outcomes = big_fixture()
    folds = folds_for(records)
    assert len(folds[0].fit_indices) == 60
    assert folds[1].apply_indices[0] == 108 and folds[2].apply_indices[0] == 156
    columns = oof_category_aggregates(records, outcomes, folds)

    # Fold 1: 58 eligible A in the fit block -> global p75 of that fit block.
    fit1 = [outcomes[j].resolution_hours for j in folds[1].fit_indices]
    threshold1 = reference_p75(fit1)
    a1 = [outcomes[j].resolution_hours for j in folds[1].fit_indices if records[j].label == "A"]
    assert row(columns, 108) == pytest.approx(
        (sum(a1) / len(a1), sum(h > threshold1 for h in a1) / len(a1)), rel=0, abs=1e-12
    )
    assert row(columns, 108)[1] == 0.0

    # Fold 2: 106 eligible A -> A's own p75 over its fit-block values.
    a2 = [outcomes[j].resolution_hours for j in folds[2].fit_indices if records[j].label == "A"]
    assert len(a2) == 106
    own2 = reference_p75(a2)
    assert row(columns, 156) == pytest.approx(
        (sum(a2) / len(a2), sum(h > own2 for h in a2) / len(a2)), rel=0, abs=1e-12
    )
    assert row(columns, 156)[1] == pytest.approx(27 / 106, rel=0, abs=1e-12)


def test_open_requests_keep_a_category_below_one_hundred_eligible():
    """T2a: fold 2's fit block still holds 106 A records, but seven are open, so
    only 99 are eligible and A falls back to the fit block's global p75."""
    records, outcomes = big_fixture(open_a_rows=range(60, 67))
    folds = folds_for(records)
    columns = oof_category_aggregates(records, outcomes, folds)

    fit2 = [outcomes[j].resolution_hours for j in folds[2].fit_indices]
    eligible = [h for h in fit2 if h is not None]
    a2 = [
        outcomes[j].resolution_hours
        for j in folds[2].fit_indices
        if records[j].label == "A" and outcomes[j].resolution_hours is not None
    ]
    assert len(a2) == 99
    global2 = reference_p75(eligible)
    assert row(columns, 156) == pytest.approx(
        (sum(a2) / len(a2), sum(h > global2 for h in a2) / len(a2)), rel=0, abs=1e-12
    )


# --- D31 Q3: open requests ------------------------------------------------------------------


def test_open_requests_contribute_nothing_but_still_receive_aggregates():
    """fold 0, fit rows 0-1 both open:     no eligible observation -> NaN, NaN
    fold 1, fit eligible {20}:              mean 20; p75 20, 20 > 20 is false -> 0
    fold 2, fit eligible {20, 30}:          mean 25; p75 27.5 -> 30 breaches -> 1/2
    fold 3, fit eligible {20, 30, 50}:      mean 100/3; p75 40 -> 50 -> 1/3
    fold 4, fit eligible {20, 30, 50, 60}:  mean 40; p75 52.5 -> 60 -> 1/4
    Row 4 is itself open and still receives fold 1's values.
    """
    records, outcomes = build(OPEN)
    columns = oof_category_aggregates(records, outcomes, folds_for(records))

    assert all(math.isnan(v) for v in row(columns, 2))
    assert row(columns, 3) == (20.0, 0.0)
    assert row(columns, 4) == (20.0, 0.0)
    assert row(columns, 5) == pytest.approx((25.0, 1 / 2), rel=0, abs=1e-12)
    assert row(columns, 6) == pytest.approx((100 / 3, 1 / 3), rel=0, abs=1e-12)
    assert row(columns, 8) == pytest.approx((40.0, 1 / 4), rel=0, abs=1e-12)


# --- D31 E1: a category present only as open requests ---------------------------------------


def test_a_category_present_only_as_open_requests_gets_the_fit_block_global():
    """Row 2 (B, fold 0, fit rows 0-1): B's only fit-block record is open, so B
    is not seen; the fit block's global values apply — mean 10, and with p75 10
    the single eligible 10 is not a breach -> 0. E2 would have given NaN.
    Row 4 (B, fold 1, fit rows 0-2): still no eligible B -> the same global.
    Row 5 (B, fold 2, fit rows 0-4): B now has one eligible 40 -> B's own mean
    40; the fit block's p75 of {10, 20, 40} is 30, so 40 breaches -> 1.
    """
    records, outcomes = build(ALL_OPEN)
    columns = oof_category_aggregates(records, outcomes, folds_for(records))

    assert row(columns, 2) == (10.0, 0.0)
    assert row(columns, 4) == (10.0, 0.0)
    assert not any(math.isnan(v) for v in row(columns, 2) + row(columns, 4))
    assert row(columns, 5) == (40.0, 1.0)


def test_a_training_category_with_only_open_requests_gets_the_training_global():
    """Frozen path. B is never eligible in training, so it is absent from both
    category maps and validation/test B rows receive the training globals:
    mean 20, and with p75 25 the 30 breaches -> 1/2."""
    records, outcomes = build([(0, "A", 10.0), (1, "A", 30.0), (2, "B", None), (3, "B", None)])
    frozen = fit_category_aggregates(records, outcomes)

    assert "B" not in frozen.mean_resolution_hours_by_category
    assert "B" not in frozen.breach_rate_by_category
    held_out, _ = build([(10, "B", 1.0)])
    columns = apply_category_aggregates(frozen, held_out)
    assert row(columns, 0) == (20.0, 0.5)
    assert row(columns, 0) == (frozen.global_mean_resolution_hours, frozen.global_breach_rate)


# --- the frozen path: validation and test --------------------------------------------------


def test_frozen_aggregates_are_fitted_on_the_whole_training_period():
    records, outcomes = build(MULTI)
    frozen = fit_category_aggregates(records, outcomes)

    hours = [o.resolution_hours for o in outcomes]
    threshold = reference_p75(hours)  # every category has < 100 eligible -> global p75
    by = {}
    for r, h in zip(records, hours, strict=True):
        by.setdefault(r.label, []).append(h)

    assert frozen.global_mean_resolution_hours == pytest.approx(sum(hours) / len(hours), abs=1e-12)
    assert frozen.global_breach_rate == pytest.approx(
        sum(h > threshold for h in hours) / len(hours), abs=1e-12
    )
    for category, values in by.items():
        assert frozen.mean_resolution_hours_by_category[category] == pytest.approx(
            sum(values) / len(values), abs=1e-12
        )
        assert frozen.breach_rate_by_category[category] == pytest.approx(
            sum(h > threshold for h in values) / len(values), abs=1e-12
        )


def test_apply_uses_category_values_and_the_train_global_for_unseen_categories():
    train_records, train_outcomes = build(MULTI)
    frozen = fit_category_aggregates(train_records, train_outcomes)
    held_out, _ = build([(20, "A", 1.0), (21, "Z", 1.0), (22, "C", 1.0)])

    columns = apply_category_aggregates(frozen, held_out)

    assert row(columns, 0) == (
        frozen.mean_resolution_hours_by_category["A"],
        frozen.breach_rate_by_category["A"],
    )
    assert row(columns, 1) == (frozen.global_mean_resolution_hours, frozen.global_breach_rate)
    assert row(columns, 2) == (
        frozen.mean_resolution_hours_by_category["C"],
        frozen.breach_rate_by_category["C"],
    )


def test_frozen_aggregates_with_no_eligible_training_observation_are_nan():
    records, outcomes = build([(0, "A", None), (1, "B", None)])
    frozen = fit_category_aggregates(records, outcomes)
    columns = apply_category_aggregates(frozen, records)

    assert math.isnan(frozen.global_mean_resolution_hours)
    assert math.isnan(frozen.global_breach_rate)
    assert all(math.isnan(v) for i in range(2) for v in row(columns, i))


def split_fixture():
    spec = [
        (i // 3, "ABCDE"[i % 5], float((i * 37) % 211 + 1) if i % 11 else None) for i in range(300)
    ]
    records, outcomes = build(spec)
    split = temporal_split([r.submitted_at for r in records])
    by_period = {
        p: [i for i, r in enumerate(records) if split.period_of(r.submitted_at) is p]
        for p in Period
    }
    return records, outcomes, split, by_period


def pick(items, indices):
    return [items[i] for i in indices]


def test_validation_aggregates_are_identical_when_validation_outcomes_are_permuted():
    records, outcomes, _, by_period = split_fixture()
    train, val = by_period[Period.TRAIN], by_period[Period.VALIDATION]

    frozen = fit_category_aggregates(pick(records, train), pick(outcomes, train))
    before = apply_category_aggregates(frozen, pick(records, val))

    shuffled_hours = [outcomes[i].resolution_hours for i in val]
    random.Random(11).shuffle(shuffled_hours)
    permuted = list(outcomes)
    for i, hours in zip(val, shuffled_hours, strict=True):
        permuted[i] = outcome(outcomes[i].external_id, records[i].submitted_at, hours)

    frozen_again = fit_category_aggregates(pick(records, train), pick(permuted, train))
    after = apply_category_aggregates(frozen_again, pick(records, val))
    assert frozen_again == frozen
    assert after == before


def test_an_unseen_test_category_receives_the_train_global_mean():
    records, outcomes, _, by_period = split_fixture()
    train = by_period[Period.TRAIN]
    frozen = fit_category_aggregates(pick(records, train), pick(outcomes, train))

    unseen, _ = build([(10_000, "NEVER-IN-TRAIN", 5.0)])
    columns = apply_category_aggregates(frozen, unseen)
    assert row(columns, 0) == (frozen.global_mean_resolution_hours, frozen.global_breach_rate)


# --- Task 9 and Task 10 composition -------------------------------------------------------


def test_validation_and_test_targets_never_reach_training_or_frozen_aggregates():
    records, outcomes, split, by_period = split_fixture()
    train = by_period[Period.TRAIN]
    held_out = by_period[Period.VALIDATION] + by_period[Period.TEST]
    train_records = pick(records, train)
    folds = folds_for(train_records)

    oof_before = oof_category_aggregates(train_records, pick(outcomes, train), folds)
    frozen_before = fit_category_aggregates(train_records, pick(outcomes, train))

    contaminated = list(outcomes)
    for i in held_out:
        contaminated[i] = outcome(outcomes[i].external_id, records[i].submitted_at, 1e6)

    assert columns_same(
        oof_category_aggregates(train_records, pick(contaminated, train), folds), oof_before
    )
    assert fit_category_aggregates(train_records, pick(contaminated, train)) == frozen_before
    assert all(records[i].submitted_at <= split.train_end for i in train)


def test_the_full_pipeline_composes_split_folds_and_aggregates():
    records, outcomes, split, by_period = split_fixture()
    train = by_period[Period.TRAIN]
    train_records = pick(records, train)
    folds = folds_for(train_records)

    columns = oof_category_aggregates(train_records, pick(outcomes, train), folds)

    assert len(columns.category_mean_resolution_hours) == len(train) == split.counts[Period.TRAIN]
    for i in folds[0].fit_indices:
        assert all(math.isnan(v) for v in row(columns, i))
    for fold in folds:
        assert fold.apply_end <= split.train_end
    # Task 9 and Task 10 results are unchanged by building aggregates.
    assert temporal_split([r.submitted_at for r in records]) == split
    assert folds_for(train_records) == folds


# --- validation of inputs ---------------------------------------------------------------------


def test_mismatched_lengths_raise():
    records, outcomes = build(SIMPLE)
    folds = folds_for(records)
    with pytest.raises(ValueError):
        oof_category_aggregates(records, outcomes[:-1], folds)
    with pytest.raises(ValueError):
        fit_category_aggregates(records, outcomes[:-1])


def test_mismatched_external_ids_raise():
    records, outcomes = build(SIMPLE)
    swapped = list(outcomes)
    swapped[3], swapped[4] = swapped[4], swapped[3]
    with pytest.raises(ValueError):
        oof_category_aggregates(records, swapped, folds_for(records))
    with pytest.raises(ValueError):
        fit_category_aggregates(records, swapped)


def test_a_non_nyc311_outcome_raises():
    records, outcomes = build(SIMPLE)
    wrong = list(outcomes)
    wrong[5] = CFPBOutcome(external_id="r5", timely_response=True, sent_to_company_at=None)
    with pytest.raises(ValueError):
        oof_category_aggregates(records, wrong, folds_for(records))
    with pytest.raises(ValueError):
        fit_category_aggregates(records, wrong)


def replace_fold(folds, index, **changes):
    changed = list(folds)
    changed[index] = dataclasses.replace(folds[index], **changes)
    return changed


def test_folds_that_do_not_partition_the_rows_raise():
    records, outcomes = build(SIMPLE)
    folds = folds_for(records)

    missing = replace_fold(folds, 4, apply_indices=(8,))
    duplicated = replace_fold(folds, 4, apply_indices=(8, 9, 9))
    out_of_range = replace_fold(folds, 4, apply_indices=(8, 9, 10))
    for bad in (missing, duplicated, out_of_range, [], folds_for(records[:9])):
        with pytest.raises(ValueError):
            oof_category_aggregates(records, outcomes, bad)


def test_a_fold_whose_fit_block_reaches_into_its_apply_block_raises():
    records, outcomes = build(SIMPLE)
    folds = folds_for(records)
    leaky = replace_fold(folds, 2, fit_indices=folds[2].fit_indices + (5,))
    with pytest.raises(ValueError):
        oof_category_aggregates(records, outcomes, leaky)


def tampered_fold_collections():
    """Each keeps the warm-up and apply blocks intact — so the positions are still
    partitioned — while breaking the fit-block rule in a different way."""
    records, outcomes = build(SIMPLE)
    folds = folds_for(records)
    assert folds[3].fit_indices == (0, 1, 2, 3, 4, 5)
    return (
        records,
        outcomes,
        folds,
        {
            "own-apply-row": replace_fold(folds, 2, fit_indices=folds[2].fit_indices + (5,)),
            "missing-preceding-apply-row": replace_fold(folds, 3, fit_indices=(0, 1, 3, 4, 5)),
            "later-block-row": replace_fold(folds, 1, fit_indices=folds[1].fit_indices + (8,)),
            "missing-warm-up-row": replace_fold(folds, 2, fit_indices=(1, 2, 3, 4)),
        },
    )


def test_each_tampered_collection_still_partitions_the_positions():
    """Guards the guard: these are rejected by the fit-block rule, not by the
    partition check, which on its own would accept every one of them."""
    records, _, _, tampered = tampered_fold_collections()
    for name, folds in tampered.items():
        covered = list(folds[0].fit_indices) + [i for f in folds for i in f.apply_indices]
        assert sorted(covered) == list(range(len(records))), name


@pytest.mark.parametrize(
    "name",
    ["own-apply-row", "missing-preceding-apply-row", "later-block-row", "missing-warm-up-row"],
)
def test_a_fit_block_that_is_not_the_warm_up_plus_every_preceding_apply_block_raises(name):
    records, outcomes, _, tampered = tampered_fold_collections()
    with pytest.raises(ValueError):
        oof_category_aggregates(records, outcomes, tampered[name])


# --- determinism and input order ------------------------------------------------------------


def test_repeated_calls_are_identical():
    records, outcomes = build(MULTI)
    folds = folds_for(records)
    first = oof_category_aggregates(records, outcomes, folds)
    second = oof_category_aggregates(records, outcomes, folds)
    assert all(rows_same(row(first, i), row(second, i)) for i in range(len(records)))


def test_values_follow_rows_under_any_input_order_bit_for_bit():
    """Every row's two values follow that row bit-for-bit under any input order,
    including rows tied at one timestamp, whose relative order within the fit
    block depends on their input positions.

    This pins order-independence of the result. It does not by itself prove that
    ``math.fsum`` is used: on the project's pinned Python 3.13, the built-in
    ``sum`` over floats is also compensated and gives identical results for
    realistic non-negative hours, so the two cannot be told apart here. D31's
    ``math.fsum`` requirement is met by the implementation, not detected by this
    test."""
    spec = [(0, "A", 0.1), (0, "A", 0.2), (0, "A", 0.3), (0, "B", 0.7)]
    spec += [(1 + i, "AB"[i % 2], 0.1 * (i + 1)) for i in range(16)]
    records, outcomes = build(spec)
    reference = oof_category_aggregates(records, outcomes, folds_for(records))
    by_id = {r.external_id: row(reference, i) for i, r in enumerate(records)}

    order = list(range(len(records)))
    for seed in range(8):
        random.Random(seed).shuffle(order)
        permuted_records = [records[i] for i in order]
        permuted_outcomes = [outcomes[i] for i in order]
        columns = oof_category_aggregates(
            permuted_records, permuted_outcomes, folds_for(permuted_records)
        )
        for i, r in enumerate(permuted_records):
            assert rows_same(row(columns, i), by_id[r.external_id]), (seed, r.external_id)


def test_the_aggregate_module_uses_no_numpy_randomness_or_django():
    tree = ast.parse(Path("ml/training/aggregates.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = {"numpy", "pandas", "scipy", "sklearn", "django", "random", "complaints"}
    assert not {m.split(".")[0] for m in imported} & forbidden, imported
