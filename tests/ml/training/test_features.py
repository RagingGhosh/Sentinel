"""Task 12: RiskFeaturesV1 assembly, feature order, and the FeatureSpec guard.

Every expected value here is hand-computed. The timezone cases are chosen so
that a UTC-derived hour, a fixed-offset shortcut, or a Sunday-first weekday
convention each produce a different number from the documented one.
"""

import inspect
import math
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

np = pytest.importorskip("numpy", reason="numpy lives in requirements/ml.txt")

from ingest.schema import CorpusRecord, NYC311Outcome  # noqa: E402
from ml.training.aggregates import (  # noqa: E402
    AggregateColumns,
    apply_category_aggregates,
    fit_category_aggregates,
    oof_category_aggregates,
)
from ml.training.features import (  # noqa: E402
    RISK_FEATURES_V1,
    TRANSFER_FEATURES_V1,
    FeatureSpec,
    FeatureUnavailable,
    build_features,
)
from ml.training.splits import Period, forward_chaining_folds, temporal_split  # noqa: E402

NEW_YORK = ZoneInfo("America/New_York")

RISK_NAMES = (
    "submitted_hour",
    "submitted_weekday",
    "text_length",
    "category_mean_resolution_hours",
    "category_breach_rate",
)
TRANSFER_NAMES = ("submitted_hour", "submitted_weekday", "text_length")


def record(
    submitted_at: datetime,
    *,
    source: str = "nyc311",
    external_id: str = "1",
    text: str = "text",
    label: str = "Noise",
) -> CorpusRecord:
    return CorpusRecord(
        source=source,
        external_id=external_id,
        text=text,
        label=label,
        submitted_at=submitted_at,
    )


def local_311(year: int, month: int, day: int, hour: int = 12, minute: int = 0) -> datetime:
    """A New York wall clock, stored the way ingest stores it: as the UTC instant."""
    return datetime(year, month, day, hour, minute, tzinfo=NEW_YORK).astimezone(UTC)


def columns(means: tuple[float, ...], rates: tuple[float, ...]) -> AggregateColumns:
    return AggregateColumns(category_mean_resolution_hours=means, category_breach_rate=rates)


def train_and_held_out(
    records: list[CorpusRecord],
) -> tuple[list[int], list[int]]:
    """Positions of the train period and of everything after it."""
    split = temporal_split([r.submitted_at for r in records])
    periods = [split.period_of(r.submitted_at) for r in records]
    train = [i for i, period in enumerate(periods) if period is Period.TRAIN]
    held_out = [i for i, period in enumerate(periods) if period is not Period.TRAIN]
    return train, held_out


def dataset(count: int, start: datetime) -> tuple[list[CorpusRecord], list[NYC311Outcome]]:
    """A 311 dataset with distinct timestamps and two categories."""
    records, outcomes = [], []
    for index in range(count):
        external_id = f"r{index}"
        records.append(
            record(
                start + timedelta(hours=index),
                external_id=external_id,
                text="x" * (index % 7),
                label="Noise" if index % 2 else "Heat",
            )
        )
        outcomes.append(
            NYC311Outcome(
                external_id=external_id,
                closed_at=start + timedelta(hours=index + 1),
                resolution_hours=float(index % 13 + 1),
            )
        )
    return records, outcomes


# --- the two specs -----------------------------------------------------------


def test_risk_features_v1_has_exactly_five_names_in_the_documented_order():
    assert RISK_FEATURES_V1.names == RISK_NAMES
    assert len(RISK_FEATURES_V1.names) == 5


def test_transfer_features_v1_has_exactly_three_names_in_the_documented_order():
    assert TRANSFER_FEATURES_V1.names == TRANSFER_NAMES
    assert len(TRANSFER_FEATURES_V1.names) == 3


def test_transfer_features_v1_is_a_strict_subset_of_risk_features_v1():
    assert set(TRANSFER_FEATURES_V1.names) < set(RISK_FEATURES_V1.names)


def test_the_two_specs_carry_distinct_version_strings():
    assert RISK_FEATURES_V1.version == "risk_features_v1"
    assert TRANSFER_FEATURES_V1.version == "transfer_features_v1"
    assert RISK_FEATURES_V1.version != TRANSFER_FEATURES_V1.version


def test_excluded_features_are_absent_from_both_specs():
    """D15 and §3: the removals are enforced, not merely documented."""
    for excluded in (
        "sla_hours",
        "priority_rank",
        "age_hours",
        "queue_depth",
        "assignee_open_count",
        "sent_to_company_at",
    ):
        assert excluded not in RISK_FEATURES_V1.names
        assert excluded not in TRANSFER_FEATURES_V1.names


def test_a_feature_spec_is_immutable():
    with pytest.raises(AttributeError):
        RISK_FEATURES_V1.names = TRANSFER_NAMES  # type: ignore[misc]
    with pytest.raises(AttributeError):
        RISK_FEATURES_V1.version = "other"  # type: ignore[misc]


def test_a_feature_spec_carries_exactly_names_and_version():
    spec = FeatureSpec(names=("text_length",), version="v")
    assert spec.names == ("text_length",)
    assert spec.version == "v"


# --- shape, dtype and alignment ----------------------------------------------


def test_the_matrix_is_one_row_per_record_and_one_column_per_spec_name():
    records = [record(local_311(2024, 6, 1), external_id=str(i)) for i in range(4)]
    matrix = build_features(records, columns((1.0,) * 4, (0.5,) * 4), RISK_FEATURES_V1)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (4, 5)


def test_the_matrix_is_floating_point_so_it_can_carry_nan():
    records = [record(local_311(2024, 6, 1))]
    matrix = build_features(records, columns((math.nan,), (math.nan,)), RISK_FEATURES_V1)
    assert matrix.dtype == np.float64


def test_no_records_yields_an_empty_matrix_of_the_right_width():
    matrix = build_features([], columns((), ()), RISK_FEATURES_V1)
    assert matrix.shape == (0, 5)


def test_rows_align_positionally_to_the_records():
    records = [
        record(local_311(2024, 6, 1, 9, 30), external_id="a", text="x"),
        record(local_311(2024, 6, 1, 20, 0), external_id="b", text="yy"),
    ]
    matrix = build_features(records, columns((1.0, 2.0), (0.1, 0.2)), RISK_FEATURES_V1)
    assert matrix[0, 2] == 1.0
    assert matrix[1, 2] == 2.0
    assert matrix[0, 3] == 1.0
    assert matrix[1, 3] == 2.0


# --- submitted_hour and submitted_weekday ------------------------------------


def test_311_hour_and_weekday_come_from_new_york_civil_time_not_utc():
    """2024-06-01 09:30 New York is 13:30 UTC. The feature is 9, never 13."""
    records = [record(local_311(2024, 6, 1, 9, 30))]
    matrix = build_features(records, None, TRANSFER_FEATURES_V1)
    assert matrix[0, 0] == 9.0
    assert matrix[0, 1] == 5.0


def test_311_local_time_can_put_a_record_on_a_different_weekday_than_utc():
    """20:00 Saturday New York is 00:00 Sunday UTC: hour 20 not 0, weekday 5 not 6."""
    records = [record(local_311(2024, 6, 1, 20, 0))]
    matrix = build_features(records, None, TRANSFER_FEATURES_V1)
    assert matrix[0, 0] == 20.0
    assert matrix[0, 1] == 5.0


def test_311_uses_the_offset_in_force_on_the_day_not_a_fixed_one():
    """23:30 on 2024-01-15 is EST (UTC-5), so 04:30 UTC the next day, a Tuesday."""
    records = [record(local_311(2024, 1, 15, 23, 30))]
    matrix = build_features(records, None, TRANSFER_FEATURES_V1)
    assert matrix[0, 0] == 23.0
    assert matrix[0, 1] == 0.0


def test_cfpb_hour_and_weekday_come_from_the_stored_instant():
    """D32: CFPB's published offset is discarded at normalization, so the stored
    UTC instant is the only representation that exists."""
    instant = datetime(2024, 6, 1, 13, 30, tzinfo=UTC)
    records = [record(instant, source="cfpb")]
    matrix = build_features(records, None, TRANSFER_FEATURES_V1)
    assert matrix[0, 0] == 13.0
    assert matrix[0, 1] == 5.0


def test_the_same_instant_gives_different_hours_for_the_two_sources():
    instant = datetime(2024, 6, 1, 13, 30, tzinfo=UTC)
    records = [record(instant, source="nyc311"), record(instant, source="cfpb")]
    matrix = build_features(records, None, TRANSFER_FEATURES_V1)
    assert matrix[0, 0] == 9.0
    assert matrix[1, 0] == 13.0


def test_weekday_is_monday_zero_through_sunday_six():
    """Seven consecutive 311 days, each at noon local. 2024-06-03 is a Monday."""
    records = [
        record(local_311(2024, 6, 3 + offset), external_id=str(offset)) for offset in range(7)
    ]
    matrix = build_features(records, None, TRANSFER_FEATURES_V1)
    assert list(matrix[:, 1]) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_hour_stays_inside_zero_to_twenty_three_across_a_whole_day():
    records = [record(local_311(2024, 6, 1, h), external_id=str(h)) for h in range(24)]
    matrix = build_features(records, None, TRANSFER_FEATURES_V1)
    assert list(matrix[:, 0]) == [float(h) for h in range(24)]


def test_a_naive_submitted_at_is_refused_rather_than_read_in_the_machine_timezone():
    records = [record(datetime(2024, 6, 1, 9, 30))]
    with pytest.raises(ValueError, match="submitted_at"):
        build_features(records, None, TRANSFER_FEATURES_V1)


# --- text_length -------------------------------------------------------------


def test_text_length_counts_characters_of_the_record_text():
    records = [record(local_311(2024, 6, 1), text="hello")]
    matrix = build_features(records, None, TRANSFER_FEATURES_V1)
    assert matrix[0, 2] == 5.0


def test_text_length_counts_characters_not_utf8_bytes():
    """'café' is four characters and five UTF-8 bytes."""
    records = [record(local_311(2024, 6, 1), text="café")]
    matrix = build_features(records, None, TRANSFER_FEATURES_V1)
    assert matrix[0, 2] == 4.0


def test_text_length_counts_whitespace_and_does_not_normalize():
    records = [record(local_311(2024, 6, 1), text="  a  b \n")]
    matrix = build_features(records, None, TRANSFER_FEATURES_V1)
    assert matrix[0, 2] == 8.0


def test_empty_text_is_length_zero_not_nan():
    records = [record(local_311(2024, 6, 1), text="")]
    matrix = build_features(records, None, TRANSFER_FEATURES_V1)
    assert matrix[0, 2] == 0.0


# --- the two aggregate columns -----------------------------------------------


def test_aggregate_columns_are_copied_through_verbatim():
    """Task 12 reads Task 11's output. It never recomputes a statistic."""
    records = [record(local_311(2024, 6, 1), external_id=str(i)) for i in range(3)]
    absurd = columns((-7.5, 1e9, 0.0), (2.0, -1.0, 0.25))
    matrix = build_features(records, absurd, RISK_FEATURES_V1)
    assert list(matrix[:, 3]) == [-7.5, 1e9, 0.0]
    assert list(matrix[:, 4]) == [2.0, -1.0, 0.25]


def test_aggregates_shorter_than_the_records_is_an_error():
    records = [record(local_311(2024, 6, 1), external_id=str(i)) for i in range(3)]
    with pytest.raises(ValueError):
        build_features(records, columns((1.0, 2.0), (0.1, 0.2)), RISK_FEATURES_V1)


def test_mismatched_aggregate_columns_are_an_error():
    records = [record(local_311(2024, 6, 1))]
    with pytest.raises(ValueError):
        build_features(records, columns((1.0,), (0.1, 0.2)), RISK_FEATURES_V1)


def test_a_single_aggregate_value_is_not_broadcast_across_every_record():
    """A length-1 column would broadcast silently if the alignment were left to
    the array assignment. One value for many rows is a misalignment, not a fill."""
    records = [record(local_311(2024, 6, 1), external_id=str(i)) for i in range(3)]
    with pytest.raises(ValueError, match="category_mean_resolution_hours"):
        build_features(records, columns((1.0,), (0.1,)), RISK_FEATURES_V1)


def test_a_misaligned_aggregate_column_is_named_with_both_counts():
    """The check is the module's own, made before any column is built — not a
    broadcast error surfacing from the array assignment afterwards."""
    records = [record(local_311(2024, 6, 1), external_id=str(i)) for i in range(3)]
    with pytest.raises(ValueError, match=r"category_breach_rate holds 2 values for 3 records"):
        build_features(records, columns((1.0, 2.0, 3.0), (0.1, 0.2)), RISK_FEATURES_V1)


# --- NaN is preserved, never imputed -----------------------------------------


def test_warm_up_nan_survives_assembly():
    records = [record(local_311(2024, 6, 1), external_id=str(i)) for i in range(3)]
    warm_up = columns((math.nan, 4.0, 6.0), (math.nan, 0.5, 0.25))
    matrix = build_features(records, warm_up, RISK_FEATURES_V1)
    assert np.isnan(matrix[0, 3])
    assert np.isnan(matrix[0, 4])
    assert matrix[1, 3] == 4.0
    assert matrix[2, 4] == 0.25


def test_nan_is_never_replaced_by_zero():
    records = [record(local_311(2024, 6, 1))]
    matrix = build_features(records, columns((math.nan,), (math.nan,)), RISK_FEATURES_V1)
    assert not matrix[0, 3] == 0.0
    assert not matrix[0, 4] == 0.0


def test_an_all_nan_aggregate_column_stays_all_nan():
    """No eligible observation anywhere: every aggregate is undefined."""
    records = [record(local_311(2024, 6, 1), external_id=str(i)) for i in range(4)]
    matrix = build_features(records, columns((math.nan,) * 4, (math.nan,) * 4), RISK_FEATURES_V1)
    assert np.isnan(matrix[:, 3]).all()
    assert np.isnan(matrix[:, 4]).all()
    assert not np.isnan(matrix[:, :3]).any()


def test_the_number_of_nan_cells_is_exactly_what_went_in():
    records = [record(local_311(2024, 6, 1), external_id=str(i)) for i in range(5)]
    means = (math.nan, math.nan, 1.0, 2.0, 3.0)
    rates = (math.nan, 0.5, 0.5, 0.5, 0.5)
    matrix = build_features(records, columns(means, rates), RISK_FEATURES_V1)
    assert int(np.isnan(matrix).sum()) == 3


# --- feature order is the contract -------------------------------------------


def test_columns_follow_spec_names_exactly():
    records = [record(local_311(2024, 6, 1, 9, 30), text="hello")]
    matrix = build_features(records, columns((4.0,), (0.25,)), RISK_FEATURES_V1)
    assert list(matrix[0]) == [9.0, 5.0, 5.0, 4.0, 0.25]


def test_permuting_spec_names_permutes_the_columns_correspondingly():
    records = [record(local_311(2024, 6, 1, 9, 30), text="hello")]
    reversed_spec = FeatureSpec(names=tuple(reversed(RISK_NAMES)), version="reversed")
    matrix = build_features(records, columns((4.0,), (0.25,)), reversed_spec)
    assert list(matrix[0]) == [0.25, 4.0, 5.0, 5.0, 9.0]


def test_every_permutation_of_the_spec_reorders_the_same_values():
    records = [record(local_311(2024, 6, 1, 9, 30), text="hello")]
    aggregates = columns((4.0,), (0.25,))
    baseline = dict(zip(RISK_NAMES, build_features(records, aggregates, RISK_FEATURES_V1)[0]))
    for rotation in range(1, 5):
        names = RISK_NAMES[rotation:] + RISK_NAMES[:rotation]
        spec = FeatureSpec(names=names, version=f"rotated_{rotation}")
        row = build_features(records, aggregates, spec)[0]
        assert list(row) == [baseline[name] for name in names]


def test_a_narrower_spec_produces_a_narrower_matrix_in_its_own_order():
    records = [record(local_311(2024, 6, 1, 9, 30), text="hello")]
    spec = FeatureSpec(names=("text_length", "submitted_hour"), version="narrow")
    matrix = build_features(records, None, spec)
    assert matrix.shape == (1, 2)
    assert list(matrix[0]) == [5.0, 9.0]


def test_reordering_the_records_does_not_move_a_feature_between_columns():
    first = record(local_311(2024, 6, 1, 9, 30), external_id="a", text="x")
    second = record(local_311(2024, 6, 2, 20, 0), external_id="b", text="yyy")
    forward = build_features([first, second], columns((1.0, 2.0), (0.1, 0.2)), RISK_FEATURES_V1)
    backward = build_features([second, first], columns((2.0, 1.0), (0.2, 0.1)), RISK_FEATURES_V1)
    assert list(forward[0]) == list(backward[1])
    assert list(forward[1]) == list(backward[0])


# --- fail loudly on an unproducible feature ----------------------------------


def test_a_spec_naming_sla_hours_raises_feature_unavailable_naming_it():
    """D15's removal is enforced by code, not documented in prose."""
    records = [record(local_311(2024, 6, 1))]
    spec = FeatureSpec(names=("submitted_hour", "sla_hours"), version="bad")
    with pytest.raises(FeatureUnavailable, match="sla_hours"):
        build_features(records, columns((1.0,), (0.1,)), spec)


@pytest.mark.parametrize(
    "excluded", ["queue_depth", "priority_rank", "age_hours", "assignee_open_count"]
)
def test_a_spec_naming_an_excluded_serving_feature_raises_feature_unavailable(excluded):
    records = [record(local_311(2024, 6, 1))]
    spec = FeatureSpec(names=(excluded,), version="bad")
    with pytest.raises(FeatureUnavailable, match=excluded):
        build_features(records, columns((1.0,), (0.1,)), spec)


def test_an_unknown_feature_name_raises_feature_unavailable_naming_it():
    records = [record(local_311(2024, 6, 1))]
    spec = FeatureSpec(names=("submitted_hour", "moon_phase"), version="bad")
    with pytest.raises(FeatureUnavailable, match="moon_phase"):
        build_features(records, None, spec)


def test_an_aggregate_feature_without_aggregates_raises_feature_unavailable():
    """§5.4: the primary five-feature model cannot be scored on a corpus with no
    resolution times. That impossibility is an error, not a fallback."""
    records = [record(local_311(2024, 6, 1), source="cfpb")]
    with pytest.raises(FeatureUnavailable, match="category_mean_resolution_hours"):
        build_features(records, None, RISK_FEATURES_V1)


def test_the_breach_rate_alone_without_aggregates_also_raises():
    records = [record(local_311(2024, 6, 1))]
    spec = FeatureSpec(names=("category_breach_rate",), version="bad")
    with pytest.raises(FeatureUnavailable, match="category_breach_rate"):
        build_features(records, None, spec)


def test_the_transfer_spec_needs_no_aggregates_at_all():
    records = [record(local_311(2024, 6, 1), source="cfpb", text="hello")]
    matrix = build_features(records, None, TRANSFER_FEATURES_V1)
    assert matrix.shape == (1, 3)


def test_an_unavailable_feature_produces_no_partial_output_and_no_reordering():
    records = [record(local_311(2024, 6, 1))]
    spec = FeatureSpec(names=("submitted_hour", "queue_depth", "text_length"), version="bad")
    with pytest.raises(FeatureUnavailable):
        build_features(records, None, spec)


# --- determinism -------------------------------------------------------------


def test_repeated_calls_return_identical_matrices():
    records = [record(local_311(2024, 6, 1, h), external_id=str(h), text="x" * h) for h in range(6)]
    aggregates = columns(tuple(float(i) for i in range(6)), tuple(i / 10 for i in range(6)))
    first = build_features(records, aggregates, RISK_FEATURES_V1)
    second = build_features(records, aggregates, RISK_FEATURES_V1)
    assert np.array_equal(first, second)


def test_the_returned_matrix_does_not_alias_the_aggregate_tuples():
    records = [record(local_311(2024, 6, 1))]
    aggregates = columns((1.0,), (0.5,))
    matrix = build_features(records, aggregates, RISK_FEATURES_V1)
    matrix[0, 3] = 99.0
    assert aggregates.category_mean_resolution_hours == (1.0,)


# --- the train / validation / test construction paths ------------------------


def test_the_training_path_composes_split_folds_aggregates_and_assembly():
    records, outcomes = dataset(300, local_311(2024, 1, 1))
    train, _ = train_and_held_out(records)
    train_records = [records[i] for i in train]
    train_outcomes = [outcomes[i] for i in train]

    folds = forward_chaining_folds([r.submitted_at for r in train_records])
    aggregates = oof_category_aggregates(train_records, train_outcomes, folds)
    matrix = build_features(train_records, aggregates, RISK_FEATURES_V1)

    assert matrix.shape == (len(train_records), 5)
    for column, values in (
        (3, aggregates.category_mean_resolution_hours),
        (4, aggregates.category_breach_rate),
    ):
        for row, expected in enumerate(values):
            if math.isnan(expected):
                assert np.isnan(matrix[row, column])
            else:
                assert matrix[row, column] == expected

    warm_up = list(folds[0].fit_indices)
    assert np.isnan(matrix[warm_up, 3]).all()
    assert np.isnan(matrix[warm_up, 4]).all()
    assert not np.isnan(matrix[warm_up, :3]).any()


def test_validation_and_test_features_come_from_the_frozen_training_aggregates():
    records, outcomes = dataset(300, local_311(2024, 1, 1))
    train, held_out = train_and_held_out(records)

    frozen = fit_category_aggregates([records[i] for i in train], [outcomes[i] for i in train])
    held_records = [records[i] for i in held_out]
    aggregates = apply_category_aggregates(frozen, held_records)
    matrix = build_features(held_records, aggregates, RISK_FEATURES_V1)

    for row, held_record in enumerate(held_records):
        assert matrix[row, 3] == frozen.mean_resolution_hours_by_category[held_record.label]
        assert matrix[row, 4] == frozen.breach_rate_by_category[held_record.label]


def test_permuting_held_out_outcomes_changes_no_held_out_feature():
    records, outcomes = dataset(300, local_311(2024, 1, 1))
    train, held_out = train_and_held_out(records)

    frozen = fit_category_aggregates([records[i] for i in train], [outcomes[i] for i in train])
    held_records = [records[i] for i in held_out]
    before = build_features(
        held_records, apply_category_aggregates(frozen, held_records), RISK_FEATURES_V1
    )

    shuffled = list(reversed([outcomes[i] for i in held_out]))
    assert shuffled != [outcomes[i] for i in held_out]

    after = build_features(
        held_records, apply_category_aggregates(frozen, held_records), RISK_FEATURES_V1
    )
    assert np.array_equal(before, after)


def test_training_outcomes_determine_the_frozen_values():
    records, outcomes = dataset(300, local_311(2024, 1, 1))
    train, held_out = train_and_held_out(records)
    held_records = [records[i] for i in held_out]

    original = fit_category_aggregates([records[i] for i in train], [outcomes[i] for i in train])
    changed_outcomes = [
        NYC311Outcome(
            external_id=outcomes[i].external_id,
            closed_at=outcomes[i].closed_at,
            resolution_hours=(outcomes[i].resolution_hours or 0.0) + 100.0,
        )
        for i in train
    ]
    changed = fit_category_aggregates([records[i] for i in train], changed_outcomes)

    before = build_features(
        held_records, apply_category_aggregates(original, held_records), RISK_FEATURES_V1
    )
    after = build_features(
        held_records, apply_category_aggregates(changed, held_records), RISK_FEATURES_V1
    )
    assert not np.array_equal(before[:, 3], after[:, 3])
    assert np.array_equal(before[:, :3], after[:, :3])


def test_build_features_accepts_no_outcomes_at_all():
    """The signature is the guarantee: a target cannot reach feature assembly."""
    assert list(inspect.signature(build_features).parameters) == ["records", "aggregates", "spec"]
