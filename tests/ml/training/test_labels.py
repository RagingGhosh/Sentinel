"""Task 13: frozen per-type 311 SLA thresholds and the labels they produce.

Every threshold here is hand-computed from D31's formula. The eligibility cases
are chosen so that counting raw records rather than eligible observations, or
letting an open request or an undefined threshold through, each produces a
different answer from the documented one.
"""

import ast
import dataclasses
import inspect
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="numpy lives in requirements/ml.txt")

from ingest.schema import CFPBOutcome, CorpusRecord, NYC311Outcome  # noqa: E402
from ml.training.labels import (  # noqa: E402
    FrozenThresholds,
    apply_thresholds,
    breach_rate,
    fit_thresholds,
)
from ml.training.thresholds import (  # noqa: E402
    MIN_ELIGIBLE_OBSERVATIONS,
    THRESHOLD_QUANTILE,
    linear_percentile,
)

BASE = datetime(2024, 1, 1, tzinfo=UTC)


def pair(
    label: str, hours: float | None, external_id: str, offset: int = 0
) -> tuple[CorpusRecord, NYC311Outcome]:
    """One record and its 311 outcome, aligned by `external_id`."""
    submitted = BASE + timedelta(hours=offset)
    record = CorpusRecord(
        source="nyc311",
        external_id=external_id,
        text="text",
        label=label,
        submitted_at=submitted,
    )
    outcome = NYC311Outcome(
        external_id=external_id,
        closed_at=None if hours is None else submitted + timedelta(hours=hours),
        resolution_hours=hours,
    )
    return record, outcome


def dataset(
    spec: dict[str, list[float | None]],
) -> tuple[list[CorpusRecord], list[NYC311Outcome]]:
    """`{category: [resolution_hours, ...]}` into aligned record/outcome lists."""
    records, outcomes, index = [], [], 0
    for label, values in spec.items():
        for hours in values:
            record, outcome = pair(label, hours, f"e{index}", index)
            records.append(record)
            outcomes.append(outcome)
            index += 1
    return records, outcomes


# --- the public API ----------------------------------------------------------


def test_fit_thresholds_signature_is_the_documented_one():
    parameters = inspect.signature(fit_thresholds).parameters
    assert list(parameters) == ["train_records", "train_outcomes", "min_eligible"]
    assert parameters["min_eligible"].default == MIN_ELIGIBLE_OBSERVATIONS == 100


def test_apply_thresholds_signature_is_the_documented_one():
    assert list(inspect.signature(apply_thresholds).parameters) == [
        "frozen",
        "records",
        "outcomes",
    ]


def test_frozen_thresholds_carries_exactly_the_task_13_fields():
    fields = [f.name for f in dataclasses.fields(FrozenThresholds)]
    assert fields == ["per_type", "global_fallback", "fallback_type_count"]


def test_frozen_thresholds_is_immutable():
    records, outcomes = dataset({"Heat": [1.0, 2.0]})
    frozen = fit_thresholds(records, outcomes)
    with pytest.raises(dataclasses.FrozenInstanceError):
        frozen.global_fallback = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        frozen.per_type = {}  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        frozen.fallback_type_count = 0  # type: ignore[misc]


def test_the_per_type_mapping_is_read_only():
    records, outcomes = dataset({"Heat": [float(i) for i in range(100)]})
    frozen = fit_thresholds(records, outcomes)
    with pytest.raises(TypeError):
        frozen.per_type["Heat"] = 0.0  # type: ignore[index]


def test_apply_thresholds_returns_a_boolean_array_aligned_to_the_records():
    records, outcomes = dataset({"Heat": [float(i) for i in range(100)]})
    frozen = fit_thresholds(records, outcomes)
    labels = apply_thresholds(frozen, records, outcomes)
    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.bool_
    assert labels.shape == (100,)


# --- threshold values, hand-computed -----------------------------------------


def test_p75_of_one_to_one_hundred_is_the_interpolated_seventy_five_point_two_five():
    """n=100: h = 99 x 0.75 = 74.25, so x[74] + 0.25 x (x[75] - x[74]) = 75.25."""
    values = [float(i) for i in range(1, 101)]
    records, outcomes = dataset({"Heat": values})
    frozen = fit_thresholds(records, outcomes)
    assert frozen.per_type["Heat"] == 75.25


def test_the_threshold_matches_the_shared_primitive_exactly():
    values = [float(i) * 1.5 for i in range(1, 121)]
    records, outcomes = dataset({"Heat": values})
    frozen = fit_thresholds(records, outcomes)
    assert frozen.per_type["Heat"] == linear_percentile(values, THRESHOLD_QUANTILE)
    assert frozen.global_fallback == linear_percentile(values, THRESHOLD_QUANTILE)


def test_a_type_with_exactly_one_hundred_eligible_observations_gets_its_own_p75():
    records, outcomes = dataset({"Heat": [float(i) for i in range(1, 101)]})
    frozen = fit_thresholds(records, outcomes)
    assert "Heat" in frozen.per_type
    assert frozen.fallback_type_count == 0


def test_a_type_with_ninety_nine_eligible_observations_falls_back_and_is_counted():
    heat = [float(i) for i in range(1, 100)]
    noise = [float(i) for i in range(1, 201)]
    records, outcomes = dataset({"Heat": heat, "Noise": noise})
    frozen = fit_thresholds(records, outcomes)
    assert "Heat" not in frozen.per_type
    assert "Noise" in frozen.per_type
    assert frozen.fallback_type_count == 1
    assert frozen.threshold_for("Heat") == frozen.global_fallback


def test_multiple_types_each_get_their_own_p75():
    heat = [float(i) for i in range(1, 101)]
    noise = [float(i) * 10 for i in range(1, 101)]
    records, outcomes = dataset({"Heat": heat, "Noise": noise})
    frozen = fit_thresholds(records, outcomes)
    assert frozen.per_type["Heat"] == 75.25
    assert frozen.per_type["Noise"] == 752.5
    assert frozen.fallback_type_count == 0


def test_the_global_fallback_pools_every_eligible_observation():
    heat = [float(i) for i in range(1, 100)]
    noise = [float(i) for i in range(1, 100)]
    records, outcomes = dataset({"Heat": heat, "Noise": noise})
    frozen = fit_thresholds(records, outcomes)
    assert frozen.global_fallback == linear_percentile(heat + noise, THRESHOLD_QUANTILE)
    assert frozen.fallback_type_count == 2


# --- eligibility -------------------------------------------------------------


def test_open_requests_do_not_contribute_to_a_threshold():
    resolved = [float(i) for i in range(1, 101)]
    records, outcomes = dataset({"Heat": resolved + [None] * 50})
    frozen = fit_thresholds(records, outcomes)
    assert frozen.per_type["Heat"] == 75.25


def test_open_requests_do_not_count_toward_the_hundred():
    """101 records, 99 of them resolved: the type still falls back (D31)."""
    records, outcomes = dataset(
        {"Heat": [float(i) for i in range(1, 100)] + [None, None], "Noise": [1.0] * 150}
    )
    frozen = fit_thresholds(records, outcomes)
    assert "Heat" not in frozen.per_type
    assert frozen.fallback_type_count == 1


def test_a_type_seen_only_as_open_requests_uses_the_global_and_is_counted():
    """D33: fallback_type_count includes zero-eligible types, which the
    primitive's fallback_categories (1-99 eligible) deliberately omits."""
    records, outcomes = dataset({"Heat": [1.0] * 150, "Ghost": [None] * 20})
    frozen = fit_thresholds(records, outcomes)
    assert "Ghost" not in frozen.per_type
    assert frozen.threshold_for("Ghost") == frozen.global_fallback
    assert frozen.fallback_type_count == 1


def test_zero_eligible_observations_anywhere_leaves_the_global_undefined():
    records, outcomes = dataset({"Heat": [None] * 10, "Noise": [None] * 10})
    frozen = fit_thresholds(records, outcomes)
    assert math.isnan(frozen.global_fallback)
    assert dict(frozen.per_type) == {}
    assert frozen.fallback_type_count == 2


def test_no_records_at_all_fits_an_empty_undefined_set_of_thresholds():
    frozen = fit_thresholds([], [])
    assert math.isnan(frozen.global_fallback)
    assert dict(frozen.per_type) == {}
    assert frozen.fallback_type_count == 0


# --- breach classification ---------------------------------------------------


def test_slower_than_the_threshold_is_a_breach():
    records, outcomes = dataset({"Heat": [float(i) for i in range(1, 101)]})
    frozen = fit_thresholds(records, outcomes)
    probe_records, probe_outcomes = dataset({"Heat": [75.26, 1000.0]})
    assert list(apply_thresholds(frozen, probe_records, probe_outcomes)) == [True, True]


def test_exactly_the_threshold_is_not_a_breach():
    records, outcomes = dataset({"Heat": [float(i) for i in range(1, 101)]})
    frozen = fit_thresholds(records, outcomes)
    probe_records, probe_outcomes = dataset({"Heat": [75.25]})
    assert list(apply_thresholds(frozen, probe_records, probe_outcomes)) == [False]


def test_faster_than_the_threshold_is_not_a_breach():
    records, outcomes = dataset({"Heat": [float(i) for i in range(1, 101)]})
    frozen = fit_thresholds(records, outcomes)
    probe_records, probe_outcomes = dataset({"Heat": [0.5, 75.24]})
    assert list(apply_thresholds(frozen, probe_records, probe_outcomes)) == [False, False]


def test_each_type_is_judged_against_its_own_threshold():
    heat = [float(i) for i in range(1, 101)]
    noise = [float(i) * 10 for i in range(1, 101)]
    records, outcomes = dataset({"Heat": heat, "Noise": noise})
    frozen = fit_thresholds(records, outcomes)
    probe_records, probe_outcomes = dataset({"Heat": [100.0], "Noise": [100.0]})
    assert list(apply_thresholds(frozen, probe_records, probe_outcomes)) == [True, False]


# --- fallback at application time --------------------------------------------


def test_an_unseen_type_is_judged_against_the_training_global():
    records, outcomes = dataset({"Heat": [float(i) for i in range(1, 101)]})
    frozen = fit_thresholds(records, outcomes)
    probe_records, probe_outcomes = dataset({"Brand New": [frozen.global_fallback + 1.0]})
    assert list(apply_thresholds(frozen, probe_records, probe_outcomes)) == [True]


def test_a_fallback_type_is_judged_against_the_global_not_its_own_values():
    heat = [1000.0] * 99
    noise = [float(i) for i in range(1, 201)]
    records, outcomes = dataset({"Heat": heat, "Noise": noise})
    frozen = fit_thresholds(records, outcomes)
    probe_records, probe_outcomes = dataset({"Heat": [frozen.global_fallback + 0.1]})
    assert list(apply_thresholds(frozen, probe_records, probe_outcomes)) == [True]


# --- refusals: open requests and undefined thresholds ------------------------


def test_applying_to_an_unresolved_request_raises_rather_than_labelling_it_false():
    """D33: a bool has no 'no label' state; coercing to False would manufacture a
    'not breached' label for a request that was never resolved."""
    records, outcomes = dataset({"Heat": [float(i) for i in range(1, 101)]})
    frozen = fit_thresholds(records, outcomes)
    probe_records, probe_outcomes = dataset({"Heat": [1.0, None, 2.0]})
    with pytest.raises(ValueError, match="resolution_hours"):
        apply_thresholds(frozen, probe_records, probe_outcomes)


def test_the_refusal_names_the_offending_record():
    records, outcomes = dataset({"Heat": [float(i) for i in range(1, 101)]})
    frozen = fit_thresholds(records, outcomes)
    probe_records, probe_outcomes = dataset({"Heat": [1.0, None]})
    with pytest.raises(ValueError, match="e1"):
        apply_thresholds(frozen, probe_records, probe_outcomes)


def test_applying_an_undefined_threshold_raises_rather_than_labelling_everything_false():
    """Every comparison with NaN is false, so an undefined threshold would
    silently report a clean period."""
    records, outcomes = dataset({"Heat": [None] * 10})
    frozen = fit_thresholds(records, outcomes)
    assert math.isnan(frozen.global_fallback)
    probe_records, probe_outcomes = dataset({"Heat": [1.0, 1e6]})
    with pytest.raises(ValueError, match="undefined|NaN|nan"):
        apply_thresholds(frozen, probe_records, probe_outcomes)


def test_an_undefined_threshold_is_refused_even_for_a_single_record():
    frozen = fit_thresholds([], [])
    probe_records, probe_outcomes = dataset({"Heat": [5.0]})
    with pytest.raises(ValueError):
        apply_thresholds(frozen, probe_records, probe_outcomes)


# --- freeze semantics --------------------------------------------------------


def test_applying_does_not_mutate_the_frozen_object():
    records, outcomes = dataset({"Heat": [float(i) for i in range(1, 101)]})
    frozen = fit_thresholds(records, outcomes)
    before = (dict(frozen.per_type), frozen.global_fallback, frozen.fallback_type_count)
    later_records, later_outcomes = dataset({"Heat": [1e6] * 500, "Other": [1e6] * 500})
    apply_thresholds(frozen, later_records, later_outcomes)
    after = (dict(frozen.per_type), frozen.global_fallback, frozen.fallback_type_count)
    assert before == after


def test_fitting_twice_on_the_same_data_is_deterministic():
    records, outcomes = dataset({"Heat": [float(i) for i in range(1, 101)], "Noise": [2.0] * 50})
    first = fit_thresholds(records, outcomes)
    second = fit_thresholds(records, outcomes)
    assert dict(first.per_type) == dict(second.per_type)
    assert first.global_fallback == second.global_fallback
    assert first.fallback_type_count == second.fallback_type_count


def test_the_order_records_arrive_in_does_not_change_a_threshold():
    records, outcomes = dataset({"Heat": [float(i) for i in range(1, 101)]})
    forward = fit_thresholds(records, outcomes)
    backward = fit_thresholds(list(reversed(records)), list(reversed(outcomes)))
    assert forward.per_type["Heat"] == backward.per_type["Heat"]


# --- training-only: no validation or test outcome may move a threshold -------


def test_permuting_validation_outcomes_cannot_move_a_frozen_threshold():
    train_records, train_outcomes = dataset({"Heat": [float(i) for i in range(1, 101)]})
    frozen = fit_thresholds(train_records, train_outcomes)
    baseline = frozen.per_type["Heat"]

    # A validation period whose values would dominate the p75 if they reached it.
    val_records, val_outcomes = dataset({"Heat": [1e6] * 400})
    apply_thresholds(frozen, val_records, val_outcomes)
    assert frozen.per_type["Heat"] == baseline
    assert fit_thresholds(train_records, train_outcomes).per_type["Heat"] == baseline


def test_a_test_period_outcome_cannot_move_a_frozen_threshold():
    train_records, train_outcomes = dataset({"Heat": [float(i) for i in range(1, 101)]})
    frozen = fit_thresholds(train_records, train_outcomes)
    baseline = frozen.per_type["Heat"]
    test_records, test_outcomes = dataset({"Heat": [0.001] * 1000})
    apply_thresholds(frozen, test_records, test_outcomes)
    assert frozen.per_type["Heat"] == baseline


def test_only_the_rows_passed_to_fit_thresholds_can_influence_it():
    """The adversarial check: the same records, fitted with and without a later
    block whose values would materially move the p75."""
    train_records, train_outcomes = dataset({"Heat": [float(i) for i in range(1, 101)]})
    held_records, held_outcomes = dataset({"Heat": [1e6] * 100})
    train_only = fit_thresholds(train_records, train_outcomes)
    contaminated = fit_thresholds(train_records + held_records, train_outcomes + held_outcomes)
    assert train_only.per_type["Heat"] == 75.25
    assert contaminated.per_type["Heat"] != train_only.per_type["Heat"]


# --- the per-period breach-rate helper ---------------------------------------


def test_breach_rate_is_the_share_of_true_labels():
    labels = np.array([True, False, True, False], dtype=bool)
    assert breach_rate(labels) == 0.5


def test_breach_rate_of_no_breaches_is_zero():
    assert breach_rate(np.array([False, False], dtype=bool)) == 0.0


def test_breach_rate_of_all_breaches_is_one():
    assert breach_rate(np.array([True, True], dtype=bool)) == 1.0


def test_breach_rate_of_an_empty_period_is_undefined():
    assert math.isnan(breach_rate(np.array([], dtype=bool)))


def test_breach_rate_accepts_the_output_of_apply_thresholds():
    records, outcomes = dataset({"Heat": [float(i) for i in range(1, 101)]})
    frozen = fit_thresholds(records, outcomes)
    labels = apply_thresholds(frozen, records, outcomes)
    # x[74] = 75 with values 1..100, so 75.25 leaves 25 of them strictly above.
    assert breach_rate(labels) == 0.25


# --- input validation --------------------------------------------------------


def test_mismatched_lengths_are_refused_by_fit():
    records, outcomes = dataset({"Heat": [1.0, 2.0]})
    with pytest.raises(ValueError):
        fit_thresholds(records, outcomes[:1])


def test_mismatched_lengths_are_refused_by_apply():
    records, outcomes = dataset({"Heat": [float(i) for i in range(1, 101)]})
    frozen = fit_thresholds(records, outcomes)
    with pytest.raises(ValueError):
        apply_thresholds(frozen, records, outcomes[:5])


def test_mismatched_external_ids_are_refused():
    records, outcomes = dataset({"Heat": [1.0, 2.0]})
    swapped = [outcomes[1], outcomes[0]]
    with pytest.raises(ValueError, match="does not match"):
        fit_thresholds(records, swapped)


def test_a_non_311_outcome_is_refused():
    records, _ = dataset({"Heat": [1.0]})
    wrong = [CFPBOutcome(external_id="e0", timely_response=True, sent_to_company_at=None)]
    with pytest.raises(ValueError, match="NYC311Outcome"):
        fit_thresholds(records, wrong)


def test_mismatched_external_ids_are_refused_by_apply():
    """Pairing a record with another record's outcome would judge it against the
    wrong type's threshold, and the array would still come out the right length."""
    records, outcomes = dataset({"Heat": [float(i) for i in range(1, 101)]})
    frozen = fit_thresholds(records, outcomes)
    probe_records, probe_outcomes = dataset({"Heat": [1.0], "Noise": [500.0]})
    swapped = [probe_outcomes[1], probe_outcomes[0]]
    with pytest.raises(ValueError, match="does not match"):
        apply_thresholds(frozen, probe_records, swapped)


def test_a_non_311_outcome_is_refused_by_apply():
    records, outcomes = dataset({"Heat": [float(i) for i in range(1, 101)]})
    frozen = fit_thresholds(records, outcomes)
    probe_records, _ = dataset({"Heat": [1.0]})
    wrong = [CFPBOutcome(external_id="e0", timely_response=True, sent_to_company_at=None)]
    with pytest.raises(ValueError, match="NYC311Outcome"):
        apply_thresholds(frozen, probe_records, wrong)


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_a_non_positive_minimum_is_refused(bad):
    records, outcomes = dataset({"Heat": [1.0, 2.0]})
    with pytest.raises(ValueError, match="min_eligible"):
        fit_thresholds(records, outcomes, min_eligible=bad)


def test_a_minimum_of_one_lets_every_seen_type_keep_its_own_threshold():
    records, outcomes = dataset({"Heat": [1.0, 3.0], "Noise": [5.0]})
    frozen = fit_thresholds(records, outcomes, min_eligible=1)
    assert set(frozen.per_type) == {"Heat", "Noise"}
    assert frozen.fallback_type_count == 0


# --- the shared primitive is reused, not reproduced ---------------------------


def test_labels_module_imports_the_shared_primitive():
    source = Path("ml/training/labels.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "ml.training.thresholds" in imported


def test_labels_module_does_not_restate_the_percentile_arithmetic():
    """D33: no second implementation of the percentile, interpolation,
    minimum-count, global-fallback or breach comparison."""
    source = Path("ml/training/labels.py").read_text(encoding="utf-8")
    body = "\n".join(line for line in source.splitlines() if not line.strip().startswith("#"))
    for forbidden in ("math.floor", "math.ceil", "sorted(", "0.75", "THRESHOLD_QUANTILE *"):
        assert forbidden not in body, f"{forbidden!r} suggests duplicated threshold math"


def test_the_task_11_primitive_still_behaves_exactly_as_before():
    """Task 13 must not have altered the shared semantics."""
    assert linear_percentile([1.0, 2.0, 3.0, 4.0], 0.75) == 3.25
    assert linear_percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.75) == 4.0
    assert MIN_ELIGIBLE_OBSERVATIONS == 100
    assert THRESHOLD_QUANTILE == 0.75
