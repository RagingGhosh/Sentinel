"""Task 17: the NYC 311 SLA risk experiment (plan §L, plan Task 17, D37).

RED phase. ``ml/training/experiments/risk.py`` does not exist, and neither does
the outcome sidecar D37.1 requires, so every test that reaches either fails. The
production module is imported late, inside `experiment()`, so the fixture and the
corpus-shape tests still run and prove the fixture itself is sound.

Every corpus is generated into pytest's ``tmp_path`` and nothing is committed.
Nothing is downloaded and no socket is opened; one test proves the latter.
"""

import ast
import importlib
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="numpy lives in requirements/ml.txt")
pytest.importorskip("joblib", reason="joblib lives in requirements/ml.txt")
pytest.importorskip("pyarrow", reason="pyarrow lives in requirements/train.txt")
sklearn = pytest.importorskip("sklearn", reason="scikit-learn lives in requirements/ml.txt")

from ingest.manifest import (  # noqa: E402
    CorpusManifest,
    build_manifest,
    load_corpus,
    write_manifest,
)
from ingest.schema import CorpusRecord, NYC311Outcome  # noqa: E402
from ingest.storage import write_partition  # noqa: E402
from ml.training.artifacts import load_artifact  # noqa: E402
from ml.training.features import RISK_FEATURES_V1, FeatureUnavailable, build_features  # noqa: E402
from ml.training.metrics import ConfusionMatrix, MinorityReport, ScoreResult  # noqa: E402
from ml.training.splits import Period, forward_chaining_folds, temporal_split  # noqa: E402
from ml.training.thresholds import MIN_ELIGIBLE_OBSERVATIONS  # noqa: E402

pytestmark = pytest.mark.ml

SEED = 17


def experiment():
    """The Task 17 experiment module, imported late so the fixture tests still run."""
    return importlib.import_module("ml.training.experiments.risk")


def storage():
    from ingest import storage as module

    return module


# --- the fixture corpus ----------------------------------------------------------------
#
# Four complaint types, chosen so every branch of D31/D33 is exercised:
#
#   Noise     — far above `min_eligible` in training, so it earns its own p75
#   Heat      — resolved but below `min_eligible`, so it takes the global fallback
#   Ghost     — present only through open requests: zero eligible, still fallback
#   Newcomer  — absent from training entirely, so val/test meet an unseen category
#
# Open requests appear in every period. A three-way timestamp tie straddles the
# train boundary. The warm-up falls out of `forward_chaining_folds`' 20% default.

NOISE = "Noise"
HEAT = "Heat or Hot Water"
GHOST = "Ghost Requests"
NEWCOMER = "Newcomer Type"

CORPUS_START = datetime(2024, 3, 1, 9, 0, tzinfo=UTC)
WINDOW_START = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
WINDOW_END = datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC)

#: (label, resolution_hours) per period. ``None`` is an open request.
#: Noise carries 130 eligible training observations, comfortably over the
#: hundred `MIN_ELIGIBLE_OBSERVATIONS` wants, with a wide spread so its p75 is a
#: genuine interpolation rather than an order statistic.
TRAIN_PLAN = (
    [(NOISE, float(3 + (index * 7) % 240)) for index in range(130)]
    + [(NOISE, None) for _ in range(10)]
    + [(HEAT, float(10 + (index * 5) % 90)) for index in range(40)]
    + [(HEAT, None) for _ in range(6)]
    + [(GHOST, None) for _ in range(18)]
)
VALIDATION_PLAN = (
    [(NOISE, float(4 + (index * 11) % 200)) for index in range(18)]
    + [(NOISE, None) for _ in range(3)]
    + [(HEAT, float(12 + (index * 9) % 80)) for index in range(8)]
    + [(NEWCOMER, float(6 + (index * 13) % 150)) for index in range(12)]
    + [(NEWCOMER, None) for _ in range(2)]
)
TEST_PLAN = (
    [(NOISE, float(5 + (index * 13) % 220)) for index in range(18)]
    + [(NOISE, None) for _ in range(3)]
    + [(HEAT, float(15 + (index * 7) % 70)) for index in range(8)]
    + [(NEWCOMER, float(8 + (index * 17) % 160)) for index in range(12)]
    + [(GHOST, None) for _ in range(2)]
)

FULL_PLAN = TRAIN_PLAN + VALIDATION_PLAN + TEST_PLAN

#: Three records share one timestamp, placed so the tie straddles the 70% cut.
TIED_INDICES = (len(TRAIN_PLAN) - 2, len(TRAIN_PLAN) - 1, len(TRAIN_PLAN))
TIED_HOUR = len(TRAIN_PLAN) - 2

TIMESTAMP_DIAGNOSTIC = {
    "verdict": "suspicious_insufficient_evidence",
    "not_directly_testable": True,
    "reason": "fixture corpus; no provenance measurement is claimed",
}


def fixture_timestamp(index: int) -> datetime:
    hours = TIED_HOUR if index in TIED_INDICES else index
    return CORPUS_START + timedelta(hours=hours)


def fixture_rows() -> list[tuple[CorpusRecord, NYC311Outcome]]:
    rows = []
    for index, (label, hours) in enumerate(FULL_PLAN):
        external_id = f"{index:05d}"
        rows.append(
            (
                CorpusRecord(
                    source="nyc311",
                    external_id=external_id,
                    text=f"{label.lower()} report {index} on block {index % 17}",
                    label=label,
                    submitted_at=fixture_timestamp(index),
                ),
                NYC311Outcome(
                    external_id=external_id,
                    # D37: the actual normalised close instant for a resolved
                    # request, and nothing at all for an open one.
                    closed_at=(
                        None if hours is None else fixture_timestamp(index) + timedelta(hours=hours)
                    ),
                    resolution_hours=hours,
                ),
            )
        )
    return rows


@dataclass(frozen=True)
class Fixture:
    root: Path
    manifest: CorpusManifest
    rows: tuple[tuple[CorpusRecord, NYC311Outcome], ...]

    @property
    def records(self) -> list[CorpusRecord]:
        return [record for record, _ in self.rows]

    @property
    def outcomes(self) -> list[NYC311Outcome]:
        return [outcome for _, outcome in self.rows]

    def split(self):
        return temporal_split([record.submitted_at for record in self.records])

    def of(self, period: Period):
        split = self.split()
        return [row for row in self.rows if split.period_of(row[0].submitted_at) is period]


def build_fixture_corpus(root: Path, rows=None) -> CorpusManifest:
    """A real 311 corpus with its outcome sidecar, written through the real path."""
    rows = fixture_rows() if rows is None else rows
    by_year: dict[int, list[tuple[CorpusRecord, NYC311Outcome]]] = {}
    for record, outcome in rows:
        by_year.setdefault(record.submitted_at.year, []).append((record, outcome))
    for year, group in sorted(by_year.items()):
        # Two parts per year, so the merge path and multi-part checksums are real.
        half = max(len(group) // 2, 1)
        for index, block in enumerate((group[:half], group[half:])):
            if not block:
                continue
            write_partition([r for r, _ in block], "nyc311", year, index, root=root)
            storage().write_outcome_partition(
                [o for _, o in block], "nyc311", year, index, root=root
            )
    manifest = build_manifest(
        source="nyc311",
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        source_api_version="fixture-v1",
        limit=None,
        timestamp_diagnostic=TIMESTAMP_DIAGNOSTIC,
        root=root,
        ingested_at=datetime(2026, 9, 20, 12, 0, tzinfo=UTC),
    )
    write_manifest(manifest, root=root)
    return manifest


def _rotated_hours(rows, *, start: int):
    """Permute the resolved hours from ``start`` onward, leaving earlier rows alone.

    A rotation changes which records breach without emptying either class, so the
    period stays scoreable: `pr_auc` and `roc_auc` raise on a wholly one-class
    population by design (D34), and flattening every outcome to one value would
    test that refusal rather than the invariant these callers are after.
    """
    moved = list(rows)
    resolved = [index for index in range(start, len(moved)) if moved[index][1].resolution_hours]
    values = [moved[index][1].resolution_hours for index in resolved]
    values = values[len(values) // 2 :] + values[: len(values) // 2]
    for index, hours in zip(resolved, values, strict=True):
        record, outcome = moved[index]
        moved[index] = (record, NYC311Outcome(outcome.external_id, outcome.closed_at, hours))
    return moved


@pytest.fixture
def corpus(tmp_path) -> Fixture:
    root = tmp_path / "corpus"
    manifest = build_fixture_corpus(root)
    return Fixture(root=root, manifest=manifest, rows=tuple(fixture_rows()))


@pytest.fixture
def artifact_root(tmp_path) -> Path:
    return tmp_path / "artifacts"


def run(corpus: Fixture, artifact_root: Path, **kwargs):
    return experiment().run_experiment(
        corpus_root=corpus.root, artifact_root=artifact_root, seed=SEED, **kwargs
    )


def dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


# --- the fixture itself ------------------------------------------------------------------


def test_the_fixture_exercises_every_threshold_branch():
    """Without this the threshold tests below could pass by never reaching a branch."""
    split = temporal_split([r.submitted_at for r, _ in fixture_rows()])
    train = [
        (record.label, outcome.resolution_hours)
        for record, outcome in fixture_rows()
        if split.period_of(record.submitted_at) is Period.TRAIN
    ]
    eligible = {}
    for label, hours in train:
        eligible.setdefault(label, 0)
        if hours is not None:
            eligible[label] += 1
    assert eligible[NOISE] >= MIN_ELIGIBLE_OBSERVATIONS, "Noise must earn its own p75"
    assert 0 < eligible[HEAT] < MIN_ELIGIBLE_OBSERVATIONS, "Heat must hit the fallback"
    assert eligible[GHOST] == 0, "Ghost must be seen only through open requests"
    assert NEWCOMER not in eligible, "Newcomer must be unseen in training"


def test_the_fixture_has_open_requests_in_every_period():
    split = temporal_split([r.submitted_at for r, _ in fixture_rows()])
    for period in Period:
        rows = [
            outcome
            for record, outcome in fixture_rows()
            if split.period_of(record.submitted_at) is period
        ]
        assert any(o.resolution_hours is None for o in rows), period
        assert any(o.resolution_hours is not None for o in rows), period


def test_the_fixture_ties_a_timestamp_across_the_train_boundary():
    rows = fixture_rows()
    tied = [rows[index][0] for index in TIED_INDICES]
    assert len({record.submitted_at for record in tied}) == 1
    split = temporal_split([r.submitted_at for r, _ in rows])
    assert len({split.period_of(record.submitted_at) for record in tied}) == 1


def test_the_fixture_produces_a_non_empty_warmup():
    rows = fixture_rows()
    split = temporal_split([r.submitted_at for r, _ in rows])
    train = [r for r, _ in rows if split.period_of(r.submitted_at) is Period.TRAIN]
    folds = forward_chaining_folds([r.submitted_at for r in train])
    assert len(folds[0].fit_indices) > 0, "there must be warm-up rows to carry NaN"


def test_the_fixture_window_is_inside_the_decided_window():
    rows = fixture_rows()
    assert all(WINDOW_START <= r.submitted_at <= WINDOW_END for r, _ in rows)


def test_the_fixture_outcomes_follow_the_corrected_schema():
    """D37: resolved rows carry a real `closed_at`; open rows carry neither field."""
    for record, outcome in fixture_rows():
        assert outcome.external_id == record.external_id
        if outcome.resolution_hours is None:
            assert outcome.closed_at is None
        else:
            assert outcome.closed_at is not None
            assert outcome.closed_at.tzinfo is not None
            assert outcome.closed_at > record.submitted_at


def test_the_loaded_outcomes_are_real_instances_the_task_apis_accept(corpus):
    """D37: Task 11 and Task 13 check the type, so the loader must return the type."""
    from ingest.manifest import load_outcomes
    from ml.training.labels import _validated_pairs

    _, records = load_corpus("nyc311", root=corpus.root)
    _, outcomes = load_outcomes("nyc311", root=corpus.root)
    records, outcomes = list(records), list(outcomes)
    assert all(type(item) is NYC311Outcome for item in outcomes)
    _validated_pairs(records, outcomes)


# --- loading: records and outcomes ---------------------------------------------------------


def test_the_experiment_loads_records_and_outcomes_through_the_manifest(corpus, artifact_root):
    """D37.1: both streams are authoritative and manifest-gated."""
    result = run(corpus, artifact_root)
    assert len(result.records) == len(corpus.records)
    assert result.manifest.corpus_id == corpus.manifest.corpus_id


def test_a_corpus_without_an_outcome_sidecar_stops_the_run(corpus, artifact_root, tmp_path):
    """D37.16: absence is a typed error, never an empty iterator silently accepted."""
    from ingest.manifest import OutcomeSidecarNotFound

    bare = tmp_path / "bare"
    rows = fixture_rows()
    write_partition([r for r, _ in rows], "nyc311", 2024, 0, root=bare)
    write_manifest(
        build_manifest(
            source="nyc311",
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            source_api_version="fixture-v1",
            limit=None,
            timestamp_diagnostic=TIMESTAMP_DIAGNOSTIC,
            root=bare,
        ),
        root=bare,
    )
    with pytest.raises(OutcomeSidecarNotFound):
        experiment().run_experiment(corpus_root=bare, artifact_root=artifact_root, seed=SEED)


def test_a_tampered_outcome_part_stops_the_run(corpus, artifact_root):
    """Byte verification is not optional for the sidecar either."""
    from ingest.manifest import ChecksumMismatch

    part = next(iter(corpus.manifest.outcome_part_files))
    target = corpus.root / part
    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(ChecksumMismatch):
        run(corpus, artifact_root)


def test_the_record_outcome_join_is_total_and_bijective(corpus, artifact_root):
    """D37.1: identity is preserved and nothing is dropped on either side."""
    result = run(corpus, artifact_root)
    record_ids = [record.external_id for record in result.records]
    outcome_ids = [outcome.external_id for outcome in result.outcomes]
    assert record_ids == outcome_ids
    assert len(set(record_ids)) == len(record_ids)


@pytest.mark.parametrize("damage", ["missing", "extra", "duplicate"])
def test_a_broken_outcome_join_raises(tmp_path, artifact_root, damage):
    """Missing, extra and duplicate identities are three separate failures."""
    rows = fixture_rows()
    records = [record for record, _ in rows]
    outcomes = [outcome for _, outcome in rows]
    if damage == "missing":
        outcomes = outcomes[:-1]
    elif damage == "extra":
        outcomes = outcomes + [NYC311Outcome("99999", datetime(2024, 4, 1, tzinfo=UTC), 1.0)]
    else:
        outcomes = outcomes[:-1] + [outcomes[0]]

    root = tmp_path / "broken"
    write_partition(records, "nyc311", 2024, 0, root=root)
    storage().write_outcome_partition(outcomes, "nyc311", 2024, 0, root=root)
    write_manifest(
        build_manifest(
            source="nyc311",
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            source_api_version="fixture-v1",
            limit=None,
            timestamp_diagnostic=TIMESTAMP_DIAGNOSTIC,
            root=root,
        ),
        root=root,
    )
    with pytest.raises(ValueError):
        experiment().run_experiment(corpus_root=root, artifact_root=artifact_root, seed=SEED)


def test_the_experiment_never_reads_the_raw_cache(corpus, artifact_root, monkeypatch):
    """D37.1. Mutation: re-derive outcomes from `data/raw/`."""
    opened: list[str] = []
    real_open = Path.open

    def watched(self, *args, **kwargs):
        opened.append(str(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", watched)
    run(corpus, artifact_root)
    assert not any("raw" in Path(path).parts for path in opened)


def test_only_records_inside_the_decided_window_are_used(tmp_path, artifact_root):
    """D37.2: 2024-01-01 through 2025-12-31 inclusive. Mutation: widen the window."""
    rows = fixture_rows()
    outside = (
        CorpusRecord(
            source="nyc311",
            external_id="99999",
            text="far too early",
            label=NOISE,
            submitted_at=datetime(2023, 6, 1, tzinfo=UTC),
        ),
        NYC311Outcome("99999", datetime(2023, 6, 1, 5, tzinfo=UTC), 5.0),
    )
    root = tmp_path / "wide"
    build_fixture_corpus(root, [outside, *rows])
    result = experiment().run_experiment(corpus_root=root, artifact_root=artifact_root, seed=SEED)
    assert "99999" not in {record.external_id for record in result.records}
    assert all(WINDOW_START <= record.submitted_at <= WINDOW_END for record in result.records)


# --- the split, folds and warm-up --------------------------------------------------------


def test_every_record_is_assigned_by_the_temporal_split(corpus, artifact_root):
    """§6.1. Mutation: any shuffled split."""
    result = run(corpus, artifact_root)
    for period, records in result.periods.items():
        for record in records:
            assert result.split.period_of(record.submitted_at) is period
    train = [r.submitted_at for r in result.periods[Period.TRAIN]]
    validation = [r.submitted_at for r in result.periods[Period.VALIDATION]]
    test = [r.submitted_at for r in result.periods[Period.TEST]]
    assert max(train) <= result.split.train_end < min(validation)
    assert max(validation) <= result.split.val_end < min(test)


def test_a_tied_boundary_timestamp_cannot_straddle_two_periods(corpus, artifact_root):
    result = run(corpus, artifact_root)
    tied = [corpus.rows[index][0] for index in TIED_INDICES]
    assert len({result.split.period_of(r.submitted_at) for r in tied}) == 1


def test_folds_are_built_over_the_training_period_only(corpus, artifact_root):
    """D30. Mutation: build folds over every record."""
    result = run(corpus, artifact_root)
    covered = sum(len(fold.apply_indices) for fold in result.folds)
    train_count = result.split.counts[Period.TRAIN]
    assert covered + result.warmup_row_count == train_count
    assert len(result.folds) == 5


def test_warmup_rows_carry_nan_and_reach_training(corpus, artifact_root):
    """§J and the plan's named test. Mutation: drop or impute the warm-up rows."""
    result = run(corpus, artifact_root)
    train_matrix = dense(result.matrices[Period.TRAIN])
    aggregate_columns = train_matrix[:, 3:5]
    assert np.isnan(aggregate_columns).any(), "warm-up rows must survive as NaN"
    assert result.warmup_row_count > 0
    assert len(result.periods[Period.TRAIN]) == result.split.counts[Period.TRAIN]


def test_warmup_row_count_is_the_first_folds_fit_block(corpus, artifact_root):
    """D30 and D37.11. Mutation: write `null`, as a text model would."""
    result = run(corpus, artifact_root)
    assert result.warmup_row_count == len(result.folds[0].fit_indices)
    metadata = load_artifact(result.artifact_path).metadata
    assert metadata["warmup_row_count"] == result.warmup_row_count
    assert metadata["warmup_row_count"] is not None


# --- open requests (D37.3) -----------------------------------------------------------------


def test_open_requests_stay_in_the_split_and_fold_population(corpus, artifact_root):
    """D37.3: they inform a category's history without acquiring a label."""
    result = run(corpus, artifact_root)
    for period in Period:
        rows = result.periods[period]
        assert any(
            result.outcome_for(record.external_id).resolution_hours is None for record in rows
        ), period


def test_open_requests_are_absent_from_the_labelled_population(corpus, artifact_root):
    """D37.3. Mutation: keep them and coerce the label to False."""
    result = run(corpus, artifact_root)
    for period in (Period.VALIDATION, Period.TEST):
        labelled = result.labelled[period.value]
        assert all(
            result.outcome_for(record.external_id).resolution_hours is not None
            for record in labelled.records
        )
        assert len(labelled.records) == len(labelled.labels)
        assert len(labelled.records) < len(result.periods[period])


def test_an_open_request_never_receives_a_breach_label(corpus, artifact_root):
    """D37.3: no coercion to False, which `apply_thresholds` already refuses."""
    result = run(corpus, artifact_root)
    labelled_ids = {
        record.external_id
        for period in (Period.TRAIN, Period.VALIDATION, Period.TEST)
        for record in result.labelled[period.value].records
    }
    open_ids = {
        outcome.external_id for outcome in result.outcomes if outcome.resolution_hours is None
    }
    assert labelled_ids & open_ids == set()


def test_open_requests_contribute_to_no_threshold_statistic(corpus, artifact_root):
    """D31/D37.3: `Ghost` is seen only through open requests, so it has no own p75."""
    result = run(corpus, artifact_root)
    assert GHOST not in result.thresholds.per_type
    assert result.thresholds.threshold_for(GHOST) == result.thresholds.global_fallback


def test_open_requests_may_still_receive_aggregate_features(corpus, artifact_root):
    """D37.3: excluded from the label, not from the feature history."""
    result = run(corpus, artifact_root)
    matrix = dense(result.matrices[Period.TEST])
    open_positions = [
        index
        for index, record in enumerate(result.periods[Period.TEST])
        if result.outcome_for(record.external_id).resolution_hours is None
    ]
    assert open_positions, "the fixture must hold open test requests"
    assert not np.isnan(matrix[open_positions][:, 3:5]).all()


def test_open_requests_are_outside_every_metric_population(corpus, artifact_root):
    """D37.3. Mutation: evaluate over all rows, inflating the denominator."""
    result = run(corpus, artifact_root)
    matrix: ConfusionMatrix = result.metrics["test"]["confusion_matrix"]
    total = sum(sum(row) for row in matrix.model)
    assert total == len(result.labelled["test"].records)
    assert total < len(result.periods[Period.TEST])


def test_breach_rate_excludes_open_requests_from_the_denominator(corpus, artifact_root):
    """D37.7. Mutation: divide by every record in the period."""
    result = run(corpus, artifact_root)
    for period in ("validation", "test"):
        labelled = result.labelled[period]
        expected = float(np.count_nonzero(labelled.labels) / len(labelled.labels))
        assert result.breach_rate[period] == pytest.approx(expected)


# --- thresholds (D31, D33, §7) ---------------------------------------------------------------


def test_thresholds_are_fitted_on_the_training_period_alone(corpus, artifact_root):
    """§7. Mutation: fit on train+validation."""
    result = run(corpus, artifact_root)
    from ml.training.labels import fit_thresholds

    train = result.labelled["train"]
    expected = fit_thresholds(train.records, train.outcomes)
    assert result.thresholds.per_type == expected.per_type
    assert result.thresholds.global_fallback == pytest.approx(expected.global_fallback)


def test_permuting_test_outcomes_leaves_the_fitted_thresholds_identical(tmp_path, artifact_root):
    """§6.2. Mutation: fit thresholds over every period."""
    rows = fixture_rows()
    shifted = list(rows)
    start = len(TRAIN_PLAN) + len(VALIDATION_PLAN)
    for offset in range(start, len(shifted)):
        record, outcome = shifted[offset]
        hours = outcome.resolution_hours
        shifted[offset] = (
            record,
            NYC311Outcome(
                outcome.external_id, outcome.closed_at, None if hours is None else hours * 3 + 1
            ),
        )

    first_root, second_root = tmp_path / "a", tmp_path / "b"
    build_fixture_corpus(first_root, rows)
    build_fixture_corpus(second_root, shifted)
    first = experiment().run_experiment(
        corpus_root=first_root, artifact_root=artifact_root / "a", seed=SEED
    )
    second = experiment().run_experiment(
        corpus_root=second_root, artifact_root=artifact_root / "b", seed=SEED
    )
    assert first.thresholds.per_type == second.thresholds.per_type
    assert first.thresholds.global_fallback == pytest.approx(second.thresholds.global_fallback)


def test_a_type_above_the_minimum_gets_its_own_threshold(corpus, artifact_root):
    """§7: 130 eligible Noise observations is well past `min_eligible`."""
    result = run(corpus, artifact_root)
    assert NOISE in result.thresholds.per_type


def test_a_type_below_the_minimum_takes_the_global_fallback(corpus, artifact_root):
    """§7/D33: the hundred counts eligible observations, not raw records."""
    result = run(corpus, artifact_root)
    assert HEAT not in result.thresholds.per_type
    assert result.thresholds.threshold_for(HEAT) == result.thresholds.global_fallback
    assert result.thresholds.fallback_type_count >= 2


def test_the_threshold_is_the_interpolated_p75_of_the_task_13_primitive(corpus, artifact_root):
    """D31's shared definition, computed independently. Mutation: nearest-rank."""
    from ml.training.thresholds import linear_percentile

    result = run(corpus, artifact_root)
    train = result.labelled["train"]
    observed = [
        outcome.resolution_hours
        for record, outcome in zip(train.records, train.outcomes, strict=True)
        if record.label == NOISE
    ]
    assert result.thresholds.per_type[NOISE] == pytest.approx(linear_percentile(observed, 0.75))


def test_task_seventeen_states_no_threshold_arithmetic_of_its_own(corpus, artifact_root):
    """D37.12: the primitive is called, never restated.

    Behavioural rather than textual: a run whose threshold disagrees with the
    Task 13 API would mean a second definition exists somewhere.
    """
    from ml.training.labels import fit_thresholds

    result = run(corpus, artifact_root)
    # Every training pair, open requests included: D33 counts a type seen only
    # through open requests in `fallback_type_count`, because its applied
    # threshold is still the global fallback. A resolved-only reference would
    # drop such a type and under-count by one.
    train_records = result.periods[Period.TRAIN]
    train_outcomes = [result.outcome_for(r.external_id) for r in train_records]
    independent = fit_thresholds(train_records, train_outcomes)
    assert dict(result.thresholds.per_type) == dict(independent.per_type)
    assert result.thresholds.fallback_type_count == independent.fallback_type_count
    assert result.thresholds.fallback_type_count >= 2, (
        "the fixture must exercise both a below-minimum and an open-only type"
    )


# --- aggregates (D31) -------------------------------------------------------------------------


def test_training_aggregates_use_the_out_of_fold_path(corpus, artifact_root):
    """The plan's named test: the train path differs from the val/test construction."""
    from ml.training.aggregates import (
        apply_category_aggregates,
        fit_category_aggregates,
        oof_category_aggregates,
    )

    result = run(corpus, artifact_root)
    train = result.labelled["train"]
    all_train = result.periods[Period.TRAIN]
    expected_oof = oof_category_aggregates(
        all_train, [result.outcome_for(r.external_id) for r in all_train], result.folds
    )
    built = dense(result.matrices[Period.TRAIN])
    assert np.allclose(
        built[:, 3],
        np.array(expected_oof.category_mean_resolution_hours, dtype=float),
        equal_nan=True,
    )

    frozen = fit_category_aggregates(
        all_train, [result.outcome_for(r.external_id) for r in all_train]
    )
    whole_period = apply_category_aggregates(frozen, all_train)
    assert not np.allclose(
        np.array(expected_oof.category_mean_resolution_hours, dtype=float),
        np.array(whole_period.category_mean_resolution_hours, dtype=float),
        equal_nan=True,
    ), "the out-of-fold path must not coincide with the whole-period path"
    assert train.records


def test_validation_and_test_aggregates_come_from_the_training_period_only(corpus, artifact_root):
    """§6.3. Mutation: refit aggregates per evaluation period."""
    from ml.training.aggregates import apply_category_aggregates, fit_category_aggregates

    result = run(corpus, artifact_root)
    train = result.periods[Period.TRAIN]
    frozen = fit_category_aggregates(train, [result.outcome_for(r.external_id) for r in train])
    for period in (Period.VALIDATION, Period.TEST):
        rows = result.periods[period]
        expected = apply_category_aggregates(frozen, rows)
        built = dense(result.matrices[period])
        assert np.allclose(
            built[:, 3],
            np.array(expected.category_mean_resolution_hours, dtype=float),
            equal_nan=True,
        )


def test_permuting_test_outcomes_leaves_training_features_unchanged(tmp_path, artifact_root):
    """The plan's named leakage test. Mutation: fit aggregates over all periods."""
    rows = fixture_rows()
    shifted = list(rows)
    start = len(TRAIN_PLAN) + len(VALIDATION_PLAN)
    for offset in range(start, len(shifted)):
        record, outcome = shifted[offset]
        hours = outcome.resolution_hours
        shifted[offset] = (
            record,
            NYC311Outcome(
                outcome.external_id, outcome.closed_at, None if hours is None else hours * 5 + 2
            ),
        )

    first_root, second_root = tmp_path / "a", tmp_path / "b"
    build_fixture_corpus(first_root, rows)
    build_fixture_corpus(second_root, shifted)
    first = experiment().run_experiment(
        corpus_root=first_root, artifact_root=artifact_root / "a", seed=SEED
    )
    second = experiment().run_experiment(
        corpus_root=second_root, artifact_root=artifact_root / "b", seed=SEED
    )
    assert np.allclose(
        dense(first.matrices[Period.TRAIN]),
        dense(second.matrices[Period.TRAIN]),
        equal_nan=True,
    )


def test_a_category_unseen_in_training_receives_the_training_global(corpus, artifact_root):
    """D31/§6.3. Mutation: `NaN`, or a value derived from the evaluation period."""
    result = run(corpus, artifact_root)
    rows = result.periods[Period.VALIDATION]
    positions = [index for index, record in enumerate(rows) if record.label == NEWCOMER]
    assert positions, "the fixture must carry an unseen category in validation"
    matrix = dense(result.matrices[Period.VALIDATION])
    assert not np.isnan(matrix[positions][:, 3]).any()
    assert len(set(matrix[positions, 3])) == 1, "every unseen row takes the same global"


# --- features (D37.13) -------------------------------------------------------------------------


def test_the_feature_matrix_is_exactly_risk_features_v1_in_order(corpus, artifact_root):
    """§3.3. Mutation: reorder, or add a sixth column."""
    result = run(corpus, artifact_root)
    assert result.feature_spec == RISK_FEATURES_V1
    for period in Period:
        assert dense(result.matrices[period]).shape[1] == 5
    metadata = load_artifact(result.artifact_path).metadata
    assert list(metadata["feature_spec"]) == list(RISK_FEATURES_V1.names)
    assert metadata["feature_spec_version"] == "risk_features_v1"


def test_features_are_built_through_task_twelve(corpus, artifact_root):
    """D37.13: `build_features` with `AggregateColumns`, never assembled by hand."""
    from ml.training.aggregates import AggregateColumns

    result = run(corpus, artifact_root)
    rows = result.periods[Period.TEST]
    columns = result.aggregates[Period.TEST]
    assert isinstance(columns, AggregateColumns)
    expected = build_features(rows, columns, RISK_FEATURES_V1)
    assert np.allclose(dense(result.matrices[Period.TEST]), expected, equal_nan=True)


@pytest.mark.parametrize(
    "forbidden",
    ["sla_hours", "age_hours", "priority_rank", "queue_depth", "assignee_open_count"],
)
def test_a_forbidden_feature_is_not_producible(corpus, forbidden):
    """D15/D37.13: the vocabulary is closed, so none of these can enter a spec."""
    from ml.training.features import FeatureSpec

    with pytest.raises(FeatureUnavailable, match=forbidden):
        build_features(corpus.records[:3], None, FeatureSpec((forbidden,), "probe"))


# --- the roster, the model and banding (D37.4, D37.5, D37.10) -------------------------------------


def test_the_roster_is_false_then_true(corpus, artifact_root):
    """D37.5. Mutation: `(True, False)`, which silently swaps every column."""
    result = run(corpus, artifact_root)
    assert result.roster == (False, True)
    assert tuple(result.model.classes_) == (False, True)
    assert list(load_artifact(result.artifact_path).metadata["label_roster"]) == [False, True]


def test_the_score_is_the_probability_of_true(corpus, artifact_root):
    """D37.5. Mutation: take column 0."""
    result = run(corpus, artifact_root)
    for period in ("validation", "test"):
        matrix = result.scores[period]
        assert matrix.shape == (len(result.labelled[period].records), 2)
        assert np.allclose(matrix.sum(axis=1), 1.0)
        assert np.allclose(result.positive_scores[period], matrix[:, 1])


def test_the_candidate_grid_runs_from_five_to_ninety_five_hundredths():
    """D37.4. Mutation: start at 0.00, which would band every record high."""
    grid = experiment().BAND_GRID
    assert len(grid) == 19
    assert grid[0] == pytest.approx(0.05)
    assert grid[-1] == pytest.approx(0.95)
    assert list(grid) == pytest.approx([0.05 * (step + 1) for step in range(19)])


def test_the_decision_threshold_is_selected_on_validation_alone(corpus, artifact_root):
    """D37.4. Mutation: concatenate the test period into the selection inputs."""
    result = run(corpus, artifact_root)
    assert len(result.threshold.candidates) == len(experiment().BAND_GRID)
    assert result.threshold.population == len(result.labelled["validation"].records)
    assert result.threshold.population != (
        len(result.labelled["validation"].records) + len(result.labelled["test"].records)
    )


def test_permuting_test_labels_leaves_the_decision_threshold_unchanged(tmp_path, artifact_root):
    """§6.2 path 4. Mutation: select over validation+test."""
    rows = fixture_rows()
    flipped = _rotated_hours(rows, start=len(TRAIN_PLAN) + len(VALIDATION_PLAN))

    first_root, second_root = tmp_path / "a", tmp_path / "b"
    build_fixture_corpus(first_root, rows)
    build_fixture_corpus(second_root, flipped)
    first = experiment().run_experiment(
        corpus_root=first_root, artifact_root=artifact_root / "a", seed=SEED
    )
    second = experiment().run_experiment(
        corpus_root=second_root, artifact_root=artifact_root / "b", seed=SEED
    )
    assert first.threshold.value == second.threshold.value


def test_the_objective_is_the_positive_class_f1(corpus, artifact_root):
    """D37.4. Mutation: accuracy, or macro-F1 over both classes.

    Verified against Task 14's own minority report for the winning threshold,
    which is where positive-class F1 already lives.
    """
    from ml.training.metrics import minority_report

    result = run(corpus, artifact_root)
    validation = result.labelled["validation"]
    predicted = result.positive_scores["validation"] >= result.threshold.value
    report: MinorityReport = minority_report(
        [bool(value) for value in validation.labels],
        [bool(value) for value in predicted],
        result.roster,
        True,
        majority=result.majority,
        stratified=result.stratified["validation"],
    )
    assert result.threshold.positive_f1 == pytest.approx(report.model.f1)


def test_an_exact_tie_selects_the_lowest_threshold():
    """D37.4. Mutation: `max`, which would take the most conservative band."""
    from ml.training.metrics import majority_baseline, stratified_baseline

    roster = (False, True)
    train_labels = [False] * 8 + [True] * 4
    #: Two adjacent thresholds retain the identical predicted set, so their
    #: positive F1 is identical and only the tie-break can separate them.
    scores = np.array([[0.1, 0.9], [0.2, 0.8], [0.7, 0.3], [0.8, 0.2]])
    y_true = [True, True, False, False]
    chosen = experiment().select_decision_threshold(
        scores,
        y_true,
        roster,
        train_labels=train_labels,
        seed=SEED,
    )
    best = max(c.positive_f1 for c in chosen.candidates if c.positive_f1 is not None)
    tied = [c.threshold for c in chosen.candidates if c.positive_f1 == best]
    assert len(tied) >= 2
    assert chosen.value == pytest.approx(min(tied))
    assert stratified_baseline(train_labels, roster, 4, seed=SEED).seed == SEED
    assert majority_baseline(train_labels, roster).labels == roster


def test_the_frozen_threshold_is_applied_unchanged_to_test(corpus, artifact_root):
    """D37.4. Mutation: re-select on the test period."""
    result = run(corpus, artifact_root)
    predicted = result.positive_scores["test"] >= result.threshold.value
    assert np.array_equal(result.predictions["test"], predicted)
    metadata = load_artifact(result.artifact_path).metadata
    assert metadata["thresholds"]["decision"]["value"] == pytest.approx(result.threshold.value)


def test_the_estimator_carries_exactly_the_decided_parameters(corpus, artifact_root):
    """D37.10. Mutation: `class_weight="balanced"`, or a different learning rate."""
    model = run(corpus, artifact_root).estimator
    assert model.learning_rate == pytest.approx(0.1)
    assert model.max_iter == 100
    assert model.max_leaf_nodes == 31
    assert model.max_depth is None
    assert model.min_samples_leaf == 20
    assert model.l2_regularization == pytest.approx(0.0)
    assert model.early_stopping is False
    assert model.class_weight is None
    assert model.random_state == 17


def test_early_stopping_is_off_so_no_random_split_is_carved_out(corpus, artifact_root):
    """D37.10: `'auto'` would take an internal random validation split (§6)."""
    model = run(corpus, artifact_root).estimator
    assert model.early_stopping is not True
    assert model.early_stopping != "auto"


def test_no_hyperparameter_search_is_performed(corpus, artifact_root):
    """D37.10: nothing but the decision threshold is tuned."""
    module = Path(__file__).resolve().parents[3] / "ml" / "training" / "experiments" / "risk.py"
    tree = ast.parse(module.read_text(encoding="utf-8"))
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
        alias.asname or alias.name.split(".")[-1]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    for banned in ("GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV"):
        assert banned not in names


def test_no_resampling_and_no_class_weighting(corpus, artifact_root):
    """D37.9: residual imbalance is reported, not engineered away."""
    result = run(corpus, artifact_root)
    assert result.estimator.class_weight is None
    train = result.labelled["train"]
    assert len(train.records) == len(train.labels)
    assert len(set(train.labels)) > 1, "the fixture must hold both classes in training"


# --- published metrics (D37.6, D37.8) -------------------------------------------------------------


def test_pr_auc_is_the_headline_with_the_majority_baseline_only(corpus, artifact_root):
    """§5.5 and D34. Mutation: attach the stratified baseline to a ranking metric."""
    result = run(corpus, artifact_root)
    metadata = load_artifact(result.artifact_path).metadata
    for period in ("validation", "test"):
        published: ScoreResult = result.metrics[period]["pr_auc"]
        assert isinstance(published, ScoreResult)
        assert set(published.baselines) == {"majority"}
        # And as published: a serialiser that adds a stratified entry here would
        # invent the ranking baseline D34 forbids.
        written = metadata["metrics"][period]["pr_auc"]
        assert set(written["baselines"]) == {"majority"}
        assert set(metadata["metrics"][period]["roc_auc"]["baselines"]) == {"majority"}


def test_roc_auc_is_reported_as_secondary(corpus, artifact_root):
    """§5.5/D7: secondary, never promoted."""
    result = run(corpus, artifact_root)
    published: ScoreResult = result.metrics["test"]["roc_auc"]
    assert set(published.baselines) == {"majority"}
    metadata = load_artifact(result.artifact_path).metadata
    assert metadata["metrics"]["test"]["headline"] == "pr_auc"


def test_the_minority_report_uses_the_frozen_threshold_and_the_full_roster(corpus, artifact_root):
    """D37.6. Mutation: observed labels only, or a re-tuned threshold."""
    result = run(corpus, artifact_root)
    report: MinorityReport = result.metrics["test"]["minority_report"]
    assert report.positive_label is True
    assert set(report.baselines) == {"majority", "stratified"}
    assert report.support == int(np.count_nonzero(result.labelled["test"].labels))
    # And as published, for both periods: dropping a baseline on the way out
    # would leave a figure quoted alone, which is what section 5.1 forbids.
    metadata = load_artifact(result.artifact_path).metadata
    for period in ("validation", "test"):
        written = metadata["metrics"][period]
        assert set(written["minority_report"]["baselines"]) == {"majority", "stratified"}
        assert set(written["confusion_matrix"]["baselines"]) == {"majority", "stratified"}


def test_the_confusion_matrix_covers_the_whole_labelled_population(corpus, artifact_root):
    """D37.6. Mutation: filter the low band out before counting."""
    result = run(corpus, artifact_root)
    matrix: ConfusionMatrix = result.metrics["test"]["confusion_matrix"]
    assert matrix.labels == (False, True)
    assert set(matrix.baselines) == {"majority", "stratified"}
    assert sum(sum(row) for row in matrix.model) == len(result.labelled["test"].records)


def test_no_macro_f1_headline_is_introduced(corpus, artifact_root):
    """D37.6: triage's headline is not imported into risk."""
    metadata = load_artifact(run(corpus, artifact_root).artifact_path).metadata
    for period, published in metadata["metrics"].items():
        assert "macro_f1" not in published, period


def test_the_calibration_curve_has_ten_bins_and_the_three_fields(corpus, artifact_root):
    """D37.8. Mutation: keep empty bins, or add a Brier score."""
    result = run(corpus, artifact_root)
    for period in ("validation", "test"):
        curve = result.calibration[period]
        assert 0 < len(curve) <= experiment().CALIBRATION_BINS == 10
        for entry in curve:
            assert set(entry) == {"mean_predicted_probability", "fraction_positive", "count"}
            assert entry["count"] > 0
            assert 0.0 <= entry["fraction_positive"] <= 1.0
        assert sum(entry["count"] for entry in curve) == len(result.labelled[period].records)


def test_the_calibration_curve_carries_no_baseline_and_is_not_a_metric(corpus, artifact_root):
    """D37.8: ancillary diagnostic information only."""
    metadata = load_artifact(run(corpus, artifact_root).artifact_path).metadata
    published = metadata["metrics"]["test"]
    assert "calibration_curve" in published
    assert "brier" not in published
    for entry in published["calibration_curve"]:
        assert "baselines" not in entry


def test_baseline_priors_come_from_training_labels_only(tmp_path, artifact_root):
    """D34. Mutation: fit the prior on the evaluation labels."""
    rows = fixture_rows()
    flipped = _rotated_hours(rows, start=len(TRAIN_PLAN))

    first_root, second_root = tmp_path / "a", tmp_path / "b"
    build_fixture_corpus(first_root, rows)
    build_fixture_corpus(second_root, flipped)
    first = experiment().run_experiment(
        corpus_root=first_root, artifact_root=artifact_root / "a", seed=SEED
    )
    second = experiment().run_experiment(
        corpus_root=second_root, artifact_root=artifact_root / "b", seed=SEED
    )
    assert first.majority.prior == second.majority.prior
    assert first.majority.predicted_label == second.majority.predicted_label


def test_the_stratified_baseline_seed_is_explicit_and_recorded(corpus, artifact_root):
    """D34 and §R."""
    result = run(corpus, artifact_root)
    metadata = load_artifact(result.artifact_path).metadata
    assert result.stratified["test"].seed == metadata["seeds"]["stratified_baseline"]
    assert isinstance(metadata["seeds"]["stratified_baseline"], int)


def test_metrics_are_published_per_period(corpus, artifact_root):
    """§P. Mutation: publish the test period only."""
    metadata = load_artifact(run(corpus, artifact_root).artifact_path).metadata
    assert set(metadata["metrics"]) >= {"validation", "test"}
    for published in metadata["metrics"].values():
        assert "baselines" in published["pr_auc"]
        assert "breach_rate" in published


# --- the artifact (D37.11, D37.12, D37.14) ---------------------------------------


def test_the_artifact_is_written_to_the_decided_directory_and_loads(corpus, artifact_root):
    """D37.11 and D35's lexical directory check."""
    result = run(corpus, artifact_root)
    assert result.artifact_path.parent.name == "nyc311_sla_risk"
    assert result.artifact_path.name == "v1"
    assert "nyc311" in result.artifact_path.parts
    assert load_artifact(result.artifact_path) is not None


def test_the_identity_fields_are_exactly_the_decided_literals(corpus, artifact_root):
    """D37.11. Mutation: any drift."""
    metadata = load_artifact(run(corpus, artifact_root).artifact_path).metadata
    assert metadata["model_name"] == "nyc311_sla_risk"
    assert metadata["model_version"] == "v1"
    assert metadata["experiment_label"] == "nyc311 sla risk histgradientboosting"


def test_the_threshold_metadata_has_exactly_the_decided_shape(corpus, artifact_root):
    """D37.12. Mutation: recompute, or omit the fallback count."""
    result = run(corpus, artifact_root)
    thresholds = load_artifact(result.artifact_path).metadata["thresholds"]
    assert set(thresholds) >= {
        "min_eligible",
        "percentile",
        "per_type",
        "global_fallback",
        "fallback_type_count",
    }
    assert thresholds["min_eligible"] == 100
    assert thresholds["percentile"] == pytest.approx(0.75)
    assert dict(thresholds["per_type"]) == dict(result.thresholds.per_type)
    assert thresholds["global_fallback"] == pytest.approx(result.thresholds.global_fallback)
    assert thresholds["fallback_type_count"] == result.thresholds.fallback_type_count


def test_corpus_identity_and_window_come_from_the_loaded_manifest(corpus, artifact_root):
    """§R, D27 and D37.1: the identity binds record and outcome bytes alike."""
    result = run(corpus, artifact_root)
    metadata = load_artifact(result.artifact_path).metadata
    assert metadata["corpus_id"] == corpus.manifest.corpus_id
    assert metadata["corpus_schema_version"] == corpus.manifest.schema_version == 1
    assert metadata["source_window"]["start"] == corpus.manifest.window_start.isoformat()
    assert corpus.manifest.outcome_part_files, "the cited corpus must carry a sidecar"


def test_split_metadata_records_cuts_counts_and_both_fraction_sets(corpus, artifact_root):
    """§6.1 and finding I6: Task 19 reuses these exact boundaries."""
    result = run(corpus, artifact_root)
    split = load_artifact(result.artifact_path).metadata["split"]
    assert set(split) >= {
        "train_end",
        "val_end",
        "counts",
        "requested_fractions",
        "achieved_fractions",
    }
    assert split["train_end"] == result.split.train_end.isoformat()
    assert dict(split["counts"]) == {p.value: result.split.counts[p] for p in Period}


def test_dependency_versions_are_the_four_live_values(corpus, artifact_root):
    """D37.14: no onnxruntime, and no stale literals."""
    import joblib
    import scipy

    metadata = load_artifact(run(corpus, artifact_root).artifact_path).metadata
    versions = metadata["dependency_versions"]
    assert versions["numpy"] == np.__version__
    assert versions["scipy"] == scipy.__version__
    assert versions["scikit-learn"] == sklearn.__version__
    assert versions["joblib"] == joblib.__version__
    assert "onnxruntime" not in versions


def test_no_embedding_fields_are_written(corpus, artifact_root):
    metadata = load_artifact(run(corpus, artifact_root).artifact_path).metadata
    for field in ("embedding_dimension", "embedding_model_id", "embedding_model_sha256"):
        assert field not in metadata


def test_round_tripped_predictions_are_identical_on_fixed_input(corpus, artifact_root):
    """Plan Task 15. Mutation: refit anything at load."""
    result = run(corpus, artifact_root)
    loaded = load_artifact(result.artifact_path)
    rows = result.labelled["test"].records
    rebuilt = loaded.build_features(rows, result.aggregates_for_labelled["test"])
    assert np.allclose(loaded.model.predict_proba(rebuilt), result.scores["test"], equal_nan=True)
    assert np.array_equal(
        loaded.model.predict(rebuilt), result.model.predict(dense(result.labelled_matrices["test"]))
    )


# --- determinism and boundaries --------------------------------------------------


def test_two_runs_with_the_same_seed_agree_completely(corpus, artifact_root):
    """§R. Mutation: any unseeded shuffle."""
    first = run(corpus, artifact_root / "first")
    second = run(corpus, artifact_root / "second")
    assert first.threshold.value == second.threshold.value
    assert first.metadata["metrics"] == second.metadata["metrics"]
    assert first.warmup_row_count == second.warmup_row_count
    assert dict(first.thresholds.per_type) == dict(second.thresholds.per_type)


def test_records_are_consumed_in_corpus_order(corpus, artifact_root):
    """§R: deterministic ordering before the split."""
    result = run(corpus, artifact_root)
    seen = [r.external_id for period in Period for r in result.periods[period]]
    assert seen == sorted(seen)


def test_no_network_is_used_by_the_experiment(corpus, artifact_root, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    assert run(corpus, artifact_root).roster == (False, True)


def test_importing_the_experiment_pulls_in_no_django():
    probe = (
        "import sys; import ml.training.experiments.risk; "
        "print(','.join(sorted({m.split('.')[0] for m in sys.modules} & "
        "{'django', 'complaints', 'domains', 'accounts', 'config'})))"
    )
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=root,
        env=dict(os.environ),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_the_serving_registry_does_not_reference_the_risk_experiment():
    registry = Path(__file__).resolve().parents[3] / "ml" / "registry.py"
    source = registry.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any(name.startswith("ml.training") for name in imported)
    assert "nyc311_sla_risk" not in source


def test_this_module_is_selected_by_the_ml_marker(request):
    assert request.node.get_closest_marker("ml") is not None
