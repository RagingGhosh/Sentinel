"""Task 19: the reduced-feature cross-domain cross-target robustness probe.

RED phase. `ml.training.experiments.robustness_probe` does not exist, and neither
does the CFPB outcome sidecar its evaluation target lives in, so every test that
reaches production fails. The production module is imported late, inside
`probe()`, so the fixture and source-level guards still run.

The API surface these tests pin, all of it implied by the frozen contract at
`docs/superpowers/specs/2026-09-24-task-19-robustness-probe-contract.md` and none
of it existing yet:

    MODEL_NAME / MODEL_VERSION / EXPERIMENT_LABEL
    SOURCE / EVALUATION_SOURCE / WINDOW_START / WINDOW_END / SEED
    ROSTER / POSITIVE_LABEL / QUANTILES
    IN_DOMAIN / CROSS_DOMAIN            the two evaluation keys
    RESULT_CLASSIFICATION               verdict -> classification
    PROHIBITED_WORDINGS
    build_estimator(seed)
    run_probe(corpus_root=, artifact_root=, report_path=) -> report

Both corpora are written into `tmp_path`. Nothing is downloaded and no socket is
opened.
"""

import importlib
import inspect
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="numpy lives in requirements/ml.txt")
pytest.importorskip("sklearn", reason="scikit-learn lives in requirements/ml.txt")
pytest.importorskip("pyarrow", reason="pyarrow lives in requirements/train.txt")

from ingest.manifest import build_manifest, write_manifest  # noqa: E402
from ingest.schema import CFPBOutcome, CorpusRecord, NYC311Outcome  # noqa: E402
from ingest.storage import write_outcome_partition, write_partition  # noqa: E402
from ml.training.features import TRANSFER_FEATURES_V1  # noqa: E402
from ml.training.splits import DEFAULT_FRACTIONS, Period, temporal_split  # noqa: E402

pytestmark = pytest.mark.ml

ROOT = Path(__file__).resolve().parents[3]

# --- the contract's frozen values ----------------------------------------------------

SOURCE = "nyc311"
EVALUATION_SOURCE = "cfpb"
MODEL_NAME = "xdomain_xtarget_probe"
MODEL_VERSION = "xdomain_xtarget_probe_v1"
EXPERIMENT_LABEL = "reduced-feature cross-domain cross-target robustness probe"

FEATURE_NAMES = ("submitted_hour", "submitted_weekday", "text_length")
FEATURE_SPEC_VERSION = "transfer_features_v1"

WINDOW_START = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
WINDOW_END = datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC)
SEED = 17

ESTIMATOR_PARAMS = {
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "max_depth": None,
    "min_samples_leaf": 20,
    "l2_regularization": 0,
    "early_stopping": False,
    "class_weight": None,
    "random_state": SEED,
}

QUANTILE_POINTS = ("min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max")
SHIFT_MEASURES = ("pct_outside_source_range", "pct_outside_source_iqr")

FRAMING_FACT_KEYS = (
    "source_domain",
    "source_target",
    "evaluation_domain",
    "evaluation_target",
    "feature_set",
    "target_semantics_differ",
    "polarity_mapping",
    "analysis_type",
)

VERDICTS = {
    "strongly_suspicious_load_timestamp": "non-informative / diagnostic",
    "suspicious_insufficient_evidence": "substantive_with_stated_caveat",
    "supported_plausible_event_time": "substantive",
}

PROHIBITED = ("transfers to", "generalises to", "generalizes to", "works on cfpb")


# --- the production module, imported late --------------------------------------------


def probe():
    return importlib.import_module("ml.training.experiments.robustness_probe")


# --- the fixture corpora -------------------------------------------------------------
#
# NYC 311: 60 records inside the frozen window, one hour apart. At 70/15/15 that is
# 42 train, 9 validation and 9 test. Nine are left open -- no resolution time -- so
# "an open request never receives a fabricated label" is observable. Resolution
# hours alternate around the per-type p75 so both classes exist in every period.
#
# CFPB: 40 records, a mix of timely and untimely, with narratives an order of
# magnitude longer than the 311 descriptors so `text_length` shift is real.

NYC_START = datetime(2024, 3, 1, 9, 0, tzinfo=UTC)
NYC_COUNT = 60
OPEN_EVERY = 7

CFPB_START = datetime(2024, 4, 1, 9, 0, tzinfo=UTC)
CFPB_COUNT = 40
UNTIMELY_EVERY = 4

TIMESTAMP_DIAGNOSTIC = {
    "verdict": "supported_plausible_event_time",
    "reason": "fixture corpus; no provenance measurement is claimed",
    "deltas": {"median_hour_delta": 0.0},
    "rule_thresholds": {"median_hour_delta": 6.0},
}


def nyc_records() -> list[CorpusRecord]:
    return [
        CorpusRecord(
            source=SOURCE,
            external_id=f"n{index:04d}",
            text=f"noise descriptor {index}",
            label="Noise" if index % 2 else "Street Condition",
            submitted_at=NYC_START + timedelta(hours=index),
        )
        for index in range(NYC_COUNT)
    ]


def nyc_outcomes() -> list[NYC311Outcome]:
    outcomes = []
    for index in range(NYC_COUNT):
        if index % OPEN_EVERY == 0:
            outcomes.append(
                NYC311Outcome(external_id=f"n{index:04d}", closed_at=None, resolution_hours=None)
            )
            continue
        hours = 4.0 if index % 2 else 400.0
        outcomes.append(
            NYC311Outcome(
                external_id=f"n{index:04d}",
                closed_at=NYC_START + timedelta(hours=index + hours),
                resolution_hours=hours,
            )
        )
    return outcomes


def open_refs() -> set[str]:
    return {f"n{index:04d}" for index in range(NYC_COUNT) if index % OPEN_EVERY == 0}


def cfpb_records() -> list[CorpusRecord]:
    return [
        CorpusRecord(
            source=EVALUATION_SOURCE,
            external_id=f"c{index:04d}",
            text=f"consumer narrative {index} " + ("detail phrase " * 40),
            label="Mortgage" if index % 2 else "Credit card",
            submitted_at=CFPB_START + timedelta(hours=index),
        )
        for index in range(CFPB_COUNT)
    ]


def cfpb_outcomes() -> list[CFPBOutcome]:
    return [
        CFPBOutcome(
            external_id=f"c{index:04d}",
            timely_response=index % UNTIMELY_EVERY != 0,
            sent_to_company_at=CFPB_START + timedelta(hours=index + 24),
        )
        for index in range(CFPB_COUNT)
    ]


def write_corpus(root: Path, source: str, records, outcomes, writer) -> None:
    by_year: dict[int, list] = {}
    for record in records:
        by_year.setdefault(record.submitted_at.year, []).append(record)
    for year, group in sorted(by_year.items()):
        write_partition(group, source, year, 0, root=root)

    if outcomes is not None:
        identifiers = {record.external_id: record.submitted_at.year for record in records}
        outcomes_by_year: dict[int, list] = {}
        for outcome in outcomes:
            outcomes_by_year.setdefault(identifiers[outcome.external_id], []).append(outcome)
        for year, group in sorted(outcomes_by_year.items()):
            writer(group, source, year, 0, root=root)

    manifest = build_manifest(
        source,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        source_api_version="fixture-v1",
        limit=None,
        timestamp_diagnostic=TIMESTAMP_DIAGNOSTIC,
        root=root,
        ingested_at=datetime(2026, 9, 24, 12, 0, tzinfo=UTC),
    )
    write_manifest(manifest, root=root)


def build_corpora(root: Path, *, diagnostic=None) -> Path:
    """Both corpora under one root, as `run_probe` will load them."""
    storage = importlib.import_module("ingest.storage")
    write_corpus(root, SOURCE, nyc_records(), nyc_outcomes(), write_outcome_partition)
    write_corpus(
        root,
        EVALUATION_SOURCE,
        cfpb_records(),
        cfpb_outcomes(),
        storage.write_cfpb_outcome_partition,
    )
    if diagnostic is not None:
        _rewrite_diagnostic(root, diagnostic)
    return root


def _rewrite_diagnostic(root: Path, diagnostic: dict) -> None:
    """Re-issue the CFPB manifest with another provenance verdict."""
    manifest = build_manifest(
        EVALUATION_SOURCE,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        source_api_version="fixture-v1",
        limit=None,
        timestamp_diagnostic=diagnostic,
        root=root,
        ingested_at=datetime(2026, 9, 24, 12, 0, tzinfo=UTC),
    )
    write_manifest(manifest, root=root)


@pytest.fixture
def corpora(tmp_path) -> Path:
    return build_corpora(tmp_path / "corpus")


@pytest.fixture
def artifact_root(tmp_path) -> Path:
    return tmp_path / "artifacts"


@pytest.fixture
def report_path(tmp_path) -> Path:
    return tmp_path / "reports" / "robustness_probe.json"


def run(corpora: Path, artifact_root: Path, report_path: Path):
    return probe().run_probe(
        corpus_root=corpora, artifact_root=artifact_root, report_path=report_path
    )


def nyc_split():
    windowed = [
        record for record in nyc_records() if WINDOW_START <= record.submitted_at <= WINDOW_END
    ]
    return temporal_split([record.submitted_at for record in windowed], DEFAULT_FRACTIONS)


def cfpb_split():
    return temporal_split([record.submitted_at for record in cfpb_records()], DEFAULT_FRACTIONS)


# --- the fixture itself --------------------------------------------------------------


def test_the_fixture_has_both_classes_in_every_labelled_311_period():
    """Guard the guard: a one-class period would make PR-AUC raise by design (D34)."""
    split = nyc_split()
    hours = {o.external_id: o.resolution_hours for o in nyc_outcomes()}
    for period in Period:
        resolved = [
            hours[record.external_id]
            for record in nyc_records()
            if split.period_of(record.submitted_at) is period
            and hours[record.external_id] is not None
        ]
        assert len(set(resolved)) > 1, period


def test_the_fixture_has_untimely_cfpb_records_in_the_test_period():
    split = cfpb_split()
    timely = {o.external_id: o.timely_response for o in cfpb_outcomes()}
    test = [
        record for record in cfpb_records() if split.period_of(record.submitted_at) is Period.TEST
    ]
    assert test, "the CFPB fixture has no test period"
    assert any(not timely[record.external_id] for record in test)


def test_the_fixture_makes_text_length_shift_observable():
    """§5.4's expected finding: 311 descriptors are far shorter than CFPB narratives."""
    nyc = np.median([len(record.text) for record in nyc_records()])
    cfpb = np.median([len(record.text) for record in cfpb_records()])
    assert cfpb > nyc * 5


# --- A, B. features -------------------------------------------------------------------


def test_the_probe_uses_exactly_the_three_transfer_features_in_order():
    assert tuple(TRANSFER_FEATURES_V1.names) == FEATURE_NAMES
    assert probe().FEATURE_SPEC is TRANSFER_FEATURES_V1


def test_the_feature_spec_version_is_the_transfer_one_not_the_risk_one():
    """Versioned independently so the probe can never be read as the primary model."""
    assert probe().FEATURE_SPEC.version == FEATURE_SPEC_VERSION


def test_no_aggregate_feature_is_requested(corpora, artifact_root, report_path):
    """The probe calls `build_features(records, None, spec)`; an aggregate raises."""
    report = run(corpora, artifact_root, report_path)
    assert tuple(report.feature_names) == FEATURE_NAMES
    for banned in ("category_mean_resolution_hours", "category_breach_rate"):
        assert banned not in report.feature_names


def test_the_source_requests_no_aggregate_columns():
    """A target-derived aggregate in this probe would be leakage §5.4 forbids."""
    source = Path(probe().__file__).read_text(encoding="utf-8")
    for banned in ("category_mean_resolution_hours", "category_breach_rate"):
        assert banned not in source or "never" in source
    assert "oof_category_aggregates" not in source
    assert "fit_category_aggregates" not in source


def test_the_probe_does_not_import_the_primary_risk_experiment():
    """It reuses Task 17's frozen inputs, not Task 17's module (contract §3)."""
    source = Path(probe().__file__).read_text(encoding="utf-8")
    assert "experiments.risk" not in source
    assert "from ml.training.experiments import risk" not in source


# --- C. the 311 population ------------------------------------------------------------


def test_the_source_window_is_task_seventeens(corpora, artifact_root, report_path):
    module = probe()
    assert module.WINDOW_START == WINDOW_START
    assert module.WINDOW_END == WINDOW_END


def test_the_split_boundaries_reproduce_task_seventeens(corpora, artifact_root, report_path):
    """Finding I6: a different split would make the in-domain comparison invalid."""
    report = run(corpora, artifact_root, report_path)
    expected = nyc_split()
    recorded = report.source_training_population
    assert recorded["train_end"] == expected.train_end.isoformat()
    assert recorded["val_end"] == expected.val_end.isoformat()


def test_training_uses_the_train_period_only(corpora, artifact_root, report_path):
    report = run(corpora, artifact_root, report_path)
    split = nyc_split()
    trained_on = set(report.source_training_population["refs"])
    for record in nyc_records():
        period = split.period_of(record.submitted_at)
        if record.external_id in trained_on:
            assert period is Period.TRAIN, record.external_id


def test_the_validation_period_exists_and_is_unused(corpora, artifact_root, report_path):
    """O5: the split still produces validation; the probe tunes nothing with it."""
    split = nyc_split()
    assert split.counts[Period.VALIDATION] > 0
    report = run(corpora, artifact_root, report_path)
    assert set(report.metrics) == {probe().IN_DOMAIN, probe().CROSS_DOMAIN}


def test_the_in_domain_evaluation_is_the_311_test_period(corpora, artifact_root, report_path):
    report = run(corpora, artifact_root, report_path)
    split = nyc_split()
    evaluated = set(report.in_domain_evaluation_population["refs"])
    assert evaluated
    for record in nyc_records():
        if record.external_id in evaluated:
            assert split.period_of(record.submitted_at) is Period.TEST


def test_no_311_record_is_both_trained_on_and_evaluated(corpora, artifact_root, report_path):
    report = run(corpora, artifact_root, report_path)
    trained = set(report.source_training_population["refs"])
    evaluated = set(report.in_domain_evaluation_population["refs"])
    assert trained and evaluated
    assert trained.isdisjoint(evaluated)


# --- D. the CFPB population -----------------------------------------------------------


def test_the_cross_domain_population_is_the_cfpb_test_period(corpora, artifact_root, report_path):
    report = run(corpora, artifact_root, report_path)
    split = cfpb_split()
    evaluated = set(report.cross_domain_evaluation_population["refs"])
    assert evaluated
    for record in cfpb_records():
        if record.external_id in evaluated:
            assert split.period_of(record.submitted_at) is Period.TEST


def test_no_cfpb_train_or_validation_record_is_evaluated(corpora, artifact_root, report_path):
    """Task 16 tuned its abstention threshold on CFPB validation; none appears here."""
    report = run(corpora, artifact_root, report_path)
    split = cfpb_split()
    earlier = {
        record.external_id
        for record in cfpb_records()
        if split.period_of(record.submitted_at) is not Period.TEST
    }
    assert earlier.isdisjoint(set(report.cross_domain_evaluation_population["refs"]))


def test_only_records_with_a_persisted_outcome_are_evaluated(corpora, artifact_root, report_path):
    """ "CFPB test period" means the Task 16 test population with persisted outcomes."""
    report = run(corpora, artifact_root, report_path)
    persisted = {outcome.external_id for outcome in cfpb_outcomes()}
    assert set(report.cross_domain_evaluation_population["refs"]) <= persisted


def test_the_probe_never_fits_anything_on_cfpb(corpora, artifact_root, report_path):
    """One model, fitted once on 311 TRAIN (O5, contract §6)."""
    source = Path(probe().__file__).read_text(encoding="utf-8")
    assert source.count(".fit(") == 1


# --- E. label semantics ----------------------------------------------------------------


def test_both_evaluations_share_one_roster_and_positive_label():
    module = probe()
    assert tuple(module.ROSTER) == (False, True)
    assert module.POSITIVE_LABEL is True


def test_the_cfpb_target_enters_as_the_adverse_boolean(corpora, artifact_root, report_path):
    """Polarity mapping: adverse is breach in 311 and NOT timely in CFPB."""
    report = run(corpora, artifact_root, report_path)
    split = cfpb_split()
    timely = {o.external_id: o.timely_response for o in cfpb_outcomes()}
    evaluated = [
        record for record in cfpb_records() if split.period_of(record.submitted_at) is Period.TEST
    ]
    adverse = sum(1 for record in evaluated if not timely[record.external_id])
    assert report.metrics[probe().CROSS_DOMAIN]["minority_count"] == adverse


def test_no_field_combines_the_two_targets():
    """§4.3: no field, column, dataclass or variable unifies them."""
    source = Path(probe().__file__).read_text(encoding="utf-8")
    for banned in ("sla_or_timely", "combined_target", "unified_target", "breach_or_untimely"):
        assert banned not in source


def test_the_polarity_mapping_is_stated_in_the_report(corpora, artifact_root, report_path):
    report = run(corpora, artifact_root, report_path)
    mapping = report.framing_facts["polarity_mapping"]
    assert "nyc311_sla_breach" in str(mapping)
    assert "cfpb_timely_response" in str(mapping)


# --- F. open 311 records ----------------------------------------------------------------


def test_an_open_311_request_never_receives_a_label(corpora, artifact_root, report_path):
    """O6: no fabricated label, and never counted as a non-breach."""
    report = run(corpora, artifact_root, report_path)
    opened = open_refs()
    assert opened, "the fixture has no open requests"
    labelled = set(report.source_training_population["refs"]) | set(
        report.in_domain_evaluation_population["refs"]
    )
    assert labelled.isdisjoint(opened)


def test_open_requests_remain_in_the_split_population(corpora, artifact_root, report_path):
    """They still shape the temporal boundaries: only the labels exclude them."""
    split = nyc_split()
    assert split.total == NYC_COUNT


def test_the_labelled_population_is_smaller_than_the_split_population(
    corpora, artifact_root, report_path
):
    """Guard the guard: if it were not, the open rows would be leaking in as False."""
    report = run(corpora, artifact_root, report_path)
    labelled = len(report.source_training_population["refs"]) + len(
        report.in_domain_evaluation_population["refs"]
    )
    assert labelled < NYC_COUNT


# --- G, H, I. the estimator --------------------------------------------------------------


def test_the_estimator_carries_exactly_task_seventeens_parameters():
    """Inspected, not merely exercised: a fit would pass with any parameters."""
    params = probe().build_estimator(SEED).get_params()
    for name, expected in ESTIMATOR_PARAMS.items():
        assert params[name] == expected, name


def test_the_estimator_is_a_histogram_gradient_boosting_classifier():
    from sklearn.ensemble import HistGradientBoostingClassifier

    assert isinstance(probe().build_estimator(SEED), HistGradientBoostingClassifier)


def test_the_seed_is_seventeen():
    assert probe().SEED == SEED


def test_no_class_weighting_or_resampling_is_applied():
    """§5.4 reports the imbalance rather than engineering it away."""
    source = Path(probe().__file__).read_text(encoding="utf-8")
    assert "class_weight=None" in source
    for banned in ("SMOTE", "RandomOverSampler", "RandomUnderSampler", "resample", "balanced"):
        assert banned not in source


def test_no_preprocessing_is_applied():
    source = Path(probe().__file__).read_text(encoding="utf-8")
    for banned in ("StandardScaler", "MinMaxScaler", "SimpleImputer", "Pipeline", "nan_to_num"):
        assert banned not in source


def test_no_hyperparameter_search_is_performed():
    source = Path(probe().__file__).read_text(encoding="utf-8")
    for banned in ("GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV", "optuna"):
        assert banned not in source


def test_no_threshold_is_selected():
    """The probe publishes ranking metrics; it bands nothing (contract §6)."""
    source = Path(probe().__file__).read_text(encoding="utf-8")
    for banned in ("select_decision_threshold", "select_abstention_threshold", "threshold_grid"):
        assert banned not in source


def test_the_artifact_records_no_thresholds(corpora, artifact_root, report_path):
    report = run(corpora, artifact_root, report_path)
    assert report.metadata["thresholds"] is None


# --- J. the baseline ---------------------------------------------------------------------


def test_the_baseline_prior_comes_from_311_training_labels(corpora, artifact_root, report_path):
    """O4: the fraction of True breach labels in the 311 training population."""
    report = run(corpora, artifact_root, report_path)
    prior = report.baseline_prior
    assert 0.0 < prior < 1.0
    assert prior == pytest.approx(report.source_training_population["positive_rate"])


def test_one_frozen_baseline_scores_both_evaluations(corpora, artifact_root, report_path):
    """The same source-trained prior, unchanged, in both evaluations (O4)."""
    module = probe()
    report = run(corpora, artifact_root, report_path)
    in_domain = report.metrics[module.IN_DOMAIN]["pr_auc"].baselines["majority"]
    cross = report.metrics[module.CROSS_DOMAIN]["pr_auc"].baselines["majority"]
    assert report.metrics[module.IN_DOMAIN]["baseline_prior"] == report.baseline_prior
    assert report.metrics[module.CROSS_DOMAIN]["baseline_prior"] == report.baseline_prior
    assert isinstance(in_domain, float) and isinstance(cross, float)


def test_the_baseline_is_never_fitted_from_cfpb_labels(corpora, artifact_root, report_path):
    """Changing the CFPB outcomes must not move the frozen source prior."""
    first = run(corpora, artifact_root, report_path)
    other = build_corpora(report_path.parent / "flipped-corpus")
    storage = importlib.import_module("ingest.storage")
    flipped = [
        CFPBOutcome(
            external_id=outcome.external_id,
            timely_response=not outcome.timely_response,
            sent_to_company_at=outcome.sent_to_company_at,
        )
        for outcome in cfpb_outcomes()
    ]
    write_corpus(
        other, EVALUATION_SOURCE, cfpb_records(), flipped, storage.write_cfpb_outcome_partition
    )
    second = probe().run_probe(
        corpus_root=other,
        artifact_root=report_path.parent / "artifacts2",
        report_path=report_path.parent / "report2.json",
    )
    assert second.baseline_prior == first.baseline_prior


# --- K. metrics ---------------------------------------------------------------------------


REQUIRED_METRICS = (
    "pr_auc",
    "roc_auc",
    "minority_precision",
    "minority_recall",
    "minority_f1",
    "minority_count",
    "base_rate",
    "baseline_prior",
)


@pytest.mark.parametrize("evaluation", ("in_domain", "cross_domain"))
def test_every_required_metric_is_present(corpora, artifact_root, report_path, evaluation):
    module = probe()
    report = run(corpora, artifact_root, report_path)
    key = module.IN_DOMAIN if evaluation == "in_domain" else module.CROSS_DOMAIN
    for name in REQUIRED_METRICS:
        assert name in report.metrics[key], name


def test_pr_auc_carries_its_baseline(corpora, artifact_root, report_path):
    """§5.4: no figure is ever published alone."""
    module = probe()
    report = run(corpora, artifact_root, report_path)
    for key in (module.IN_DOMAIN, module.CROSS_DOMAIN):
        assert "majority" in report.metrics[key]["pr_auc"].baselines


def test_roc_auc_is_secondary_not_the_headline(corpora, artifact_root, report_path):
    run(corpora, artifact_root, report_path)
    assert probe().HEADLINE == "pr_auc"


def test_the_two_headline_figures_are_never_a_before_after_pair(
    corpora, artifact_root, report_path
):
    """§5.4: base rates differ ~27x, so a delta between them would be arithmetic."""
    report = run(corpora, artifact_root, report_path)
    serialized = report.as_json()
    for banned in ("improvement", "delta", "before_after", "degradation", "drop_from"):
        assert banned not in serialized


def test_the_base_rate_is_recorded_per_evaluation(corpora, artifact_root, report_path):
    module = probe()
    report = run(corpora, artifact_root, report_path)
    keys = (module.IN_DOMAIN, module.CROSS_DOMAIN)
    rates = {key: report.metrics[key]["base_rate"] for key in keys}
    for rate in rates.values():
        assert 0.0 < rate < 1.0


# --- L. distribution shift ------------------------------------------------------------------


def test_every_feature_carries_all_nine_quantiles(corpora, artifact_root, report_path):
    report = run(corpora, artifact_root, report_path)
    shift = report.feature_distribution_shift
    assert set(shift) == set(FEATURE_NAMES)
    for name in FEATURE_NAMES:
        assert set(shift[name]["source_training"]) == set(QUANTILE_POINTS), name
        assert set(shift[name]["evaluation"]) == set(QUANTILE_POINTS), name


def test_every_feature_carries_both_percentage_outside_measures(
    corpora, artifact_root, report_path
):
    report = run(corpora, artifact_root, report_path)
    for name in FEATURE_NAMES:
        entry = report.feature_distribution_shift[name]
        for measure in SHIFT_MEASURES:
            assert measure in entry, (name, measure)
            assert 0.0 <= entry[measure] <= 100.0


def test_the_out_of_range_flag_is_present_per_feature(corpora, artifact_root, report_path):
    report = run(corpora, artifact_root, report_path)
    for name in FEATURE_NAMES:
        assert isinstance(report.feature_distribution_shift[name]["out_of_range"], bool)


def test_text_length_shift_is_reported_not_normalised_away(corpora, artifact_root, report_path):
    """§5.4's expected finding is published, never rescued by rescaling."""
    report = run(corpora, artifact_root, report_path)
    entry = report.feature_distribution_shift["text_length"]
    assert entry["out_of_range"] is True
    assert entry["pct_outside_source_range"] > 0.0


def test_the_quantiles_describe_the_right_populations(corpora, artifact_root, report_path):
    """The source column is the 311 training set, not the evaluation set."""
    report = run(corpora, artifact_root, report_path)
    lengths = [len(record.text) for record in nyc_records()]
    entry = report.feature_distribution_shift["text_length"]["source_training"]
    assert min(lengths) <= entry["min"] <= max(lengths)


# --- M. framing and classification -----------------------------------------------------------


def test_all_six_framing_facts_are_present(corpora, artifact_root, report_path):
    report = run(corpora, artifact_root, report_path)
    for key in FRAMING_FACT_KEYS:
        assert key in report.framing_facts, key


def test_the_targets_are_declared_non_equivalent(corpora, artifact_root, report_path):
    report = run(corpora, artifact_root, report_path)
    assert report.framing_facts["target_semantics_differ"] is True


def test_the_analysis_type_is_exploratory_robustness(corpora, artifact_root, report_path):
    report = run(corpora, artifact_root, report_path)
    assert report.framing_facts["analysis_type"] == (
        "exploratory robustness, not same-task transfer"
    )


def test_no_prohibited_transfer_wording_appears(corpora, artifact_root, report_path):
    """D19's binding prohibition, enforced on the output rather than on prose."""
    report = run(corpora, artifact_root, report_path)
    serialized = report.as_json().lower()
    for banned in PROHIBITED:
        assert banned not in serialized, banned


def test_the_prohibited_wordings_are_declared_by_the_module():
    declared = {phrase.lower() for phrase in probe().PROHIBITED_WORDINGS}
    for banned in PROHIBITED:
        assert banned in declared


@pytest.mark.parametrize("verdict,classification", sorted(VERDICTS.items()))
def test_the_result_classification_follows_the_cfpb_verdict(tmp_path, verdict, classification):
    """The §2.3 gate: the probe classifies its own result from Task 8's verdict."""
    root = build_corpora(
        tmp_path / f"corpus-{verdict}",
        diagnostic={
            "verdict": verdict,
            "reason": "fixture",
            "deltas": {"median_hour_delta": 0.0},
            "rule_thresholds": {"median_hour_delta": 6.0},
        },
    )
    report = probe().run_probe(
        corpus_root=root,
        artifact_root=tmp_path / f"artifacts-{verdict}",
        report_path=tmp_path / f"report-{verdict}.json",
    )
    assert report.result_classification == classification


def test_the_verdict_and_its_evidence_are_copied_into_the_report(
    corpora, artifact_root, report_path
):
    """Which branch fired, and why, without opening the manifest."""
    report = run(corpora, artifact_root, report_path)
    diagnostic = report.timestamp_diagnostic
    assert diagnostic["verdict"] == "supported_plausible_event_time"
    assert "deltas" in diagnostic
    assert "rule_thresholds" in diagnostic


def test_a_manifest_without_a_verdict_is_refused(tmp_path):
    """`result_classification` has no fallback value (contract §13)."""
    root = build_corpora(tmp_path / "corpus", diagnostic={"reason": "no verdict recorded"})
    with pytest.raises(Exception):
        probe().run_probe(
            corpus_root=root,
            artifact_root=tmp_path / "artifacts",
            report_path=tmp_path / "report.json",
        )


def test_the_interpretation_limits_are_carried_in_the_report(corpora, artifact_root, report_path):
    """The design cannot separate domain shift, target semantics and weak features."""
    report = run(corpora, artifact_root, report_path)
    assert report.interpretation_limits


# --- N. the artifact ----------------------------------------------------------------------------


def test_the_artifact_identity_is_the_frozen_one(corpora, artifact_root, report_path):
    report = run(corpora, artifact_root, report_path)
    assert report.metadata["model_name"] == MODEL_NAME
    assert report.metadata["model_version"] == MODEL_VERSION
    assert report.metadata["experiment_label"] == EXPERIMENT_LABEL


def test_the_module_declares_the_frozen_identity_constants():
    module = probe()
    assert module.MODEL_NAME == MODEL_NAME
    assert module.MODEL_VERSION == MODEL_VERSION
    assert module.EXPERIMENT_LABEL == EXPERIMENT_LABEL


def test_the_artifact_round_trips_through_load_artifact(corpora, artifact_root, report_path):
    from ml.training.artifacts import load_artifact

    report = run(corpora, artifact_root, report_path)
    loaded = load_artifact(report.artifact_path)
    assert loaded.feature_spec == TRANSFER_FEATURES_V1
    assert list(loaded.metadata["feature_spec"]) == list(FEATURE_NAMES)
    assert loaded.metadata["feature_spec_version"] == FEATURE_SPEC_VERSION


def test_the_artifact_is_not_the_primary_risk_model(corpora, artifact_root, report_path):
    """Distinct name and version so it cannot be confused at load time (§5.4)."""
    report = run(corpora, artifact_root, report_path)
    assert report.metadata["model_name"] != "nyc311_sla_risk"
    assert report.metadata["feature_spec_version"] != "risk_features_v1"


# --- O. persistence -------------------------------------------------------------------------------


def test_the_report_is_written_to_the_caller_supplied_path(corpora, artifact_root, report_path):
    """O7 follows Task 18's convention: a caller-supplied path, no repository default."""
    run(corpora, artifact_root, report_path)
    assert report_path.is_file()


def test_no_default_repository_report_path_is_invented():
    """Nothing is written inside the repository by default (D38's convention)."""
    signature = inspect.signature(probe().run_probe)
    default = signature.parameters["report_path"].default
    assert default is inspect.Parameter.empty or default is None


def test_the_diagnostics_are_not_smuggled_into_artifact_metadata(
    corpora, artifact_root, report_path
):
    """D35's schema is closed; Task 19 diagnostics live in the report instead (O7)."""
    report = run(corpora, artifact_root, report_path)
    for banned in (
        "feature_distribution_shift",
        "result_classification",
        "framing_facts",
        "timestamp_diagnostic",
    ):
        assert banned not in report.metadata, banned


def test_the_written_report_carries_every_required_element(corpora, artifact_root, report_path):
    import json

    run(corpora, artifact_root, report_path)
    written = json.loads(report_path.read_text(encoding="utf-8"))
    for key in (
        "source_training_population",
        "in_domain_evaluation_population",
        "cross_domain_evaluation_population",
        "feature_names",
        "estimator",
        "baseline_prior",
        "metrics",
        "feature_distribution_shift",
        "framing_facts",
        "result_classification",
        "timestamp_diagnostic",
        "interpretation_limits",
    ):
        assert key in written, key


def test_the_written_report_states_the_feature_order(corpora, artifact_root, report_path):
    import json

    run(corpora, artifact_root, report_path)
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert list(written["feature_names"]) == list(FEATURE_NAMES)


def test_the_written_report_records_the_estimator(corpora, artifact_root, report_path):
    import json

    run(corpora, artifact_root, report_path)
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["estimator"]["random_state"] == SEED
    assert written["estimator"]["max_iter"] == ESTIMATOR_PARAMS["max_iter"]


# --- P. dependencies and boundaries ---------------------------------------------------


def test_the_probe_adds_no_dependency_outside_the_pinned_tiers():
    source = Path(probe().__file__).read_text(encoding="utf-8")
    for banned in ("import torch", "import tensorflow", "import xgboost", "import lightgbm"):
        assert banned not in source


def test_the_probe_imports_no_django():
    """`ml/training` stays Django-independent (plan §99)."""
    source = Path(probe().__file__).read_text(encoding="utf-8")
    assert "import django" not in source
    assert "from django" not in source


def test_the_probe_never_reads_the_raw_cache(corpora, artifact_root, report_path):
    """Both corpora are read through the manifest-backed loaders (§2.7, D27)."""
    source = Path(probe().__file__).read_text(encoding="utf-8")
    assert "load_corpus" in source
    assert "raw" not in source.replace("raw_root", "").replace("# raw", "")
