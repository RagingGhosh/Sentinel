"""Task 19's reduced-feature cross-domain cross-target robustness probe (§5.4, D19).

One run, end to end::

    load_corpus("nyc311") + load_outcomes("nyc311")   ->  window  ->  temporal_split
      ->  Task 13 thresholds fitted on TRAIN  ->  three structural features
      ->  one HistGradientBoostingClassifier, fitted once on 311 TRAIN
      ->  scored on 311 TEST          (in-domain reference)
      ->  scored on CFPB TEST         (cross-domain evaluation)
      ->  one frozen source-training baseline for both
      ->  feature distribution shift  ->  provenance verdict  ->  artifact + report

**What this is not.** It is not the primary risk model scored on other data.
Section 5.4 establishes that the five-feature model *cannot* be scored on CFPB at
all: two of its five features need resolution times CFPB does not publish, and
§4.3 prohibits substituting a responsiveness flag for a breach rate. The
impossibility is structural.

**The binding prohibition (§5.4, D19).** Nothing this module produces describes
the probe as evidence that the 311 SLA-risk model carries over to the CFPB task.
`PROHIBITED_WORDINGS` lists the phrasings that are defects rather than stylistic
choices, and the report is checked against it.

**Two non-equivalent targets, and one derived boolean at the metric boundary.**
The source target is `nyc311_sla_breach`; the evaluation target is
`cfpb_timely_response`. §4 defines them as non-equivalent constructs and §4.3
forbids combining them under a single name. So the model scores P(adverse), and
"adverse" is a breach in the source domain and *not* timely in the evaluation
domain. CFPB records enter every metric as `not timely_response`, computed where
the metric is called and stored in no field, column or corpus schema. That
correspondence is an interpretive choice, stated in the report, not a fact about
the data.

**Open 311 requests are separated from the labelled population, not removed**
(O6, D37.3). An open request has no resolution time, so it can carry no breach
label, but it is a real request: it reaches the window, the split and the
structural features, and leaves only at the labelled-population step. Nothing
here coerces an unresolved outcome to `False`, which is the refusal D33 built
into `apply_thresholds`.

**Nothing is recomputed that another task owns.** The window and the split
fractions are Task 17's frozen inputs, applied here rather than imported from its
module, so the in-domain reference and the primary model's in-domain figure rest
on the same population (contract §3, finding I6). The per-type p75, its global
fallback and the hundred-eligible-observation rule are Task 13's; the feature
matrix is Task 12's; every metric and the one baseline are Task 14's. No
threshold arithmetic, no metric arithmetic and no baseline fitting happens here.

**One fit, no tuning, no banding.** The validation period is produced by the
split and then left unused: this probe selects nothing, so there is nothing for
validation to decide. No feature scaling, no imputation, no sampling correction
and no class weighting — §5.4 requires the class imbalance to be reported rather
than engineered away.

Training-side and Django-independent: invoked as ``python -m``, never as a
management command, and nothing here is imported by serving.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from ingest.manifest import CorpusManifest, load_cfpb_outcomes, load_corpus, load_outcomes
from ingest.schema import CFPBOutcome, CorpusRecord, NYC311Outcome
from ml.training.artifacts import write_artifact
from ml.training.features import TRANSFER_FEATURES_V1, FeatureSpec, build_features
from ml.training.labels import FrozenThresholds, apply_thresholds, fit_thresholds
from ml.training.metrics import (
    MajorityBaseline,
    MinorityReport,
    ScoreResult,
    StratifiedBaseline,
    majority_baseline,
    minority_report,
    pr_auc,
    roc_auc,
    stratified_baseline,
)
from ml.training.splits import DEFAULT_FRACTIONS, Period, TemporalSplit, temporal_split
from ml.training.thresholds import MIN_ELIGIBLE_OBSERVATIONS, THRESHOLD_QUANTILE

SOURCE = "nyc311"
"""The training domain. The probe fits here and nowhere else."""

EVALUATION_SOURCE = "cfpb"
"""The cross-domain evaluation domain. Nothing is ever fitted on it (O5)."""

MODEL_NAME = "xdomain_xtarget_probe"
MODEL_VERSION = "xdomain_xtarget_probe_v1"
EXPERIMENT_LABEL = "reduced-feature cross-domain cross-target robustness probe"
"""O3's artifact identity. Distinct from the primary model's so the two can never
be confused at load time (§5.4)."""

WINDOW_START = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
WINDOW_END = datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC)
"""Task 17's window (D37.2), reused here as a frozen input rather than imported."""

SEED = 17
"""O2's ``random_state``, and the seed the one stratified comparison is taken at."""

FEATURE_SPEC: FeatureSpec = TRANSFER_FEATURES_V1
"""The three features computable in both corpora, versioned independently of the
primary five so the probe can never be read as the primary model (§2, D16)."""

ROSTER: tuple[bool, bool] = (False, True)
POSITIVE_LABEL = True
"""One roster and one positive label for both evaluations, which is what lets a
single frozen baseline score both (§4, §7)."""

SOURCE_TARGET = "nyc311_sla_breach"
EVALUATION_TARGET = "cfpb_timely_response"

IN_DOMAIN = "nyc311_test_in_domain"
CROSS_DOMAIN = "cfpb_test_cross_domain"
"""The two evaluation keys. There are exactly two, each scored once (O5)."""

HEADLINE = "pr_auc"
"""§5.5 and D7: ROC-AUC is secondary and is prohibited as a headline at these
class ratios."""

QUANTILES: tuple[tuple[str, float], ...] = (
    ("min", 0.0),
    ("p01", 0.01),
    ("p05", 0.05),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p95", 0.95),
    ("p99", 0.99),
    ("max", 1.0),
)
"""§8's nine points, reported for every feature in both populations."""

ESTIMATOR_PARAMETERS = (
    "learning_rate",
    "max_iter",
    "max_leaf_nodes",
    "max_depth",
    "min_samples_leaf",
    "l2_regularization",
    "early_stopping",
    "class_weight",
    "random_state",
)
"""The nine O2 fixes. Everything else takes the pinned scikit-learn default, and
the report records the version that supplied them."""

RESULT_CLASSIFICATION: Mapping[str, str] = MappingProxyType(
    {
        "strongly_suspicious_load_timestamp": "non-informative / diagnostic",
        "suspicious_insufficient_evidence": "substantive_with_stated_caveat",
        "supported_plausible_event_time": "substantive",
    }
)
"""§9's table, keyed by Task 8's CFPB provenance verdict. There is no fallback
value: an unknown or absent verdict is a refusal, because with `submitted_hour`
unusable the probe's figures would describe nothing and the report must say so."""

PROHIBITED_WORDINGS: tuple[str, ...] = (
    "transfers to",
    "generalises to",
    "generalizes to",
    "works on CFPB",
)
"""D19's binding prohibition, declared here so it can be checked against output
rather than trusted to prose."""

ANALYSIS_TYPE = "exploratory robustness, not same-task transfer"

INTERPRETATION_LIMITS: tuple[str, ...] = (
    "A reduced-feature model that scores poorly here may be failing because of "
    "domain shift, because the target means something different, because three "
    "weak features are insufficient, or any combination; this design cannot "
    "separate those causes, and none of them is established by this result.",
    "The in-domain figure is the reduced-feature in-domain reference performance, "
    "not a ceiling: a ceiling would imply the cross-domain figure measures the "
    "same quantity less well, and it does not.",
    "The two headline figures are two separate evaluations of one fitted model "
    "against two different targets on two different populations. Each is "
    "interpretable only as lift over its own baseline, beside its own base rate.",
    "The measured base rates differ by roughly an order of magnitude or more, and "
    "Average Precision is base-rate dependent, so a lower cross-domain figure is "
    "expected arithmetic before any question of distribution shift.",
    "The feature distribution shift block is evidence and context. It cannot "
    "upgrade the result classification, which follows the provenance verdict "
    "alone (contract §9).",
    "A poor result is published rather than buried, and is an observation about "
    "this probe rather than a measurement of either operational task.",
)
"""§12's limits, carried in the report structure rather than left to a reader."""


# --- the estimator -------------------------------------------------------------------


def build_estimator(seed: int) -> HistGradientBoostingClassifier:
    """O2's frozen recipe: Task 17's configuration, as a decision taken here.

    ``early_stopping=False`` is explicit and load-bearing: the estimator's
    ``'auto'`` default carves an internal **random** validation split out of the
    training rows, which is exactly the non-temporal evaluation §6 forbids.
    ``class_weight=None`` is equally deliberate — §5.4 reports the class
    imbalance rather than correcting for it.
    """
    return HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        early_stopping=False,
        class_weight=None,
        random_state=seed,
    )


class ProbeModel:
    """What the artifact stores: the fitted estimator with the roster bound to it.

    `predict_proba` returns its columns in `ROSTER` order whatever order the
    estimator learned, so one order governs the score matrix and every metric.
    `predict` is the estimator's own argmax over those two columns and selects no
    operating point: the probe publishes ranking metrics and bands nothing.
    """

    def __init__(self, estimator: HistGradientBoostingClassifier, roster: Sequence[bool]) -> None:
        self.estimator = estimator
        self.classes_ = np.asarray(tuple(roster), dtype=object)
        learned = [bool(value) for value in estimator.classes_]
        missing = [label for label in roster if bool(label) not in learned]
        if missing:
            raise ValueError(
                f"the fitted estimator learned classes {learned!r} and so cannot score "
                f"the roster label(s) {missing!r}; the training population holds only "
                "one class, and a metric over it would describe nothing (D34)"
            )
        self._columns = [learned.index(bool(label)) for label in roster]

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(X))[:, self._columns]

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# --- what one run produced -------------------------------------------------------------


@dataclass(frozen=True)
class LabelledPeriod:
    """One 311 period's resolved requests, labelled by the frozen thresholds."""

    records: tuple[CorpusRecord, ...]
    labels: np.ndarray
    positions: tuple[int, ...]
    """Where these rows sit in the period's full population, open requests included."""


@dataclass(frozen=True)
class EvaluatedPopulation:
    """The CFPB test-period records carrying a persisted outcome, with the
    adverse boolean each one enters the metrics as."""

    records: tuple[CorpusRecord, ...]
    labels: np.ndarray
    without_outcome: tuple[str, ...]
    """Test-period records the sidecar holds no outcome for. Contract §3 defines
    the evaluation population as the ones it does hold, so these are ineligible
    rather than dropped, and their count is published."""


@dataclass(frozen=True)
class ProbeReport:
    """One run's frozen report: everything §11 requires, serializable as JSON.

    The artifact carries the model and D35's closed metadata; this object carries
    the Task 19 diagnostics D35 deliberately has no field for (O7).
    """

    source_training_population: Mapping[str, Any]
    in_domain_evaluation_population: Mapping[str, Any]
    cross_domain_evaluation_population: Mapping[str, Any]
    feature_names: tuple[str, ...]
    feature_spec_version: str
    estimator: Mapping[str, Any]
    baseline_prior: float
    metrics: Mapping[str, Mapping[str, Any]]
    feature_distribution_shift: Mapping[str, Mapping[str, Any]]
    framing_facts: Mapping[str, Any]
    timestamp_diagnostic: Mapping[str, Any]
    """Task 8's CFPB verdict with the evidence that produced it, copied verbatim
    so which branch fired and why is visible without opening the manifest."""
    result_classification: str
    interpretation_limits: tuple[str, ...]
    prohibited_wordings: tuple[str, ...]
    dependency_versions: Mapping[str, str]
    metadata: Mapping[str, Any]
    """The artifact metadata this run published, exactly as written."""
    artifact_path: Path
    report_path: Path

    def as_json(self) -> str:
        """The report as strict JSON text — what `report_path` receives.

        No filesystem path is included: a path is the caller's context, not a
        finding, and a report that travels should not carry one machine's layout.
        """
        return json.dumps(_payload(self), indent=2, ensure_ascii=False, allow_nan=False) + "\n"


# --- the run ----------------------------------------------------------------------------


def run_probe(
    *,
    corpus_root: Path,
    artifact_root: Path,
    report_path: Path,
    git_sha: str | None = None,
) -> ProbeReport:
    """Fit once on 311 TRAIN, score 311 TEST and CFPB TEST, publish both.

    ``report_path`` is required and has no default: following the convention D38
    established, the report is returned as a frozen object and serialized to a
    caller-supplied path, and nothing is written inside the repository.

    Raises ``ValueError`` for an empty window, an empty labelled population, a
    non-finite feature value, a corpus whose records and outcomes do not pair, or
    a CFPB manifest carrying no usable provenance verdict.
    """
    source_manifest, split, periods, outcomes_of = _source_population(corpus_root)

    thresholds = fit_thresholds(
        periods[Period.TRAIN], outcomes_of[Period.TRAIN], min_eligible=MIN_ELIGIBLE_OBSERVATIONS
    )

    # Open requests reach the window, the split and the structural features, and
    # leave only here: with no resolution time they can carry no label, so they
    # can join no fit and no published figure (O6).
    matrices = {
        period: _finite(build_features(periods[period], None, FEATURE_SPEC), period.value)
        for period in (Period.TRAIN, Period.TEST)
    }
    labelled = {
        period: _labelled(periods[period], outcomes_of[period], thresholds)
        for period in (Period.TRAIN, Period.TEST)
    }
    labelled_matrices = {
        period: matrices[period][list(labelled[period].positions)]
        for period in (Period.TRAIN, Period.TEST)
    }
    for period in (Period.TRAIN, Period.TEST):
        if not len(labelled[period].records):
            raise ValueError(
                f"the {SOURCE} {period.value} period holds no labelled request; a metric "
                "over no data describes nothing (D34)"
            )

    train_labels = [bool(value) for value in labelled[Period.TRAIN].labels]

    estimator = build_estimator(SEED)
    estimator.fit(labelled_matrices[Period.TRAIN], labelled[Period.TRAIN].labels)
    model = ProbeModel(estimator, ROSTER)

    # Frozen before either evaluation, from the 311 training labels alone, and
    # used unchanged for both: the probe's training labels are the 311 ones, in
    # both evaluations (O4, D34).
    majority = majority_baseline(train_labels, ROSTER)
    baseline_prior = float(majority.prior[ROSTER.index(POSITIVE_LABEL)])

    evaluation_manifest, evaluated = _evaluation_population(corpus_root)
    cross_matrix = _finite(build_features(evaluated.records, None, FEATURE_SPEC), CROSS_DOMAIN)

    diagnostic = _diagnostic(evaluation_manifest)
    classification = RESULT_CLASSIFICATION[diagnostic["verdict"]]

    metrics = {
        IN_DOMAIN: _evaluation_metrics(
            labelled[Period.TEST].labels,
            labelled_matrices[Period.TEST],
            model,
            majority=majority,
            train_labels=train_labels,
            baseline_prior=baseline_prior,
        ),
        CROSS_DOMAIN: _evaluation_metrics(
            evaluated.labels,
            cross_matrix,
            model,
            majority=majority,
            train_labels=train_labels,
            baseline_prior=baseline_prior,
        ),
    }

    shift = _distribution_shift(labelled_matrices[Period.TRAIN], cross_matrix)

    metadata = _metadata(
        manifest=source_manifest,
        evaluation_manifest=evaluation_manifest,
        split=split,
        evaluation_split=evaluated,
        metrics=metrics,
        baseline_prior=baseline_prior,
        estimator=estimator,
        git_sha=git_sha if git_sha is not None else _git_sha(),
    )
    artifact_path = Path(artifact_root) / source_manifest.source_slug / MODEL_NAME / MODEL_VERSION
    write_artifact(model, metadata, artifact_path)

    report = ProbeReport(
        source_training_population=_source_block(
            split, periods[Period.TRAIN], labelled[Period.TRAIN], thresholds, source_manifest
        ),
        in_domain_evaluation_population=_in_domain_block(
            split, periods[Period.TEST], labelled[Period.TEST], source_manifest
        ),
        cross_domain_evaluation_population=_cross_domain_block(evaluated, evaluation_manifest),
        feature_names=FEATURE_SPEC.names,
        feature_spec_version=FEATURE_SPEC.version,
        estimator=_estimator_block(estimator),
        baseline_prior=baseline_prior,
        metrics=MappingProxyType(metrics),
        feature_distribution_shift=shift,
        framing_facts=_framing_facts(),
        timestamp_diagnostic=MappingProxyType(dict(diagnostic)),
        result_classification=classification,
        interpretation_limits=INTERPRETATION_LIMITS,
        prohibited_wordings=PROHIBITED_WORDINGS,
        dependency_versions=_dependency_versions(),
        metadata=MappingProxyType(metadata),
        artifact_path=artifact_path,
        report_path=Path(report_path),
    )
    _write_report(report)
    return report


# --- the two populations -----------------------------------------------------------------


def _source_population(
    corpus_root: Path,
) -> tuple[
    CorpusManifest,
    TemporalSplit,
    dict[Period, tuple[CorpusRecord, ...]],
    dict[Period, tuple[NYC311Outcome, ...]],
]:
    """The 311 corpus, windowed and split on Task 17's frozen inputs (§3, I6).

    `temporal_split` is deterministic given the same timestamps and fractions, so
    recomputing it over the same windowed population reproduces Task 17's cut
    dates exactly without importing that experiment.
    """
    manifest, record_stream = load_corpus(SOURCE, root=corpus_root)
    _, outcome_stream = load_outcomes(SOURCE, root=corpus_root)
    records = tuple(record_stream)
    by_id = _joined(records, tuple(outcome_stream))

    windowed = tuple(
        record for record in records if WINDOW_START <= record.submitted_at <= WINDOW_END
    )
    if not windowed:
        raise ValueError(
            f"{SOURCE} corpus at {corpus_root} holds no record inside "
            f"{WINDOW_START.isoformat()} .. {WINDOW_END.isoformat()}"
        )

    split = temporal_split([record.submitted_at for record in windowed], DEFAULT_FRACTIONS)
    periods = {
        period: tuple(
            record for record in windowed if split.period_of(record.submitted_at) is period
        )
        for period in Period
    }
    outcomes_of = {
        period: tuple(by_id[record.external_id] for record in rows)
        for period, rows in periods.items()
    }
    return manifest, split, periods, outcomes_of


def _evaluation_population(corpus_root: Path) -> tuple[CorpusManifest, EvaluatedPopulation]:
    """The CFPB Task 16 test population carrying persisted outcomes (§3, O1).

    Task 16 splits the whole CFPB corpus at `DEFAULT_FRACTIONS` and windows
    nothing, so the same cut reproduces its periods. No record that tuned Task
    16's abstention threshold appears here: that threshold was selected on CFPB
    validation, and only the test period is evaluated.

    The adverse boolean is `not timely_response`, derived here at the metric
    boundary. It is written into no field, no column and no corpus schema, so
    §4.3's prohibition on combining the two targets holds.
    """
    manifest, record_stream = load_corpus(EVALUATION_SOURCE, root=corpus_root)
    _, outcome_stream = load_cfpb_outcomes(root=corpus_root)
    records = tuple(record_stream)
    outcomes = _cfpb_outcomes_by_id(records, tuple(outcome_stream))

    split = temporal_split([record.submitted_at for record in records], DEFAULT_FRACTIONS)
    test_period = tuple(
        record for record in records if split.period_of(record.submitted_at) is Period.TEST
    )
    evaluated = tuple(record for record in test_period if record.external_id in outcomes)
    if not evaluated:
        raise ValueError(
            f"the {EVALUATION_SOURCE} test period holds no record with a persisted "
            "outcome; a metric over no data describes nothing (D34)"
        )

    labels = np.array(
        [not outcomes[record.external_id].timely_response for record in evaluated], dtype=bool
    )
    return manifest, EvaluatedPopulation(
        records=evaluated,
        labels=labels,
        without_outcome=tuple(
            record.external_id for record in test_period if record.external_id not in outcomes
        ),
    )


def _joined(
    records: Sequence[CorpusRecord], outcomes: Sequence[NYC311Outcome]
) -> dict[str, NYC311Outcome]:
    """A total bijection between 311 records and outcomes, keyed by `external_id`.

    Nothing is dropped or reordered to force a match: a missing, extra or repeated
    identity is a corpus that does not describe what it claims to, and quietly
    discarding the odd row would change the population underneath every published
    figure (D37.1).
    """
    seen: dict[str, NYC311Outcome] = {}
    for outcome in outcomes:
        if outcome.external_id in seen:
            raise ValueError(f"the outcome sidecar repeats external_id {outcome.external_id!r}")
        seen[outcome.external_id] = outcome

    identifiers = [record.external_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"the {SOURCE} corpus repeats an external_id")
    missing = [external_id for external_id in identifiers if external_id not in seen]
    if missing:
        raise ValueError(f"{len(missing)} record(s) have no outcome, starting with {missing[0]!r}")
    extra = sorted(set(seen) - set(identifiers))
    if extra:
        raise ValueError(f"{len(extra)} outcome(s) match no record, starting with {extra[0]!r}")
    return seen


def _cfpb_outcomes_by_id(
    records: Sequence[CorpusRecord], outcomes: Sequence[CFPBOutcome]
) -> dict[str, CFPBOutcome]:
    """CFPB outcomes keyed by identity, refusing a sidecar that cannot be joined.

    A repeated identity, on either side, or an outcome matching no record, means
    the two do not describe one another — an impossible join rather than a
    tolerable gap. A record the sidecar holds no outcome for is the one permitted
    case: §3 defines the evaluation population as the records whose outcome is
    present, and the count of the others is published rather than swallowed.

    The corpus-side duplicate check is not redundant with the sidecar one. Nothing
    in `ingest` enforces identity uniqueness within a corpus, so a repeated record
    would join twice to one outcome and enter the evaluation population twice,
    inflating every published cross-domain figure while the sidecar itself stayed
    consistent. The 311 side already refuses this; both sides now fail closed the
    same way.
    """
    seen: dict[str, CFPBOutcome] = {}
    for outcome in outcomes:
        if outcome.external_id in seen:
            raise ValueError(
                f"the {EVALUATION_SOURCE} outcome sidecar repeats external_id "
                f"{outcome.external_id!r}"
            )
        seen[outcome.external_id] = outcome

    identifiers = [record.external_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"the {EVALUATION_SOURCE} corpus repeats an external_id")

    extra = sorted(set(seen) - set(identifiers))
    if extra:
        raise ValueError(
            f"{len(extra)} {EVALUATION_SOURCE} outcome(s) match no record, starting with "
            f"{extra[0]!r}; the sidecar does not describe this corpus"
        )
    return seen


def _labelled(
    records: Sequence[CorpusRecord],
    outcomes: Sequence[NYC311Outcome],
    thresholds: FrozenThresholds,
) -> LabelledPeriod:
    """The resolved subset of one 311 period, labelled by the frozen thresholds.

    An open request is left out rather than labelled: `apply_thresholds` refuses
    an unresolved outcome precisely so that no caller can turn "we could not
    tell" into "resolved in time", and this function never catches that refusal —
    it simply never offers it an open row (O6, D33).
    """
    positions = tuple(
        index for index, outcome in enumerate(outcomes) if outcome.resolution_hours is not None
    )
    resolved_records = tuple(records[index] for index in positions)
    resolved_outcomes = tuple(outcomes[index] for index in positions)
    return LabelledPeriod(
        records=resolved_records,
        labels=apply_thresholds(thresholds, resolved_records, resolved_outcomes),
        positions=positions,
    )


def _finite(matrix: np.ndarray, where: str) -> np.ndarray:
    """The three features are finite by construction; a non-finite value is a defect.

    Contract §13: a non-finite feature value indicates an upstream problem and
    raises rather than being filled in, because filling one would publish a
    figure computed from a number nothing measured.
    """
    if matrix.size and not np.isfinite(matrix).all():
        offending = [
            FEATURE_SPEC.names[column]
            for column in range(matrix.shape[1])
            if not np.isfinite(matrix[:, column]).all()
        ]
        raise ValueError(
            f"{where}: non-finite feature value(s) in {', '.join(offending)}; the three "
            "structural features are finite by construction, so this is an upstream defect"
        )
    return matrix


# --- metrics -------------------------------------------------------------------------------


def _evaluation_metrics(
    labels: np.ndarray,
    matrix: np.ndarray,
    model: ProbeModel,
    *,
    majority: MajorityBaseline,
    train_labels: Sequence[bool],
    baseline_prior: float,
) -> Mapping[str, Any]:
    """§7's set for one evaluation, every figure beside the frozen source baseline.

    PR-AUC is the headline and ROC-AUC secondary, both ranking metrics carrying
    the majority baseline only, because D34 gives a single seeded label
    assignment no ranking meaning. The minority figures come from Task 14's
    `minority_report` against the estimator's own argmax: no operating point is
    selected here, and none is published.
    """
    truth = [bool(value) for value in labels]
    scores = model.predict_proba(matrix)
    predicted = [bool(value) for value in model.predict(matrix)]
    stratified: StratifiedBaseline = stratified_baseline(
        train_labels, ROSTER, len(truth), seed=SEED
    )
    report: MinorityReport = minority_report(
        truth,
        predicted,
        ROSTER,
        POSITIVE_LABEL,
        majority=majority,
        stratified=stratified,
    )
    return MappingProxyType(
        {
            "headline": HEADLINE,
            "pr_auc": pr_auc(truth, scores, ROSTER, POSITIVE_LABEL, majority=majority),
            "roc_auc": roc_auc(truth, scores, ROSTER, POSITIVE_LABEL, majority=majority),
            "minority_precision": float(report.model.precision),
            "minority_recall": float(report.model.recall),
            "minority_f1": float(report.model.f1),
            "minority_count": int(report.support),
            "minority_baselines": MappingProxyType(dict(report.baselines)),
            "base_rate": float(np.count_nonzero(labels) / len(truth)),
            "baseline_prior": baseline_prior,
            "population": len(truth),
        }
    )


# --- distribution shift ----------------------------------------------------------------------


def _distribution_shift(
    source: np.ndarray, evaluation: np.ndarray
) -> Mapping[str, Mapping[str, Any]]:
    """§8's block: nine quantiles per feature in both populations, and the spread.

    The source column is the 311 **training** population the model was fitted on,
    and the evaluation column is the CFPB test population it was scored on.
    `text_length` is expected to flag, and that finding is published: no ad-hoc
    rescaling, standardisation or per-domain transformation is applied to rescue
    the comparison (§8).
    """
    shift: dict[str, Mapping[str, Any]] = {}
    for column, name in enumerate(FEATURE_SPEC.names):
        source_column = source[:, column]
        evaluation_column = evaluation[:, column]
        source_quantiles = _quantiles(source_column)
        low, high = source_quantiles["min"], source_quantiles["max"]
        lower, upper = source_quantiles["p25"], source_quantiles["p75"]
        evaluation_quantiles = _quantiles(evaluation_column)
        total = evaluation_column.size
        shift[name] = MappingProxyType(
            {
                "source_training": source_quantiles,
                "evaluation": evaluation_quantiles,
                "pct_outside_source_range": _percentage(
                    (evaluation_column < low) | (evaluation_column > high), total
                ),
                "pct_outside_source_iqr": _percentage(
                    (evaluation_column < lower) | (evaluation_column > upper), total
                ),
                "out_of_range": bool(
                    evaluation_quantiles["p50"] < lower or evaluation_quantiles["p50"] > upper
                ),
            }
        )
    return MappingProxyType(shift)


def _quantiles(values: np.ndarray) -> Mapping[str, float]:
    """The nine points of `QUANTILES`, by the same linear interpolation between
    adjacent order statistics that Task 13's percentile primitive defines."""
    points = np.quantile(values, [point for _, point in QUANTILES], method="linear")
    return MappingProxyType(
        {name: float(value) for (name, _), value in zip(QUANTILES, points, strict=True)}
    )


def _percentage(mask: np.ndarray, total: int) -> float:
    return float(100.0 * np.count_nonzero(mask) / total) if total else 0.0


# --- framing, provenance and the report payload ------------------------------------------------


def _framing_facts() -> Mapping[str, Any]:
    """§10's six facts, emitted as report structure rather than only as prose."""
    return MappingProxyType(
        {
            "source_domain": SOURCE,
            "source_target": MappingProxyType(
                {
                    "name": SOURCE_TARGET,
                    "rule": (
                        "resolution strictly slower than the request type's own p"
                        f"{int(THRESHOLD_QUANTILE * 100)} of resolution hours, fitted on the "
                        "311 training period only and then frozen"
                    ),
                }
            ),
            "evaluation_domain": EVALUATION_SOURCE,
            "evaluation_target": MappingProxyType(
                {
                    "name": EVALUATION_TARGET,
                    "measures": (
                        "whether the company replied inside CFPB's published response "
                        "window; a fact about a reply, not a measure of resolution"
                    ),
                }
            ),
            "feature_set": MappingProxyType(
                {
                    "names": list(FEATURE_SPEC.names),
                    "version": FEATURE_SPEC.version,
                    "why_the_five_feature_set_was_unusable": (
                        "two of the primary model's five features are derived from "
                        "resolution times CFPB does not publish, and substituting a "
                        "responsiveness flag for a breach rate is prohibited (§4.3, §5.4)"
                    ),
                }
            ),
            "target_semantics_differ": True,
            "polarity_mapping": MappingProxyType(
                {
                    "model_output": "probability of the adverse class",
                    "adverse_in_source": f"{SOURCE_TARGET} == True",
                    "adverse_in_evaluation": f"{EVALUATION_TARGET} == False",
                    "status": (
                        "an interpretive choice stated here, not a fact about the data, "
                        f"and not a claim that {SOURCE_TARGET} and {EVALUATION_TARGET} "
                        "measure the same thing (§4, §4.3)"
                    ),
                }
            ),
            "analysis_type": ANALYSIS_TYPE,
        }
    )


def _diagnostic(manifest: CorpusManifest) -> Mapping[str, Any]:
    """Task 8's CFPB provenance verdict, refused rather than defaulted (§9, §13).

    The verdict gates `result_classification` downward only: the field-delta rule
    Task 8 applied is what decided it, and this module copies that decision and
    its evidence without re-deciding. A histogram shape can raise doubt there; it
    can never establish artifact status, and nothing here reads one.
    """
    diagnostic = manifest.timestamp_diagnostic
    verdict = diagnostic.get("verdict") if isinstance(diagnostic, Mapping) else None
    if verdict is None:
        raise ValueError(
            f"the {EVALUATION_SOURCE} manifest records no timestamp_diagnostic verdict; "
            "result_classification has no fallback value, so the probe cannot classify "
            "its own result (contract §9, §13)"
        )
    if verdict not in RESULT_CLASSIFICATION:
        raise ValueError(
            f"the {EVALUATION_SOURCE} manifest records the unknown timestamp_diagnostic "
            f"verdict {verdict!r}; the three §9 verdicts are "
            f"{', '.join(sorted(RESULT_CLASSIFICATION))}"
        )
    return diagnostic


def _source_block(
    split: TemporalSplit,
    period_records: Sequence[CorpusRecord],
    labelled: LabelledPeriod,
    thresholds: FrozenThresholds,
    manifest: CorpusManifest,
) -> Mapping[str, Any]:
    """The population the one fit used, and the boundaries that produced it."""
    positive = int(np.count_nonzero(labelled.labels))
    return MappingProxyType(
        {
            "domain": SOURCE,
            "target": SOURCE_TARGET,
            "period": Period.TRAIN.value,
            "corpus_id": manifest.corpus_id,
            "window": MappingProxyType(
                {"start": WINDOW_START.isoformat(), "end": WINDOW_END.isoformat()}
            ),
            "train_end": split.train_end.isoformat(),
            "val_end": split.val_end.isoformat(),
            "split_counts": MappingProxyType(
                {period.value: split.counts[period] for period in Period}
            ),
            "requested_fractions": MappingProxyType(
                {period.value: split.requested_fractions[period] for period in Period}
            ),
            "period_record_count": len(period_records),
            "count": len(labelled.records),
            "unlabelled_open_count": len(period_records) - len(labelled.records),
            "positive_count": positive,
            "positive_rate": float(positive / len(labelled.records)),
            "thresholds": MappingProxyType(
                {
                    "percentile": THRESHOLD_QUANTILE,
                    "min_eligible": MIN_ELIGIBLE_OBSERVATIONS,
                    "per_type_count": len(thresholds.per_type),
                    "fallback_type_count": int(thresholds.fallback_type_count),
                    "fitted_on": "the 311 training period only, then frozen",
                }
            ),
            "refs": tuple(record.external_id for record in labelled.records),
        }
    )


def _in_domain_block(
    split: TemporalSplit,
    period_records: Sequence[CorpusRecord],
    labelled: LabelledPeriod,
    manifest: CorpusManifest,
) -> Mapping[str, Any]:
    """The 311 TEST period: the reduced-feature in-domain reference population."""
    positive = int(np.count_nonzero(labelled.labels))
    return MappingProxyType(
        {
            "domain": SOURCE,
            "target": SOURCE_TARGET,
            "period": Period.TEST.value,
            "corpus_id": manifest.corpus_id,
            "starts_after": split.val_end.isoformat(),
            "period_record_count": len(period_records),
            "count": len(labelled.records),
            "unlabelled_open_count": len(period_records) - len(labelled.records),
            "positive_count": positive,
            "evaluated_once": True,
            "refs": tuple(record.external_id for record in labelled.records),
        }
    )


def _cross_domain_block(
    evaluated: EvaluatedPopulation, manifest: CorpusManifest
) -> Mapping[str, Any]:
    """The CFPB TEST period restricted to records with a persisted outcome."""
    return MappingProxyType(
        {
            "domain": EVALUATION_SOURCE,
            "target": EVALUATION_TARGET,
            "period": Period.TEST.value,
            "corpus_id": manifest.corpus_id,
            "count": len(evaluated.records),
            "adverse_count": int(np.count_nonzero(evaluated.labels)),
            "without_persisted_outcome_count": len(evaluated.without_outcome),
            "adverse_derivation": f"not {EVALUATION_TARGET}, derived at the metric boundary",
            "evaluated_once": True,
            "fitted_on": False,
            "refs": tuple(record.external_id for record in evaluated.records),
        }
    )


def _estimator_block(estimator: HistGradientBoostingClassifier) -> Mapping[str, Any]:
    """The nine frozen parameters, read back from the estimator that was fitted."""
    parameters = estimator.get_params()
    block: dict[str, Any] = {"estimator_class": type(estimator).__name__}
    block.update({name: parameters[name] for name in ESTIMATOR_PARAMETERS})
    block["unnamed_parameters"] = f"scikit-learn {sklearn.__version__} defaults"
    block["fits"] = 1
    block["tuning"] = "none: no search, no operating point, no threshold"
    return MappingProxyType(block)


def _dependency_versions() -> Mapping[str, str]:
    return MappingProxyType(
        {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
        }
    )


# --- the artifact ------------------------------------------------------------------------------


def _metadata(
    *,
    manifest: CorpusManifest,
    evaluation_manifest: CorpusManifest,
    split: TemporalSplit,
    evaluation_split: EvaluatedPopulation,
    metrics: Mapping[str, Mapping[str, Any]],
    baseline_prior: float,
    estimator: HistGradientBoostingClassifier,
    git_sha: str,
) -> dict[str, Any]:
    """Plan §P's fields under D35's closed schema, extended by nothing (O7).

    `thresholds` is ``null`` because the probe bands nothing, and
    `warmup_row_count` is ``null`` because it has no out-of-fold construction to
    count. Task 19's own diagnostics — the shift block, the framing facts, the
    provenance verdict and the result classification — live in the standalone
    report, because D35 refuses unknown top-level keys and O7 declines to widen
    it.
    """
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "corpus_id": manifest.corpus_id,
        "corpus_schema_version": manifest.schema_version,
        "source_window": {
            "start": WINDOW_START.isoformat(),
            "end": WINDOW_END.isoformat(),
            "corpus_window": {
                "start": manifest.window_start.isoformat(),
                "end": manifest.window_end.isoformat(),
            },
        },
        "split": {
            "train_end": split.train_end.isoformat(),
            "val_end": split.val_end.isoformat(),
            "counts": {period.value: split.counts[period] for period in Period},
            "requested_fractions": {
                period.value: split.requested_fractions[period] for period in Period
            },
            "achieved_fractions": {
                period.value: split.achieved_fractions[period] for period in Period
            },
            "cross_domain_evaluation": {
                "corpus_id": evaluation_manifest.corpus_id,
                "source": EVALUATION_SOURCE,
                "period": Period.TEST.value,
                "count": len(evaluation_split.records),
            },
        },
        "feature_spec": list(FEATURE_SPEC.names),
        "feature_spec_version": FEATURE_SPEC.version,
        "label_roster": [bool(label) for label in ROSTER],
        # The probe publishes ranking metrics and bands nothing (§6, O3).
        "thresholds": None,
        "metrics": {key: _metrics_payload(block) for key, block in metrics.items()},
        # No out-of-fold construction, so there is no realised warm-up to count.
        "warmup_row_count": None,
        "seeds": {"classifier": SEED, "stratified_baseline": SEED},
        "dependency_versions": dict(_dependency_versions()),
        "experiment_label": EXPERIMENT_LABEL,
    }


# --- serialization ------------------------------------------------------------------------------


def _payload(report: ProbeReport) -> dict[str, Any]:
    """The report as strict-JSON-ready data, in the order §11 lists it."""
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "experiment_label": EXPERIMENT_LABEL,
        "analysis_type": ANALYSIS_TYPE,
        "headline": HEADLINE,
        "source_training_population": _plain(report.source_training_population),
        "in_domain_evaluation_population": _plain(report.in_domain_evaluation_population),
        "cross_domain_evaluation_population": _plain(report.cross_domain_evaluation_population),
        "feature_names": list(report.feature_names),
        "feature_spec_version": report.feature_spec_version,
        "estimator": _plain(report.estimator),
        "baseline_prior": report.baseline_prior,
        "baseline": {
            "kind": "majority/prior",
            "fitted_on": f"{SOURCE} TRAIN labels only",
            "frozen_before_evaluation": True,
            "used_for": [IN_DOMAIN, CROSS_DOMAIN],
        },
        "metrics": {key: _metrics_payload(block) for key, block in report.metrics.items()},
        "feature_distribution_shift": _plain(report.feature_distribution_shift),
        "framing_facts": _plain(report.framing_facts),
        "timestamp_diagnostic": _plain(report.timestamp_diagnostic),
        "result_classification": report.result_classification,
        "interpretation_limits": list(report.interpretation_limits),
        "dependency_versions": _plain(report.dependency_versions),
    }


def _metrics_payload(block: Mapping[str, Any]) -> dict[str, Any]:
    """One evaluation as strict JSON, each figure beside the baseline it carries."""
    return {
        "headline": block["headline"],
        "pr_auc": _score_payload(block["pr_auc"]),
        "roc_auc": _score_payload(block["roc_auc"]),
        "minority_precision": block["minority_precision"],
        "minority_recall": block["minority_recall"],
        "minority_f1": block["minority_f1"],
        "minority_count": block["minority_count"],
        "minority_baselines": {
            name: _class_payload(scores) for name, scores in block["minority_baselines"].items()
        },
        "base_rate": block["base_rate"],
        "baseline_prior": block["baseline_prior"],
        "population": block["population"],
    }


def _score_payload(result: ScoreResult) -> dict[str, Any]:
    return {
        "score": float(result.score),
        "baselines": {name: float(score) for name, score in result.baselines.items()},
    }


def _class_payload(scores: Any) -> dict[str, Any]:
    return {
        "precision": float(scores.precision),
        "recall": float(scores.recall),
        "f1": float(scores.f1),
        "support": int(scores.support),
    }


def _plain(value: Any) -> Any:
    """Read-only mappings and tuples as the plain structures JSON accepts."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_report(report: ProbeReport) -> Path:
    """Serialize the report to the caller's path, whole or not at all (D27).

    Written through a temporary file in the same directory and moved into place,
    so a reader finds either a complete report or no report.
    """
    target = report.report_path
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".report-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(report.as_json())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def _git_sha() -> str:
    """The commit this run was produced from. Never invented (§R, D35)."""
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            f"cannot resolve git_sha from {root}: {result.stderr.strip() or 'no output'}. "
            "Pass git_sha= explicitly rather than publishing an artifact without provenance."
        )
    return result.stdout.strip()
