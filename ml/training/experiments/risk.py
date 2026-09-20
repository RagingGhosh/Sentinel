"""The NYC 311 SLA risk experiment: will a request resolve slower than its type's p75?

One run, end to end::

    load_corpus("nyc311") + load_outcomes("nyc311")  ->  total bijective join
      ->  temporal_split  ->  thresholds fitted on train  ->  forward-chaining folds
      ->  out-of-fold aggregates for train, frozen train aggregates for val/test
      ->  RiskFeaturesV1  ->  resolved-only model population  ->  gradient boosting
      ->  decision band chosen on validation  ->  frozen  ->  evaluated  ->  artifact

**Open requests are split from the labelled population, not removed** (D37.3). A
request still open at ingest has no `resolution_hours`, so it can carry no breach
label -- but it is a real request that happened, and dropping it before the split
or before aggregate construction would quietly change what a category's history
says. So the ordering is deliberate and is the single most important thing in
this module::

    all records -> temporal split -> thresholds and aggregates
                -> feature construction -> resolved-only model population

Open rows therefore reach the split, the folds and the aggregates, and may
receive aggregate feature values; they reach no threshold statistic (Task 11 and
Task 13 already count only eligible observations), no classifier fit and no
published metric. Nothing here coerces an unresolved outcome to `False`, which is
the refusal D33 built into `apply_thresholds`.

**Nothing is recomputed that another task owns** (D37.12, D37.13). The per-type
p75, its global fallback, the hundred-eligible-observation rule and the breach
comparison all live in Task 13; the out-of-fold construction lives in Task 11;
the feature matrix is Task 12's. This module calls them and does no arithmetic of
its own beyond the decision band and the calibration curve.

**The target is boolean and the roster order is fixed** (D37.5): `(False, True)`,
with `True` the breach class, and the score is always the probability of `True`.
That one order governs the classifier's classes, the probability columns, the
thresholded predictions, every metric and the recorded roster.

Training-side and Django-independent: invoked as ``python -m``, never as a
management command, and nothing here is imported by serving.
"""

from __future__ import annotations

import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from ingest.manifest import CorpusManifest, load_corpus, load_outcomes
from ingest.schema import CorpusRecord, NYC311Outcome
from ml.training.aggregates import (
    AggregateColumns,
    apply_category_aggregates,
    fit_category_aggregates,
    oof_category_aggregates,
)
from ml.training.artifacts import write_artifact
from ml.training.features import RISK_FEATURES_V1, build_features
from ml.training.labels import FrozenThresholds, apply_thresholds, breach_rate, fit_thresholds
from ml.training.metrics import (
    ConfusionMatrix,
    MajorityBaseline,
    MinorityReport,
    ScoreResult,
    StratifiedBaseline,
    confusion_matrix,
    majority_baseline,
    minority_report,
    pr_auc,
    roc_auc,
    stratified_baseline,
)
from ml.training.splits import (
    DEFAULT_FRACTIONS,
    Fold,
    Period,
    TemporalSplit,
    forward_chaining_folds,
    temporal_split,
)
from ml.training.thresholds import MIN_ELIGIBLE_OBSERVATIONS, THRESHOLD_QUANTILE

SOURCE = "nyc311"

MODEL_NAME = "nyc311_sla_risk"
MODEL_VERSION = "v1"
EXPERIMENT_LABEL = "nyc311 sla risk histgradientboosting"
"""D37.11's artifact identity, giving the version directory
``<domain>/nyc311_sla_risk/v1/`` that D35's lexical check compares against."""

WINDOW_START = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
WINDOW_END = datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC)
"""D37.2: 2024-01-01 through 2025-12-31 inclusive, a decision taken there rather
than inherited -- section 1's window is CFPB's."""

ROSTER: tuple[bool, bool] = (False, True)
POSITIVE_LABEL = True
"""D37.5: `True` is the breach class, and this order governs every column."""

BAND_GRID: tuple[float, ...] = tuple(round(0.05 * (step + 1), 2) for step in range(19))
"""D37.4: nineteen candidates, 0.05 through 0.95. It starts at 0.05 rather than
0.00 because a threshold of zero bands every record high and decides nothing."""

DECISION_QUANTITY = "probability_of_true"
"""What the decision threshold thresholds, recorded beside its value."""

CALIBRATION_BINS = 10
"""D37.8: ten uniform probability bins, populated ones only."""

EVALUATION_PERIODS = (Period.VALIDATION, Period.TEST)
"""Train is fitted on, not reported on. The band comes from validation and is
applied unchanged to test (section 6.2)."""

HEADLINE = "pr_auc"
"""Section 5.5. ROC-AUC is secondary and macro-F1 is not a risk headline (D7, D37.6)."""


# --- the decision band ----------------------------------------------------------------


@dataclass(frozen=True)
class DecisionCandidate:
    """One grid point and the positive-class F1 it achieves."""

    threshold: float
    positive_f1: float | None


@dataclass(frozen=True)
class DecisionThreshold:
    """The selected band and the table it came from."""

    value: float
    positive_f1: float
    population: int
    """How many records the selection saw. Validation's size, never validation
    plus test -- the cheapest way to see that no test row reached the choice."""
    candidates: tuple[DecisionCandidate, ...]


def select_decision_threshold(
    scores: Sequence[Sequence[float]] | np.ndarray,
    y_true: Sequence[bool] | np.ndarray,
    roster: Sequence[bool],
    *,
    train_labels: Sequence[bool],
    seed: int,
) -> DecisionThreshold:
    """Choose the decision band on the validation period alone (D37.4).

    A record is banded high when its probability of `True` is at or above the
    candidate, and the objective is Task 14's positive-class F1 for `True` over
    the complete roster. The highest-scoring candidate wins and an exact tie takes
    the **lowest** threshold, which bands the most records high.

    ``train_labels`` and ``seed`` are keyword-only and mandatory because the score
    runs through `minority_report`, whose baselines D34 fits from training labels
    only. Nothing here may see a test label.

    Raises ``ValueError`` for a shape mismatch, an empty population, no training
    labels or a training label outside the roster.
    """
    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(roster):
        raise ValueError(
            f"scores must be one row per record and one column per roster label; "
            f"got {matrix.shape} for a roster of {len(roster)}"
        )
    if matrix.shape[0] != len(y_true):
        raise ValueError(f"{matrix.shape[0]} score rows but {len(y_true)} labels")
    if len(y_true) == 0:
        raise ValueError("the validation population is empty; no band can be chosen")

    # Fitted once, before any candidate, from the training labels alone (D34).
    # Doing it here is what turns empty or out-of-roster training labels into an
    # error rather than a silent fallback onto the population being scored.
    majority = majority_baseline(train_labels, roster)
    stratified = stratified_baseline(train_labels, roster, len(y_true), seed=seed)

    positive_column = list(roster).index(POSITIVE_LABEL)
    positive_scores = matrix[:, positive_column]
    truth = [bool(value) for value in y_true]

    candidates = []
    for threshold in BAND_GRID:
        predicted = [bool(score >= threshold) for score in positive_scores]
        report = minority_report(
            truth,
            predicted,
            roster,
            POSITIVE_LABEL,
            majority=majority,
            stratified=stratified,
        )
        candidates.append(DecisionCandidate(threshold=threshold, positive_f1=report.model.f1))

    best = max(candidate.positive_f1 for candidate in candidates)  # type: ignore[type-var]
    chosen = min(
        (candidate for candidate in candidates if candidate.positive_f1 == best),
        key=lambda candidate: candidate.threshold,
    )
    return DecisionThreshold(
        value=chosen.threshold,
        positive_f1=chosen.positive_f1,  # type: ignore[arg-type]
        population=len(y_true),
        candidates=tuple(candidates),
    )


# --- the stored model ------------------------------------------------------------------


class RiskModel:
    """What the artifact stores: the fitted estimator with the roster bound to it.

    `predict_proba` returns its columns in `ROSTER` order whatever order the
    estimator learned, so one order governs the score matrix, the banding and
    every metric (D37.5).
    """

    def __init__(self, estimator: HistGradientBoostingClassifier, roster: Sequence[bool]) -> None:
        self.estimator = estimator
        self.classes_ = np.asarray(tuple(roster), dtype=object)
        learned = [bool(value) for value in estimator.classes_]
        self._columns = [learned.index(bool(label)) for label in roster]

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(X))[:, self._columns]

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def build_estimator(seed: int) -> HistGradientBoostingClassifier:
    """D37.10's frozen recipe. No search, no calibration wrapper, no weighting.

    ``early_stopping=False`` is explicit and load-bearing: the estimator's
    ``'auto'`` default carves an internal **random** validation split out of the
    training rows, which is exactly the non-temporal evaluation section 6 forbids.
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


# --- the experiment ---------------------------------------------------------------------


@dataclass(frozen=True)
class LabelledPeriod:
    """One period's resolved requests, with the labels the frozen thresholds gave."""

    records: tuple[CorpusRecord, ...]
    outcomes: tuple[NYC311Outcome, ...]
    labels: np.ndarray
    positions: tuple[int, ...]
    """Where these rows sit in the period's full population, open requests included."""


@dataclass(frozen=True)
class RiskResult:
    """Everything one run produced, including what it wrote."""

    manifest: CorpusManifest
    records: tuple[CorpusRecord, ...]
    outcomes: tuple[NYC311Outcome, ...]
    roster: tuple[bool, bool]
    split: TemporalSplit
    periods: Mapping[Period, tuple[CorpusRecord, ...]]
    folds: tuple[Fold, ...]
    warmup_row_count: int
    thresholds: FrozenThresholds
    aggregates: Mapping[Period, AggregateColumns]
    matrices: Mapping[Period, np.ndarray]
    labelled: Mapping[str, LabelledPeriod]
    labelled_matrices: Mapping[str, np.ndarray]
    aggregates_for_labelled: Mapping[str, AggregateColumns]
    estimator: HistGradientBoostingClassifier
    model: RiskModel
    feature_spec: Any
    threshold: DecisionThreshold
    scores: Mapping[str, np.ndarray]
    positive_scores: Mapping[str, np.ndarray]
    predictions: Mapping[str, np.ndarray]
    metrics: Mapping[str, Mapping[str, Any]]
    calibration: Mapping[str, tuple[dict[str, Any], ...]]
    breach_rate: Mapping[str, float]
    majority: MajorityBaseline
    stratified: Mapping[str, StratifiedBaseline]
    metadata: Mapping[str, Any]
    artifact_path: Path
    _outcomes_by_id: Mapping[str, NYC311Outcome]

    def outcome_for(self, external_id: str) -> NYC311Outcome:
        return self._outcomes_by_id[external_id]


def run_experiment(
    *,
    corpus_root: Path,
    artifact_root: Path,
    seed: int,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    decision_threshold: float | None = None,
    git_sha: str | None = None,
) -> RiskResult:
    """Train, evaluate and publish the 311 SLA risk artifact for one corpus.

    ``decision_threshold`` overrides validation selection so a caller can evaluate
    the same fitted model at another operating point; everything else is fixed.
    """
    manifest, record_stream = load_corpus(SOURCE, root=corpus_root)
    _, outcome_stream = load_outcomes(SOURCE, root=corpus_root)
    records = tuple(record_stream)
    outcomes = tuple(outcome_stream)
    by_id = _joined(records, outcomes)

    # D37.2's window, applied before anything is fitted or split.
    windowed = tuple(
        record for record in records if WINDOW_START <= record.submitted_at <= WINDOW_END
    )
    if not windowed:
        raise ValueError(
            f"{SOURCE} corpus at {corpus_root} holds no record inside "
            f"{WINDOW_START.isoformat()} .. {WINDOW_END.isoformat()}"
        )

    split = temporal_split([record.submitted_at for record in windowed], fractions)
    # Every record, open requests included: they belong to the split, the folds
    # and the aggregates, and only leave at the labelled-population step (D37.3).
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

    train_records = periods[Period.TRAIN]
    train_outcomes = outcomes_of[Period.TRAIN]

    # Task 13 owns every part of this: the per-type p75, the global fallback and
    # the hundred-eligible-observation rule. Open requests reach it and are
    # excluded by its own eligibility rule, not by a filter here.
    thresholds = fit_thresholds(train_records, train_outcomes, min_eligible=MIN_ELIGIBLE)

    folds = tuple(forward_chaining_folds([record.submitted_at for record in train_records]))
    aggregates = {
        Period.TRAIN: oof_category_aggregates(train_records, train_outcomes, folds),
    }
    frozen_aggregates = fit_category_aggregates(train_records, train_outcomes)
    for period in EVALUATION_PERIODS:
        aggregates[period] = apply_category_aggregates(frozen_aggregates, periods[period])

    matrices = {
        period: build_features(periods[period], aggregates[period], RISK_FEATURES_V1)
        for period in Period
    }

    # Only here do open requests leave: they take no label, so they can join no
    # fit and no metric. Their features were built above and stay built.
    labelled = {
        period.value: _labelled(periods[period], outcomes_of[period], thresholds)
        for period in Period
    }
    labelled_matrices = {
        period.value: matrices[period][list(labelled[period.value].positions)] for period in Period
    }
    aggregates_for_labelled = {
        period.value: _subset(aggregates[period], labelled[period.value].positions)
        for period in Period
    }

    estimator = build_estimator(seed)
    estimator.fit(labelled_matrices["train"], labelled["train"].labels)
    model = RiskModel(estimator, ROSTER)

    scores = {
        period.value: model.predict_proba(labelled_matrices[period.value])
        for period in EVALUATION_PERIODS
    }
    positive_scores = {
        name: matrix[:, ROSTER.index(POSITIVE_LABEL)] for name, matrix in scores.items()
    }

    train_labels = [bool(value) for value in labelled["train"].labels]
    majority = majority_baseline(train_labels, ROSTER)
    stratified = {
        period.value: stratified_baseline(
            train_labels, ROSTER, len(labelled[period.value].records), seed=seed
        )
        for period in EVALUATION_PERIODS
    }

    validation = Period.VALIDATION.value
    if decision_threshold is None:
        threshold = select_decision_threshold(
            scores[validation],
            labelled[validation].labels,
            ROSTER,
            train_labels=train_labels,
            seed=seed,
        )
    else:
        threshold = DecisionThreshold(
            value=float(decision_threshold),
            positive_f1=math.nan,
            population=len(labelled[validation].records),
            candidates=(),
        )

    # Frozen from here: the test period only ever applies the value above.
    predictions = {name: values >= threshold.value for name, values in positive_scores.items()}
    rates = {
        period.value: breach_rate(labelled[period.value].labels) for period in EVALUATION_PERIODS
    }
    calibration = {
        period.value: _calibration_curve(
            positive_scores[period.value], labelled[period.value].labels
        )
        for period in EVALUATION_PERIODS
    }
    metrics = {
        period.value: _published_metrics(
            labelled[period.value].labels,
            predictions[period.value],
            scores[period.value],
            majority=majority,
            stratified=stratified[period.value],
        )
        for period in EVALUATION_PERIODS
    }

    metadata = _metadata(
        manifest=manifest,
        split=split,
        thresholds=thresholds,
        decision=threshold,
        metrics=metrics,
        calibration=calibration,
        rates=rates,
        warmup_row_count=len(folds[0].fit_indices),
        seed=seed,
        git_sha=git_sha if git_sha is not None else _git_sha(),
    )

    artifact_path = Path(artifact_root) / manifest.source_slug / MODEL_NAME / MODEL_VERSION
    write_artifact(model, metadata, artifact_path)

    return RiskResult(
        manifest=manifest,
        # The windowed population: what the run actually trained and scored on,
        # aligned position for position with its outcomes (D37.2).
        records=windowed,
        outcomes=tuple(by_id[record.external_id] for record in windowed),
        roster=ROSTER,
        split=split,
        periods=periods,
        folds=folds,
        warmup_row_count=len(folds[0].fit_indices),
        thresholds=thresholds,
        aggregates=aggregates,
        matrices=matrices,
        labelled=labelled,
        labelled_matrices=labelled_matrices,
        aggregates_for_labelled=aggregates_for_labelled,
        estimator=estimator,
        model=model,
        feature_spec=RISK_FEATURES_V1,
        threshold=threshold,
        scores=scores,
        positive_scores=positive_scores,
        predictions=predictions,
        metrics=metrics,
        calibration=calibration,
        breach_rate=rates,
        majority=majority,
        stratified=stratified,
        metadata=metadata,
        artifact_path=artifact_path,
        _outcomes_by_id=by_id,
    )


MIN_ELIGIBLE = MIN_ELIGIBLE_OBSERVATIONS
"""D37.12 publishes the hundred beside the thresholds it produced."""


# --- joining, labelling and the ancillary curve -------------------------------------------


def _joined(
    records: Sequence[CorpusRecord], outcomes: Sequence[NYC311Outcome]
) -> dict[str, NYC311Outcome]:
    """A total bijection between records and outcomes, keyed by `external_id`.

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

    record_ids = [record.external_id for record in records]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("the corpus repeats an external_id")

    missing = [external_id for external_id in record_ids if external_id not in seen]
    if missing:
        raise ValueError(f"{len(missing)} record(s) have no outcome, starting with {missing[0]!r}")
    extra = sorted(set(seen) - set(record_ids))
    if extra:
        raise ValueError(f"{len(extra)} outcome(s) match no record, starting with {extra[0]!r}")
    return seen


def _labelled(
    records: Sequence[CorpusRecord],
    outcomes: Sequence[NYC311Outcome],
    thresholds: FrozenThresholds,
) -> LabelledPeriod:
    """The resolved subset of one period, labelled by the frozen thresholds (D37.3).

    An open request is left out rather than labelled: `apply_thresholds` refuses an
    unresolved outcome precisely so that no caller can turn "we could not tell"
    into "resolved in time", and this function never catches that refusal -- it
    simply never offers it an open row.
    """
    positions = tuple(
        index for index, outcome in enumerate(outcomes) if outcome.resolution_hours is not None
    )
    resolved_records = tuple(records[index] for index in positions)
    resolved_outcomes = tuple(outcomes[index] for index in positions)
    labels = apply_thresholds(thresholds, resolved_records, resolved_outcomes)
    return LabelledPeriod(
        records=resolved_records,
        outcomes=resolved_outcomes,
        labels=labels,
        positions=positions,
    )


def _subset(columns: AggregateColumns, positions: Sequence[int]) -> AggregateColumns:
    """The same aggregate values, restricted to the rows that carry a label."""
    return AggregateColumns(
        category_mean_resolution_hours=tuple(
            columns.category_mean_resolution_hours[index] for index in positions
        ),
        category_breach_rate=tuple(columns.category_breach_rate[index] for index in positions),
    )


def _calibration_curve(
    positive_scores: np.ndarray, labels: np.ndarray
) -> tuple[dict[str, Any], ...]:
    """D37.8's ancillary diagnostic: ten uniform bins, populated ones only.

    Not a metric and not a headline, so it carries no baseline: it describes how
    the probabilities line up against outcomes, which is a different question from
    how well the ranking separates them.
    """
    truth = np.asarray(labels, dtype=bool)
    indices = np.minimum(
        (np.asarray(positive_scores, dtype=float) * CALIBRATION_BINS).astype(int),
        CALIBRATION_BINS - 1,
    )
    curve = []
    for position in range(CALIBRATION_BINS):
        selected = indices == position
        count = int(np.count_nonzero(selected))
        if count == 0:
            continue
        curve.append(
            {
                "mean_predicted_probability": float(np.mean(positive_scores[selected])),
                "fraction_positive": float(np.count_nonzero(truth[selected]) / count),
                "count": count,
            }
        )
    return tuple(curve)


def _published_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    *,
    majority: MajorityBaseline,
    stratified: StratifiedBaseline,
) -> Mapping[str, Any]:
    """Section 5.5's set over the complete labelled population (D37.6).

    PR-AUC is the headline and ROC-AUC secondary, both ranking metrics carrying the
    majority baseline only, because D34 gives a single seeded draw no ranking
    meaning. The two thresholded metrics carry both baselines.
    """
    truth = [bool(value) for value in labels]
    predicted = [bool(value) for value in predictions]
    return {
        "pr_auc": pr_auc(truth, scores, ROSTER, POSITIVE_LABEL, majority=majority),
        "roc_auc": roc_auc(truth, scores, ROSTER, POSITIVE_LABEL, majority=majority),
        "minority_report": minority_report(
            truth,
            predicted,
            ROSTER,
            POSITIVE_LABEL,
            majority=majority,
            stratified=stratified,
        ),
        "confusion_matrix": confusion_matrix(
            truth, predicted, ROSTER, majority=majority, stratified=stratified
        ),
    }


# --- metadata ------------------------------------------------------------------------------


def _metadata(
    *,
    manifest: CorpusManifest,
    split: TemporalSplit,
    thresholds: FrozenThresholds,
    decision: DecisionThreshold,
    metrics: Mapping[str, Mapping[str, Any]],
    calibration: Mapping[str, tuple[dict[str, Any], ...]],
    rates: Mapping[str, float],
    warmup_row_count: int,
    seed: int,
    git_sha: str,
) -> dict[str, Any]:
    """Plan section P's fields, with nothing invented and nothing recomputed (D35)."""
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "corpus_id": manifest.corpus_id,
        "corpus_schema_version": manifest.schema_version,
        "source_window": {
            "start": manifest.window_start.isoformat(),
            "end": manifest.window_end.isoformat(),
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
        },
        "feature_spec": list(RISK_FEATURES_V1.names),
        "feature_spec_version": RISK_FEATURES_V1.version,
        "label_roster": [bool(label) for label in ROSTER],
        # Straight from Task 13's frozen object (D37.12); no threshold arithmetic
        # is repeated here, and the decision band sits beside it rather than in a
        # new top-level field.
        "thresholds": {
            "min_eligible": MIN_ELIGIBLE,
            "percentile": THRESHOLD_QUANTILE,
            "per_type": {str(key): float(value) for key, value in thresholds.per_type.items()},
            "global_fallback": float(thresholds.global_fallback),
            "fallback_type_count": int(thresholds.fallback_type_count),
            "decision": {"value": decision.value, "quantity": DECISION_QUANTITY},
        },
        "metrics": {
            period: _metrics_payload(published, calibration[period], rates[period])
            for period, published in metrics.items()
        },
        # D37.11: unlike a text model, this one has an out-of-fold construction,
        # so it has a realised warm-up to count (D30).
        "warmup_row_count": warmup_row_count,
        "seeds": {"classifier": seed, "stratified_baseline": seed},
        "dependency_versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "experiment_label": EXPERIMENT_LABEL,
    }


def _metrics_payload(
    published: Mapping[str, Any],
    calibration: Sequence[Mapping[str, Any]],
    rate: float,
) -> dict[str, Any]:
    """One period as strict JSON, each scored metric beside the baselines it carries."""
    report: MinorityReport = published["minority_report"]
    matrix: ConfusionMatrix = published["confusion_matrix"]
    return {
        "headline": HEADLINE,
        "pr_auc": _score_payload(published["pr_auc"]),
        "roc_auc": _score_payload(published["roc_auc"]),
        "minority_report": {
            "positive_label": bool(report.positive_label),
            "support": int(report.support),
            "model": _class_payload(report.model),
            "baselines": {
                name: _class_payload(scores) for name, scores in report.baselines.items()
            },
        },
        "confusion_matrix": {
            "labels": [bool(label) for label in matrix.labels],
            "model": _matrix_payload(matrix.model),
            "baselines": {name: _matrix_payload(rows) for name, rows in matrix.baselines.items()},
        },
        "breach_rate": float(rate),
        # Ancillary and baseline-free on purpose (D37.8).
        "calibration_curve": [dict(entry) for entry in calibration],
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


def _matrix_payload(matrix: Sequence[Sequence[int]]) -> list[list[int]]:
    return [[int(count) for count in row] for row in matrix]


def _git_sha() -> str:
    """The commit this run was produced from. Never invented (section R, D35)."""
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
