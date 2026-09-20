"""The CFPB triage experiment: narrative text to its source category (plan §M, D36).

One run, end to end::

    load_corpus("cfpb")  ->  temporal_split  ->  TF-IDF fitted on train text only
      ->  calibrated logistic regression  ->  abstention threshold chosen on
      validation  ->  frozen  ->  evaluated on test  ->  artifact

**The roster is derived, never transcribed** (§1.1). It is
``tuple(sorted(manifest.label_roster))``, and that one order governs the
classifier's classes, the score-matrix columns, the confusion matrix, the
per-class report, macro-F1, top-3 ranking and tie-breaking, and the roster written
to metadata (D34, D36). No category name from the source taxonomy appears in this
file.

**Nothing but training text fits anything** (§6.2). Both vectorisers are fitted on
the training period alone; validation and test are only ever transformed. The
abstention threshold is chosen on validation and applied to test unchanged.

**Abstention changes no published figure** (D36). Confidence is the maximum
calibrated class probability and a record is abstained below the threshold, but
macro-F1, the per-class report, the confusion matrix and top-3 accuracy are all
computed over the *complete* evaluation population from argmax predictions. The
abstention rate is reported beside them as an ancillary number, with no baseline,
and the retained-subset score that chose the threshold is never published.

**The features are blocks, not columns** (D36). ``feature_spec`` names two ordered
TF-IDF blocks -- word first, character second -- whose fitted vectorisers travel
inside the artifact. They stay in the sparse representation scikit-learn produced;
nothing is densified to satisfy an annotation.

Training-side and Django-independent: invoked as ``python -m``, never as a
management command, and nothing here is imported by serving.
"""

from __future__ import annotations

import subprocess
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import scipy
import sklearn
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ingest.manifest import CorpusManifest, load_corpus
from ingest.schema import CorpusRecord
from ml.training.artifacts import TRIAGE_TFIDF_V1, write_artifact
from ml.training.metrics import (
    ConfusionMatrix,
    MajorityBaseline,
    PerClassReport,
    ScoreResult,
    StratifiedBaseline,
    confusion_matrix,
    macro_f1,
    majority_baseline,
    per_class_report,
    stratified_baseline,
    top_k_accuracy,
)
from ml.training.splits import DEFAULT_FRACTIONS, Period, TemporalSplit, temporal_split

SOURCE = "cfpb"

MODEL_NAME = "cfpb_triage_tfidf"
MODEL_VERSION = "v1"
EXPERIMENT_LABEL = "cfpb triage tfidf logistic regression"
"""D36's artifact identity. The version directory is ``<MODEL_NAME>/<MODEL_VERSION>``,
which is what D35's lexical directory check compares against."""

TOP_K = 3
"""§5.2: the UI presents a shortlist, so top-3 accuracy is reported."""

ABSTENTION_GRID: tuple[float, ...] = tuple(round(step * 0.05, 2) for step in range(20))
"""D36: twenty candidates, 0.00 through 0.95 in 0.05 steps. Fixed rather than
derived from the data, so the search space cannot drift with the corpus."""

MIN_RETAINED_COVERAGE = 0.70
"""D36's project design constant. Without a floor the selection is degenerate: the
highest thresholds retain only the most confident handful and score near-perfectly
on them."""

ABSTENTION_QUANTITY = "max_calibrated_probability"
"""What the threshold thresholds, recorded beside its value (D36)."""

EVALUATION_PERIODS = (Period.VALIDATION, Period.TEST)
"""Train is fitted on, not reported on. The threshold comes from validation and is
applied unchanged to test (§6.2)."""


# --- the abstention threshold ------------------------------------------------------------


@dataclass(frozen=True)
class AbstentionCandidate:
    """One grid point, with everything D36 judges it by."""

    threshold: float
    retained: int
    coverage: float
    feasible: bool
    retained_macro_f1: float | None
    """``None`` exactly when the retained set is empty, which D36 calls invalid:
    macro-F1 over no data would describe nothing."""


@dataclass(frozen=True)
class AbstentionThreshold:
    """The selected threshold and the table it was selected from."""

    value: float
    coverage: float
    retained_macro_f1: float
    candidates: tuple[AbstentionCandidate, ...]


def select_abstention_threshold(
    confidences: Sequence[float] | np.ndarray,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    roster: Sequence[str],
    *,
    train_labels: Sequence[str],
    seed: int,
) -> AbstentionThreshold:
    """Choose the abstention threshold on the validation period alone (D36).

    A candidate retains the records whose confidence is at or above it, is feasible
    when it retains at least `MIN_RETAINED_COVERAGE` of them, and is scored by
    Task 14's macro-F1 over the **complete** ``roster`` -- so a class the filter
    empties still enters the macro average at D34's explicit ``0.0`` rather than
    quietly leaving the denominator. The highest-scoring feasible candidate wins,
    and an exact tie goes to the **lowest** threshold, which retains the most.

    ``train_labels`` and ``seed`` are keyword-only and mandatory because the score
    runs through Task 14, whose baselines D34 fits from training labels only, with
    the stratified draw redrawn per candidate to match its retained count. Nothing
    here may see a validation or test label except as something to score.

    Raises ``ValueError`` for mismatched lengths, no training labels, a training
    label outside the roster, or a grid with no feasible candidate at all.
    """
    if not (len(confidences) == len(y_true) == len(y_pred)):
        raise ValueError(
            f"confidences, y_true and y_pred must be the same length; got "
            f"{len(confidences)}, {len(y_true)} and {len(y_pred)}"
        )
    if len(confidences) == 0:
        raise ValueError("the validation population is empty; no threshold can be chosen")

    # Fitted once, before any candidate: it depends only on the training labels, and
    # fitting it here is what turns empty or out-of-roster training labels into an
    # error rather than a silent fallback onto the population being scored.
    majority = majority_baseline(train_labels, roster)

    candidates = []
    for threshold in ABSTENTION_GRID:
        keep = [index for index, value in enumerate(confidences) if value >= threshold]
        coverage = len(keep) / len(confidences)
        score = None
        if keep:
            score = macro_f1(
                [y_true[index] for index in keep],
                [y_pred[index] for index in keep],
                roster,
                majority=majority,
                stratified=stratified_baseline(train_labels, roster, len(keep), seed=seed),
            ).score
        candidates.append(
            AbstentionCandidate(
                threshold=threshold,
                retained=len(keep),
                coverage=coverage,
                feasible=bool(keep) and coverage >= MIN_RETAINED_COVERAGE,
                retained_macro_f1=score,
            )
        )

    # A feasible candidate always retains something, so it always has a score; the
    # explicit check says so rather than leaving it to be assumed.
    scored = [
        (candidate.retained_macro_f1, candidate)
        for candidate in candidates
        if candidate.feasible and candidate.retained_macro_f1 is not None
    ]
    if not scored:
        raise ValueError(
            f"no candidate threshold retains at least {MIN_RETAINED_COVERAGE:.0%} of the "
            "validation population"
        )
    best = max(score for score, _ in scored)
    chosen = min(
        (candidate for score, candidate in scored if score == best),
        key=lambda candidate: candidate.threshold,
    )
    return AbstentionThreshold(
        value=chosen.threshold,
        coverage=chosen.coverage,
        retained_macro_f1=best,
        candidates=tuple(candidates),
    )


# --- the stored model ----------------------------------------------------------------------


class TriageModel:
    """What the artifact stores: the fitted vectorisers and the calibrated classifier.

    It is the whole of Task 16's model-side behaviour, kept here rather than in
    `ml.training.artifacts` so that the artifact loader needs no knowledge of
    scikit-learn's pipeline internals -- it asks for `build_feature_blocks` and
    nothing else (D36).

    ``classes_`` is the derived roster, and `predict_proba` returns its columns in
    exactly that order whatever order the underlying estimator learned, so one
    roster order governs the score matrix, top-k ranking and every metric.
    """

    def __init__(
        self,
        word_vectorizer: TfidfVectorizer,
        char_vectorizer: TfidfVectorizer,
        classifier: CalibratedClassifierCV,
        roster: Sequence[str],
    ) -> None:
        self.word_vectorizer = word_vectorizer
        self.char_vectorizer = char_vectorizer
        self.classifier = classifier
        self.classes_ = np.asarray(tuple(roster), dtype=object)
        learned = list(classifier.classes_)
        self._columns = [learned.index(label) for label in roster]

    def build_feature_blocks(self, texts: Sequence[str]) -> Any:
        """``[word TF-IDF | char TF-IDF]``, in that order, sparse as fitted (D36)."""
        return sparse.hstack(
            [self.word_vectorizer.transform(texts), self.char_vectorizer.transform(texts)],
            format="csr",
        )

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.classifier.predict_proba(X))[:, self._columns]

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def build_vectorizers() -> tuple[TfidfVectorizer, TfidfVectorizer]:
    """D36's frozen word and character blocks. No search, no tuned parameters."""
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
    return word, char


def build_classifier(seed: int) -> CalibratedClassifierCV:
    """D36's frozen estimator.

    `CalibratedClassifierCV` takes no ``random_state`` in the pinned scikit-learn,
    and needs none: an integer ``cv`` selects a non-shuffled `StratifiedKFold`, so
    calibration is deterministic given the deterministic corpus order plan §R
    guarantees. ``lbfgs`` is deterministic too, so the inner ``random_state`` is
    recorded for provenance rather than because it moves a result.
    """
    inner = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight=None,
        random_state=seed,
    )
    return CalibratedClassifierCV(estimator=inner, method="sigmoid", cv=5, ensemble=True)


# --- the experiment ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriageResult:
    """Everything one run produced, including what it wrote."""

    roster: tuple[str, ...]
    split: TemporalSplit
    periods: Mapping[Period, tuple[CorpusRecord, ...]]
    matrices: Mapping[Period, Any]
    """Sparse, as the vectorisers produced them."""
    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer
    calibrated_classifier: CalibratedClassifierCV
    model: TriageModel
    threshold: AbstentionThreshold
    scores: Mapping[str, np.ndarray]
    """Calibrated class probabilities per evaluation period, in roster column order."""
    metrics: Mapping[str, Mapping[str, Any]]
    abstention: Mapping[str, float]
    majority: MajorityBaseline
    stratified: StratifiedBaseline
    metadata: Mapping[str, Any]
    artifact_path: Path


def run_experiment(
    *,
    corpus_root: Path,
    artifact_root: Path,
    seed: int,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    abstention_threshold: float | None = None,
    git_sha: str | None = None,
) -> TriageResult:
    """Train, evaluate and publish the triage artifact for one corpus.

    ``abstention_threshold`` overrides validation selection and exists so a caller
    can evaluate the same fitted model at a different operating point; the
    published metrics are identical either way, which is D36's point.
    """
    manifest, stream = load_corpus(SOURCE, root=corpus_root)
    records = tuple(stream)
    if not records:
        raise ValueError(f"{SOURCE} corpus at {corpus_root} holds no records")

    roster = tuple(sorted(manifest.label_roster))
    split = temporal_split([record.submitted_at for record in records], fractions)
    periods = {
        period: tuple(
            record for record in records if split.period_of(record.submitted_at) is period
        )
        for period in Period
    }
    labels = {period: [record.label for record in rows] for period, rows in periods.items()}
    texts = {period: [record.text for record in rows] for period, rows in periods.items()}

    # §6.2, the single easiest mistake here: both vectorisers see training text and
    # nothing else. Every other period is only ever transformed.
    word_vectorizer, char_vectorizer = build_vectorizers()
    word_vectorizer.fit(texts[Period.TRAIN])
    char_vectorizer.fit(texts[Period.TRAIN])

    def blocks(period: Period) -> Any:
        return sparse.hstack(
            [
                word_vectorizer.transform(texts[period]),
                char_vectorizer.transform(texts[period]),
            ],
            format="csr",
        )

    matrices = {period: blocks(period) for period in Period}

    calibrated = build_classifier(seed)
    calibrated.fit(matrices[Period.TRAIN], labels[Period.TRAIN])
    model = TriageModel(word_vectorizer, char_vectorizer, calibrated, roster)

    scores = {period.value: model.predict_proba(matrices[period]) for period in EVALUATION_PERIODS}
    predictions = {
        period.value: [roster[index] for index in scores[period.value].argmax(axis=1)]
        for period in EVALUATION_PERIODS
    }

    # D34: both baseline priors come from the training labels, never from the
    # population being scored. The majority baseline is prior-only, so one serves
    # every period; the stratified draw is redrawn to each period's row count.
    majority = majority_baseline(labels[Period.TRAIN], roster)
    stratified = {
        period.value: stratified_baseline(
            labels[Period.TRAIN], roster, len(labels[period]), seed=seed
        )
        for period in EVALUATION_PERIODS
    }

    validation = Period.VALIDATION.value
    if abstention_threshold is None:
        threshold = select_abstention_threshold(
            scores[validation].max(axis=1),
            labels[Period.VALIDATION],
            predictions[validation],
            roster,
            train_labels=labels[Period.TRAIN],
            seed=seed,
        )
    else:
        threshold = _supplied_threshold(abstention_threshold, scores[validation].max(axis=1))

    # Frozen from here: the test period only ever applies the value above.
    abstention = {
        period.value: float(np.mean(scores[period.value].max(axis=1) < threshold.value))
        for period in EVALUATION_PERIODS
    }

    metrics = {
        period.value: _published_metrics(
            labels[period],
            predictions[period.value],
            scores[period.value],
            roster,
            majority=majority,
            stratified=stratified[period.value],
            abstention_rate=abstention[period.value],
        )
        for period in EVALUATION_PERIODS
    }

    metadata = _metadata(
        manifest=manifest,
        roster=roster,
        split=split,
        threshold=threshold,
        metrics=metrics,
        seed=seed,
        git_sha=git_sha if git_sha is not None else _git_sha(),
    )

    artifact_path = Path(artifact_root) / manifest.source_slug / MODEL_NAME / MODEL_VERSION
    write_artifact(model, metadata, artifact_path)

    return TriageResult(
        roster=roster,
        split=split,
        periods=periods,
        matrices=matrices,
        word_vectorizer=word_vectorizer,
        char_vectorizer=char_vectorizer,
        calibrated_classifier=calibrated,
        model=model,
        threshold=threshold,
        scores=scores,
        metrics=metrics,
        abstention=abstention,
        majority=majority,
        stratified=stratified[Period.TEST.value],
        metadata=metadata,
        artifact_path=artifact_path,
    )


def _supplied_threshold(value: float, confidences: np.ndarray) -> AbstentionThreshold:
    """Wrap a caller-supplied operating point in the same shape selection returns."""
    retained = int(np.count_nonzero(confidences >= value))
    coverage = retained / len(confidences)
    return AbstentionThreshold(
        value=float(value),
        coverage=coverage,
        retained_macro_f1=float("nan"),
        candidates=(),
    )


def _published_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    scores: np.ndarray,
    roster: Sequence[str],
    *,
    majority: MajorityBaseline,
    stratified: StratifiedBaseline,
    abstention_rate: float,
) -> Mapping[str, Any]:
    """§5.2's four metrics over the complete population, plus the ancillary rate.

    Every row is scored, abstained or not (D36): abstention is an operating choice
    reported beside these figures, never a filter applied before them.
    """
    return {
        "macro_f1": macro_f1(y_true, y_pred, roster, majority=majority, stratified=stratified),
        "per_class": per_class_report(
            y_true, y_pred, roster, majority=majority, stratified=stratified
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, roster, majority=majority, stratified=stratified
        ),
        "top_3_accuracy": top_k_accuracy(y_true, scores, roster, TOP_K, majority=majority),
        "abstention_rate": abstention_rate,
    }


# --- metadata ---------------------------------------------------------------------------------


def _metadata(
    *,
    manifest: CorpusManifest,
    roster: tuple[str, ...],
    split: TemporalSplit,
    threshold: AbstentionThreshold,
    metrics: Mapping[str, Mapping[str, Any]],
    seed: int,
    git_sha: str,
) -> dict[str, Any]:
    """Plan §P's fields, with nothing invented and nothing recomputed (D35)."""
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
        "feature_spec": list(TRIAGE_TFIDF_V1.names),
        "feature_spec_version": TRIAGE_TFIDF_V1.version,
        "label_roster": list(roster),
        "thresholds": {"abstention": {"value": threshold.value, "quantity": ABSTENTION_QUANTITY}},
        "metrics": {period: _metrics_payload(published) for period, published in metrics.items()},
        # D36: triage derives no out-of-fold aggregate, so there is no warm-up
        # prefix to count. This is D35's "not applicable to this kind of artifact".
        "warmup_row_count": None,
        "seeds": {"classifier": seed, "stratified_baseline": seed},
        "dependency_versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "experiment_label": EXPERIMENT_LABEL,
    }


def _metrics_payload(published: Mapping[str, Any]) -> dict[str, Any]:
    """One period's metrics as strict JSON, each beside the baselines it carries."""
    macro: ScoreResult = published["macro_f1"]
    report: PerClassReport = published["per_class"]
    matrix: ConfusionMatrix = published["confusion_matrix"]
    top_k: ScoreResult = published["top_3_accuracy"]
    return {
        "macro_f1": _score_payload(macro),
        "per_class": {
            "labels": list(report.labels),
            "model": _report_payload(report.model),
            "baselines": {
                name: _report_payload(scores) for name, scores in report.baselines.items()
            },
        },
        "confusion_matrix": {
            "labels": list(matrix.labels),
            "model": _matrix_payload(matrix.model),
            "baselines": {name: _matrix_payload(rows) for name, rows in matrix.baselines.items()},
        },
        "top_3_accuracy": {**_score_payload(top_k), "k": TOP_K},
        # Ancillary, and a plain number on purpose: it carries no baseline because it
        # describes the operating point rather than being a score to beat (D36).
        "abstention_rate": float(published["abstention_rate"]),
    }


def _score_payload(result: ScoreResult) -> dict[str, Any]:
    return {
        "score": float(result.score),
        "baselines": {name: float(score) for name, score in result.baselines.items()},
    }


def _report_payload(scores: Mapping[Hashable, Any]) -> dict[str, Any]:
    return {
        str(label): {
            "precision": float(entry.precision),
            "recall": float(entry.recall),
            "f1": float(entry.f1),
            "support": int(entry.support),
        }
        for label, entry in scores.items()
    }


def _matrix_payload(matrix: Sequence[Sequence[int]]) -> list[list[int]]:
    return [[int(count) for count in row] for row in matrix]


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
