"""Imbalance-aware metrics with mandatory baselines (plan Task 14, §5, D7, D17, D34).

**No metric is ever a bare float.** Every public metric returns a frozen result
carrying the model's figure beside its baselines, so a score cannot be quoted
without the do-nothing comparison §5.1 requires. The majority-class baseline is a
required keyword argument on every metric; the stratified-random baseline is
required on the four label metrics and absent from the three ranking metrics.

**Baselines are generated in one place and consumed everywhere else.**
`majority_baseline` and `stratified_baseline` fit a class prior from **training
labels only** — neither takes an evaluation target — and every metric scores
what they produce against the evaluation population with the same arithmetic it
applies to the model. No metric reconstructs a baseline classifier.

* **Majority** is the training class-prior vector, in roster order. As a label it
  predicts the class with the highest prior, ties to the earliest roster label.
  As scores it is that whole vector, identical for every row, so its Average
  Precision is the evaluation base rate, its ROC-AUC is 0.5, and its top-k ranks
  the k classes with the highest training prior.
* **Stratified-random** is exactly one draw from
  ``numpy.random.default_rng(seed)`` over the training prior, with a required
  seed. One draw yields one label per row and defines no score or ranking, so it
  is reported only for ``macro_f1``, ``per_class_report``, ``confusion_matrix``
  and ``minority_report``. Using its 0/1 draw as a score would let the seed alone
  move a 99:1 Average Precision anywhere between 0.01 and 1.0.

**The caller owns the taxonomy.** Every function takes the complete ordered
roster and, where polarity matters, a ``positive_label``. Nothing here infers a
class, an order, a minority class or a polarity, and a label outside the roster
raises. That one order fixes the confusion matrix, the per-class report, the
macro average, the top-k score columns and every tie-break.

**Definitions** (D34):

* per-class precision, recall and F1 resolve a zero denominator to ``0.0``, and a
  class absent from a non-empty evaluation population stays in the report;
* macro-F1 is the mean over the **full** roster, never only the classes present;
* the confusion matrix has true classes as rows and predicted classes as columns;
* top-k counts a hit when fewer than k classes outrank the true one, a tie going
  to the earlier roster label, and a ``k`` wider than the roster raises;
* ``pr_auc`` is step-wise Average Precision, the sum over distinct score
  thresholds of (Rₙ − Rₙ₋₁) × Pₙ, with tied scores entering together — never
  the trapezoidal area, which is optimistic at these imbalances;
* ``roc_auc`` is secondary only (D7): the probability a positive outranks a
  negative, ties counting one half.

**Refusals, never misleading numbers.** An empty evaluation population raises for
every metric; ``pr_auc`` raises with no positive example and ``roc_auc`` with no
positive or no negative example. Mismatched lengths, unknown labels, a
non-finite score, a baseline fitted on another roster and an empty training
population all raise too.

``recall_at_k`` is deliberately absent: §5.3 defines no retrieval baseline, and
plan Task 18 owns recall@k over ``RecordRef`` (D34). ``precision_at_k`` is absent
per D17.

NumPy and the standard library only, and Django-independent.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

Matrix = tuple[tuple[int, ...], ...]


# --- results ------------------------------------------------------------------


@dataclass(frozen=True)
class ClassScores:
    """One class's precision, recall and F1, with its support in ``y_true``."""

    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ScoreResult:
    """A scalar metric beside its named baseline scores."""

    score: float
    baselines: Mapping[str, float]
    """Read-only. Always includes ``"majority"``."""


@dataclass(frozen=True)
class PerClassReport:
    labels: tuple[Hashable, ...]
    model: Mapping[Hashable, ClassScores]
    """Read-only, keyed in roster order."""
    baselines: Mapping[str, Mapping[Hashable, ClassScores]]


@dataclass(frozen=True)
class ConfusionMatrix:
    """Rows are true classes and columns are predicted classes, in roster order."""

    labels: tuple[Hashable, ...]
    model: Matrix
    baselines: Mapping[str, Matrix]
    """Each baseline's own matrix — never reduced to a scalar."""


@dataclass(frozen=True)
class MinorityReport:
    """Precision, recall and F1 for ``positive_label``, with its absolute count."""

    positive_label: Hashable
    support: int
    """How many evaluation records carry ``positive_label`` (§5.4)."""
    model: ClassScores
    baselines: Mapping[str, ClassScores]


# --- baselines ------------------------------------------------------------------


@dataclass(frozen=True)
class MajorityBaseline:
    """The training class prior, usable as a label or as a score vector."""

    labels: tuple[Hashable, ...]
    prior: tuple[float, ...]
    """Training class shares, in roster order."""
    predicted_label: Hashable
    """The highest training prior, ties to the earliest roster label."""

    def predictions(self, count: int) -> tuple[Hashable, ...]:
        return (self.predicted_label,) * _validated_count(count)

    def scores(self, count: int) -> np.ndarray:
        """The prior vector repeated for every row: columns follow ``labels``."""
        return np.tile(np.asarray(self.prior, dtype=float), (_validated_count(count), 1))


@dataclass(frozen=True)
class StratifiedBaseline:
    """One seeded draw of labels from the training class prior."""

    labels: tuple[Hashable, ...]
    prior: tuple[float, ...]
    seed: int
    predictions: tuple[Hashable, ...]


def majority_baseline(
    train_labels: Sequence[Hashable], labels: Sequence[Hashable]
) -> MajorityBaseline:
    """Fit the majority-class baseline from training labels alone (D34).

    Raises ``ValueError`` for an empty or repeating roster, an unknown training
    label, or no training labels at all.
    """
    roster = _validated_roster(labels)
    prior = _training_prior(train_labels, roster)
    predicted = roster[prior.index(max(prior))]
    return MajorityBaseline(labels=roster, prior=prior, predicted_label=predicted)


def stratified_baseline(
    train_labels: Sequence[Hashable],
    labels: Sequence[Hashable],
    count: int,
    *,
    seed: int,
) -> StratifiedBaseline:
    """Draw ``count`` labels from the training prior in one seeded draw (D34).

    ``seed`` is required and must be a non-negative integer: a missing or
    ``None`` seed would draw from operating-system entropy and could never be
    reproduced. The caller records the seed in experiment metadata.

    Raises ``ValueError`` for an invalid seed or count, an empty or repeating
    roster, an unknown training label, or no training labels at all.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer; got {seed!r}")
    roster = _validated_roster(labels)
    prior = _training_prior(train_labels, roster)
    size = _validated_count(count)
    drawn = np.random.default_rng(int(seed)).choice(len(roster), size=size, p=prior)
    return StratifiedBaseline(
        labels=roster,
        prior=prior,
        seed=int(seed),
        predictions=tuple(roster[int(position)] for position in drawn),
    )


# --- label metrics ---------------------------------------------------------------


def macro_f1(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    labels: Sequence[Hashable],
    *,
    majority: MajorityBaseline,
    stratified: StratifiedBaseline,
) -> ScoreResult:
    """Mean F1 over the full roster — the triage headline (§5.2)."""
    matrices = _label_matrices("macro_f1", y_true, y_pred, labels, majority, stratified)
    scores = {name: _macro(_class_scores(matrix)) for name, matrix in matrices.items()}
    return ScoreResult(score=scores.pop("model"), baselines=MappingProxyType(scores))


def per_class_report(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    labels: Sequence[Hashable],
    *,
    majority: MajorityBaseline,
    stratified: StratifiedBaseline,
) -> PerClassReport:
    """Precision, recall, F1 and support for every roster class, in roster order."""
    roster = _validated_roster(labels)
    matrices = _label_matrices("per_class_report", y_true, y_pred, roster, majority, stratified)
    reports = {
        name: MappingProxyType(dict(zip(roster, _class_scores(matrix), strict=True)))
        for name, matrix in matrices.items()
    }
    return PerClassReport(
        labels=roster, model=reports.pop("model"), baselines=MappingProxyType(reports)
    )


def confusion_matrix(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    labels: Sequence[Hashable],
    *,
    majority: MajorityBaseline,
    stratified: StratifiedBaseline,
) -> ConfusionMatrix:
    """Counts with true classes as rows and predicted classes as columns."""
    roster = _validated_roster(labels)
    matrices = _label_matrices("confusion_matrix", y_true, y_pred, roster, majority, stratified)
    return ConfusionMatrix(
        labels=roster, model=matrices.pop("model"), baselines=MappingProxyType(matrices)
    )


def minority_report(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    labels: Sequence[Hashable],
    positive_label: Hashable,
    *,
    majority: MajorityBaseline,
    stratified: StratifiedBaseline,
) -> MinorityReport:
    """Precision, recall and F1 for the caller-named minority class, with its count.

    The minority class is never inferred: §5.4 makes the polarity mapping a stated
    interpretive choice, so the caller names it.
    """
    roster = _validated_roster(labels)
    column = _positive_column(roster, positive_label)
    matrices = _label_matrices("minority_report", y_true, y_pred, roster, majority, stratified)
    reports = {name: _class_scores(matrix)[column] for name, matrix in matrices.items()}
    model = reports.pop("model")
    return MinorityReport(
        positive_label=positive_label,
        support=model.support,
        model=model,
        baselines=MappingProxyType(reports),
    )


# --- ranking metrics ---------------------------------------------------------------


def top_k_accuracy(
    y_true: Sequence[Hashable],
    scores: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[Hashable],
    k: int,
    *,
    majority: MajorityBaseline,
) -> ScoreResult:
    """Share of records whose true class ranks inside the top ``k`` (§5.2).

    ``scores`` has one row per record and one column per roster label, in roster
    order. Carries the majority baseline only (D34).
    """
    roster = _validated_roster(labels)
    _require_nonempty("top_k_accuracy", y_true)
    width = _validated_k(k, len(roster))
    true_positions = np.asarray(_positions(y_true, roster, "y_true"))
    _require_majority(majority, roster)
    matrix = _score_matrix("top_k_accuracy", scores, len(y_true), len(roster))

    model = _top_k(matrix, true_positions, width)
    baseline = _top_k(majority.scores(len(y_true)), true_positions, width)
    return ScoreResult(score=model, baselines=MappingProxyType({"majority": baseline}))


def pr_auc(
    y_true: Sequence[Hashable],
    scores: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[Hashable],
    positive_label: Hashable,
    *,
    majority: MajorityBaseline,
) -> ScoreResult:
    """Step-wise Average Precision for ``positive_label`` — the imbalanced headline.

    Raises ``ValueError`` when the evaluation population holds no positive example.
    """
    positive, column, matrix = _ranking_inputs(
        "pr_auc", y_true, scores, labels, positive_label, majority
    )
    if not positive.any():
        raise ValueError(
            f"pr_auc is undefined: the evaluation population holds no positive example "
            f"of positive_label {positive_label!r}"
        )
    model = _average_precision(positive, matrix[:, column])
    baseline = _average_precision(positive, majority.scores(positive.size)[:, column])
    return ScoreResult(score=model, baselines=MappingProxyType({"majority": baseline}))


def roc_auc(
    y_true: Sequence[Hashable],
    scores: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[Hashable],
    positive_label: Hashable,
    *,
    majority: MajorityBaseline,
) -> ScoreResult:
    """ROC-AUC for ``positive_label``. Secondary only, never a headline (D7).

    Raises ``ValueError`` when the evaluation population holds no positive or no
    negative example.
    """
    positive, column, matrix = _ranking_inputs(
        "roc_auc", y_true, scores, labels, positive_label, majority
    )
    if not positive.any():
        raise ValueError(
            f"roc_auc is undefined: the evaluation population holds no positive example "
            f"of positive_label {positive_label!r}"
        )
    if positive.all():
        raise ValueError(
            f"roc_auc is undefined: the evaluation population holds no negative example; "
            f"every record is positive_label {positive_label!r}"
        )
    model = _roc_auc(positive, matrix[:, column])
    baseline = _roc_auc(positive, majority.scores(positive.size)[:, column])
    return ScoreResult(score=model, baselines=MappingProxyType({"majority": baseline}))


# --- arithmetic ----------------------------------------------------------------------


def _class_scores(matrix: Matrix) -> tuple[ClassScores, ...]:
    """Per-class scores from a confusion matrix; zero denominators give 0.0 (D34)."""
    size = len(matrix)
    result = []
    for c in range(size):
        tp = matrix[c][c]
        support = sum(matrix[c])
        predicted = sum(matrix[r][c] for r in range(size))
        fp, fn = predicted - tp, support - tp
        result.append(
            ClassScores(
                precision=tp / predicted if predicted else 0.0,
                recall=tp / support if support else 0.0,
                f1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
                support=support,
            )
        )
    return tuple(result)


def _macro(scores: tuple[ClassScores, ...]) -> float:
    """Mean F1 over every class in the roster, present or not."""
    return math.fsum(s.f1 for s in scores) / len(scores)


def _top_k(matrix: np.ndarray, true_positions: np.ndarray, k: int) -> float:
    """A hit when fewer than ``k`` classes outrank the true one; ties go to the earlier
    roster label."""
    rows = np.arange(matrix.shape[0])
    own = matrix[rows, true_positions][:, None]
    columns = np.arange(matrix.shape[1])[None, :]
    outranked_by = (matrix > own) | ((matrix == own) & (columns < true_positions[:, None]))
    rank = outranked_by.sum(axis=1)
    return int(np.count_nonzero(rank < k)) / matrix.shape[0]


def _average_precision(positive: np.ndarray, scores: np.ndarray) -> float:
    """Σ (Rₙ − Rₙ₋₁) × Pₙ over distinct thresholds, highest first; ties enter together."""
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    true_positives = np.cumsum(positive[order])
    last_of_each_threshold = np.r_[np.flatnonzero(np.diff(ordered)), ordered.size - 1]
    tp = true_positives[last_of_each_threshold]
    precision = tp / (last_of_each_threshold + 1)
    recall = tp / int(positive.sum())
    return math.fsum(np.diff(np.r_[0.0, recall]) * precision)


def _roc_auc(positive: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney U over average ranks: ties between a positive and a negative count
    one half."""
    order = np.argsort(scores, kind="stable")
    ordered = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered)) + 1]
    ends = np.r_[starts[1:], ordered.size]
    ranks = np.empty(ordered.size)
    ranks[order] = np.repeat((starts + ends + 1) / 2, ends - starts)
    positives = int(positive.sum())
    negatives = positive.size - positives
    u = math.fsum(ranks[positive]) - positives * (positives + 1) / 2
    return u / (positives * negatives)


# --- validation ------------------------------------------------------------------------


def _label_matrices(
    metric: str,
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    labels: Sequence[Hashable],
    majority: MajorityBaseline,
    stratified: StratifiedBaseline,
) -> dict[str, Matrix]:
    """The model's and both baselines' confusion matrices over one validated population."""
    roster = _validated_roster(labels)
    _require_nonempty(metric, y_true)
    if len(y_pred) != len(y_true):
        raise ValueError(f"{metric}: {len(y_true)} targets but {len(y_pred)} predictions")
    true_positions = _positions(y_true, roster, "y_true")
    predicted = _positions(y_pred, roster, "y_pred")
    _require_majority(majority, roster)
    _require_stratified(stratified, roster, len(y_true))
    return {
        "model": _confusion(true_positions, predicted, len(roster)),
        "majority": _confusion(
            true_positions,
            _positions(majority.predictions(len(y_true)), roster, "majority"),
            len(roster),
        ),
        "stratified": _confusion(
            true_positions, _positions(stratified.predictions, roster, "stratified"), len(roster)
        ),
    }


def _confusion(true_positions: list[int], predicted: list[int], size: int) -> Matrix:
    counts = [[0] * size for _ in range(size)]
    for truth, guess in zip(true_positions, predicted, strict=True):
        counts[truth][guess] += 1
    return tuple(tuple(row) for row in counts)


def _ranking_inputs(
    metric: str,
    y_true: Sequence[Hashable],
    scores: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[Hashable],
    positive_label: Hashable,
    majority: MajorityBaseline,
) -> tuple[np.ndarray, int, np.ndarray]:
    roster = _validated_roster(labels)
    _require_nonempty(metric, y_true)
    column = _positive_column(roster, positive_label)
    true_positions = np.asarray(_positions(y_true, roster, "y_true"))
    _require_majority(majority, roster)
    matrix = _score_matrix(metric, scores, len(y_true), len(roster))
    return true_positions == column, column, matrix


def _validated_roster(labels: Sequence[Hashable]) -> tuple[Hashable, ...]:
    roster = tuple(labels)
    if not roster:
        raise ValueError("the class roster is empty")
    if len(set(roster)) != len(roster):
        raise ValueError(f"the class roster repeats a label: {roster!r}")
    return roster


def _positions(values: Sequence[Hashable], roster: tuple[Hashable, ...], what: str) -> list[int]:
    index = {label: position for position, label in enumerate(roster)}
    positions = []
    for value in values:
        try:
            positions.append(index[value])
        except (KeyError, TypeError):
            raise ValueError(
                f"{what} contains {value!r}, which is not in the class roster"
            ) from None
    return positions


def _training_prior(
    train_labels: Sequence[Hashable], roster: tuple[Hashable, ...]
) -> tuple[float, ...]:
    counts = [0] * len(roster)
    for position in _positions(train_labels, roster, "train_labels"):
        counts[position] += 1
    total = sum(counts)
    if total == 0:
        raise ValueError(
            "train_labels is empty: a baseline prior is fitted from training labels only, "
            "and there are none"
        )
    return tuple(count / total for count in counts)


def _positive_column(roster: tuple[Hashable, ...], positive_label: Hashable) -> int:
    if positive_label not in roster:
        raise ValueError(f"positive_label {positive_label!r} is not in the class roster")
    return roster.index(positive_label)


def _require_nonempty(metric: str, y_true: Sequence[Hashable]) -> None:
    if len(y_true) == 0:
        raise ValueError(f"{metric}: the evaluation population is empty")


def _require_majority(majority: MajorityBaseline, roster: tuple[Hashable, ...]) -> None:
    if not isinstance(majority, MajorityBaseline):
        raise TypeError(f"majority must be a MajorityBaseline, not {type(majority).__name__}")
    if majority.labels != roster:
        raise ValueError("the majority baseline was fitted on a different class roster")


def _require_stratified(
    stratified: StratifiedBaseline, roster: tuple[Hashable, ...], count: int
) -> None:
    if not isinstance(stratified, StratifiedBaseline):
        raise TypeError(f"stratified must be a StratifiedBaseline, not {type(stratified).__name__}")
    if stratified.labels != roster:
        raise ValueError("the stratified baseline was fitted on a different class roster")
    if len(stratified.predictions) != count:
        raise ValueError(
            f"the stratified baseline drew {len(stratified.predictions)} labels for "
            f"{count} evaluation records"
        )


def _score_matrix(
    metric: str, scores: Sequence[Sequence[float]] | np.ndarray, rows: int, columns: int
) -> np.ndarray:
    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{metric}: scores must be a matrix with one column per roster label")
    if matrix.shape[0] != rows:
        raise ValueError(f"{metric}: {rows} targets but {matrix.shape[0]} score rows")
    if matrix.shape[1] != columns:
        raise ValueError(
            f"{metric}: scores have {matrix.shape[1]} columns for {columns} roster labels"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{metric}: scores contain a non-finite value")
    return matrix


def _validated_k(k: int, classes: int) -> int:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError(f"k must be an integer; got {k!r}")
    if k < 1:
        raise ValueError(f"k must be at least 1; got {k}")
    if k > classes:
        raise ValueError(f"k={k} exceeds the {classes} roster classes; k is never clamped")
    return int(k)


def _validated_count(count: int) -> int:
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError(f"count must be a non-negative integer; got {count!r}")
    return int(count)
