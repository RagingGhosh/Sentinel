"""Task 14: imbalance-aware metrics with mandatory baselines (plan Task 14, §5, D34).

Every expected value here was computed independently with exact fractions before
the implementation existed. Fixtures are chosen so that the mistakes D34 guards
against each move a number: micro instead of macro averaging, an absent class
dropped from the macro mean, a transposed confusion matrix, a top-k boundary off
by one, trapezoidal instead of step-wise Average Precision, and a baseline prior
taken from evaluation rather than training labels.
"""

import inspect
import math
from fractions import Fraction
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="numpy lives in requirements/ml.txt")

from ml.training import metrics  # noqa: E402
from ml.training.metrics import (  # noqa: E402
    ClassScores,
    ConfusionMatrix,
    MajorityBaseline,
    MinorityReport,
    PerClassReport,
    ScoreResult,
    StratifiedBaseline,
    confusion_matrix,
    macro_f1,
    majority_baseline,
    minority_report,
    per_class_report,
    pr_auc,
    roc_auc,
    stratified_baseline,
    top_k_accuracy,
)

LABELS = ("a", "b", "c", "d")
Y_TRUE = ["a", "a", "a", "b", "b", "c"]
Y_PRED = ["a", "a", "b", "b", "c", "c"]

BINARY = ("F", "T")


def near(value: Fraction) -> object:
    return pytest.approx(float(value), abs=1e-12)


def label_baselines(
    train_for_majority: list[str], train_for_stratified: list[str], count: int, seed: int = 0
) -> dict[str, object]:
    return {
        "majority": majority_baseline(train_for_majority, LABELS),
        "stratified": stratified_baseline(train_for_stratified, LABELS, count, seed=seed),
    }


def fixture_baselines() -> dict[str, object]:
    """Majority fitted on [b,b,a,a,c]: a and b tie, roster order picks 'a'.
    Stratified fitted on training labels that are all 'b', so every draw is 'b'."""
    return label_baselines(["b", "b", "a", "a", "c"], ["b"] * 5, len(Y_TRUE))


def binary_scores(positive_scores: list[float]) -> "np.ndarray":
    """Columns follow BINARY = (F, T): column 1 is the positive class."""
    t = np.asarray(positive_scores, dtype=float)
    return np.column_stack([1.0 - t, t])


# --- public API ---------------------------------------------------------------


def test_public_signatures_are_the_documented_ones():
    assert list(inspect.signature(majority_baseline).parameters) == ["train_labels", "labels"]
    stratified = inspect.signature(stratified_baseline).parameters
    assert list(stratified) == ["train_labels", "labels", "count", "seed"]
    assert stratified["seed"].kind is inspect.Parameter.KEYWORD_ONLY
    assert stratified["seed"].default is inspect.Parameter.empty
    for function in (macro_f1, per_class_report, confusion_matrix):
        assert list(inspect.signature(function).parameters) == [
            "y_true",
            "y_pred",
            "labels",
            "majority",
            "stratified",
        ]
    assert list(inspect.signature(minority_report).parameters) == [
        "y_true",
        "y_pred",
        "labels",
        "positive_label",
        "majority",
        "stratified",
    ]
    assert list(inspect.signature(top_k_accuracy).parameters) == [
        "y_true",
        "scores",
        "labels",
        "k",
        "majority",
    ]
    for function in (pr_auc, roc_auc):
        assert list(inspect.signature(function).parameters) == [
            "y_true",
            "scores",
            "labels",
            "positive_label",
            "majority",
        ]


def test_baselines_are_keyword_only_and_required_on_every_metric():
    for function in (
        macro_f1,
        per_class_report,
        confusion_matrix,
        minority_report,
        top_k_accuracy,
        pr_auc,
        roc_auc,
    ):
        for name, parameter in inspect.signature(function).parameters.items():
            if name in ("majority", "stratified"):
                assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
                assert parameter.default is inspect.Parameter.empty


def test_recall_at_k_is_deferred_to_task_18():
    """D34: no retrieval baseline is defined, so recall@k is not implemented here."""
    assert not hasattr(metrics, "recall_at_k")
    assert "def recall_at_k" not in Path("ml/training/metrics.py").read_text(encoding="utf-8")


def test_precision_at_k_is_absent():
    """D17: precision@k is removed from the required risk metrics."""
    assert not hasattr(metrics, "precision_at_k")


# --- no bare float, ever -------------------------------------------------------


def test_no_public_metric_returns_a_bare_float():
    base = fixture_baselines()
    scores = np.tile([0.1, 0.4, 0.3, 0.2], (len(Y_TRUE), 1))
    majority = base["majority"]
    results = [
        macro_f1(Y_TRUE, Y_PRED, LABELS, **base),
        per_class_report(Y_TRUE, Y_PRED, LABELS, **base),
        confusion_matrix(Y_TRUE, Y_PRED, LABELS, **base),
        top_k_accuracy(Y_TRUE, scores, LABELS, 2, majority=majority),
    ]
    b_true = ["T", "F", "T", "F", "F"]
    b_base = {
        "majority": majority_baseline(["F", "T"], BINARY),
        "stratified": stratified_baseline(["F", "T"], BINARY, 5, seed=0),
    }
    results.append(minority_report(b_true, ["T", "T", "F", "F", "F"], BINARY, "T", **b_base))
    b_scores = binary_scores([0.9, 0.1, 0.8, 0.2, 0.3])
    results.append(pr_auc(b_true, b_scores, BINARY, "T", majority=b_base["majority"]))
    results.append(roc_auc(b_true, b_scores, BINARY, "T", majority=b_base["majority"]))
    for result in results:
        assert not isinstance(result, (float, int))
        assert "majority" in result.baselines


def test_results_are_immutable():
    base = fixture_baselines()
    result = macro_f1(Y_TRUE, Y_PRED, LABELS, **base)
    assert isinstance(result, ScoreResult)
    with pytest.raises(AttributeError):
        result.score = 1.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.baselines["majority"] = 1.0  # type: ignore[index]


def test_per_class_report_mappings_are_immutable():
    report = per_class_report(Y_TRUE, Y_PRED, LABELS, **fixture_baselines())
    with pytest.raises(AttributeError):
        report.model = {}  # type: ignore[misc]
    with pytest.raises(TypeError):
        report.model["a"] = ClassScores(1.0, 1.0, 1.0, 3)  # type: ignore[index]
    with pytest.raises(TypeError):
        del report.model["a"]  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        report.baselines["majority"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        del report.baselines["stratified"]  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        report.baselines["majority"]["a"] = ClassScores(1.0, 1.0, 1.0, 3)  # type: ignore[index]
    with pytest.raises(TypeError):
        report.baselines["stratified"]["b"] = ClassScores(1.0, 1.0, 1.0, 2)  # type: ignore[index]


def test_confusion_matrix_mappings_are_immutable():
    result = confusion_matrix(Y_TRUE, Y_PRED, LABELS, **fixture_baselines())
    with pytest.raises(AttributeError):
        result.baselines = {}  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.baselines["majority"] = ((0,),)  # type: ignore[index]
    with pytest.raises(TypeError):
        del result.baselines["stratified"]  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        result.model[0] = (9, 9, 9, 9)  # type: ignore[index]
    with pytest.raises(TypeError):
        result.baselines["majority"][0] = (9, 9, 9, 9)  # type: ignore[index]


def test_minority_report_mappings_are_immutable():
    y_true = ["T", "F", "T", "F", "F"]
    y_pred = ["T", "T", "F", "F", "F"]
    base = {
        "majority": majority_baseline(["F", "F", "F", "T"], BINARY),
        "stratified": stratified_baseline(["T"] * 4, BINARY, 5, seed=0),
    }
    report = minority_report(y_true, y_pred, BINARY, "T", **base)
    with pytest.raises(AttributeError):
        report.baselines = {}  # type: ignore[misc]
    with pytest.raises(TypeError):
        report.baselines["majority"] = ClassScores(1.0, 1.0, 1.0, 2)  # type: ignore[index]
    with pytest.raises(TypeError):
        del report.baselines["stratified"]  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        report.model.recall = 1.0  # type: ignore[misc]


# --- macro-F1 -----------------------------------------------------------------


def test_macro_f1_matches_the_hand_computed_value():
    """(4/5 + 1/2 + 2/3 + 0) / 4 = 59/120."""
    result = macro_f1(Y_TRUE, Y_PRED, LABELS, **fixture_baselines())
    assert result.score == near(Fraction(59, 120))


def test_macro_f1_averages_over_the_full_roster_including_absent_classes():
    """Dropping absent 'd' would give 59/90; micro averaging would give 2/3."""
    result = macro_f1(Y_TRUE, Y_PRED, LABELS, **fixture_baselines())
    assert result.score != near(Fraction(59, 90))
    assert result.score != near(Fraction(2, 3))


def test_macro_f1_carries_both_label_baselines():
    """Majority predicts 'a' (tie broken by roster): (2/3)/4 = 1/6.
    Stratified trained only on 'b' predicts 'b' everywhere: (1/2)/4 = 1/8."""
    result = macro_f1(Y_TRUE, Y_PRED, LABELS, **fixture_baselines())
    assert set(result.baselines) == {"majority", "stratified"}
    assert result.baselines["majority"] == near(Fraction(1, 6))
    assert result.baselines["stratified"] == near(Fraction(1, 8))


# --- per-class report -----------------------------------------------------------


def test_per_class_precision_recall_f1_and_support():
    report = per_class_report(Y_TRUE, Y_PRED, LABELS, **fixture_baselines())
    assert isinstance(report, PerClassReport)
    expected = {
        "a": (Fraction(1), Fraction(2, 3), Fraction(4, 5), 3),
        "b": (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), 2),
        "c": (Fraction(1, 2), Fraction(1), Fraction(2, 3), 1),
        "d": (Fraction(0), Fraction(0), Fraction(0), 0),
    }
    for label, (precision, recall, f1, support) in expected.items():
        scores = report.model[label]
        assert isinstance(scores, ClassScores)
        assert scores.precision == near(precision)
        assert scores.recall == near(recall)
        assert scores.f1 == near(f1)
        assert scores.support == support


def test_an_absent_class_is_represented_with_zero_scores_and_zero_support():
    report = per_class_report(Y_TRUE, Y_PRED, LABELS, **fixture_baselines())
    assert "d" in report.model
    assert report.model["d"] == ClassScores(precision=0.0, recall=0.0, f1=0.0, support=0)


def test_zero_denominators_resolve_to_zero_not_nan():
    """'b' is never predicted by the majority baseline: precision is 0/0."""
    report = per_class_report(Y_TRUE, Y_PRED, LABELS, **fixture_baselines())
    b = report.baselines["majority"]["b"]
    assert b.precision == 0.0 and not math.isnan(b.precision)
    assert b.recall == 0.0
    assert b.f1 == 0.0


def test_per_class_report_follows_roster_order():
    report = per_class_report(Y_TRUE, Y_PRED, LABELS, **fixture_baselines())
    assert report.labels == LABELS
    assert tuple(report.model) == LABELS


def test_per_class_report_preserves_the_evaluation_population():
    report = per_class_report(Y_TRUE, Y_PRED, LABELS, **fixture_baselines())
    assert sum(s.support for s in report.model.values()) == len(Y_TRUE)


# --- confusion matrix -----------------------------------------------------------


def test_confusion_matrix_rows_are_true_and_columns_are_predicted():
    result = confusion_matrix(Y_TRUE, Y_PRED, LABELS, **fixture_baselines())
    assert isinstance(result, ConfusionMatrix)
    assert result.labels == LABELS
    assert result.model == ((2, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 0), (0, 0, 0, 0))


def test_confusion_matrix_shape_is_roster_by_roster():
    result = confusion_matrix(Y_TRUE, Y_PRED, LABELS, **fixture_baselines())
    assert len(result.model) == len(LABELS)
    assert all(len(row) == len(LABELS) for row in result.model)


def test_confusion_matrix_follows_the_supplied_roster_order():
    reversed_labels = tuple(reversed(LABELS))
    base = {
        "majority": majority_baseline(["b", "b", "a", "a", "c"], reversed_labels),
        "stratified": stratified_baseline(["b"] * 5, reversed_labels, len(Y_TRUE), seed=0),
    }
    result = confusion_matrix(Y_TRUE, Y_PRED, reversed_labels, **base)
    assert result.model == ((0, 0, 0, 0), (0, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 2))


def test_confusion_matrix_carries_both_baseline_matrices_never_a_scalar():
    result = confusion_matrix(Y_TRUE, Y_PRED, LABELS, **fixture_baselines())
    assert set(result.baselines) == {"majority", "stratified"}
    assert result.baselines["majority"] == (
        (3, 0, 0, 0),
        (2, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 0, 0, 0),
    )
    assert result.baselines["stratified"] == (
        (0, 3, 0, 0),
        (0, 2, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 0),
    )


# --- top-k ----------------------------------------------------------------------

TOP_K_TRUE = ["a", "c", "b", "d"]
TOP_K_ROW = [0.1, 0.4, 0.3, 0.2]


def top_k_majority() -> MajorityBaseline:
    return majority_baseline(["b", "b", "b", "c", "c", "a"], LABELS)


@pytest.mark.parametrize(
    ("k", "expected"),
    [(1, Fraction(1, 4)), (2, Fraction(1, 2)), (3, Fraction(3, 4)), (4, Fraction(1))],
)
def test_top_k_accuracy_hand_computed_at_every_k(k, expected):
    scores = np.tile(TOP_K_ROW, (4, 1))
    result = top_k_accuracy(TOP_K_TRUE, scores, LABELS, k, majority=top_k_majority())
    assert result.score == near(expected)


def test_top_k_boundary_membership_is_strict():
    """'d' sits at rank index 2: inside top-3, outside top-2."""
    scores = np.tile(TOP_K_ROW, (1, 1))
    majority = top_k_majority()
    assert top_k_accuracy(["d"], scores, LABELS, 3, majority=majority).score == 1.0
    assert top_k_accuracy(["d"], scores, LABELS, 2, majority=majority).score == 0.0


def test_top_k_ties_break_by_roster_order():
    """All four tied: roster order ranks a, b, c, d. 'c' is rank 2."""
    tied = np.full((1, 4), 0.5)
    majority = top_k_majority()
    assert top_k_accuracy(["c"], tied, LABELS, 2, majority=majority).score == 0.0
    assert top_k_accuracy(["c"], tied, LABELS, 3, majority=majority).score == 1.0
    assert top_k_accuracy(["a"], tied, LABELS, 1, majority=majority).score == 1.0


def test_top_k_majority_ranks_by_training_prior():
    """Training prior b > c > a > d: top-1 hits b (2 of 6), top-2 hits b and c (3 of 6)."""
    scores = np.tile(TOP_K_ROW, (len(Y_TRUE), 1))
    majority = top_k_majority()
    one = top_k_accuracy(Y_TRUE, scores, LABELS, 1, majority=majority)
    two = top_k_accuracy(Y_TRUE, scores, LABELS, 2, majority=majority)
    assert one.baselines["majority"] == near(Fraction(1, 3))
    assert two.baselines["majority"] == near(Fraction(1, 2))


def test_top_k_carries_no_stratified_baseline():
    """D34: one seeded label draw defines no ranking."""
    scores = np.tile(TOP_K_ROW, (4, 1))
    result = top_k_accuracy(TOP_K_TRUE, scores, LABELS, 2, majority=top_k_majority())
    assert set(result.baselines) == {"majority"}


@pytest.mark.parametrize("bad_k", [0, -1, 5, True, 2.0])
def test_invalid_k_raises_and_is_never_clamped(bad_k):
    scores = np.tile(TOP_K_ROW, (4, 1))
    with pytest.raises(ValueError, match="k"):
        top_k_accuracy(TOP_K_TRUE, scores, LABELS, bad_k, majority=top_k_majority())


def test_k_equal_to_the_roster_width_is_allowed():
    scores = np.tile(TOP_K_ROW, (4, 1))
    assert top_k_accuracy(TOP_K_TRUE, scores, LABELS, 4, majority=top_k_majority()).score == 1.0


def test_top_k_score_width_must_match_the_roster():
    with pytest.raises(ValueError):
        top_k_accuracy(
            TOP_K_TRUE, np.tile([0.1, 0.2, 0.3], (4, 1)), LABELS, 1, majority=top_k_majority()
        )


def test_top_k_score_rows_must_match_the_targets():
    with pytest.raises(ValueError):
        top_k_accuracy(TOP_K_TRUE, np.tile(TOP_K_ROW, (3, 1)), LABELS, 1, majority=top_k_majority())


# --- Average Precision and ROC-AUC -----------------------------------------------

SMALL_TRUE = ["T", "F", "T", "F", "T"]
SMALL_T_SCORES = [0.9, 0.8, 0.8, 0.4, 0.2]


def binary_majority(train: list[str]) -> MajorityBaseline:
    return majority_baseline(train, BINARY)


def test_average_precision_is_step_wise():
    """Distinct thresholds 0.9, 0.8, 0.4, 0.2 give 34/45. Trapezoidal would be 143/180."""
    scores = binary_scores(SMALL_T_SCORES)
    result = pr_auc(SMALL_TRUE, scores, BINARY, "T", majority=binary_majority(["F", "T"]))
    assert result.score == near(Fraction(34, 45))
    assert result.score != near(Fraction(143, 180))


def test_average_precision_does_not_depend_on_row_order_within_tied_scores():
    """Rows 1 (negative) and 2 (positive) share 0.8; swapping them must not change AP."""
    swapped_true = ["T", "T", "F", "F", "T"]
    swapped_scores = binary_scores([0.9, 0.8, 0.8, 0.4, 0.2])
    majority = binary_majority(["F", "T"])
    original = pr_auc(SMALL_TRUE, binary_scores(SMALL_T_SCORES), BINARY, "T", majority=majority)
    swapped = pr_auc(swapped_true, swapped_scores, BINARY, "T", majority=majority)
    assert original.score == swapped.score


def test_roc_auc_counts_ties_as_half():
    """Six positive/negative pairs, one tie: 3.5 / 6 = 7/12."""
    scores = binary_scores(SMALL_T_SCORES)
    result = roc_auc(SMALL_TRUE, scores, BINARY, "T", majority=binary_majority(["F", "T"]))
    assert result.score == near(Fraction(7, 12))


NINETY_NINE_TRUE = ["F"] * 10 + ["T"] + ["F"] * 89
NINETY_NINE_T_SCORES = [(100 - i) / 100 for i in range(100)]


def test_pr_auc_on_a_ninety_nine_to_one_fixture():
    """One positive ranked eleventh: AP = 1/11."""
    scores = binary_scores(NINETY_NINE_T_SCORES)
    majority = binary_majority(["F"] * 99 + ["T"])
    result = pr_auc(NINETY_NINE_TRUE, scores, BINARY, "T", majority=majority)
    assert result.score == near(Fraction(1, 11))


def test_pr_auc_differs_sharply_from_roc_auc_at_ninety_nine_to_one():
    """Same ranking: AP 1/11 (0.09) against ROC-AUC 89/99 (0.90). This is why the
    headline changed (D7)."""
    scores = binary_scores(NINETY_NINE_T_SCORES)
    majority = binary_majority(["F"] * 99 + ["T"])
    ap = pr_auc(NINETY_NINE_TRUE, scores, BINARY, "T", majority=majority)
    roc = roc_auc(NINETY_NINE_TRUE, scores, BINARY, "T", majority=majority)
    assert roc.score == near(Fraction(89, 99))
    assert roc.score - ap.score > 0.8


def test_majority_ranking_baselines_are_base_rate_and_one_half():
    """A constant prior score: AP equals the evaluation base rate, ROC-AUC is 0.5."""
    scores = binary_scores(NINETY_NINE_T_SCORES)
    majority = binary_majority(["F"] * 99 + ["T"])
    ap = pr_auc(NINETY_NINE_TRUE, scores, BINARY, "T", majority=majority)
    roc = roc_auc(NINETY_NINE_TRUE, scores, BINARY, "T", majority=majority)
    assert ap.baselines["majority"] == near(Fraction(1, 100))
    assert roc.baselines["majority"] == 0.5


def test_ranking_metrics_carry_no_stratified_baseline():
    scores = binary_scores(SMALL_T_SCORES)
    majority = binary_majority(["F", "T"])
    assert set(pr_auc(SMALL_TRUE, scores, BINARY, "T", majority=majority).baselines) == {"majority"}
    assert set(roc_auc(SMALL_TRUE, scores, BINARY, "T", majority=majority).baselines) == {
        "majority"
    }


def test_pr_auc_uses_the_positive_label_column():
    """Naming 'F' as positive scores column 0 against 'F' targets instead."""
    scores = binary_scores(SMALL_T_SCORES)
    majority = binary_majority(["F", "T"])
    as_t = pr_auc(SMALL_TRUE, scores, BINARY, "T", majority=majority)
    as_f = pr_auc(SMALL_TRUE, scores, BINARY, "F", majority=majority)
    assert as_t.score != as_f.score


def test_pr_auc_raises_without_a_positive_example():
    scores = binary_scores([0.2, 0.3])
    with pytest.raises(ValueError, match=r"pr_auc.*positive"):
        pr_auc(["F", "F"], scores, BINARY, "T", majority=binary_majority(["F", "T"]))


def test_roc_auc_raises_without_a_positive_example():
    scores = binary_scores([0.2, 0.3])
    with pytest.raises(ValueError, match=r"roc_auc.*positive"):
        roc_auc(["F", "F"], scores, BINARY, "T", majority=binary_majority(["F", "T"]))


def test_roc_auc_raises_without_a_negative_example():
    scores = binary_scores([0.2, 0.3])
    with pytest.raises(ValueError, match=r"roc_auc.*negative"):
        roc_auc(["T", "T"], scores, BINARY, "T", majority=binary_majority(["F", "T"]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_scores_are_refused(bad):
    scores = binary_scores([0.9, bad, 0.1])
    with pytest.raises(ValueError):
        pr_auc(["T", "F", "F"], scores, BINARY, "T", majority=binary_majority(["F", "T"]))


# --- minority report -------------------------------------------------------------


def test_minority_report_with_absolute_support():
    y_true = ["T", "F", "T", "F", "F"]
    y_pred = ["T", "T", "F", "F", "F"]
    base = {
        "majority": majority_baseline(["F", "F", "F", "T"], BINARY),
        "stratified": stratified_baseline(["T"] * 4, BINARY, 5, seed=0),
    }
    report = minority_report(y_true, y_pred, BINARY, "T", **base)
    assert isinstance(report, MinorityReport)
    assert report.positive_label == "T"
    assert report.support == 2
    assert report.model == ClassScores(precision=0.5, recall=0.5, f1=0.5, support=2)
    assert report.baselines["majority"] == ClassScores(0.0, 0.0, 0.0, 2)
    stratified = report.baselines["stratified"]
    assert stratified.precision == near(Fraction(2, 5))
    assert stratified.recall == 1.0
    assert stratified.f1 == near(Fraction(4, 7))


def test_majority_baseline_on_ninety_nine_to_one_scores_high_accuracy_and_zero_recall():
    """Plan Task 14: 0.99 accuracy and near-zero minority recall."""
    majority = majority_baseline(["F"] * 99 + ["T"], BINARY)
    predictions = majority.predictions(len(NINETY_NINE_TRUE))
    accuracy = sum(p == t for p, t in zip(predictions, NINETY_NINE_TRUE)) / len(NINETY_NINE_TRUE)
    assert accuracy == 0.99
    base = {
        "majority": majority,
        "stratified": stratified_baseline(["F"] * 99 + ["T"], BINARY, 100, seed=0),
    }
    report = minority_report(NINETY_NINE_TRUE, NINETY_NINE_TRUE, BINARY, "T", **base)
    assert report.baselines["majority"].recall == 0.0
    assert report.support == 1


# --- majority baseline -------------------------------------------------------------


def test_majority_baseline_prior_comes_from_training_labels():
    majority = majority_baseline(["c", "c", "c", "a"], LABELS)
    assert isinstance(majority, MajorityBaseline)
    assert majority.labels == LABELS
    assert majority.prior == (0.25, 0.0, 0.75, 0.0)
    assert majority.predicted_label == "c"


def test_majority_baseline_ignores_the_evaluation_majority():
    """Evaluation's majority is 'a' (3 of 6); training's is 'c'. Training wins."""
    majority = majority_baseline(["c", "c", "c", "a"], LABELS)
    base = {
        "majority": majority,
        "stratified": stratified_baseline(["c"], LABELS, len(Y_TRUE), seed=0),
    }
    result = confusion_matrix(Y_TRUE, Y_PRED, LABELS, **base)
    assert all(row[2] == sum(row) for row in result.baselines["majority"])


def test_permuting_evaluation_labels_does_not_change_the_fitted_majority():
    majority = majority_baseline(["b", "b", "a"], LABELS)
    before = (majority.prior, majority.predicted_label)
    for evaluation in (Y_TRUE, list(reversed(Y_TRUE)), ["d"] * 6):
        base = {
            "majority": majority,
            "stratified": stratified_baseline(["b"], LABELS, len(evaluation), seed=0),
        }
        macro_f1(evaluation, evaluation, LABELS, **base)
    assert (majority.prior, majority.predicted_label) == before


def test_majority_ties_resolve_by_roster_order():
    assert majority_baseline(["b", "a"], LABELS).predicted_label == "a"
    assert majority_baseline(["b", "a"], ("b", "a", "c", "d")).predicted_label == "b"


def test_majority_scores_are_the_prior_vector_repeated_for_every_row():
    majority = majority_baseline(["c", "c", "c", "a"], LABELS)
    scores = majority.scores(3)
    assert scores.shape == (3, 4)
    assert all(tuple(row) == majority.prior for row in scores)


def test_majority_predictions_repeat_the_predicted_label():
    assert majority_baseline(["c", "c", "a"], LABELS).predictions(3) == ("c", "c", "c")


# --- stratified-random baseline ---------------------------------------------------


def test_stratified_baseline_requires_a_seed():
    with pytest.raises(TypeError):
        stratified_baseline(["a"], LABELS, 3)  # type: ignore[call-arg]


@pytest.mark.parametrize("bad_seed", [None, "1", 1.5, True])
def test_stratified_baseline_refuses_a_non_integer_seed(bad_seed):
    """`default_rng(None)` would draw from OS entropy: not reproducible."""
    with pytest.raises(ValueError, match="seed"):
        stratified_baseline(["a"], LABELS, 3, seed=bad_seed)


def test_stratified_baseline_draws_only_from_the_training_distribution():
    """Training holds only 'c'; every draw is 'c' whatever the seed."""
    for seed in (0, 1, 12345):
        drawn = stratified_baseline(["c"] * 10, LABELS, 50, seed=seed)
        assert isinstance(drawn, StratifiedBaseline)
        assert set(drawn.predictions) == {"c"}
        assert drawn.prior == (0.0, 0.0, 1.0, 0.0)


def test_stratified_baseline_is_one_draw_from_default_rng():
    """D34: exactly one call on numpy.random.default_rng(seed)."""
    train = ["a", "a", "a", "b", "c", "c"]
    drawn = stratified_baseline(train, LABELS, 40, seed=7)
    prior = [3 / 6, 1 / 6, 2 / 6, 0.0]
    reference = np.random.default_rng(7).choice(len(LABELS), size=40, p=prior)
    assert drawn.predictions == tuple(LABELS[i] for i in reference)
    assert drawn.seed == 7


def test_the_same_seed_reproduces_the_same_draw():
    train = ["a", "b"] * 5
    first = stratified_baseline(train, LABELS, 100, seed=42)
    second = stratified_baseline(train, LABELS, 100, seed=42)
    assert first.predictions == second.predictions


def test_different_seeds_may_produce_different_draws():
    train = ["a", "b"] * 5
    assert (
        stratified_baseline(train, LABELS, 200, seed=0).predictions
        != stratified_baseline(train, LABELS, 200, seed=1).predictions
    )


def test_stratified_baseline_takes_no_evaluation_labels():
    """The signature is the guarantee: an evaluation target cannot shape the draw."""
    assert "y_true" not in inspect.signature(stratified_baseline).parameters
    assert "y_true" not in inspect.signature(majority_baseline).parameters


def test_stratified_draw_length_must_match_the_evaluation_population():
    base = {
        "majority": majority_baseline(["a"], LABELS),
        "stratified": stratified_baseline(["a"], LABELS, 3, seed=0),
    }
    with pytest.raises(ValueError):
        macro_f1(Y_TRUE, Y_PRED, LABELS, **base)


# --- input validation ----------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda b: macro_f1([], [], LABELS, **b),
        lambda b: per_class_report([], [], LABELS, **b),
        lambda b: confusion_matrix([], [], LABELS, **b),
        lambda b: minority_report([], [], LABELS, "a", **b),
        lambda b: top_k_accuracy([], np.empty((0, 4)), LABELS, 1, majority=b["majority"]),
        lambda b: pr_auc([], np.empty((0, 4)), LABELS, "a", majority=b["majority"]),
        lambda b: roc_auc([], np.empty((0, 4)), LABELS, "a", majority=b["majority"]),
    ],
)
def test_every_public_metric_refuses_an_empty_evaluation_population(call):
    base = {
        "majority": majority_baseline(["a"], LABELS),
        "stratified": stratified_baseline(["a"], LABELS, 1, seed=0),
    }
    with pytest.raises(ValueError, match="empty"):
        call(base)


def test_length_mismatch_is_refused():
    base = fixture_baselines()
    with pytest.raises(ValueError):
        macro_f1(Y_TRUE, Y_PRED[:-1], LABELS, **base)


def test_an_unknown_target_label_is_refused():
    base = fixture_baselines()
    with pytest.raises(ValueError, match="'z'"):
        macro_f1(["a", "a", "a", "b", "b", "z"], Y_PRED, LABELS, **base)


def test_an_unknown_predicted_label_is_refused():
    base = fixture_baselines()
    with pytest.raises(ValueError, match="'z'"):
        per_class_report(Y_TRUE, ["a", "a", "b", "b", "c", "z"], LABELS, **base)


def test_an_unknown_training_label_is_refused():
    with pytest.raises(ValueError, match="'z'"):
        majority_baseline(["a", "z"], LABELS)
    with pytest.raises(ValueError, match="'z'"):
        stratified_baseline(["a", "z"], LABELS, 3, seed=0)


def test_an_empty_training_population_is_refused():
    """D34: priors are fitted from training labels; with none there is no prior."""
    with pytest.raises(ValueError):
        majority_baseline([], LABELS)
    with pytest.raises(ValueError):
        stratified_baseline([], LABELS, 3, seed=0)


def test_an_unknown_positive_label_is_refused():
    majority = binary_majority(["F", "T"])
    with pytest.raises(ValueError, match="'X'"):
        pr_auc(SMALL_TRUE, binary_scores(SMALL_T_SCORES), BINARY, "X", majority=majority)


@pytest.mark.parametrize("bad_labels", [(), ("a", "a", "b")])
def test_an_empty_or_duplicated_roster_is_refused(bad_labels):
    with pytest.raises(ValueError):
        majority_baseline(["a"], bad_labels)


def test_a_baseline_fitted_on_a_different_roster_is_refused():
    base = {
        "majority": majority_baseline(["a"], ("a", "b", "c")),
        "stratified": stratified_baseline(["a"], LABELS, len(Y_TRUE), seed=0),
    }
    with pytest.raises(ValueError, match="roster"):
        macro_f1(Y_TRUE, Y_PRED, LABELS, **base)


# --- determinism -------------------------------------------------------------------


def test_repeated_calls_are_identical():
    first = per_class_report(Y_TRUE, Y_PRED, LABELS, **fixture_baselines())
    second = per_class_report(Y_TRUE, Y_PRED, LABELS, **fixture_baselines())
    assert first == second
