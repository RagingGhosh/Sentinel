"""Task 16: the CFPB triage experiment (plan §M, plan Task 16, D36).

RED phase. ``ml/training/experiments/triage.py`` does not exist yet, so every test
that reaches the production module fails with `ModuleNotFoundError`. The module is
imported late, inside `experiment()`, rather than at the top of this file, so that
the corpus fixture, the roster derivation and the split assertions still run and
prove the fixture itself is sound before anything is built on top of it.

Every corpus is generated into pytest's ``tmp_path`` and nothing is committed.
Nothing is downloaded and no socket is opened; one test proves the latter by
making every socket connection raise.
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
sparse = pytest.importorskip("scipy.sparse", reason="scipy arrives with scikit-learn")

from ingest.manifest import (  # noqa: E402
    CorpusManifest,
    build_manifest,
    load_corpus,
    write_manifest,
)
from ingest.schema import CorpusRecord  # noqa: E402
from ingest.storage import write_partition  # noqa: E402
from ml.training.artifacts import load_artifact  # noqa: E402
from ml.training.metrics import (  # noqa: E402
    ConfusionMatrix,
    PerClassReport,
    ScoreResult,
    macro_f1,
    majority_baseline,
    stratified_baseline,
)
from ml.training.splits import Period, temporal_split  # noqa: E402

pytestmark = pytest.mark.ml

SEED = 17

# --- the production module, imported late ------------------------------------------


def experiment():
    """The Task 16 experiment module.

    Imported here rather than at module scope on purpose: until the production
    module exists only the tests that genuinely need it fail, and the fixture
    tests below still run.
    """
    return importlib.import_module("ml.training.experiments.triage")


# --- the fixture corpus -------------------------------------------------------------
#
# Four labels, chosen so that three different orderings disagree:
#
#   alphabetical        Apple cards, Kiwi transfers, Mango reporting, Zebra loans
#   first appearance    Mango reporting, Kiwi transfers, Zebra loans, Apple cards
#   descending count    Zebra loans (13), Mango reporting (11), Apple cards (9),
#                       Kiwi transfers (7)
#
# D36 fixes the alphabetical order. A model or metric ordered by either of the
# other two is a mutation these tests must catch.

ZEBRA = "Zebra loans"
MANGO = "Mango reporting"
APPLE = "Apple cards"
KIWI = "Kiwi transfers"

ALPHABETICAL = (APPLE, KIWI, MANGO, ZEBRA)
FIRST_APPEARANCE = (MANGO, KIWI, ZEBRA, APPLE)
DESCENDING_COUNT = (ZEBRA, MANGO, APPLE, KIWI)

#: Deterministic per-label vocabulary. Deliberately free of ``q`` and ``z`` so the
#: two sentinels below are the only source of those characters in the corpus.
LABEL_WORDS = {
    ZEBRA: "loan repayment interest arrears",
    MANGO: "report bureau inaccurate dispute",
    APPLE: "card charge statement billing",
    KIWI: "transfer wire recipient routing",
}

VALIDATION_SENTINEL = "vvvvalidationonly"
TEST_SENTINEL = "zzzztestonly"

CORPUS_START = datetime(2024, 3, 1, 9, 0, tzinfo=UTC)
WINDOW_START = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
WINDOW_END = datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC)

#: 28 training records, round-robin from ``MANGO`` so first appearance differs from
#: both alphabetical and frequency order. Seven of each label, which is what
#: ``CalibratedClassifierCV(cv=5)`` needs: `StratifiedKFold(5)` requires at least
#: five members of every class.
TRAIN_LABELS = [MANGO, KIWI, ZEBRA, APPLE] * 7
VALIDATION_LABELS = [ZEBRA, ZEBRA, ZEBRA, MANGO, MANGO, APPLE]
#: ``KIWI`` appears in no evaluation period, so the full-roster rules of D34 and
#: D36 are exercised by a class that has training support and no test support.
TEST_LABELS = [ZEBRA, ZEBRA, ZEBRA, MANGO, MANGO, APPLE]

#: Records 26, 27 and 28 share one timestamp, placed so the tie straddles the 70%
#: train boundary. Task 9 must keep all three in the same period, and because three
#: records cannot be split at 70% of forty the achieved fractions must then differ
#: from the requested ones -- which is what makes recording both of them meaningful.
TIED_INDICES = (26, 27, 28)
TIED_HOUR = 26

#: The split that follows puts indices 0-25 in train, 26-33 in validation and 34-39
#: in test. Asserted by ``test_the_fixture_splits_where_these_tests_assume``.
FIRST_NON_TRAIN_INDEX = 26
FIRST_TEST_INDEX = 34

TIMESTAMP_DIAGNOSTIC = {
    "verdict": "supported_plausible_event_time",
    "reason": "fixture corpus; no provenance measurement is claimed",
}


def fixture_timestamp(index: int) -> datetime:
    hours = TIED_HOUR if index in TIED_INDICES else index
    return CORPUS_START + timedelta(hours=hours)


def fixture_text(index: int, label: str) -> str:
    """Deterministic text: label vocabulary, a case number, and period sentinels."""
    words = LABEL_WORDS[label]
    text = f"{words} case {index:04d} handled by branch {index % 5}"
    if len(TRAIN_LABELS) <= index < len(TRAIN_LABELS) + len(VALIDATION_LABELS):
        text = f"{text} {VALIDATION_SENTINEL}"
    elif index >= len(TRAIN_LABELS) + len(VALIDATION_LABELS):
        text = f"{text} {TEST_SENTINEL}"
    return text


def fixture_records() -> list[CorpusRecord]:
    labels = TRAIN_LABELS + VALIDATION_LABELS + TEST_LABELS
    return [
        CorpusRecord(
            source="cfpb",
            external_id=f"{index:04d}",
            text=fixture_text(index, label),
            label=label,
            submitted_at=fixture_timestamp(index),
        )
        for index, label in enumerate(labels)
    ]


@dataclass(frozen=True)
class Fixture:
    """A real corpus tree on disk, with the records that produced it."""

    root: Path
    manifest: CorpusManifest
    records: tuple[CorpusRecord, ...]

    def texts_of(self, period: Period, split) -> list[str]:
        return [r.text for r in self.records if split.period_of(r.submitted_at) is period]


def build_fixture_corpus(root: Path, records: list[CorpusRecord] | None = None) -> CorpusManifest:
    """Write a real corpus: two part files, then a manifest describing them."""
    records = fixture_records() if records is None else records
    half = len(records) // 2
    write_partition(records[:half], "cfpb", 2024, 0, root=root)
    write_partition(records[half:], "cfpb", 2024, 1, root=root)
    manifest = build_manifest(
        "cfpb",
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        source_api_version="fixture-v1",
        limit=None,
        timestamp_diagnostic=TIMESTAMP_DIAGNOSTIC,
        root=root,
        ingested_at=datetime(2026, 9, 18, 12, 0, tzinfo=UTC),
    )
    write_manifest(manifest, root=root)
    return manifest


@pytest.fixture
def corpus(tmp_path) -> Fixture:
    root = tmp_path / "corpus"
    manifest = build_fixture_corpus(root)
    return Fixture(root=root, manifest=manifest, records=tuple(fixture_records()))


@pytest.fixture
def artifact_root(tmp_path) -> Path:
    return tmp_path / "artifacts"


def run(corpus: Fixture, artifact_root: Path, **kwargs):
    return experiment().run_experiment(
        corpus_root=corpus.root, artifact_root=artifact_root, seed=SEED, **kwargs
    )


def dense(matrix):
    """The design matrix as a dense array, whether it arrived sparse or dense."""
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


# --- the corpus ---------------------------------------------------------------------


def test_the_fixture_is_a_real_corpus_readable_through_load_corpus(corpus):
    """The fixture is not a shortcut: it verifies against its own manifest."""
    manifest, records = load_corpus("cfpb", root=corpus.root)
    loaded = list(records)
    assert manifest.corpus_id == corpus.manifest.corpus_id
    assert len(loaded) == len(corpus.records) == 40
    assert len(manifest.part_files) == 2
    assert [r.external_id for r in loaded] == [f"{i:04d}" for i in range(40)]


def test_the_fixture_gives_every_label_at_least_five_training_records(corpus):
    """``CalibratedClassifierCV(cv=5)`` needs five members of each class in train."""
    split = temporal_split([r.submitted_at for r in corpus.records])
    train = [r.label for r in corpus.records if split.period_of(r.submitted_at) is Period.TRAIN]
    for label in ALPHABETICAL:
        assert train.count(label) >= 5, f"{label} has too few training records for cv=5"


def test_the_fixture_splits_where_these_tests_assume(corpus):
    """Several tests rewrite records by index, so the boundary indices are pinned."""
    split = temporal_split([r.submitted_at for r in corpus.records])
    periods = [split.period_of(r.submitted_at) for r in corpus.records]
    assert periods[FIRST_NON_TRAIN_INDEX - 1] is Period.TRAIN
    assert periods[FIRST_NON_TRAIN_INDEX] is Period.VALIDATION
    assert periods[FIRST_TEST_INDEX - 1] is Period.VALIDATION
    assert periods[FIRST_TEST_INDEX] is Period.TEST
    assert split.counts[Period.TRAIN] == FIRST_NON_TRAIN_INDEX


def test_the_fixture_orderings_genuinely_disagree(corpus):
    """Without this the roster-order tests below could pass by coincidence."""
    counts = corpus.manifest.label_roster
    assert tuple(sorted(counts)) == ALPHABETICAL
    assert tuple(sorted(counts, key=lambda label: -counts[label])) == DESCENDING_COUNT
    seen: list[str] = []
    for record in corpus.records:
        if record.label not in seen:
            seen.append(record.label)
    assert tuple(seen) == FIRST_APPEARANCE
    assert ALPHABETICAL != DESCENDING_COUNT != FIRST_APPEARANCE != ALPHABETICAL


def test_a_tampered_part_file_stops_the_run(corpus, artifact_root):
    """Byte verification is not optional: `corpus_id` means nothing without it."""
    part = next((corpus.root / "cfpb").rglob("*.parquet"))
    part.write_bytes(part.read_bytes() + b"tampered")
    from ingest.manifest import ChecksumMismatch

    with pytest.raises(ChecksumMismatch):
        run(corpus, artifact_root)


# --- the roster -----------------------------------------------------------------------


def test_the_roster_is_the_alphabetically_sorted_manifest_roster(corpus, artifact_root):
    """D36. Mutation: order by descending count, or by first appearance."""
    result = run(corpus, artifact_root)
    assert result.roster == ALPHABETICAL
    assert result.roster != DESCENDING_COUNT
    assert result.roster != FIRST_APPEARANCE


def test_no_production_cfpb_label_is_written_into_the_experiment_module():
    """§1.1: the roster is derived, never transcribed. Mutation: hardcode the list."""
    module = Path(__file__).resolve().parents[3] / "ml" / "training" / "experiments" / "triage.py"
    source = module.read_text(encoding="utf-8")
    for label in (
        "Credit reporting",
        "Credit card",
        "Debt collection",
        "Checking or savings account",
        "Mortgage",
        "Student loan",
        "Vehicle loan or lease",
        "Payday loan",
        "Money transfer",
    ):
        assert label not in source, f"{label!r} is transcribed into the experiment module"


def test_one_roster_order_governs_classes_metrics_matrix_and_score_columns(corpus, artifact_root):
    """D34's single-order rule, proved end to end rather than by inspection."""
    result = run(corpus, artifact_root)
    assert tuple(result.model.classes_) == ALPHABETICAL
    assert result.metrics["test"]["per_class"].labels == ALPHABETICAL
    assert result.metrics["test"]["confusion_matrix"].labels == ALPHABETICAL
    assert result.scores["test"].shape[1] == len(ALPHABETICAL)
    assert tuple(result.metadata["label_roster"]) == ALPHABETICAL


def test_the_artifact_label_roster_matches_the_trained_roster_and_order(corpus, artifact_root):
    """Plan Task 16's named test. Mutation: write ``sorted(set(y_train))``."""
    result = run(corpus, artifact_root)
    loaded = load_artifact(result.artifact_path)
    assert tuple(loaded.metadata["label_roster"]) == result.roster == ALPHABETICAL


# --- the temporal split -----------------------------------------------------------------


def test_every_record_is_assigned_to_its_period_by_timestamp(corpus, artifact_root):
    """§6.1. Mutation: any shuffled or stratified split."""
    result = run(corpus, artifact_root)
    split = result.split
    for period, records in result.periods.items():
        for record in records:
            assert split.period_of(record.submitted_at) is period
    train_times = [r.submitted_at for r in result.periods[Period.TRAIN]]
    val_times = [r.submitted_at for r in result.periods[Period.VALIDATION]]
    test_times = [r.submitted_at for r in result.periods[Period.TEST]]
    assert max(train_times) <= split.train_end < min(val_times)
    assert max(val_times) <= split.val_end < min(test_times)


def test_a_tied_boundary_timestamp_cannot_straddle_two_periods(corpus, artifact_root):
    """Task 9's date-cut guarantee, surfaced through the experiment."""
    result = run(corpus, artifact_root)
    tied = [corpus.records[index] for index in TIED_INDICES]
    assert len({r.submitted_at for r in tied}) == 1, "the fixture must really tie these records"
    periods = {result.split.period_of(r.submitted_at) for r in tied}
    assert len(periods) == 1


def test_split_metadata_records_cuts_counts_and_both_fraction_sets(corpus, artifact_root):
    """§6.1 and §P: whatever was asked for, the achieved figures are recorded."""
    result = run(corpus, artifact_root)
    split = result.metadata["split"]
    assert set(split) >= {
        "train_end",
        "val_end",
        "counts",
        "requested_fractions",
        "achieved_fractions",
    }
    assert split["train_end"] == result.split.train_end.isoformat()
    assert split["val_end"] == result.split.val_end.isoformat()
    assert split["counts"] == {p.value: result.split.counts[p] for p in Period}
    assert set(split["requested_fractions"]) == {p.value for p in Period}
    achieved = split["achieved_fractions"]
    assert achieved != split["requested_fractions"], "the tie must move the boundary"
    assert sum(achieved.values()) == pytest.approx(1.0)
    total = result.split.total
    assert achieved == {period: count / total for period, count in split["counts"].items()}


# --- TF-IDF fitted on training text only (§6.2, §I path 2) -------------------------------


def test_a_token_only_in_test_text_is_absent_from_the_fitted_vocabulary(corpus, artifact_root):
    """The plan's named leakage test. Mutation: fit the vectoriser on all text."""
    result = run(corpus, artifact_root)
    assert TEST_SENTINEL in " ".join(corpus.texts_of(Period.TEST, result.split))
    assert TEST_SENTINEL not in result.word_vectorizer.vocabulary_


def test_a_token_only_in_validation_text_is_absent_from_the_fitted_vocabulary(
    corpus, artifact_root
):
    """Validation is not training either. Mutation: fit on train+val."""
    result = run(corpus, artifact_root)
    assert VALIDATION_SENTINEL in " ".join(corpus.texts_of(Period.VALIDATION, result.split))
    assert VALIDATION_SENTINEL not in result.word_vectorizer.vocabulary_


def test_replacing_every_validation_and_test_document_changes_no_fitted_statistic(
    tmp_path, artifact_root
):
    """Stronger than vocabulary membership: the IDF weights must not move either.

    Mutation: ``fit_transform`` over the whole corpus, which keeps the training
    tokens present but reweights every one of them.
    """
    baseline_records = fixture_records()
    rewritten = [
        record
        if index < FIRST_NON_TRAIN_INDEX
        else CorpusRecord(
            source=record.source,
            external_id=record.external_id,
            text=f"entirely different wording {index} unrelated to anything trained",
            label=record.label,
            submitted_at=record.submitted_at,
        )
        for index, record in enumerate(baseline_records)
    ]

    first_root, second_root = tmp_path / "a", tmp_path / "b"
    build_fixture_corpus(first_root, baseline_records)
    build_fixture_corpus(second_root, rewritten)

    first = experiment().run_experiment(
        corpus_root=first_root, artifact_root=artifact_root / "a", seed=SEED
    )
    second = experiment().run_experiment(
        corpus_root=second_root, artifact_root=artifact_root / "b", seed=SEED
    )

    assert first.word_vectorizer.vocabulary_ == second.word_vectorizer.vocabulary_
    assert np.array_equal(first.word_vectorizer.idf_, second.word_vectorizer.idf_)
    assert np.array_equal(first.char_vectorizer.idf_, second.char_vectorizer.idf_)


def test_changing_one_training_document_does_change_the_fitted_statistics(tmp_path, artifact_root):
    """Positive control. Without it the three tests above could pass vacuously."""
    baseline_records = fixture_records()
    altered = list(baseline_records)
    altered[0] = CorpusRecord(
        source=altered[0].source,
        external_id=altered[0].external_id,
        text=altered[0].text + " unmistakablenewtrainingtoken",
        label=altered[0].label,
        submitted_at=altered[0].submitted_at,
    )

    first_root, second_root = tmp_path / "a", tmp_path / "b"
    build_fixture_corpus(first_root, baseline_records)
    build_fixture_corpus(second_root, altered)

    first = experiment().run_experiment(
        corpus_root=first_root, artifact_root=artifact_root / "a", seed=SEED
    )
    second = experiment().run_experiment(
        corpus_root=second_root, artifact_root=artifact_root / "b", seed=SEED
    )

    assert "unmistakablenewtrainingtoken" in second.word_vectorizer.vocabulary_
    assert first.word_vectorizer.vocabulary_ != second.word_vectorizer.vocabulary_


def test_the_character_block_is_fitted_on_training_text_only_as_well(corpus, artifact_root):
    """The char vectoriser is not exempt from §6.2. Mutation: fit char on all text."""
    result = run(corpus, artifact_root)
    char_vocabulary = result.char_vectorizer.vocabulary_
    assert not any("zzz" in gram for gram in char_vocabulary), (
        "'zzz' occurs only in the test-period sentinel and must not be fitted"
    )
    assert not any("vvv" in gram for gram in char_vocabulary), (
        "'vvv' occurs only in the validation-period sentinel and must not be fitted"
    )


# --- feature blocks ------------------------------------------------------------------------


def test_the_word_block_precedes_the_character_block(corpus, artifact_root):
    """D36 block order. Mutation: swap the two blocks in the union."""
    result = run(corpus, artifact_root)
    train_texts = corpus.texts_of(Period.TRAIN, result.split)
    word = dense(result.word_vectorizer.transform(train_texts))
    char = dense(result.char_vectorizer.transform(train_texts))
    natural = result.matrices[Period.TRAIN]
    built = dense(natural)

    assert sparse.issparse(natural), "the blocks keep the representation sklearn produced"
    assert built.shape == (len(train_texts), word.shape[1] + char.shape[1])
    assert np.allclose(built[:, : word.shape[1]], word)
    assert np.allclose(built[:, word.shape[1] :], char)


def test_the_feature_blocks_keep_their_sparse_representation(corpus, artifact_root):
    """D36: no densification merely to satisfy an annotation.

    A fitted `TfidfVectorizer` produces a SciPy sparse matrix and the fitted
    classifier consumes one directly, so every stage keeps it. Mutation: a
    ``.toarray()`` anywhere between the vectorisers and the artifact.
    """
    result = run(corpus, artifact_root)
    for period in Period:
        assert sparse.issparse(result.matrices[period]), period

    loaded = load_artifact(result.artifact_path)
    rebuilt = loaded.build_features(result.periods[Period.TRAIN])
    assert sparse.issparse(rebuilt)
    assert rebuilt.shape == result.matrices[Period.TRAIN].shape
    blocks = loaded.model.build_feature_blocks(
        [record.text for record in result.periods[Period.TRAIN]]
    )
    assert sparse.issparse(blocks)


def test_the_feature_spec_is_exactly_the_two_named_blocks(corpus, artifact_root):
    """D36. Mutation: emit column names, or a single fused block."""
    result = run(corpus, artifact_root)
    assert result.metadata["feature_spec"] == ["tfidf_word_1_2", "tfidf_char_3_5"]
    assert result.metadata["feature_spec_version"] == "triage_tfidf_v1"


def test_the_loaded_artifact_rebuilds_the_training_matrix_without_refitting(corpus, artifact_root):
    """The seam rebuilds from the stored vectorisers. Mutation: refit on load."""
    result = run(corpus, artifact_root)
    loaded = load_artifact(result.artifact_path)
    assert loaded.feature_spec.names == ("tfidf_word_1_2", "tfidf_char_3_5")
    assert loaded.feature_spec.version == "triage_tfidf_v1"
    rebuilt = loaded.build_features(result.periods[Period.TRAIN])
    assert rebuilt.shape == result.matrices[Period.TRAIN].shape
    assert np.allclose(dense(rebuilt), dense(result.matrices[Period.TRAIN]))


# --- the abstention selector (D36) -----------------------------------------------------------
#
# Every expectation below was computed with `ml.training.metrics.macro_f1` over the
# complete roster, so the numbers pin D36's rule rather than an implementation.

SELECTOR_ROSTER = ("a", "b", "c", "d")
SELECTOR_TRAIN = ["a"] * 8 + ["b"] * 6 + ["c"] * 4 + ["d"] * 2
SELECTOR_SEED = 7

#: Confidences descending in 0.05 steps, so coverage at threshold t is exactly the
#: number of records at or above it.
CONF_TEN = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
TRUE_TEN = ["a", "b", "c", "d", "a", "b", "c", "d", "a", "b"]

#: Wrong at 0.65, 0.55 and 0.50. Feasible scores: 0.00 → 0.7083, 0.55 → 0.7833,
#: 0.60 → 0.8667, 0.65 → 0.8667. The maximum is shared by 0.60 and 0.65, so D36's
#: "lowest wins" rule selects 0.60 — which is neither the lowest feasible candidate
#: (0.00) nor the highest (0.65), so a mutation to either is caught.
PRED_INTERIOR = ["a", "b", "c", "d", "a", "b", "a", "d", "b", "c"]

#: Wrong at 0.65 and below. The best-scoring candidate overall is 0.70 at a perfect
#: 1.0 — but its coverage is 0.60, below D36's floor, so 0.65 must win instead.
PRED_TOP_CLEAN = ["a", "b", "c", "d", "a", "b", "a", "a", "b", "c"]

#: A seven-way tie at 1.0 across 0.35 … 0.65, so the tie-break is unmistakable.
CONF_TIE = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.30, 0.25]
TRUE_TIE = ["a", "b", "c", "d", "a", "b", "c", "d", "a", "b"]
PRED_TIE = ["a", "b", "c", "d", "a", "b", "c", "d", "b", "a"]

#: Class ``d`` lives only in the low-confidence tail, so at 0.65 the retained subset
#: holds no ``d`` at all. Over the complete roster that candidate scores 0.75; over
#: only the observed classes it scores 1.0.
TRUE_ROSTER = ["a", "b", "c", "a", "b", "c", "a", "d", "d", "a"]
PRED_ROSTER = ["a", "b", "c", "a", "b", "c", "a", "d", "a", "a"]

#: Nothing reaches 0.50, so every candidate from 0.50 upward retains nothing.
CONF_LOW = [0.45, 0.40, 0.40, 0.35, 0.35, 0.30, 0.30, 0.25, 0.25, 0.20]


def select(confidences, y_true, y_pred, roster=SELECTOR_ROSTER, train=None):
    return experiment().select_abstention_threshold(
        confidences,
        y_true,
        y_pred,
        roster,
        train_labels=SELECTOR_TRAIN if train is None else train,
        seed=SELECTOR_SEED,
    )


def oracle_macro_f1(y_true, y_pred, roster, train):
    """Task 14, called directly, as the independent reference for the fixtures."""
    return macro_f1(
        y_true,
        y_pred,
        roster,
        majority=majority_baseline(train, roster),
        stratified=stratified_baseline(train, roster, len(y_true), seed=SELECTOR_SEED),
    ).score


def candidate_at(result, threshold):
    matching = [c for c in result.candidates if c.threshold == pytest.approx(threshold)]
    assert len(matching) == 1, f"no single candidate at {threshold}"
    return matching[0]


def test_the_selector_takes_four_positional_arguments_and_two_keyword_only_ones():
    """D36 fixes the shape. Mutation: make ``train_labels`` or ``seed`` positional."""
    import inspect

    parameters = inspect.signature(experiment().select_abstention_threshold).parameters
    positional = [
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    keyword_only = [
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    assert positional == ["confidences", "y_true", "y_pred", "roster"]
    assert keyword_only == ["train_labels", "seed"]
    assert all(parameters[name].default is inspect.Parameter.empty for name in keyword_only), (
        "neither may default: D34 gives the prior no fallback and the seed no entropy"
    )


def test_training_labels_and_seed_cannot_be_passed_positionally():
    """Keyword-only is the contract, so a positional call is a `TypeError`."""
    with pytest.raises(TypeError):
        experiment().select_abstention_threshold(
            CONF_TEN, TRUE_TEN, PRED_INTERIOR, SELECTOR_ROSTER, SELECTOR_TRAIN, SELECTOR_SEED
        )


def test_the_selector_will_not_fit_a_prior_without_training_labels():
    """D34: the prior comes from training labels only, never the population scored.

    Empty ``train_labels`` has no fallback. Mutation: derive the prior from
    ``y_true``, which would make this call succeed instead of raising.
    """
    with pytest.raises(ValueError):
        select(CONF_TEN, TRUE_TEN, PRED_INTERIOR, train=[])


def test_the_selector_rejects_a_training_label_outside_the_roster():
    """A prior fitted over an unknown class would silently widen the taxonomy."""
    with pytest.raises(ValueError):
        select(CONF_TEN, TRUE_TEN, PRED_INTERIOR, train=SELECTOR_TRAIN + ["not_in_roster"])


def test_the_candidate_grid_is_twenty_values_from_zero_to_ninety_five():
    """D36's fixed grid. Mutation: step 0.1, or run through 1.00, or derive it."""
    grid = experiment().ABSTENTION_GRID
    assert len(grid) == 20
    assert grid[0] == pytest.approx(0.0)
    assert grid[-1] == pytest.approx(0.95)
    assert list(grid) == pytest.approx([i * 0.05 for i in range(20)])


def test_the_minimum_retained_coverage_is_seventy_percent():
    """D36's project design constant."""
    assert experiment().MIN_RETAINED_COVERAGE == pytest.approx(0.70)


def test_a_confidence_equal_to_the_threshold_is_retained():
    """Abstain iff confidence < threshold, so equality retains. Mutation: ``<=``."""
    result = select(CONF_TEN, TRUE_TEN, PRED_INTERIOR)
    assert candidate_at(result, 0.65).retained == 7


def test_exactly_seventy_percent_coverage_is_feasible():
    """The floor is inclusive. Mutation: ``>`` instead of ``>=``."""
    result = select(CONF_TEN, TRUE_TEN, PRED_INTERIOR)
    boundary = candidate_at(result, 0.65)
    assert boundary.coverage == pytest.approx(0.70)
    assert boundary.feasible is True
    below = candidate_at(result, 0.70)
    assert below.coverage == pytest.approx(0.60)
    assert below.feasible is False


def test_a_candidate_below_the_coverage_floor_cannot_be_selected():
    """D36. The best-scoring candidate here is infeasible and must lose anyway.

    Mutation: drop the coverage filter, which would select 0.70 and its 1.0.
    """
    result = select(CONF_TEN, TRUE_TEN, PRED_TOP_CLEAN)
    best_infeasible = candidate_at(result, 0.70)
    assert best_infeasible.retained_macro_f1 == pytest.approx(1.0)
    assert best_infeasible.coverage == pytest.approx(0.60)
    assert result.value == pytest.approx(0.65)
    assert result.coverage >= experiment().MIN_RETAINED_COVERAGE


def test_the_feasible_candidate_with_the_highest_retained_macro_f1_is_selected():
    """D36's objective. Mutation: take the first feasible, the lowest, or argmin."""
    result = select(CONF_TEN, TRUE_TEN, PRED_INTERIOR)
    assert candidate_at(result, 0.0).retained_macro_f1 == pytest.approx(0.7083333333333333)
    assert candidate_at(result, 0.55).retained_macro_f1 == pytest.approx(0.7833333333333333)
    assert candidate_at(result, 0.60).retained_macro_f1 == pytest.approx(0.8666666666666667)
    assert result.value == pytest.approx(0.60)
    assert result.value != pytest.approx(0.0)
    assert result.value != pytest.approx(0.65)


def test_an_exact_tie_selects_the_lowest_threshold():
    """D36's tie-break. Mutation: ``max`` returning the last maximum."""
    result = select(CONF_TIE, TRUE_TIE, PRED_TIE)
    tied = [
        c for c in result.candidates if c.feasible and c.retained_macro_f1 == pytest.approx(1.0)
    ]
    assert len(tied) >= 2
    assert result.value == pytest.approx(min(c.threshold for c in tied))
    assert result.value == pytest.approx(0.35)


def test_the_retained_score_uses_the_complete_roster():
    """D34 and D36: a class filtered out still enters the macro average at 0.0.

    Mutation: score against ``sorted(set(retained))``, which would read 1.0 here.
    """
    result = select(CONF_ROSTER_CONF := CONF_TEN, TRUE_ROSTER, PRED_ROSTER)
    keep = [i for i, c in enumerate(CONF_ROSTER_CONF) if c >= 0.65]
    retained_true = [TRUE_ROSTER[i] for i in keep]
    assert "d" not in retained_true, "the fixture must really drop a roster class"

    full = oracle_macro_f1(
        retained_true, [PRED_ROSTER[i] for i in keep], SELECTOR_ROSTER, SELECTOR_TRAIN
    )
    observed_only = oracle_macro_f1(
        retained_true,
        [PRED_ROSTER[i] for i in keep],
        ("a", "b", "c"),
        [label for label in SELECTOR_TRAIN if label != "d"],
    )
    assert full == pytest.approx(0.75)
    assert observed_only == pytest.approx(1.0)
    assert candidate_at(result, 0.65).retained_macro_f1 == pytest.approx(full)


def test_a_candidate_retaining_nothing_is_invalid_and_never_selected():
    """D36: an empty retained population is invalid, not merely infeasible."""
    result = select(CONF_LOW, TRUE_TEN, PRED_INTERIOR)
    empty = candidate_at(result, 0.95)
    assert empty.retained == 0
    assert empty.feasible is False
    assert empty.retained_macro_f1 is None
    assert result.value == pytest.approx(0.30)


def test_selection_is_deterministic_across_repeated_calls():
    """No hidden RNG. Mutation: iterate the grid from a set."""
    first = select(CONF_TEN, TRUE_TEN, PRED_INTERIOR)
    second = select(CONF_TEN, TRUE_TEN, PRED_INTERIOR)
    assert first.value == second.value
    assert [c.retained_macro_f1 for c in first.candidates] == [
        c.retained_macro_f1 for c in second.candidates
    ]


def test_the_selected_threshold_is_a_member_of_the_grid():
    """Mutation: interpolate between two candidates."""
    result = select(CONF_TEN, TRUE_TEN, PRED_INTERIOR)
    assert any(result.value == pytest.approx(t) for t in experiment().ABSTENTION_GRID)


# --- the threshold is frozen before test (§6.2 path 4) --------------------------------------


def test_the_threshold_is_unchanged_when_test_labels_are_permuted(tmp_path, artifact_root):
    """The plan's named leakage test. Mutation: select over validation+test."""
    baseline_records = fixture_records()
    rotated_test = list(baseline_records)
    start = FIRST_TEST_INDEX
    permuted = TEST_LABELS[1:] + TEST_LABELS[:1]
    for offset, label in enumerate(permuted):
        record = rotated_test[start + offset]
        rotated_test[start + offset] = CorpusRecord(
            source=record.source,
            external_id=record.external_id,
            text=record.text,
            label=label,
            submitted_at=record.submitted_at,
        )

    first_root, second_root = tmp_path / "a", tmp_path / "b"
    build_fixture_corpus(first_root, baseline_records)
    build_fixture_corpus(second_root, rotated_test)

    first = experiment().run_experiment(
        corpus_root=first_root, artifact_root=artifact_root / "a", seed=SEED
    )
    second = experiment().run_experiment(
        corpus_root=second_root, artifact_root=artifact_root / "b", seed=SEED
    )
    assert first.threshold.value == second.threshold.value


def test_the_candidate_table_covers_the_validation_period_and_nothing_else(corpus, artifact_root):
    """D36: selection sees the validation period alone.

    At threshold 0.00 nothing is abstained, so the first candidate retains exactly
    the validation period. Mutation: concatenate the test period into the selection
    inputs, which makes that count larger while leaving the chosen threshold --
    and therefore a permutation test -- able to come out unchanged by luck.
    """
    result = run(corpus, artifact_root)
    validation = result.split.counts[Period.VALIDATION]
    first = result.threshold.candidates[0]
    assert first.threshold == pytest.approx(0.0)
    assert first.coverage == pytest.approx(1.0)
    assert first.retained == validation
    assert first.retained != validation + result.split.counts[Period.TEST]
    assert all(candidate.retained <= validation for candidate in result.threshold.candidates)


def test_the_recorded_threshold_is_the_one_selected_on_validation(corpus, artifact_root):
    """§6.2 "applied unchanged". Mutation: re-select on the test period."""
    result = run(corpus, artifact_root)
    loaded = load_artifact(result.artifact_path)
    assert loaded.metadata["thresholds"]["abstention"]["value"] == pytest.approx(
        result.threshold.value
    )
    assert any(result.threshold.value == pytest.approx(t) for t in experiment().ABSTENTION_GRID)


def test_the_test_period_abstention_rate_uses_the_frozen_threshold(corpus, artifact_root):
    """Mutation: recompute a per-period threshold."""
    result = run(corpus, artifact_root)
    confidences = result.scores["test"].max(axis=1)
    expected = float(np.mean(confidences < result.threshold.value))
    assert result.abstention["test"] == pytest.approx(expected)


# --- published metrics cover the complete test population (D36) -------------------------------


def test_published_metrics_cover_every_test_record(corpus, artifact_root):
    """D36. Mutation: drop abstained rows before calling any Task 14 metric.

    This is the sharpest defect available in Task 16: filtering would raise the
    headline without the published number saying so.
    """
    result = run(corpus, artifact_root)
    total = result.split.counts[Period.TEST]
    report: PerClassReport = result.metrics["test"]["per_class"]
    matrix: ConfusionMatrix = result.metrics["test"]["confusion_matrix"]
    assert sum(scores.support for scores in report.model.values()) == total
    assert sum(sum(row) for row in matrix.model) == total


def test_published_metrics_are_identical_at_two_different_thresholds(corpus, artifact_root):
    """Abstention influences no published figure. Mutation: any subset scoring."""
    lenient = run(corpus, artifact_root / "lenient", abstention_threshold=0.0)
    strict = run(corpus, artifact_root / "strict", abstention_threshold=0.95)
    for name in ("macro_f1", "top_3_accuracy"):
        assert lenient.metrics["test"][name].score == pytest.approx(
            strict.metrics["test"][name].score
        )
    assert lenient.metrics["test"]["confusion_matrix"].model == (
        strict.metrics["test"]["confusion_matrix"].model
    )
    assert lenient.threshold.value != strict.threshold.value
    assert lenient.abstention["test"] == pytest.approx(0.0)
    assert strict.abstention["test"] >= lenient.abstention["test"]


def test_no_abstain_class_enters_the_roster_or_any_metric(corpus, artifact_root):
    """D36 prohibition. Mutation: add an ``abstain`` label."""
    result = run(corpus, artifact_root)
    assert "abstain" not in result.roster
    assert "abstain" not in result.metrics["test"]["per_class"].labels
    assert "abstain" not in result.metadata["label_roster"]
    assert len(result.metrics["test"]["confusion_matrix"].labels) == len(ALPHABETICAL)


def test_a_roster_class_absent_from_test_is_still_reported(corpus, artifact_root):
    """D34's full-roster rule. Mutation: pass only the observed labels."""
    result = run(corpus, artifact_root)
    report: PerClassReport = result.metrics["test"]["per_class"]
    assert KIWI in report.model
    assert report.model[KIWI].support == 0
    assert report.model[KIWI].f1 == pytest.approx(0.0)


def test_top_three_accuracy_uses_the_calibrated_scores_in_roster_order(corpus, artifact_root):
    """§5.2 and D34. Mutation: k=1, or columns in a different order."""
    result = run(corpus, artifact_root)
    scores = result.scores["test"]
    assert scores.shape == (result.split.counts[Period.TEST], len(ALPHABETICAL))
    assert np.allclose(scores.sum(axis=1), 1.0)
    assert np.allclose(scores, result.model.predict_proba(result.matrices[Period.TEST]))
    assert result.metadata["metrics"]["test"]["top_3_accuracy"]["k"] == 3


def test_the_four_required_metrics_are_all_present(corpus, artifact_root):
    """§5.2's set. Mutation: drop the confusion matrix."""
    result = run(corpus, artifact_root)
    for period in ("validation", "test"):
        published = result.metrics[period]
        assert isinstance(published["macro_f1"], ScoreResult)
        assert isinstance(published["per_class"], PerClassReport)
        assert isinstance(published["confusion_matrix"], ConfusionMatrix)
        assert isinstance(published["top_3_accuracy"], ScoreResult)


def test_no_ranking_metric_is_introduced_as_a_triage_headline(corpus, artifact_root):
    """§5.2 authorises neither PR-AUC nor ROC-AUC here. Mutation: add PR-AUC."""
    result = run(corpus, artifact_root)
    for period, published in result.metadata["metrics"].items():
        assert "pr_auc" not in published, period
        assert "roc_auc" not in published, period


TASK_14_METRICS = ("macro_f1", "per_class", "confusion_matrix", "top_3_accuracy")


def test_the_abstention_rate_is_ancillary_and_not_a_task_14_metric(corpus, artifact_root):
    """D36: recorded under ``metrics`` because the schema is closed, but not one of them.

    It carries no baseline and is not the headline, and the retained-subset score
    that chose the threshold is never published at all.
    """
    result = run(corpus, artifact_root)
    for period in ("validation", "test"):
        published = result.metadata["metrics"][period]
        rate = published["abstention_rate"]
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0
        assert "retained_macro_f1" not in published
        assert "abstention_rate" not in TASK_14_METRICS
        scored = {name for name, value in published.items() if isinstance(value, dict)}
        assert scored == set(TASK_14_METRICS), (
            "every scored entry is one of the four; the rate is a plain number"
        )
        assert rate == pytest.approx(result.abstention[period])


def test_the_abstention_rate_follows_the_frozen_threshold_for_every_period(corpus, artifact_root):
    """D36: computed after the validation-selected threshold is applied, per period."""
    result = run(corpus, artifact_root)
    for period in (Period.VALIDATION, Period.TEST):
        confidences = result.scores[period.value].max(axis=1)
        expected = float(np.mean(confidences < result.threshold.value))
        assert result.abstention[period.value] == pytest.approx(expected)


def test_no_top_level_metadata_field_is_added_for_abstention(corpus, artifact_root):
    """D35's schema stays closed. Mutation: a new top-level ``abstention`` field."""
    metadata = load_artifact(run(corpus, artifact_root).artifact_path).metadata
    assert "abstention" not in metadata
    assert "abstention_rate" not in metadata
    assert set(metadata["thresholds"]) == {"abstention"}


# --- baselines (§5.1, D34) ----------------------------------------------------------------------


def test_every_published_metric_carries_its_required_baselines(corpus, artifact_root):
    """§5.1: nothing is quoted alone. Mutation: strip baselines when serialising."""
    result = run(corpus, artifact_root)
    published = result.metadata["metrics"]["test"]
    for name in ("macro_f1", "per_class", "confusion_matrix"):
        assert set(published[name]["baselines"]) == {"majority", "stratified"}
    assert set(published["top_3_accuracy"]["baselines"]) == {"majority"}


def test_top_k_carries_the_majority_baseline_only(corpus, artifact_root):
    """D34: a single seeded draw is neither a score nor a ranking."""
    result = run(corpus, artifact_root)
    assert set(result.metrics["test"]["top_3_accuracy"].baselines) == {"majority"}


def test_baseline_priors_do_not_move_when_evaluation_labels_are_permuted(tmp_path, artifact_root):
    """D34: priors are fitted from training labels only. Mutation: fit on test."""
    baseline_records = fixture_records()
    permuted = list(baseline_records)
    start = FIRST_NON_TRAIN_INDEX
    rotated = (VALIDATION_LABELS + TEST_LABELS)[1:] + (VALIDATION_LABELS + TEST_LABELS)[:1]
    for offset, label in enumerate(rotated):
        record = permuted[start + offset]
        permuted[start + offset] = CorpusRecord(
            source=record.source,
            external_id=record.external_id,
            text=record.text,
            label=label,
            submitted_at=record.submitted_at,
        )

    first_root, second_root = tmp_path / "a", tmp_path / "b"
    build_fixture_corpus(first_root, baseline_records)
    build_fixture_corpus(second_root, permuted)

    first = experiment().run_experiment(
        corpus_root=first_root, artifact_root=artifact_root / "a", seed=SEED
    )
    second = experiment().run_experiment(
        corpus_root=second_root, artifact_root=artifact_root / "b", seed=SEED
    )
    assert first.majority.prior == second.majority.prior
    assert first.majority.predicted_label == second.majority.predicted_label


def test_the_stratified_baseline_seed_is_explicit_and_recorded(corpus, artifact_root):
    """D34 requires a seed and plan §R requires it recorded."""
    result = run(corpus, artifact_root)
    assert isinstance(result.metadata["seeds"]["stratified_baseline"], int)
    assert result.stratified.seed == result.metadata["seeds"]["stratified_baseline"]


# --- the artifact ---------------------------------------------------------------------------------


def test_the_artifact_is_written_to_the_decided_directory_and_loads_back(corpus, artifact_root):
    """Task 16's acceptance criterion, and D35's lexical directory check."""
    result = run(corpus, artifact_root)
    assert result.artifact_path.parent.name == "cfpb_triage_tfidf"
    assert result.artifact_path.name == "v1"
    assert (result.artifact_path / "model.joblib").is_file()
    assert (result.artifact_path / "metadata.json").is_file()
    assert load_artifact(result.artifact_path) is not None


def test_the_identity_fields_are_exactly_the_decided_literals(corpus, artifact_root):
    """D36. Mutation: any drift in name, version or label."""
    metadata = load_artifact(run(corpus, artifact_root).artifact_path).metadata
    assert metadata["model_name"] == "cfpb_triage_tfidf"
    assert metadata["model_version"] == "v1"
    assert metadata["experiment_label"] == "cfpb triage tfidf logistic regression"


def test_corpus_identity_comes_from_the_loaded_manifest(corpus, artifact_root):
    """§R and D27. Mutation: recompute, or hardcode."""
    result = run(corpus, artifact_root)
    metadata = load_artifact(result.artifact_path).metadata
    assert metadata["corpus_id"] == corpus.manifest.corpus_id
    assert metadata["corpus_schema_version"] == corpus.manifest.schema_version


def test_the_source_window_comes_from_the_manifest(corpus, artifact_root):
    """§P. Mutation: record the run's own dates."""
    metadata = load_artifact(run(corpus, artifact_root).artifact_path).metadata
    assert metadata["source_window"]["start"] == corpus.manifest.window_start.isoformat()
    assert metadata["source_window"]["end"] == corpus.manifest.window_end.isoformat()


def test_the_thresholds_field_has_exactly_the_decided_shape(corpus, artifact_root):
    """D36. Mutation: a bare float, or a new top-level artifact field."""
    result = run(corpus, artifact_root)
    metadata = load_artifact(result.artifact_path).metadata
    thresholds = metadata["thresholds"]
    assert set(thresholds) == {"abstention"}
    assert set(thresholds["abstention"]) == {"value", "quantity"}
    assert thresholds["abstention"]["quantity"] == "max_calibrated_probability"
    assert thresholds["abstention"]["value"] == pytest.approx(result.threshold.value)


def test_warmup_row_count_is_null(corpus, artifact_root):
    """D36: triage has no forward-chaining aggregates. Mutation: write 0."""
    metadata = load_artifact(run(corpus, artifact_root).artifact_path).metadata
    assert metadata["warmup_row_count"] is None


def test_seeds_and_dependency_versions_are_recorded_and_true(corpus, artifact_root):
    """§R. Mutation: stale literals that no longer describe the environment."""
    metadata = load_artifact(run(corpus, artifact_root).artifact_path).metadata
    assert set(metadata["seeds"]) >= {"classifier", "stratified_baseline"}
    versions = metadata["dependency_versions"]
    assert versions["numpy"] == np.__version__
    assert versions["scikit-learn"] == sklearn.__version__
    assert metadata["git_sha"]
    assert metadata["trained_at"].endswith(("+00:00", "Z")) or "+" in metadata["trained_at"]


def test_no_embedding_fields_are_written(corpus, artifact_root):
    """§P scopes those to embedder artifacts. Mutation: write a dimension."""
    metadata = load_artifact(run(corpus, artifact_root).artifact_path).metadata
    for field in ("embedding_dimension", "embedding_model_id", "embedding_model_sha256"):
        assert field not in metadata


def test_metrics_are_recorded_per_period_each_beside_its_baseline(corpus, artifact_root):
    """§P. Mutation: record the test period only."""
    metadata = load_artifact(run(corpus, artifact_root).artifact_path).metadata
    assert set(metadata["metrics"]) >= {"validation", "test"}
    for published in metadata["metrics"].values():
        assert "baselines" in published["macro_f1"]


def test_round_tripped_predictions_are_identical_on_fixed_input(corpus, artifact_root):
    """Plan Task 15 and §P. Mutation: refit anything at load."""
    result = run(corpus, artifact_root)
    loaded = load_artifact(result.artifact_path)
    records = result.periods[Period.TEST]
    rebuilt = loaded.build_features(records)
    assert np.array_equal(
        loaded.model.predict(rebuilt), result.model.predict(result.matrices[Period.TEST])
    )
    assert np.allclose(loaded.model.predict_proba(rebuilt), result.scores["test"])


# --- the frozen recipe (D36) ---------------------------------------------------------------------


def test_the_vectorisers_carry_exactly_the_decided_parameters(corpus, artifact_root):
    """D36. Mutation: change an n-gram range or an analyzer."""
    result = run(corpus, artifact_root)
    assert result.word_vectorizer.analyzer == "word"
    assert result.word_vectorizer.ngram_range == (1, 2)
    assert result.char_vectorizer.analyzer == "char"
    assert result.char_vectorizer.ngram_range == (3, 5)


def test_the_classifier_carries_exactly_the_decided_parameters(corpus, artifact_root):
    """D36. Mutation: ``class_weight="balanced"``, or a different C."""
    result = run(corpus, artifact_root)
    calibrated = result.calibrated_classifier
    assert calibrated.method == "sigmoid"
    assert calibrated.cv == 5
    assert calibrated.ensemble is True
    inner = calibrated.estimator
    assert inner.C == pytest.approx(1.0)
    assert inner.solver == "lbfgs"
    assert inner.max_iter == 1000
    assert inner.class_weight is None
    assert inner.random_state is not None


def test_calibrated_classifier_cv_still_rejects_random_state_under_the_pin():
    """D36's determinism note is a claim about this version, so it is checked.

    If a future scikit-learn adds the parameter, D36's rationale must be revisited
    rather than silently outgrown.
    """
    import inspect

    from sklearn.calibration import CalibratedClassifierCV

    assert "random_state" not in inspect.signature(CalibratedClassifierCV.__init__).parameters


def test_no_hyperparameter_search_is_performed():
    """D36: the abstention threshold is the only quantity tuned. Mutation: GridSearchCV."""
    module = Path(__file__).resolve().parents[3] / "ml" / "training" / "experiments" / "triage.py"
    tree = ast.parse(module.read_text(encoding="utf-8"))
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
        alias.asname or alias.name.split(".")[-1]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    for banned in ("GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV", "optuna"):
        assert banned not in names


# --- determinism and ordering (plan §R) ------------------------------------------------


def test_two_runs_with_the_same_seed_produce_the_same_threshold_and_metrics(corpus, artifact_root):
    """§R same-platform reproducibility. Mutation: an unseeded shuffle anywhere."""
    first = run(corpus, artifact_root / "first")
    second = run(corpus, artifact_root / "second")
    assert first.threshold.value == second.threshold.value
    assert first.metrics["test"]["macro_f1"].score == second.metrics["test"]["macro_f1"].score
    assert first.metadata["metrics"] == second.metadata["metrics"]


def test_records_are_consumed_in_corpus_order(corpus, artifact_root):
    """§R: ordering is deterministic before the split. Mutation: sort by label."""
    result = run(corpus, artifact_root)
    seen = [r.external_id for period in Period for r in result.periods[period]]
    assert seen == sorted(seen)


# --- boundaries -------------------------------------------------------------------------


def test_no_network_is_used_by_the_experiment(corpus, artifact_root, monkeypatch):
    """§Q: tests never touch the network, and neither does the experiment."""

    def forbidden(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    assert run(corpus, artifact_root).roster == ALPHABETICAL


def test_importing_the_experiment_pulls_in_no_django():
    """Phase 2 boundary: `ml/training/` is Django-independent."""
    probe = (
        "import sys; import ml.training.experiments.triage; "
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


def test_the_serving_registry_does_not_reference_the_experiment():
    """Phase 2 wires nothing into serving."""
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
    assert "triage_tfidf" not in source


def test_this_module_is_selected_by_the_ml_marker(request):
    """Task 16 gives the `ml` marker its first users, so CI's tolerance can go."""
    assert request.node.get_closest_marker("ml") is not None
