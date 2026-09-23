"""Task 18: the duplicate-retrieval benchmark under one frozen config (D38).

RED phase. `ml.training.experiments.dedup` does not exist, so every test that
reaches it fails. The corpus is written into `tmp_path`; nothing is downloaded and
no socket is opened. Tests that need both arms require the pinned MiniLM assets
and skip when they are absent, naming the path they wanted.

The API surface these tests pin, all of it implied by D38 and none of it existing
yet:

    BENCHMARK_CONFIG            the single frozen instance both arms receive
    BenchmarkConfig             its frozen dataclass
    index_population(records, split)   / query_population(records, split, config)
    build_perturbations(records, config)
    build_index(refs, vectors)  -> RetrievalIndex(.dimension, .rank)
    recall_at_k(rankings, expected, k)
    random_ranking_baseline(expected, candidates, k, *, seed)
    EmbeddingDimensionMismatch
    run_benchmark(corpus_root=..., config=...)

`ml/training/metrics.py` is untouched: D34 keeps `recall_at_k` out of it, and
Task 18 owns the retrieval definition instead.
"""

import dataclasses
import hashlib
import importlib
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="numpy lives in requirements/ml.txt")
pytest.importorskip("sklearn", reason="scikit-learn lives in requirements/ml.txt")
pytest.importorskip("pyarrow", reason="pyarrow lives in requirements/train.txt")

from ingest.identity import RecordRef, make_ref  # noqa: E402
from ingest.manifest import build_manifest, load_corpus, write_manifest  # noqa: E402
from ingest.schema import CorpusRecord  # noqa: E402
from ingest.storage import write_partition  # noqa: E402
from ml.training.splits import DEFAULT_FRACTIONS, Period, temporal_split  # noqa: E402

pytestmark = pytest.mark.ml

ROOT = Path(__file__).resolve().parents[3]

# --- D38's frozen benchmark values --------------------------------------------------

SOURCE = "cfpb"
SEED = 18
HEADLINE_K = 10
REPORTED_KS = (1, 5, 10)
QUERY_POPULATION = 500
TUNING_BUDGET = 0
PERTURBATION_TYPES = ("synonym", "truncation", "typo")
TRUNCATION_FRACTION = 0.60
TYPO_RATE = 0.02
SYNONYM_MAX_FRACTION = 0.20
ARMS = ("tfidf", "minilm")

ASSET_DIR_ENV = "SENTINEL_MINILM_DIR"
REQUIRE_ENV = "SENTINEL_REQUIRE_MINILM"
DEFAULT_ASSET_DIR = Path("ml/artifacts/embedders/all_minilm_l6_v2/v1")
ONNX_SHA256 = "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
TOKENIZER_SHA256 = "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"


# --- the production module, imported late -------------------------------------------


def dedup():
    return importlib.import_module("ml.training.experiments.dedup")


def require_real_assets() -> Path:
    """Both-arm tests need the pinned assets; D38 lets them skip, or fail on demand."""
    directory = Path(os.environ.get(ASSET_DIR_ENV) or ROOT / DEFAULT_ASSET_DIR)
    problems = []
    for name, expected in (("model.onnx", ONNX_SHA256), ("tokenizer.json", TOKENIZER_SHA256)):
        path = directory / name
        if not path.is_file():
            problems.append(f"{path} is missing")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            problems.append(f"{path} does not match the pinned digest")
    if not problems:
        return directory
    detail = "; ".join(problems)
    if os.environ.get(REQUIRE_ENV) == "1":
        pytest.fail(f"{REQUIRE_ENV}=1 but the pinned MiniLM assets are unusable: {detail}")
    pytest.skip(f"pinned MiniLM assets unavailable: {detail}")


@pytest.fixture
def assets() -> Path:
    return require_real_assets()


# --- the fixture corpus -------------------------------------------------------------
#
# Forty CFPB records, one hour apart, with period sentinels. At 70/15/15 that is
# 28 train, 6 validation and 6 test, which is enough to make "fitted on training
# text only" and "no future record in the index" observable.

CORPUS_START = datetime(2024, 3, 1, 9, 0, tzinfo=UTC)
WINDOW_START = datetime(2024, 1, 1, tzinfo=UTC)
WINDOW_END = datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC)
RECORD_COUNT = 40
TEST_SENTINEL = "qqqtestonlyqqq"

TIMESTAMP_DIAGNOSTIC = {
    "verdict": "supported_plausible_event_time",
    "reason": "fixture corpus; no provenance measurement is claimed",
}

SUBJECTS = (
    "billing statement dispute with the card issuer",
    "mortgage escrow shortage notice from the loan servicer",
    "credit report inaccurate tradeline from the bureau",
    "wire transfer routing error at the receiving bank",
)


def fixture_records() -> list[CorpusRecord]:
    records = []
    for index in range(RECORD_COUNT):
        text = f"{SUBJECTS[index % len(SUBJECTS)]} case {index:04d}"
        if index >= 34:
            text = f"{text} {TEST_SENTINEL}"
        records.append(
            CorpusRecord(
                source=SOURCE,
                external_id=f"{index:04d}",
                text=text,
                label="Billing" if index % 2 else "Mortgage",
                submitted_at=CORPUS_START + timedelta(hours=index),
            )
        )
    return records


def build_fixture_corpus(root: Path) -> None:
    records = fixture_records()
    half = len(records) // 2
    write_partition(records[:half], SOURCE, 2024, 0, root=root)
    write_partition(records[half:], SOURCE, 2024, 1, root=root)
    manifest = build_manifest(
        SOURCE,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        source_api_version="fixture-v1",
        limit=None,
        timestamp_diagnostic=TIMESTAMP_DIAGNOSTIC,
        root=root,
        ingested_at=datetime(2026, 9, 20, 12, 0, tzinfo=UTC),
    )
    write_manifest(manifest, root=root)


@pytest.fixture
def corpus_root(tmp_path) -> Path:
    root = tmp_path / "corpus"
    build_fixture_corpus(root)
    return root


@pytest.fixture
def records() -> list[CorpusRecord]:
    return fixture_records()


@pytest.fixture
def split(records):
    return temporal_split([record.submitted_at for record in records])


def unit_vectors(count: int, width: int, *, offset: int = 0) -> np.ndarray:
    """`count` distinct L2-normalised rows, deterministic and easy to rank by hand."""
    matrix = np.zeros((count, width), dtype=np.float32)
    for row in range(count):
        matrix[row, (row + offset) % width] = 1.0
    return matrix


def refs(count: int, *, prefix: str = SOURCE) -> list[RecordRef]:
    return [RecordRef(source=prefix, external_id=f"{index:04d}") for index in range(count)]


# --- the frozen configuration -------------------------------------------------------


def test_there_is_exactly_one_frozen_benchmark_config():
    config = dedup().BENCHMARK_CONFIG
    assert isinstance(config, dedup().BenchmarkConfig)


def test_the_config_is_frozen():
    """A mutable config would let one arm edit what the other reads."""
    config = dedup().BENCHMARK_CONFIG
    with pytest.raises(Exception):  # FrozenInstanceError is a subclass of AttributeError
        config.k = 3


def test_the_config_carries_d38_s_values():
    config = dedup().BENCHMARK_CONFIG
    assert config.source == SOURCE
    assert tuple(config.fractions) == tuple(DEFAULT_FRACTIONS) == (0.70, 0.15, 0.15)
    assert config.query_population == QUERY_POPULATION
    assert config.seed == SEED
    assert config.k == HEADLINE_K
    assert tuple(config.reported_ks) == REPORTED_KS
    assert config.tuning_budget == TUNING_BUDGET
    assert tuple(config.perturbation_types) == PERTURBATION_TYPES


def test_the_similarity_is_cosine_over_l2_normalised_vectors():
    """On unit vectors cosine is the dot product; the config's function must agree."""
    similarity = dedup().BENCHMARK_CONFIG.similarity
    left = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    right = np.array([[1.0, 0.0]], dtype=np.float32)
    scores = np.asarray(similarity(right, left)).ravel()
    assert np.allclose(scores, [1.0, 0.0], atol=1e-6)


def test_an_equal_but_distinct_config_is_not_the_frozen_instance():
    """Why the arms are compared by identity: equality cannot tell a copy apart."""
    twin = dataclasses.replace(dedup().BENCHMARK_CONFIG)
    assert twin == dedup().BENCHMARK_CONFIG
    assert twin is not dedup().BENCHMARK_CONFIG


def test_the_tuning_budget_is_zero_for_the_whole_benchmark():
    """§5.3 holds effort equal; D38 sets it to zero rather than to equal searches."""
    assert dedup().BENCHMARK_CONFIG.tuning_budget == 0


# --- temporal boundaries and leakage (§6.2) -----------------------------------------


def test_the_index_holds_no_record_later_than_the_query_period(records, split):
    """Leakage path 6: an index over all periods lets a test query retrieve the future."""
    population = dedup().index_population(records, split)
    latest_test = max(
        record.submitted_at
        for record in records
        if split.period_of(record.submitted_at) is Period.TEST
    )
    assert population, "the index population is empty"
    assert max(record.submitted_at for record in population) <= latest_test


def test_every_query_record_comes_from_the_test_period(records, split):
    config = dedup().BENCHMARK_CONFIG
    queries = dedup().query_population(records, split, config)
    assert queries, "no query records were selected"
    for record in queries:
        assert split.period_of(record.submitted_at) is Period.TEST


def test_every_query_original_is_addressable_in_the_index(records, split):
    """Recall is undefined if the answer is not in the candidate population."""
    config = dedup().BENCHMARK_CONFIG
    indexed = {make_ref(record) for record in dedup().index_population(records, split)}
    for record in dedup().query_population(records, split, config):
        assert make_ref(record) in indexed


def test_the_query_population_is_capped_by_the_configured_size(records, split):
    config = dataclasses.replace(dedup().BENCHMARK_CONFIG, query_population=3)
    assert len(dedup().query_population(records, split, config)) == 3


def test_the_query_selection_is_deterministic(records, split):
    config = dedup().BENCHMARK_CONFIG
    first = [make_ref(r) for r in dedup().query_population(records, split, config)]
    second = [make_ref(r) for r in dedup().query_population(records, split, config)]
    assert first == second


# --- perturbations ------------------------------------------------------------------


def test_exactly_the_three_frozen_perturbation_types_are_produced(records, split):
    config = dedup().BENCHMARK_CONFIG
    queries = dedup().query_population(records, split, config)
    produced = dedup().build_perturbations(queries, config)
    assert set(produced) == set(PERTURBATION_TYPES)


def test_all_three_types_use_the_same_source_records(records, split):
    """D38: the same records receive all three perturbations, so types stay comparable."""
    config = dedup().BENCHMARK_CONFIG
    queries = dedup().query_population(records, split, config)
    produced = dedup().build_perturbations(queries, config)
    expected = [make_ref(record) for record in queries]
    for name, perturbed in produced.items():
        assert [query.ref for query in perturbed] == expected, name


def test_truncation_keeps_the_leading_sixty_percent(records, split):
    config = dedup().BENCHMARK_CONFIG
    queries = dedup().query_population(records, split, config)
    produced = dedup().build_perturbations(queries, config)
    for original, query in zip(queries, produced["truncation"], strict=True):
        expected = max(1, int(len(original.text) * TRUNCATION_FRACTION))
        assert query.text == original.text[:expected]


def test_truncation_never_produces_empty_text():
    """The protocol refuses empty strings, so the perturbation must not create one."""
    config = dedup().BENCHMARK_CONFIG
    one_character = [
        CorpusRecord(
            source=SOURCE,
            external_id="short",
            text="a",
            label="Billing",
            submitted_at=CORPUS_START,
        )
    ]
    produced = dedup().build_perturbations(one_character, config)
    assert produced["truncation"][0].text == "a"


def test_typo_injection_transposes_adjacent_characters(records, split):
    """Two percent of characters, and the multiset of characters is preserved."""
    config = dedup().BENCHMARK_CONFIG
    queries = dedup().query_population(records, split, config)
    produced = dedup().build_perturbations(queries, config)
    for original, query in zip(queries, produced["typo"], strict=True):
        assert len(query.text) == len(original.text)
        assert sorted(query.text) == sorted(original.text)
        changed = sum(1 for a, b in zip(original.text, query.text, strict=True) if a != b)
        assert changed <= 2 * max(1, round(len(original.text) * TYPO_RATE))


def test_synonym_substitution_uses_the_frozen_in_repo_table(records, split):
    """No downloaded lexicon: the table ships with the repository (D38)."""
    module = dedup()
    assert module.SYNONYM_TABLE, "the frozen substitution table is empty"
    config = module.BENCHMARK_CONFIG
    queries = module.query_population(records, split, config)
    produced = module.build_perturbations(queries, config)
    for original, query in zip(queries, produced["synonym"], strict=True):
        original_tokens = original.text.split()
        changed = [
            (before, after)
            for before, after in zip(original_tokens, query.text.split(), strict=True)
            if before != after
        ]
        for before, after in changed:
            assert module.SYNONYM_TABLE.get(before) == after


def test_synonym_substitution_touches_at_most_a_fifth_of_eligible_tokens(records, split):
    module = dedup()
    config = module.BENCHMARK_CONFIG
    queries = module.query_population(records, split, config)
    produced = module.build_perturbations(queries, config)
    for original, query in zip(queries, produced["synonym"], strict=True):
        tokens = original.text.split()
        eligible = [token for token in tokens if token in module.SYNONYM_TABLE]
        changed = sum(1 for a, b in zip(tokens, query.text.split(), strict=True) if a != b)
        assert changed <= max(0, int(len(eligible) * SYNONYM_MAX_FRACTION)) or changed == 0


def test_perturbation_is_deterministic_under_the_frozen_seed(records, split):
    config = dedup().BENCHMARK_CONFIG
    queries = dedup().query_population(records, split, config)
    first = dedup().build_perturbations(queries, config)
    second = dedup().build_perturbations(queries, config)
    for name in PERTURBATION_TYPES:
        assert [q.text for q in first[name]] == [q.text for q in second[name]], name


def test_a_different_seed_produces_different_perturbations(records, split):
    """Guards the guard: a seed that changed nothing would make determinism vacuous."""
    module = dedup()
    config = module.BENCHMARK_CONFIG
    queries = module.query_population(records, split, config)
    frozen = module.build_perturbations(queries, config)
    other = module.build_perturbations(queries, dataclasses.replace(config, seed=SEED + 1))
    assert [q.text for q in frozen["typo"]] != [q.text for q in other["typo"]]


def test_no_perturbation_yields_empty_or_non_string_text(records, split):
    config = dedup().BENCHMARK_CONFIG
    queries = dedup().query_population(records, split, config)
    for name, perturbed in dedup().build_perturbations(queries, config).items():
        for query in perturbed:
            assert isinstance(query.text, str) and query.text.strip(), name


# --- the retrieval index ------------------------------------------------------------


def test_the_index_records_the_dimension_it_was_built_with():
    index = dedup().build_index(refs(4), unit_vectors(4, 6))
    assert index.dimension == 6


def test_a_duplicate_record_ref_in_the_index_raises():
    """A silent de-duplication is a silent change of denominator (D38)."""
    duplicated = [*refs(3), RecordRef(source=SOURCE, external_id="0000")]
    with pytest.raises(ValueError):
        dedup().build_index(duplicated, unit_vectors(4, 6))


def test_a_narrower_embedding_is_refused():
    index = dedup().build_index(refs(4), unit_vectors(4, 6))
    with pytest.raises(dedup().EmbeddingDimensionMismatch):
        index.rank(unit_vectors(1, 4)[0])


def test_a_wider_embedding_is_refused():
    index = dedup().build_index(refs(4), unit_vectors(4, 6))
    with pytest.raises(dedup().EmbeddingDimensionMismatch):
        index.rank(unit_vectors(1, 8)[0])


def test_the_mismatch_error_is_a_value_error():
    """Like the rest of `ml.training`, so a caller can catch the family (D35)."""
    assert issubclass(dedup().EmbeddingDimensionMismatch, ValueError)


def test_the_mismatch_message_names_both_widths():
    index = dedup().build_index(refs(4), unit_vectors(4, 6))
    with pytest.raises(dedup().EmbeddingDimensionMismatch) as raised:
        index.rank(unit_vectors(1, 8)[0])
    message = str(raised.value)
    assert "6" in message and "8" in message


def test_ranking_returns_record_refs_not_positions():
    index = dedup().build_index(refs(4), unit_vectors(4, 6))
    ranked = index.rank(unit_vectors(1, 6, offset=2)[0])
    assert all(isinstance(item, RecordRef) for item in ranked)
    assert ranked[0] == RecordRef(source=SOURCE, external_id="0002")


def test_ranking_covers_the_whole_candidate_population():
    index = dedup().build_index(refs(4), unit_vectors(4, 6))
    assert len(index.rank(unit_vectors(1, 6)[0])) == 4


# --- recall@k -----------------------------------------------------------------------


def test_recall_at_k_counts_a_hit_inside_the_top_k():
    rankings = [refs(5)]
    assert dedup().recall_at_k(rankings, [refs(5)[2]], k=3) == pytest.approx(1.0)


def test_recall_at_k_counts_a_miss_outside_the_top_k():
    rankings = [refs(5)]
    assert dedup().recall_at_k(rankings, [refs(5)[4]], k=3) == pytest.approx(0.0)


def test_recall_at_k_is_the_fraction_of_queries_that_hit():
    candidates = refs(4)
    rankings = [candidates, candidates, candidates, candidates]
    expected = [candidates[0], candidates[0], candidates[3], candidates[3]]
    assert dedup().recall_at_k(rankings, expected, k=1) == pytest.approx(0.5)


def test_identity_is_the_record_ref_and_not_a_position():
    """Two sources numbering records independently must not collide (`ingest.identity`)."""
    ranked = [RecordRef(source="nyc311", external_id="0000")]
    expected = [RecordRef(source="cfpb", external_id="0000")]
    assert dedup().recall_at_k([ranked], expected, k=1) == pytest.approx(0.0)


def test_a_positional_integer_is_not_accepted_as_an_identity():
    with pytest.raises(ValueError):
        dedup().recall_at_k([[0, 1, 2]], [0], k=1)


def test_a_k_wider_than_the_candidate_population_raises():
    """Clamping would make recall trivially perfect; Task 14 refuses the same shape."""
    with pytest.raises(ValueError):
        dedup().recall_at_k([refs(3)], [refs(3)[0]], k=10)


def test_a_zero_or_negative_k_raises():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            dedup().recall_at_k([refs(3)], [refs(3)[0]], k=bad)


def test_mismatched_query_and_expected_counts_raise():
    with pytest.raises(ValueError):
        dedup().recall_at_k([refs(3), refs(3)], [refs(3)[0]], k=1)


def test_recall_at_k_is_absent_from_task_fourteen_s_metrics():
    """D34 keeps retrieval out of `metrics.py`; D38 keeps it that way."""
    metrics = importlib.import_module("ml.training.metrics")
    assert not hasattr(metrics, "recall_at_k")


# --- the baseline -------------------------------------------------------------------


def test_the_baseline_is_a_single_seeded_random_ranking():
    module = dedup()
    candidates = refs(20)
    expected = [candidates[0]] * 5
    first = module.random_ranking_baseline(expected, candidates, k=HEADLINE_K, seed=SEED)
    second = module.random_ranking_baseline(expected, candidates, k=HEADLINE_K, seed=SEED)
    assert first == second


def test_the_baseline_changes_with_its_seed():
    module = dedup()
    candidates = refs(50)
    expected = [candidates[7]] * 20
    assert module.random_ranking_baseline(
        expected, candidates, k=HEADLINE_K, seed=SEED
    ) != module.random_ranking_baseline(expected, candidates, k=HEADLINE_K, seed=SEED + 1)


def test_the_baseline_requires_its_seed():
    with pytest.raises(TypeError):
        dedup().random_ranking_baseline(refs(3), refs(3), k=1)


def test_the_baseline_is_a_recall_in_the_unit_interval():
    module = dedup()
    candidates = refs(30)
    score = module.random_ranking_baseline([candidates[1]] * 10, candidates, k=5, seed=SEED)
    assert 0.0 <= score <= 1.0


# --- the benchmark run --------------------------------------------------------------


def test_the_benchmark_fails_closed_without_the_minilm_assets(corpus_root, tmp_path, monkeypatch):
    """D38: a run never skips an arm, unlike a test."""
    monkeypatch.setenv(ASSET_DIR_ENV, str(tmp_path / "absent"))
    minilm = importlib.import_module("ml.embedders.minilm")
    with pytest.raises(minilm.ModelAssetUnavailable):
        dedup().run_benchmark(corpus_root=corpus_root)


def test_both_arms_receive_the_same_config_object(assets, corpus_root):
    """Identity, not equality: a copy would pass an equality assertion unchanged."""
    report = dedup().run_benchmark(corpus_root=corpus_root)
    config = dedup().BENCHMARK_CONFIG
    for arm in ARMS:
        assert report.arms[arm].config is config, f"{arm} received a different config object"


def test_both_arms_receive_the_same_perturbed_texts(assets, corpus_root):
    """Generated once, outside both arms, so an arm cannot perturb for itself."""
    report = dedup().run_benchmark(corpus_root=corpus_root)
    for name in PERTURBATION_TYPES:
        assert report.arms["tfidf"].queries[name] is report.arms["minilm"].queries[name]


def test_both_arms_report_the_same_k_seed_and_budget(assets, corpus_root):
    report = dedup().run_benchmark(corpus_root=corpus_root)
    tfidf, minilm = report.arms["tfidf"], report.arms["minilm"]
    assert tfidf.config.k == minilm.config.k == HEADLINE_K
    assert tfidf.config.seed == minilm.config.seed == SEED
    assert tfidf.config.tuning_budget == minilm.config.tuning_budget == TUNING_BUDGET


def test_both_arms_search_the_same_index_population(assets, corpus_root):
    report = dedup().run_benchmark(corpus_root=corpus_root)
    assert report.arms["tfidf"].candidate_refs == report.arms["minilm"].candidate_refs


def test_recall_is_reported_per_perturbation_type_and_never_pooled(assets, corpus_root):
    """§5.3 reports each perturbation separately; a pooled figure hides the difference."""
    report = dedup().run_benchmark(corpus_root=corpus_root)
    for arm in ARMS:
        assert set(report.arms[arm].recall) == set(PERTURBATION_TYPES)


def test_every_reported_k_is_present_for_every_type(assets, corpus_root):
    report = dedup().run_benchmark(corpus_root=corpus_root)
    for arm in ARMS:
        for name in PERTURBATION_TYPES:
            assert set(report.arms[arm].recall[name]) == set(REPORTED_KS)


def test_every_recall_figure_carries_its_baseline(assets, corpus_root):
    """Plan §P: a published number beside its baseline, never alone."""
    report = dedup().run_benchmark(corpus_root=corpus_root)
    for arm in ARMS:
        for name in PERTURBATION_TYPES:
            assert set(report.arms[arm].baseline[name]) == set(REPORTED_KS)


def test_the_report_records_the_seed_it_used(assets, corpus_root):
    assert dedup().run_benchmark(corpus_root=corpus_root).seed == SEED


def test_the_report_records_the_corpus_id_it_loaded(assets, corpus_root):
    """D38 makes `corpus_id` run-level provenance, so the figure is tied to exact bytes.

    Read from the manifest independently rather than from the report itself: a
    report echoing its own value would assert nothing about which corpus was read.
    """
    manifest, _ = load_corpus(SOURCE, root=corpus_root)
    report = dedup().run_benchmark(corpus_root=corpus_root)
    assert report.corpus_id == manifest.corpus_id


def test_the_report_is_labelled_synthetic(assets, corpus_root):
    """§5.3: a synthetic duplicate-retrieval benchmark, not real-world accuracy."""
    assert "synthetic" in dedup().run_benchmark(corpus_root=corpus_root).label.lower()


def test_each_arm_records_its_own_provenance(assets, corpus_root):
    """D38's provenance table, including the two deliberate nulls for TF-IDF."""
    report = dedup().run_benchmark(corpus_root=corpus_root)
    tfidf, minilm = report.arms["tfidf"], report.arms["minilm"]

    assert tfidf.model_version == "tfidf_char_wb_3_5_v1"
    assert tfidf.embedding_model_id == "tfidf_char_wb_3_5_v1"
    assert tfidf.embedding_model_sha256 is None
    assert tfidf.tokenizer_sha256 is None

    assert minilm.model_version == "all_minilm_l6_v2_onnx_v1"
    assert minilm.embedding_model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert minilm.embedding_model_sha256 == ONNX_SHA256
    assert minilm.tokenizer_sha256 == TOKENIZER_SHA256


def test_each_arm_records_its_observed_dimension(assets, corpus_root):
    report = dedup().run_benchmark(corpus_root=corpus_root)
    for arm in ARMS:
        assert isinstance(report.arms[arm].embedding_dimension, int)
        assert report.arms[arm].embedding_dimension > 0


def test_the_benchmark_writes_no_artifact(assets, corpus_root, tmp_path):
    """D38: Task 18 publishes a report, not a model. Phase 3 writes the first artifact."""
    dedup().run_benchmark(corpus_root=corpus_root)
    assert not (ROOT / "ml" / "artifacts" / SOURCE).exists()
    assert not list(tmp_path.rglob("model.joblib"))
    assert not list(tmp_path.rglob("metadata.json"))
