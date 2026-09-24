"""The synthetic duplicate-retrieval benchmark: TF-IDF against MiniLM (D38, §5.3).

One frozen configuration, two arms, and the representation as the only permitted
difference between them. The configuration is a single object both arms receive,
the perturbed queries are generated once outside both arms, and the candidate
population is built once and searched by both -- so divergence requires editing
what both arms read rather than one arm's code.

The measurement is honest about what it is: perturbed copies of held-out records
retrieved against their originals. That is a synthetic retrieval exercise, not
real-world duplicate-detection accuracy, and the report says so in its own label.

Nothing here writes an artifact. Task 18 publishes a report; Phase 3 writes the
first embedder artifact when it wires the winner behind `DedupIndex`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from ingest.identity import RecordRef, make_ref
from ingest.manifest import load_corpus
from ingest.schema import CorpusRecord
from ml.embedders.minilm import EmbeddingDimensionMismatch, load_minilm
from ml.embedders.tfidf import TFIDF_CONFIG, fit_tfidf
from ml.training.splits import DEFAULT_FRACTIONS, Period, TemporalSplit, temporal_split

__all__ = [
    "ARMS",
    "BENCHMARK_CONFIG",
    "ArmResult",
    "BenchmarkConfig",
    "BenchmarkReport",
    "EmbeddingDimensionMismatch",
    "PerturbedQuery",
    "RetrievalIndex",
    "build_index",
    "build_perturbations",
    "index_population",
    "query_population",
    "random_ranking_baseline",
    "recall_at_k",
    "run_benchmark",
]

SOURCE = "cfpb"
SEED = 18
HEADLINE_K = 10
REPORTED_KS: tuple[int, ...] = (1, 5, 10)
QUERY_POPULATION = 500
TUNING_BUDGET = 0
"""Zero for both arms. §5.3 holds effort equal rather than optimal, and equal at
zero is the only point where that is verifiable rather than argued."""

ARM_TFIDF = "tfidf"
ARM_MINILM = "minilm"
ARMS: tuple[str, ...] = (ARM_TFIDF, ARM_MINILM)

LABEL = "synthetic duplicate-retrieval benchmark"
"""§5.3 requires the result to be labelled for what it is, in the report itself
rather than only in prose around it."""

SYNONYM = "synonym"
TRUNCATION = "truncation"
TYPO = "typo"
PERTURBATION_TYPES: tuple[str, ...] = (SYNONYM, TRUNCATION, TYPO)

TRUNCATION_FRACTION = 0.60
TYPO_RATE = 0.02
SYNONYM_MAX_FRACTION = 0.20

EMBED_BATCH = 64
"""How many texts are handed to an arm at once. Pooling is mask-weighted, so
batching changes no vector; this only bounds peak memory."""

SYNONYM_TABLE: Mapping[str, str] = {
    "account": "profile",
    "agency": "bureau",
    "amount": "sum",
    "bank": "institution",
    "billing": "invoicing",
    "bureau": "agency",
    "called": "phoned",
    "card": "instrument",
    "charge": "fee",
    "company": "firm",
    "complaint": "grievance",
    "credit": "lending",
    "dispute": "challenge",
    "error": "mistake",
    "escrow": "impound",
    "fee": "charge",
    "inaccurate": "incorrect",
    "issuer": "provider",
    "loan": "advance",
    "mortgage": "homeloan",
    "notice": "notification",
    "payment": "remittance",
    "received": "got",
    "report": "record",
    "reported": "recorded",
    "requested": "asked",
    "sent": "mailed",
    "servicer": "administrator",
    "statement": "summary",
    "transfer": "transmission",
    "wire": "cable",
}
"""A frozen, in-repo substitution table. Deliberately small and shipped with the
source: a downloaded lexicon would make the benchmark irreproducible in exactly
the way §5.3 exists to prevent, and would need a dependency D38 does not allow."""


def cosine_similarity(queries: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Cosine over L2-normalised vectors, which on unit rows is the dot product.

    Both arms produce normalised rows by contract, so no renormalisation happens
    here: silently rescaling would hide an arm that stopped normalising.
    """
    return np.asarray(queries, dtype=np.float32) @ np.asarray(candidates, dtype=np.float32).T


@dataclass(frozen=True)
class BenchmarkConfig:
    """Everything held constant across the arms (§5.3's table).

    Frozen, and shared by identity rather than by value: an arm handed a copy
    could drift from the other without any test of equality noticing.
    """

    source: str
    fractions: tuple[float, float, float]
    query_population: int
    seed: int
    k: int
    reported_ks: tuple[int, ...]
    tuning_budget: int
    perturbation_types: tuple[str, ...]
    similarity: Callable[[np.ndarray, np.ndarray], np.ndarray]
    truncation_fraction: float = TRUNCATION_FRACTION
    typo_rate: float = TYPO_RATE
    synonym_max_fraction: float = SYNONYM_MAX_FRACTION


BENCHMARK_CONFIG = BenchmarkConfig(
    source=SOURCE,
    fractions=DEFAULT_FRACTIONS,
    query_population=QUERY_POPULATION,
    seed=SEED,
    k=HEADLINE_K,
    reported_ks=REPORTED_KS,
    tuning_budget=TUNING_BUDGET,
    perturbation_types=PERTURBATION_TYPES,
    similarity=cosine_similarity,
)
"""The one instance. Both arms receive this object, and a test asserts identity
rather than equality, so a per-arm copy is a visible failure."""


@dataclass(frozen=True)
class PerturbedQuery:
    """One perturbed copy, still addressed by the original record's reference."""

    ref: RecordRef
    text: str


# --- the retrieval index ---------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalIndex:
    """Candidate embeddings, addressed by `RecordRef` and by nothing else.

    The width the index was built with is recorded, and every query is checked
    against it: comparing vectors of different widths is either a crash in the
    wrong place or, worse, a broadcast that quietly scores nonsense (D18).
    """

    refs: tuple[RecordRef, ...]
    vectors: np.ndarray
    dimension: int
    similarity: Callable[[np.ndarray, np.ndarray], np.ndarray] = field(default=cosine_similarity)

    def rank(self, vector: np.ndarray) -> tuple[RecordRef, ...]:
        """Every candidate, most similar first, as references."""
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise EmbeddingDimensionMismatch(
                f"this index was built with embeddings of width {self.dimension} "
                f"but was queried with one of width {query.shape[1]}; widths are "
                "never broadcast, truncated or padded to make a comparison work"
            )
        scores = np.asarray(self.similarity(query, self.vectors)).ravel()
        order = np.argsort(-scores, kind="stable")
        return tuple(self.refs[position] for position in order)


def build_index(
    refs: Sequence[RecordRef],
    vectors: np.ndarray,
    *,
    similarity: Callable[[np.ndarray, np.ndarray], np.ndarray] = cosine_similarity,
) -> RetrievalIndex:
    """Bind references to embeddings, refusing a population that repeats one.

    A duplicate reference is refused rather than de-duplicated: silently dropping
    one would change the denominator every recall figure is computed against.

    `similarity` is taken from the shared configuration by the benchmark, so the
    function both arms rank with is one the config actually governs rather than a
    default each index happens to hold.
    """
    references = tuple(refs)
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D embedding matrix, got shape {matrix.shape}")
    if len(references) != matrix.shape[0]:
        raise ValueError(
            f"{len(references)} references but {matrix.shape[0]} embeddings; "
            "every candidate must carry exactly one vector"
        )
    _require_record_refs(references, "the candidate population")
    seen: set[RecordRef] = set()
    for ref in references:
        if ref in seen:
            raise ValueError(
                f"{ref} appears more than once in the candidate population; "
                "a repeated reference would change the recall denominator"
            )
        seen.add(ref)
    return RetrievalIndex(
        refs=references,
        vectors=matrix,
        dimension=int(matrix.shape[1]),
        similarity=similarity,
    )


# --- the metric and its baseline -------------------------------------------------------


def recall_at_k(
    rankings: Sequence[Sequence[RecordRef]],
    expected: Sequence[RecordRef],
    k: int,
) -> float:
    """The fraction of queries whose original reference is in the top `k` (D38).

    Lives here rather than in `ml/training/metrics.py`: D34 left retrieval out of
    Task 14 because a classifier baseline has no meaning for it, and this
    definition is retrieval-specific.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if len(rankings) != len(expected):
        raise ValueError(
            f"{len(rankings)} rankings but {len(expected)} expected references; "
            "every query answers for exactly one original"
        )
    if not rankings:
        raise ValueError("recall@k over no queries describes nothing")
    _require_record_refs(expected, "the expected originals")
    hits = 0
    for ranking, original in zip(rankings, expected, strict=True):
        if k > len(ranking):
            raise ValueError(
                f"k={k} is wider than the candidate population ({len(ranking)}); "
                "clamping it would make recall trivially perfect"
            )
        top = tuple(ranking[:k])
        _require_record_refs(top, "a ranking")
        hits += original in top
    return hits / len(expected)


def random_ranking_baseline(
    expected: Sequence[RecordRef],
    candidates: Sequence[RecordRef],
    k: int,
    *,
    seed: int,
) -> float:
    """Recall@k for one seeded random ranking of the candidate population.

    Exactly one draw, as D34 fixed for the stratified baseline: a single call to
    the generator produces the scores every query is ranked by. No repeated Monte
    Carlo rounds and no analytic k/N stand-in, which D34 explicitly declined to
    invent. `seed` is keyword-only and mandatory, so a published baseline always
    has a reproducible one recorded beside it.
    """
    population = tuple(candidates)
    _require_record_refs(population, "the candidate population")
    _require_record_refs(expected, "the expected originals")
    scores = np.random.default_rng(seed).random((len(expected), len(population)))
    rankings = [
        tuple(population[position] for position in np.argsort(-row, kind="stable"))
        for row in scores
    ]
    return recall_at_k(rankings, expected, k)


# --- populations -----------------------------------------------------------------------


def index_population(
    records: Sequence[CorpusRecord],
    split: TemporalSplit,
    *,
    evaluation_end: datetime,
) -> tuple[CorpusRecord, ...]:
    """Candidates a test-period query may legitimately retrieve (§6.2, path 6).

    The boundary is the **evaluation window's end**, which the benchmark takes
    from the manifest, and never the newest record or the newest sampled query.
    Deriving it from the records would be circular -- the split partitions the
    very records handed in, so the newest of them is always inside it and the
    rule could never exclude anything. Deriving it from the queries would make
    the candidate population depend on which 500 records were sampled, and would
    drop legitimate candidates from later in the same period.

    `split` is consulted only to refuse a boundary that falls before the test
    period has begun: an evaluation window that ends inside validation has no
    test period to query for.
    """
    if evaluation_end <= split.val_end:
        raise ValueError(
            f"the evaluation window ends at {evaluation_end.isoformat()}, at or before "
            f"the validation cut {split.val_end.isoformat()}; there is no test period "
            "left to query for"
        )
    population = tuple(record for record in records if record.submitted_at <= evaluation_end)
    if not population:
        raise ValueError(
            f"no record falls at or before the evaluation window's end "
            f"({evaluation_end.isoformat()}), so there is nothing to retrieve"
        )
    return population


def query_population(
    records: Sequence[CorpusRecord], split: TemporalSplit, config: BenchmarkConfig
) -> tuple[CorpusRecord, ...]:
    """The test-period records that become queries, chosen deterministically.

    Queries come from the test period alone. When the period is larger than the
    configured population the sample is drawn from the configured seed, so both
    arms are handed the same records without either one choosing them.
    """
    period = [record for record in records if split.period_of(record.submitted_at) is Period.TEST]
    if len(period) <= config.query_population:
        return tuple(period)
    chosen = np.random.default_rng(config.seed).choice(
        len(period), size=config.query_population, replace=False
    )
    return tuple(period[position] for position in sorted(int(p) for p in chosen))


# --- perturbations ---------------------------------------------------------------------


def build_perturbations(
    records: Sequence[CorpusRecord], config: BenchmarkConfig
) -> dict[str, tuple[PerturbedQuery, ...]]:
    """The three perturbed copies of every query record, generated once.

    Built here and handed to both arms, never inside one: an arm that perturbed
    for itself is the likeliest way two arms stop measuring the same thing. The
    same records receive all three types, so per-type figures stay comparable
    with each other as well as across arms.
    """
    rng = np.random.default_rng(config.seed)
    return {
        SYNONYM: tuple(
            PerturbedQuery(make_ref(record), _substituted(record.text, rng, config))
            for record in records
        ),
        TRUNCATION: tuple(
            PerturbedQuery(make_ref(record), _truncated(record.text, config)) for record in records
        ),
        TYPO: tuple(
            PerturbedQuery(make_ref(record), _transposed(record.text, rng, config))
            for record in records
        ),
    }


def _substituted(text: str, rng: np.random.Generator, config: BenchmarkConfig) -> str:
    """Replace at most a fifth of the tokens the frozen table knows."""
    tokens = text.split()
    eligible = [position for position, token in enumerate(tokens) if token in SYNONYM_TABLE]
    budget = int(len(eligible) * config.synonym_max_fraction)
    if budget <= 0:
        return text
    chosen = rng.choice(len(eligible), size=budget, replace=False)
    for position in (eligible[int(index)] for index in chosen):
        tokens[position] = SYNONYM_TABLE[tokens[position]]
    return " ".join(tokens)


def _truncated(text: str, config: BenchmarkConfig) -> str:
    """Keep the leading fraction, never less than one character.

    The floor matters: an embedder refuses empty text, so a perturbation that
    could empty a short record would fail the run rather than measure it.
    """
    return text[: max(1, int(len(text) * config.truncation_fraction))]


def _transposed(text: str, rng: np.random.Generator, config: BenchmarkConfig) -> str:
    """Swap adjacent characters at a rate, preserving length and character counts."""
    if len(text) < 2:
        return text
    swaps = max(1, round(len(text) * config.typo_rate))
    characters = list(text)
    positions = rng.choice(len(text) - 1, size=min(swaps, len(text) - 1), replace=False)
    for position in (int(value) for value in positions):
        characters[position], characters[position + 1] = (
            characters[position + 1],
            characters[position],
        )
    return "".join(characters)


# --- the report ------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmResult:
    """One representation's side of the comparison, with its own provenance.

    `config`, `queries` and `candidate_refs` are the shared objects rather than
    copies of them, so a test can assert that both arms really did read the same
    configuration, the same perturbed queries and the same candidates.
    """

    name: str
    config: BenchmarkConfig
    model_version: str
    embedding_model_id: str
    embedding_model_sha256: str | None
    tokenizer_sha256: str | None
    embedding_dimension: int
    candidate_refs: tuple[RecordRef, ...]
    queries: Mapping[str, tuple[PerturbedQuery, ...]]
    recall: Mapping[str, Mapping[int, float]]
    baseline: Mapping[str, Mapping[int, float]]
    truncated_input_count: int | None = None
    """The MiniLM arm's count for this run alone (D38); `None` where truncation
    is not a concept the arm has."""


@dataclass(frozen=True)
class BenchmarkReport:
    """What the benchmark publishes. Not an artifact, and not a model."""

    label: str
    corpus_id: str
    seed: int
    config: BenchmarkConfig
    arms: Mapping[str, ArmResult]


def run_benchmark(
    *,
    corpus_root: str | os.PathLike[str] | None = None,
    config: BenchmarkConfig = BENCHMARK_CONFIG,
) -> BenchmarkReport:
    """Run both arms over one corpus under one configuration.

    Fails closed rather than skipping: both arms are constructed before any
    measurement, so a missing or mismatched MiniLM asset aborts the run instead
    of publishing a one-armed comparison that looks like a comparison.
    """
    manifest, stream = (
        load_corpus(config.source, root=Path(corpus_root))
        if corpus_root is not None
        else load_corpus(config.source)
    )
    records = tuple(stream)
    split = temporal_split([record.submitted_at for record in records], config.fractions)

    # The boundary is the window the manifest recorded at ingest: frozen, and
    # independent of both the records loaded and the queries sampled.
    candidates = index_population(records, split, evaluation_end=manifest.window_end)
    candidate_refs = tuple(make_ref(record) for record in candidates)
    queries = build_perturbations(query_population(records, split, config), config)
    originals = [query.ref for query in queries[config.perturbation_types[0]]]

    train_texts = [
        record.text for record in records if split.period_of(record.submitted_at) is Period.TRAIN
    ]
    # Both arms are constructed up front: the MiniLM assets are verified before
    # any work is done, so an unusable asset costs nothing and hides nothing.
    minilm = load_minilm()
    tfidf = fit_tfidf(train_texts, TFIDF_CONFIG)

    before = minilm.truncated_input_count
    tfidf_result = _arm(
        ARM_TFIDF,
        tfidf,
        config=config,
        candidates=candidates,
        candidate_refs=candidate_refs,
        queries=queries,
        originals=originals,
        embedding_model_sha256=None,
        tokenizer_sha256=None,
    )
    minilm_result = _arm(
        ARM_MINILM,
        minilm,
        config=config,
        candidates=candidates,
        candidate_refs=candidate_refs,
        queries=queries,
        originals=originals,
        embedding_model_sha256=_minilm_constant("ONNX_SHA256"),
        tokenizer_sha256=_minilm_constant("TOKENIZER_SHA256"),
    )
    # Read after the arm has run, not as an argument to it: an argument is
    # evaluated before the call it belongs to, so the difference would be taken
    # before a single input had been embedded and every run would report zero.
    # `before` is this run's starting point, which `load_minilm` leaves at zero,
    # so the dimension probe is excluded and no earlier run can leak in.
    minilm_truncations = minilm.truncated_input_count - before

    return BenchmarkReport(
        label=LABEL,
        corpus_id=manifest.corpus_id,
        seed=config.seed,
        config=config,
        arms={
            ARM_TFIDF: tfidf_result,
            ARM_MINILM: replace(minilm_result, truncated_input_count=minilm_truncations),
        },
    )


def _arm(
    name: str,
    embedder: object,
    *,
    config: BenchmarkConfig,
    candidates: Sequence[CorpusRecord],
    candidate_refs: tuple[RecordRef, ...],
    queries: Mapping[str, tuple[PerturbedQuery, ...]],
    originals: Sequence[RecordRef],
    embedding_model_sha256: str | None,
    tokenizer_sha256: str | None,
    truncated_input_count: int | None = None,
) -> ArmResult:
    """Measure one arm over the shared candidates and the shared perturbed queries."""
    index = build_index(
        candidate_refs,
        _embedded(embedder, [record.text for record in candidates]),
        similarity=config.similarity,
    )
    recall: dict[str, dict[int, float]] = {}
    baseline: dict[str, dict[int, float]] = {}
    for kind in config.perturbation_types:
        perturbed = queries[kind]
        vectors = _embedded(embedder, [query.text for query in perturbed])
        rankings = [index.rank(vector) for vector in vectors]
        recall[kind] = {k: recall_at_k(rankings, originals, k) for k in config.reported_ks}
        baseline[kind] = {
            k: random_ranking_baseline(originals, candidate_refs, k, seed=config.seed)
            for k in config.reported_ks
        }
    return ArmResult(
        name=name,
        config=config,
        model_version=embedder.model_version,  # type: ignore[attr-defined]
        embedding_model_id=embedder.embedding_model_id,  # type: ignore[attr-defined]
        embedding_model_sha256=embedding_model_sha256,
        tokenizer_sha256=tokenizer_sha256,
        embedding_dimension=int(embedder.embedding_dimension),  # type: ignore[attr-defined]
        candidate_refs=candidate_refs,
        queries=queries,
        recall=recall,
        baseline=baseline,
        truncated_input_count=truncated_input_count,
    )


def _embedded(embedder: object, texts: Sequence[str]) -> np.ndarray:
    """Embed in batches, which bounds memory and changes no vector."""
    batches = [
        np.asarray(embedder.embed(list(texts[start : start + EMBED_BATCH])))  # type: ignore[attr-defined]
        for start in range(0, len(texts), EMBED_BATCH)
    ]
    return np.vstack(batches) if batches else np.zeros((0, 0), dtype=np.float32)


def _minilm_constant(name: str) -> str:
    """The pinned digest the arm was verified against, read from its own module."""
    import ml.embedders.minilm as module

    return str(getattr(module, name))


def _require_record_refs(values: Iterable[object], what: str) -> None:
    """Identity is a `RecordRef` throughout, never a position in some array.

    A positional identity would silently collide across sources, which is exactly
    what `ingest.identity` exists to prevent.
    """
    for value in values:
        if not isinstance(value, RecordRef):
            raise ValueError(
                f"{what} must hold RecordRef values, got {type(value).__name__}; "
                "a positional index is not an identity"
            )
