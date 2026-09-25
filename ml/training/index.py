"""The retrieval index, separated from corpus and Parquet I/O (D40.1).

`cosine_similarity`, `RetrievalIndex` and `build_index` were Task 18's, defined
beside the benchmark that first used them in
`ml/training/experiments/dedup.py`. That module imports `ingest.manifest` to load
a corpus, which imports `ingest.storage`, which imports `pyarrow` -- so the index
could not be imported at all without the training dependency tier installed.

Task 20 must measure index construction and query cost in an environment built
from `requirements/ml.txt` alone, where `pandas` and `pyarrow` are absent by
design and the measurement harness refuses to run if either is importable. This
module exists **solely** so that environment can measure the real Sentinel index
rather than a copy of it: the definitions moved unchanged, `dedup.py` imports
them from here, and there is one implementation in one place.

Nothing else moved. `recall_at_k` and `random_ranking_baseline` stay with the
benchmark, because Task 20 measures cost and not quality.

**`EmbeddingDimensionMismatch` stays in `ml/embedders/minilm.py`, and `rank`
resolves it on that module rather than capturing it.** The embedder's own test
suite reloads the module, which rebinds the class to a new object; a class
captured here at import time would then differ from the one a caller reads off
the module, and `except` would miss it. Resolving at raise time is what keeps the
exception Task 18's tests catch the same one this module raises.

NumPy and the standard library only, plus `RecordRef` and that one embedder
module. No `ingest.manifest`, no `ingest.storage`, no `pandas`, no `pyarrow`, no
Django. Training-side like the rest of `ml.training`: nothing here is imported by
serving.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ingest.identity import RecordRef
from ml.embedders import minilm

__all__ = ["RetrievalIndex", "build_index", "cosine_similarity"]


def cosine_similarity(queries: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Cosine over L2-normalised vectors, which on unit rows is the dot product.

    Both arms produce normalised rows by contract, so no renormalisation happens
    here: silently rescaling would hide an arm that stopped normalising.
    """
    return np.asarray(queries, dtype=np.float32) @ np.asarray(candidates, dtype=np.float32).T


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
            # Resolved on the module at raise time, not captured at import time.
            # `tests/ml/embedders/test_minilm.py` reloads that module, which rebinds
            # the class to a new object; a captured one would then differ from the
            # class a caller reads, and `except` would miss it. The exception type
            # and its message are unchanged -- this is how they stay unchanged.
            raise minilm.EmbeddingDimensionMismatch(
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
