"""The TF-IDF arm of the duplicate-retrieval benchmark (D38, addendum §5.3).

One frozen character-n-gram block, fitted on training-period text alone. The
recipe is this benchmark's own: a representation chosen for a classifier is not
thereby a representation for retrieval, so nothing here is inherited from the
Task 16 experiment even though both use `TfidfVectorizer`.

`max_features` is load-bearing rather than cosmetic. It bounds the observed
`embedding_dimension`, which is what lets `embed` return the dense matrix the
`TextEmbedder` protocol promises instead of a sparse one. The width is still an
observation -- read from what the fitted vectoriser actually produced -- and
never a declared constant.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_VERSION = "tfidf_char_wb_3_5_v1"
"""D38's version for this arm: it names the recipe, so changing the recipe
without changing this string is impossible to do quietly."""

EMBEDDING_MODEL_ID = MODEL_VERSION
"""D38's provenance table. This arm has no model file, so its identity is the
frozen configuration below together with the ``corpus_id`` it was fitted on --
there is no digest to record, and the benchmark report writes null for one."""


@dataclass(frozen=True)
class TfidfConfig:
    """The frozen vectoriser parameters. Frozen so an arm cannot retune itself."""

    analyzer: str
    ngram_range: tuple[int, int]
    lowercase: bool
    min_df: int
    max_df: float
    max_features: int
    norm: str
    sublinear_tf: bool
    dtype: type


TFIDF_CONFIG = TfidfConfig(
    analyzer="char_wb",
    ngram_range=(3, 5),
    lowercase=True,
    min_df=2,
    max_df=1.0,
    max_features=2048,
    norm="l2",
    sublinear_tf=False,
    dtype=np.float32,
)
"""D38's values exactly. Every parameter not named here takes the pinned
scikit-learn default, which `dependency_versions` records so a default that
moves between releases is visible rather than silent."""


class TfidfEmbedder:
    """A fitted `TextEmbedder` over character n-grams.

    Constructed only by `fit_tfidf`, and already fitted when it exists: there is
    no public `fit`, because the period an embedder learned from is a decision
    made once, not per call.
    """

    def __init__(self, vectorizer: Any, embedding_dimension: int) -> None:
        self.vectorizer = vectorizer
        self.embedding_dimension = embedding_dimension
        self.model_version = MODEL_VERSION
        self.embedding_model_id = EMBEDDING_MODEL_ID

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Dense, `float32`, L2-normalised, one row per input, in input order.

        `transform` never refits, so evaluation text is projected through the
        training vocabulary rather than extending it (§6.2).
        """
        validated = _validated(texts)
        matrix = self.vectorizer.transform(validated)
        return np.asarray(matrix.toarray(), dtype=self.vectorizer.dtype)


def fit_tfidf(train_texts: Sequence[str], config: TfidfConfig) -> TfidfEmbedder:
    """Fit the vocabulary and IDF weights on `train_texts` and nothing else.

    The construction seam D38 names. Building the vocabulary over the whole
    corpus would leak future token statistics into the past -- §6.2 calls it the
    easiest mistake to make here -- so the caller passes the training period and
    the fitted object never sees another one.
    """
    texts = _validated(train_texts)
    vectorizer = TfidfVectorizer(
        analyzer=config.analyzer,
        ngram_range=config.ngram_range,
        lowercase=config.lowercase,
        min_df=config.min_df,
        max_df=config.max_df,
        max_features=config.max_features,
        norm=config.norm,
        sublinear_tf=config.sublinear_tf,
        dtype=config.dtype,
    )
    # The width comes from what was actually produced, not from `max_features`:
    # a vocabulary smaller than the cap is the ordinary case (D18).
    fitted = vectorizer.fit_transform(texts)
    return TfidfEmbedder(vectorizer, int(fitted.shape[1]))


def _validated(texts: Sequence[str]) -> list[str]:
    """Every element checked before any work, so a refusal embeds nothing.

    A bare string is refused too: it is a sequence of characters, and accepting
    it would silently embed one row per letter.
    """
    if isinstance(texts, str):
        raise ValueError("texts must be a sequence of strings, not a single string")
    values = list(texts)
    for position, text in enumerate(values):
        if not isinstance(text, str):
            raise ValueError(
                f"texts[{position}] is {type(text).__name__}, not a string; "
                "the embedder accepts strings only"
            )
        if not text.strip():
            raise ValueError(
                f"texts[{position}] is empty or whitespace only; an all-zero row "
                "has no defined cosine similarity"
            )
    return values
