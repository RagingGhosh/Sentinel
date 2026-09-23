"""Inference interfaces.

Every result object is frozen and carries the model_version that produced it,
so a prediction can always be traced to a specific artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    # Django imports this module at startup, so numpy must never follow it here.
    # Annotations are strings under the future import above, so nothing below
    # needs these at run time (D38).
    from collections.abc import Sequence

    import numpy as np


@dataclass(frozen=True)
class TriagePrediction:
    category_slug: str | None
    confidence: float
    model_version: str


@dataclass(frozen=True)
class Match:
    complaint_id: int
    similarity: float
    model_version: str


@dataclass(frozen=True)
class RiskScore:
    score: float
    band: str
    model_version: str


@dataclass(frozen=True)
class RiskFeatures:
    """Domain-independent by construction.

    Categories enter through how they *behave* (sla_hours, mean resolution,
    breach rate), never through which category they *are*. A category identity
    feature would be meaningless when a model trained on one domain serves
    another.
    """

    sla_hours: int
    category_mean_resolution_hours: float
    category_breach_rate: float
    priority_rank: float
    age_hours: float
    submitted_hour: int
    submitted_weekday: int
    text_length: int
    queue_depth: int
    assignee_open_count: int


class TriageModel(Protocol):
    def predict(self, text: str) -> TriagePrediction: ...


class DedupIndex(Protocol):
    def query(self, text: str, k: int) -> list[Match]: ...


class RiskModel(Protocol):
    def predict(self, features: RiskFeatures) -> RiskScore: ...


class TextEmbedder(Protocol):
    """Text in, vector out, plus enough to say what produced it (§2.2, D38).

    Benchmarked in Phase 2, not served by it: the duplicate-retrieval benchmark
    measures embedders over corpus records, while `DedupIndex` keeps returning
    `Match` over live complaints and is unchanged. Phase 3 wires the winning
    embedder behind it.

    There is deliberately no `fit`. Each implementation is constructed already
    fitted, so an unfitted embedder is not a state a caller can reach, and the
    training text an implementation saw is decided at construction rather than
    at call time. `embedding_dimension` is read from the implementation's own
    output and is never a constant.
    """

    model_version: str
    embedding_dimension: int
    embedding_model_id: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...
