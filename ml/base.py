"""Inference interfaces.

Every result object is frozen and carries the model_version that produced it,
so a prediction can always be traced to a specific artifact.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TriagePrediction:
    category_slug: str | None
    confidence: float
    model_version: str


@dataclass(frozen=True)
class Match:
    complaint_id: int
    similarity: float


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
