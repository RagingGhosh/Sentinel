"""Null implementations.

These are the behavior of the system when no artifacts are installed, and the
fallback when a real model raises. Abstaining is always preferred to guessing.
"""

from ml.base import Match, RiskFeatures, RiskScore, TriagePrediction

NULL_VERSION = "null"


class NullTriageModel:
    def predict(self, text: str) -> TriagePrediction:
        return TriagePrediction(category_slug=None, confidence=0.0, model_version=NULL_VERSION)


class NullDedupIndex:
    def query(self, text: str, k: int) -> list[Match]:
        return []


class NullRiskModel:
    def predict(self, features: RiskFeatures) -> RiskScore:
        return RiskScore(score=0.0, band="unknown", model_version=NULL_VERSION)
