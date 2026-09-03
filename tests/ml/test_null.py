from ml.base import RiskFeatures
from ml.null import NullDedupIndex, NullRiskModel, NullTriageModel

FEATURES = RiskFeatures(
    sla_hours=72,
    category_mean_resolution_hours=40.0,
    category_breach_rate=0.1,
    priority_rank=0.5,
    age_hours=2.0,
    submitted_hour=9,
    submitted_weekday=2,
    text_length=400,
    queue_depth=12,
    assignee_open_count=3,
)


def test_null_triage_abstains_rather_than_guessing():
    prediction = NullTriageModel().predict("my mortgage servicer lost my payment")
    assert prediction.category_slug is None
    assert prediction.confidence == 0.0
    assert prediction.model_version == "null"


def test_null_dedup_returns_no_matches():
    assert NullDedupIndex().query("anything", k=5) == []


def test_null_risk_returns_an_unknown_band():
    score = NullRiskModel().predict(FEATURES)
    assert score.band == "unknown"
    assert score.model_version == "null"


def test_results_are_immutable():
    """Prediction results are evidence. Nothing downstream may mutate them."""
    import dataclasses

    import pytest

    prediction = NullTriageModel().predict("text")
    with pytest.raises(dataclasses.FrozenInstanceError):
        prediction.confidence = 0.99  # type: ignore[misc]
