from ml.null import NullDedupIndex, NullRiskModel, NullTriageModel
from ml.registry import get_dedup_index, get_risk_model, get_triage_model, registry_status


def test_missing_artifacts_yield_null_implementations():
    """A fresh clone with no artifacts must still run."""
    assert isinstance(get_triage_model("cfpb"), NullTriageModel)
    assert isinstance(get_dedup_index("cfpb"), NullDedupIndex)
    assert isinstance(get_risk_model("cfpb"), NullRiskModel)


def test_registry_status_reports_null_state_per_domain():
    status = registry_status()
    assert status["cfpb"]["triage"] == "null"
    assert status["nyc311"]["dedup"] == "null"
